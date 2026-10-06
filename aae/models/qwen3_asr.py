"""Qwen3-ASR-1.7B (Qwen, Apache-2.0) through transformers' native implementation (main environment)."""
from __future__ import annotations

import numpy as np
import torch

from aae.models.base import Adapter, new_tokens_limit, target_labels


class Qwen3ASRAdapter(Adapter):
    LANG_NAME = {"en": "English", "zh": "Chinese"}

    def __init__(self, device: str, path: str, dtype=torch.float16):
        super().__init__(device)
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(path)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        tok = self.processor.tokenizer
        self.eos_ids = [151643, 151645]                   # <|endoftext|>, <|im_end|>
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else 151645

    def _inputs(self, wavs, prefills=None, padding_side="left"):
        self.processor.tokenizer.padding_side = padding_side
        audio = [w.astype(np.float32) for w in wavs]
        if prefills is None:
            inputs = self.processor.apply_transcription_request(audio=audio, padding=True)
        else:
            convs = [[{"role": "user", "content": [{"type": "audio", "audio": a}]},
                      {"role": "assistant", "content": [{"type": "text", "text": p}]}]
                     for a, p in zip(audio, prefills, strict=True)]
            inputs = self.processor.apply_chat_template(convs, tokenize=True, return_dict=True,
                                                        continue_final_message=True, padding=True)
        inputs = inputs.to(self.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.dtype)
        return inputs

    @staticmethod
    def extract(raw: str) -> str:
        return raw.split("<asr_text>", 1)[1].strip() if "<asr_text>" in raw else raw.strip()

    @torch.no_grad()
    def transcribe(self, wavs):
        inputs = self._inputs(wavs, padding_side="left")
        out = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                  max_new_tokens=new_tokens_limit(max(len(w) for w in wavs)),
                                  eos_token_id=self.eos_ids, pad_token_id=self.pad_id)
        raws = self.processor.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
        texts = []
        for r in raws:
            for stop in ("<|im_end|>", "<|endoftext|>"):
                r = r.split(stop, 1)[0]
            texts.append(self.extract(r))
        return texts

    def loss(self, wavs, texts, langs=None):
        """CE over the assistant turn: 'language <Lang><asr_text>TEXT' for speech, 'language None<asr_text>' for ""."""
        langs = langs or ["en"] * len(wavs)
        targets = [(f"language {self.LANG_NAME.get(lang, 'English')}<asr_text>{t}" if t else "language None<asr_text>")
                   for t, lang in zip(texts, langs, strict=True)]
        prompt = self._inputs(wavs, prefills=[""] * len(wavs), padding_side="right")
        full = self._inputs(wavs, prefills=[t + "<|im_end|>" for t in targets], padding_side="right")
        return self.model(**full, labels=target_labels(full, prompt)).loss
