"""ARK-ASR-3B (AutoArk-AI, now Edge0; Apache-2.0): Whisper-style encoder, MLP adapter, Qwen decoder.

Remote code written for transformers 4.x, so it runs in the legacy environment. Prompt and decoding follow the model
card (greedy, the same special and added tokens banned); the generation cap is ``new_tokens_limit`` instead of the
card's 256 new tokens.
"""
from __future__ import annotations

import numpy as np
import torch

from aae.models.base import Adapter, new_tokens_limit


def right_pad_concat(prompt_ids, prompt_mask, target_ids: list[list[int]], pad_id: int):
    """Per row: the real prompt tokens, then the target tokens, right-padded; labels cover the target only."""
    rows, labs = [], []
    for i, tgt in enumerate(target_ids):
        real = prompt_ids[i][prompt_mask[i].bool()].tolist()
        rows.append(real + list(tgt))
        labs.append([-100] * len(real) + list(tgt))
    width = max(len(r) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, (r, lab) in enumerate(zip(rows, labs, strict=True)):
        ids[i, : len(r)] = torch.tensor(r)
        labels[i, : len(r)] = torch.tensor(lab)
        mask[i, : len(r)] = 1
    return ids, mask, labels


class ARKAdapter(Adapter):
    INSTRUCTION = "Please transcribe this audio."

    def __init__(self, device: str, path: str, dtype=torch.float16):
        super().__init__(device)
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=dtype,
                                                          attn_implementation="sdpa").to(self.device).eval()
        self.dtype = dtype
        self.eos_id = self.tokenizer.eos_token_id
        banned = set(self.tokenizer.all_special_ids) - {self.eos_id}
        banned.update(tid for tok, tid in self.tokenizer.get_added_vocab().items()
                      if tok.startswith("<") and tok.endswith(">") and tid != self.eos_id)
        self.bad_words_ids = [[t] for t in sorted(banned)]

    def _prompt(self, wavs):
        conv = [[{"role": "user", "content": [{"type": "audio", "array": w.astype(np.float32)},
                                              {"type": "text", "text": self.INSTRUCTION}]}] for w in wavs]
        self.tokenizer.padding_side = "left"
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt",
                                                    sampling_rate=16000, audio_padding="longest",
                                                    text_kwargs={"padding": "longest"}, audio_max_length=30 * 16000)
        inputs = inputs.to(self.device)
        inputs["audios"] = inputs["audios"].to(self.dtype)
        return inputs

    @torch.no_grad()
    def transcribe(self, wavs):
        inputs = self._prompt(wavs)
        out = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                  max_new_tokens=new_tokens_limit(max(len(w) for w in wavs)),
                                  pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.eos_id,
                                  bad_words_ids=self.bad_words_ids)
        return [t.strip() for t in self.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                                                skip_special_tokens=True)]

    def loss(self, wavs, texts, langs=None):
        p = self._prompt(wavs)
        tgt = [self.tokenizer(t, add_special_tokens=False)["input_ids"] + [self.eos_id] for t in texts]
        ids, mask, labels = right_pad_concat(p["input_ids"].cpu(), p["attention_mask"].cpu(), tgt,
                                             self.tokenizer.pad_token_id)
        return self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), audios=p["audios"],
                          labels=labels.to(self.device)).loss
