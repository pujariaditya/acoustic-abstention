"""Hojo-ASR-V1 (HojoAI, Apache-2.0): Qwen3-Omni audio encoder, Conformer bottleneck, Qwen3-4B decoder.

Runs in the legacy environment with the ``hojo-asr`` package. The package defaults to 4-beam search and to
flash-attention in the encoder; here decoding is greedy like every other model, and the encoder uses SDPA so the model
also runs on GPUs without flash-attention.
"""
from __future__ import annotations

import numpy as np
import torch

from aae.models.base import Adapter, new_tokens_limit


class HojoAdapter(Adapter):
    def __init__(self, device: str, path: str, dtype=torch.float16):
        super().__init__(device)
        from hojo_asr import HOJO_ASR

        m = HOJO_ASR.load_model(path, device="cpu")
        m.speech_encoder.config._attn_implementation = "sdpa"
        self.model = m.to(dtype).to(self.device).eval()
        self.dtype = dtype
        self.tok = m.tokenizer
        self.eos_id = self.tok.eos_token_id

    def _speech(self, wavs):
        fe = self.model.feat_extractor
        feats = [fe(w.astype(np.float32), sampling_rate=16000, return_tensors="pt", padding=False)
                 .input_features[0].transpose(0, 1) for w in wavs]
        lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.int64)
        spec = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0).to(self.device, self.dtype)
        emb, attn = self.model.encode_speech(spec, lens.to(self.device))
        bos = self.model.decoder_model.model.embed_tokens(
            torch.full((len(wavs), 1), self.model.bos_token_id, dtype=torch.long, device=self.device))
        return torch.cat([bos, emb.to(bos.dtype)], dim=1), torch.cat([attn[:, :1], attn], dim=1).long()

    @torch.no_grad()
    def transcribe(self, wavs):
        embeds, mask = self._speech(wavs)
        limit = min(new_tokens_limit(max(len(w) for w in wavs)), int(embeds.shape[1] * 2) + 10)
        out = self.model.decoder_model.generate(inputs_embeds=embeds, attention_mask=mask, do_sample=False, num_beams=1,
                                                max_new_tokens=max(limit, 10), eos_token_id=self.eos_id,
                                                pad_token_id=self.tok.pad_token_id)
        return [t.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                for t in self.tok.batch_decode(out, skip_special_tokens=True)]

    @torch.no_grad()
    def transcribe_default(self, wavs):
        """The released generation settings of ``config.yaml`` (as ``HOJO_ASR`` applies them): 4-beam search,
        repetition penalty 2.0, length penalty 1, at least one new token, at most min(200, 2 x speech frames + 10)."""
        embeds, mask = self._speech(wavs)
        limit = max(min(200, int((embeds.shape[1] - 1) * 2) + 10), 10)
        out = self.model.decoder_model.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=limit,
                                                eos_token_id=self.eos_id, pad_token_id=self.tok.pad_token_id,
                                                num_beams=4, do_sample=False, min_length=1, temperature=1.0, top_p=0.9,
                                                repetition_penalty=2.0, length_penalty=1.0)
        return [t.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                for t in self.tok.batch_decode(out, skip_special_tokens=True)]

    def loss(self, wavs, texts, langs=None):
        embeds, mask = self._speech(wavs)
        tgt = [self.tok(t, add_special_tokens=False)["input_ids"] + [self.eos_id] for t in texts]
        width = max(len(t) for t in tgt)
        B, P = mask.shape
        ids = torch.full((B, width), self.tok.pad_token_id, dtype=torch.long, device=self.device)
        tmask = torch.zeros((B, width), dtype=torch.long, device=self.device)
        for i, t in enumerate(tgt):
            ids[i, : len(t)] = torch.tensor(t, device=self.device)
            tmask[i, : len(t)] = 1
        full = torch.cat([embeds, self.model.decoder_model.model.embed_tokens(ids).to(embeds.dtype)], dim=1)
        fmask = torch.cat([mask, tmask], dim=1)
        labels = torch.cat([torch.full((B, P), -100, dtype=torch.long, device=self.device),
                            ids.masked_fill(tmask == 0, -100)], dim=1)
        pos = (fmask.cumsum(1) - 1).clamp(min=0)
        return self.model.decoder_model(inputs_embeds=full, attention_mask=fmask, position_ids=pos, labels=labels).loss
