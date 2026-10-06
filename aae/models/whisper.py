"""Whisper large-v3 (OpenAI, MIT) through the openai-whisper package.

Decoding: one pass of ``whisper.decode``, greedy (temperature 0), language detected per clip, timestamp tokens on; no
temperature fallback and no skip rule.
With timestamps on, an empty transcript is ``<|0.00|><|endoftext|>``; without them Whisper cannot end a transcript
before its first word, so the hallucination rate would be 100% by construction.
"""
from __future__ import annotations

import numpy as np
import torch

from aae.models.base import Adapter

LANG_TOKEN = {"en": "en", "zh": "zh"}


class WhisperAdapter(Adapter):
    def __init__(self, device: str, path: str):
        super().__init__(device)
        import encodings.cp437  # noqa: F401  (imported lazily inside whisper's mel code)

        import whisper
        from whisper.audio import mel_filters
        from whisper.tokenizer import get_tokenizer

        self.whisper = whisper
        model = whisper.load_model(path, device="cpu")   # cast on the CPU: fp32 on the GPU costs 6 GB
        for m in model.modules():                         # LayerNorm stays fp32, as in whisper's own fp16 path
            if isinstance(m, torch.nn.Linear | torch.nn.Conv1d | torch.nn.Embedding):
                m.half()
        self.model = model.to(self.device).eval()
        self.tok = get_tokenizer(multilingual=True, num_languages=100, language="en", task="transcribe")
        self.n_mels = self.model.dims.n_mels
        for dev in {str(self.device), "cpu"}:
            mel_filters(dev, self.n_mels)

    def _mel(self, wav: np.ndarray) -> torch.Tensor:
        from whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim

        a = pad_or_trim(torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)), N_SAMPLES)
        return log_mel_spectrogram(a, n_mels=self.n_mels)

    def mels(self, wavs) -> torch.Tensor:
        """Per-clip log-mels (whisper normalizes each input by its own maximum), stacked as fp16."""
        return torch.stack([self._mel(w) for w in wavs]).to(self.device).half()

    def text(self, token_ids) -> str:
        """Text tokens only: stop at end-of-text, drop sot/language/task/timestamp tokens, strip."""
        out = []
        for t in token_ids:
            t = int(t)
            if t == self.tok.eot:
                break
            if t < self.tok.eot:
                out.append(t)
        return self.tok.decode(out).strip()

    def decoding_options(self):
        from whisper.decoding import DecodingOptions

        return DecodingOptions(task="transcribe", language=None, temperature=0.0, beam_size=None, best_of=None,
                               fp16=True, without_timestamps=False)

    @torch.no_grad()
    def decode(self, wavs):
        res = self.whisper.decode(self.model, self.mels(wavs), self.decoding_options())
        return res if isinstance(res, list) else [res]

    def transcribe(self, wavs: list[np.ndarray]) -> list[str]:
        return [self.text(r.tokens) for r in self.decode(wavs)]

    @torch.no_grad()
    def transcribe_default(self, wavs: list[np.ndarray]) -> list[str]:
        """Whisper's default pipeline, ``whisper.transcribe`` with its default arguments, one clip at a time:
        temperature fallback (0 to 1 in steps of 0.2; compression ratio above 2.4 or average log-probability below -1),
        the skip rule (no-speech probability above 0.6 unless the average log-probability exceeds -1), conditioning
        on previous text, language detection and timestamps."""
        return [self.whisper.transcribe(self.model, np.ascontiguousarray(w, dtype=np.float32), verbose=None,
                                        fp16=True)["text"].strip() for w in wavs]

    def sequences(self, texts, langs):
        """Token ids ``<|sot|><|lang|><|transcribe|><|0.00|> TEXT <|eot|>`` and the labels of the target span (-100
        elsewhere); for "" the target is ``<|0.00|><|endoftext|>``, exactly the empty transcript decoding can produce."""
        ts0 = self.tok.timestamp_begin
        seqs = []
        for txt, lang in zip(texts, langs, strict=True):
            sot = [self.tok.sot, self.tok.to_language_token(LANG_TOKEN.get(lang, "en")), self.tok.transcribe, ts0]
            body = self.tok.encoding.encode(" " + txt.strip()) if txt.strip() else []
            seqs.append((sot, body + [self.tok.eot]))
        width = max(len(a) + len(b) for a, b in seqs)
        ids = torch.full((len(seqs), width), self.tok.eot, dtype=torch.long, device=self.device)
        labels = torch.full((len(seqs), width), -100, dtype=torch.long, device=self.device)
        for i, (sot, tgt) in enumerate(seqs):
            row = sot + tgt
            ids[i, : len(row)] = torch.tensor(row, device=self.device)
            labels[i, len(sot) - 1: len(row) - 1] = torch.tensor(tgt, device=self.device)
        return ids, labels

    def loss(self, wavs, texts, langs=None):
        """Teacher-forced CE over the target span of ``sequences``."""
        ids, labels = self.sequences(texts, langs or ["en"] * len(wavs))
        logits = self.model(tokens=ids, mel=self.mels(wavs))
        return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                                 ignore_index=-100)

    def decoder_logits(self, wavs, texts, langs, features=None):
        """Teacher-forced logits along ``sequences`` and their labels, plus the encoder features; pass ``features``
        back in to run the decoder again on the same input (a decoder edit leaves the features unchanged)."""
        ids, labels = self.sequences(texts, langs)
        if features is None:
            with torch.no_grad():
                features = self.model.encoder(self.mels(wavs))
        return self.model.decoder(ids, features), labels, features
