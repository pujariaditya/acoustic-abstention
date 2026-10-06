"""Encoder edit versus decoder edit on Whisper large-v3: the margin argument of docs/RESULTS.md (section 5), measured.

    python -m aae.locus --out runs/locus/whisper-large-v3.json

Inputs are the complete LibriSpeech test-clean and test-other sets (speech) and the complete UrbanSound8K set
(non-speech). The original model decodes each utterance with the evaluation protocol (greedy, temperature 0, language
detection, timestamps). Three edited models then run on the same inputs, teacher-forced along the original model's
token path, so every edit is compared with the original on identical prefixes:

  aae               encoder layers 16 to 31, abstention + feature preservation (the released model)
  transcript-ce     encoder layers 16 to 31, abstention + transcript cross-entropy on m (ablation)
  decoder-adapter   self-attention of all 32 decoder layers, abstention + transcript cross-entropy on m (ablation)

Speech, per utterance and step k of the original path: the original margin gamma_k (filtered logit of the chosen token
minus the best other filtered logit), each edit's largest logit change delta_k = max_v |z'_v - z_v|, the KL divergence
between the original and the edited next-token distributions, and whether the edit's filtered argmax is still the
original token (if it is at every step, the edit reproduces the original greedy transcript exactly); the same over the
word steps alone (the original token is a word piece, not a timestamp or end-of-text). Decision step, for every
UrbanSound8K clip and again for every speech utterance: the margin g of the best text token over end-of-text at the
step after <|sot|><|lang|><|transcribe|><|0.00|>, where the empty transcript is decided (g > 0: the model writes
text), and the KL divergence there. For the encoder edits, the relative feature drift
sqrt(mean_t ||F_t - F0_t||^2 / ||F0_t||^2) over the frames that cover the audio, and ||F - F0|| / ||F0|| over the
whole 30-s window that the decoder reads.

Runs in envs/main; about 2.5 hours on one Quadro RTX 8000 shared with other jobs (5,559 utterances, 8,732 clips).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from aae import audio, models, weights
from aae.evaluate import MANIFESTS
from aae.prepare import read_jsonl

MODEL = "whisper-large-v3"
EDITS = {"aae": f"{MODEL}/aae", "transcript-ce": f"{MODEL}/transcript-ce",
         "decoder-adapter": f"{MODEL}/decoder-adapter"}
ENCODER_EDITS = ("aae", "transcript-ce")            # the decoder edit leaves the features unchanged by construction
ROWS = ("original", *EDITS)                          # row order of the stacked logits
SPEECH_SETS = ("ls_clean", "ls_other")
FRAMES_PER_S = 50                                    # Whisper's encoder: 1,500 frames per 30 s
# UrbanSound8K clip ids are fsID-classID-occurrence-slice
URBAN_CLASSES = ("air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling",
                 "gun_shot", "jackhammer", "siren", "street_music")


def urban_class(clip_id: str) -> str:
    return URBAN_CLASSES[int(clip_id.split("-")[1])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    from safetensors.torch import load_file
    from whisper.decoding import DecodingOptions, DecodingTask, detect_language

    torch.manual_seed(0)
    ad = models.load(MODEL, models.cuda())
    model, dev, tk = ad.model, ad.device, ad.tok
    params = dict(model.named_parameters())
    tensors = {}                       # edited weights, kept in pinned CPU memory (the GPUs may be shared)
    for e, ref in EDITS.items():
        loaded = load_file(str(weights.resolve(ref) / "model.safetensors"))
        tensors[e] = {k: v.to(dtype=params[k].dtype).pin_memory() for k, v in loaded.items()}
    keys = sorted({k for t in tensors.values() for k in t})
    original = {k: params[k].detach().cpu().pin_memory() for k in keys}
    state = {"cur": None}

    def use(edit):
        """Load one edit's weights into the model (None: the original model); only the touched tensors are copied."""
        if state["cur"] == edit:
            return
        with torch.no_grad():
            if state["cur"] is not None:
                for k in tensors[state["cur"]]:
                    params[k].copy_(original[k], non_blocking=True)
            if edit is not None:
                for k, v in tensors[edit].items():
                    params[k].copy_(v, non_blocking=True)
        torch.cuda.synchronize()
        state["cur"] = edit

    opts = ad.decoding_options()
    task = DecodingTask(model, DecodingOptions(task="transcribe", language="en", temperature=0.0, fp16=True,
                                               without_timestamps=False))
    filters, sample_begin = task.logit_filters, task.sample_begin
    assert sample_begin == 3
    eot, ts0 = tk.eot, tk.timestamp_begin

    def prompt(lang):
        return [tk.sot, tk.to_language_token(lang), tk.transcribe]

    def filter_rows(z, toks):
        """Whisper's decoding filters applied to one step's stacked logits (rows: models) for a shared prefix.
        The prefix stays on the CPU, so the rules that read it do not synchronize with the GPU."""
        z = z.clone()
        t = toks[None].expand(z.shape[0], -1)
        for f in filters:
            f.apply(z, t)
        return z

    def drifts(fe, f0, nfr):
        """Relative feature drift: per frame over the frames that cover the audio (the form of the preservation loss),
        and over the whole 30-s window that the decoder attends to, ||F - F0|| / ||F0||."""
        d = (fe.float() - f0.float()).pow(2).sum(-1)
        r = f0.float().pow(2).sum(-1).clamp_min(1e-12)
        mask = (torch.arange(d.shape[1], device=dev)[None] < torch.tensor(nfr, device=dev)[:, None]).float()
        on_audio = torch.sqrt((d / r * mask).sum(-1) / mask.sum(-1))
        window = torch.sqrt(d.sum(-1) / r.sum(-1))
        return [{"audio": x, "window": y} for x, y in zip(on_audio.tolist(), window.tolist(), strict=True)]

    def stacked_logits(inp, f0, mel, nfr, last_only=False):
        """Logits of the original model and of each edit on the same token inputs; drift of the encoder edits."""
        pick = (lambda z: z[:, -1].float()) if last_only else (lambda z: z.float())
        z, drf = {}, {}
        with torch.no_grad():
            use(None)
            z["original"] = pick(model.decoder(inp, f0))
            for e in ENCODER_EDITS:
                use(e)
                fe = model.embed_audio(mel)
                z[e] = pick(model.decoder(inp, fe))
                drf[e] = drifts(fe, f0, [min(1500, n) for n in nfr])
                del fe
            use("decoder-adapter")
            z["decoder-adapter"] = pick(model.decoder(inp, f0))
            use(None)
        return torch.stack([z[k] for k in ROWS]), drf

    def batches(rows):
        for b in range(0, len(rows), args.batch):
            chunk = rows[b: b + args.batch]
            wavs = [audio.load(r["staged"]) for r in chunk]
            yield chunk, ad.mels(wavs), [int(math.ceil(len(w) / audio.SAMPLE_RATE * FRAMES_PER_S)) for w in wavs]

    def decision_records(name):
        """Per input of a set: the margin of the best text token over end-of-text at the step after
        <|sot|><|lang|><|transcribe|><|0.00|>, where the empty transcript is decided, for the original model and
        each edit; the KL divergence at that step; and the feature drift of the encoder edits."""
        rows = read_jsonl(MANIFESTS / f"{name}.jsonl")
        recs = []
        for chunk, mel, nfr in batches(rows):
            use(None)
            with torch.no_grad():
                f0 = model.embed_audio(mel)
                lang_tokens, _ = detect_language(model, f0, tk)
            langs = [tk.decode([int(t)]).strip("<|>") for t in lang_tokens]
            seqs = [prompt(lang) + [ts0] for lang in langs]
            z, drf = stacked_logits(torch.tensor(seqs, device=dev), f0, mel, nfr, last_only=True)    # (4, B, V)
            with torch.no_grad():
                lp0 = z[0].log_softmax(-1)
                kl1 = torch.stack([(lp0.exp() * (lp0 - z[q].log_softmax(-1))).sum(-1)
                                   for q in range(1, len(ROWS))]).tolist()
            for i, (r, s) in enumerate(zip(chunk, seqs, strict=True)):
                zf = filter_rows(z[:, i], torch.tensor(s))
                zt = zf.clone()
                zt[:, eot] = -math.inf
                g = (zt.amax(-1) - zf[:, eot]).tolist()                   # > 0: a text token beats end-of-text
                recs.append({"id": r["id"], "lang": langs[i], "text_margin": dict(zip(ROWS, g, strict=True)),
                             "kl": {e: kl1[q][i] for q, e in enumerate(EDITS)},
                             "drift": {e: drf[e][i] for e in ENCODER_EDITS}})
                if name == "urban":
                    recs[-1]["class"] = urban_class(r["id"])
            del z
            if len(recs) % (50 * args.batch) < args.batch:
                print(json.dumps({"set": name, "decision": True, "done": len(recs), "of": len(rows)}), flush=True)
        return recs

    def path_records(name):
        """Per utterance: margins, logit changes, KL divergences and reproduced choices along the original path."""
        rows = read_jsonl(MANIFESTS / f"{name}.jsonl")
        recs = []
        for chunk, mel, nfr in batches(rows):
            use(None)
            with torch.no_grad():
                res = ad.whisper.decode(model, mel, opts)
                f0 = model.embed_audio(mel)
            res = res if isinstance(res, list) else [res]
            seqs = [prompt(r.language) + list(r.tokens) + [eot] for r in res]
            width = max(len(s) for s in seqs) - 1
            inp = torch.full((len(seqs), width), eot, dtype=torch.long, device=dev)
            for i, s in enumerate(seqs):
                inp[i, : len(s) - 1] = torch.tensor(s[:-1], device=dev)
            z, drf = stacked_logits(inp, f0, mel, nfr)                 # (4, B, W, V)
            with torch.no_grad():
                lp0 = z[0].log_softmax(-1)
                klv = torch.stack([(lp0.exp() * (lp0 - z[q].log_softmax(-1))).sum(-1)
                                   for q in range(1, len(ROWS))])                                 # (3, B, W)
                dlv = torch.stack([(z[q] - z[0]).abs().amax(-1) for q in range(1, len(ROWS))])    # (3, B, W)
                del lp0
            for i, (r, s) in enumerate(zip(chunk, seqs, strict=True)):
                st = torch.tensor(s)
                js = list(range(sample_begin - 1, len(s) - 1))
                chosen = torch.tensor(s[sample_begin:], device=dev)
                am, mg = [], []
                for j in js:
                    zf = filter_rows(z[:, i, j], st[: j + 1])          # (4, V)
                    am.append(zf.argmax(-1))
                    z0 = zf[0].clone()
                    c = s[j + 1]
                    top = z0[c].clone()
                    z0[c] = -math.inf
                    mg.append(top - z0.max())
                am = torch.stack(am)                                   # (steps, 4)
                hit = (am == chosen[:, None]).all(0).tolist()          # per row: every greedy choice reproduced
                word = [t < eot for t in s[sample_begin:]]             # steps whose original token is a word piece,
                wm = torch.tensor(word, device=dev)                    # not a timestamp or end-of-text
                hit_w = ((am == chosen[:, None]) | ~wm[:, None]).all(0).tolist()
                gam = torch.stack(mg).tolist()
                dl = dlv[:, i, js].tolist()
                kv = klv[:, i, js].tolist()

                def words(xs, word=word):
                    return [x for x, w in zip(xs, word, strict=True) if w] or [0.0]

                recs.append({"id": r["id"], "lang": res[i].language, "n_tokens": len(js), "tf_ok": hit[0],
                             "text": ad.text(res[i].tokens), "gamma_min": min(gam),
                             "delta_max": {e: max(dl[q]) for q, e in enumerate(EDITS)},
                             "kl_mean": {e: float(np.mean(kv[q])) for q, e in enumerate(EDITS)},
                             "kl_max": {e: max(kv[q]) for q, e in enumerate(EDITS)},
                             "certified": {e: all(d < g / 2 for d, g in zip(dl[q], gam, strict=True))
                                           for q, e in enumerate(EDITS)},
                             "kept": {e: hit[q + 1] for q, e in enumerate(EDITS)},
                             "word_steps": sum(word), "gamma_min_word": min(words(gam)) if any(word) else None,
                             "delta_max_word": {e: max(words(dl[q])) for q, e in enumerate(EDITS)},
                             "kl_mean_word": {e: float(np.mean(words(kv[q]))) for q, e in enumerate(EDITS)},
                             "kept_word": {e: hit_w[q + 1] for q, e in enumerate(EDITS)},
                             "drift": {e: drf[e][i] for e in ENCODER_EDITS}})
            del z, klv, dlv
            print(json.dumps({"set": name, "done": len(recs), "of": len(rows)}), flush=True)
        return recs

    out = {"model": MODEL, "edits": EDITS,
           "speech": {name: path_records(name) for name in SPEECH_SETS},
           "decision_speech": {name: decision_records(name) for name in SPEECH_SETS},
           "nonspeech": {"urban": decision_records("urban")}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
