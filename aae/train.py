"""Train acoustic abstention encoding, or one of the four variants of the ablation study.

    python -m aae.train --model whisper-large-v3 --variant aae --out checkpoints/mine/whisper-large-v3/aae

Without overrides, the steps, units, micro-batches, rank and Mandarin share are those of the released checkpoint of
that model and variant (``configs/checkpoints.yaml``).

Each step draws units of (non-speech clip n, speech clip s). The mixture m is s itself with probability 0.2;
otherwise, with equal probability, n is added to s at an SNR drawn from [-5, 20] dB, or s is followed by 0.3 to 1.5 s
of silence and a 1 to 4 s segment of n at an SNR from the same range. The loss is

    CE(empty transcript | n)  +  lambda * mean_t ||F_t(m) - F0_t(m)||^2 / ||F0_t(m)||^2

where the cross-entropy is the frozen decoder's, F is the feature module the decoder reads, F0 is the same network
with the adapters switched off, and t runs over frames that hold audio of m. Gradients reach only the LoRA adapters,
which are merged into the weights when the checkpoint is written.

Variants (``--variant``):
  aae                 the full recipe
  no-preservation     lambda = 0
  transcript-ce       the preservation loss replaced by cross-entropy on m toward the original model's transcript of s
  independent-noise   the noise in m drawn independently of n
  decoder-adapter     rank-9 LoRA on the decoder self-attention, transcript CE on m (encoder features cannot move)
  decoder-kl          LoRA on the decoder self-attention (rank 18: as many parameters as the encoder adapters), with
                      lambda * KL(original || edited) of the next-token distributions on m, teacher-forced along the
                      original model's transcript of s (Whisper only)

Runs resume from ``<out>/resume.pt``, which is also written at the last step, so rerunning a finished command only
rewrites the same checkpoint; a resume with a different recipe is refused. GPU training is not bit-deterministic, so a
retrained checkpoint differs from the released one in bytes and its numbers can differ slightly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from aae import lora, models, weights
from aae.evaluate import DATA
from aae.prepare import ROOT, read_jsonl

VARIANTS = ("aae", "no-preservation", "transcript-ce", "independent-noise", "decoder-adapter", "decoder-kl")
HELD_OUT_NONSPEECH, HELD_OUT_SPEECH = 48, 24
CONFIGS = ROOT / "configs" / "checkpoints.yaml"


def recipe(model: str, variant: str) -> dict:
    """The arguments of the released checkpoint ``model``/``variant``: the defaults of this script."""
    import yaml

    c = yaml.safe_load(CONFIGS.read_text())[f"{model}/{variant}"]
    return {"steps": c["steps"], "units": c["units"], "accum": c["accum"], "rank": c["rank"],
            "zh_share": float(c.get("zh_share", 0.0)), "lam": float(c.get("lam", 50.0))}


def check_resume(saved: dict, config: dict):
    """A resume state must come from the same recipe; only the number of steps may grow."""
    diff = sorted(k for k in saved if k in config and saved[k] != config[k])
    if diff:
        raise SystemExit("the resume state was written with different arguments ("
                         + ", ".join(f"{k}: {saved[k]} there, {config[k]} here" for k in diff)
                         + "); choose another --out or remove its resume.pt")


def load_wav(row, rng=None, max_s: float = 20.0) -> np.ndarray:
    """A staged clip as float32; clips longer than ``max_s`` are cropped at a random offset."""
    w = np.load(row["staged"]).astype(np.float32) / 32768.0
    n = int(16000 * max_s)
    if len(w) > n:
        s = (rng or random).randint(0, len(w) - n)
        w = w[s:s + n]
    return w


def mix(speech, noise, rng, snr_lo: float, snr_hi: float, gap_p: float = 0.5) -> np.ndarray:
    """A training mixture: either an overlay at a random SNR, or the speech, a silent gap and a noise tail."""
    n = noise.astype(np.float32)
    if rng.random() < gap_p:
        gap = np.zeros(int(16000 * rng.uniform(0.3, 1.5)), dtype=np.float32)
        tail = n[: int(16000 * rng.uniform(1.0, 4.0))]
        rs = float(np.sqrt(np.mean(speech ** 2)) + 1e-6)
        rn = float(np.sqrt(np.mean(tail ** 2)) + 1e-6)
        out = np.concatenate([speech, gap, tail * (rs / rn) * 10 ** (-rng.uniform(snr_lo, snr_hi) / 20)])
    else:
        if len(n) < len(speech):
            n = np.tile(n, int(np.ceil(len(speech) / max(len(n), 1))))
        s0 = rng.randint(0, len(n) - len(speech)) if len(n) > len(speech) else 0
        seg = n[s0:s0 + len(speech)]
        rs = float(np.sqrt(np.mean(speech ** 2)) + 1e-6)
        rn = float(np.sqrt(np.mean(seg ** 2)) + 1e-6)
        out = speech + seg * (rs / rn) * 10 ** (-rng.uniform(snr_lo, snr_hi) / 20)
    peak = float(np.max(np.abs(out)) + 1e-6)
    return (out / peak * 0.95 if peak > 1.0 else out).astype(np.float32)[: 16000 * 30]


class _Stop(Exception):
    pass


class FeatureTap:
    """Forward hook on the module whose output the decoder reads. With ``stop`` set, the forward pass is aborted right
    after that module, so a features-only pass costs the encoder and nothing more."""

    def __init__(self, model, name):
        self.stop, self.out = False, None
        model.get_submodule(name).register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.out = out[0] if isinstance(out, tuple | list) else out
        if self.stop:
            raise _Stop


def features(ad, tap, wavs):
    tap.stop, tap.out = True, None
    try:
        ad.loss(wavs, ["x"] * len(wavs), ["en"] * len(wavs))
    except _Stop:
        pass
    finally:
        tap.stop = False
    return tap.out


def preservation(f1, f0, wavs, whisper: bool):
    """Per-frame relative feature error, averaged over the frames that hold audio. Whisper pads every input to 30 s;
    a clip-level ratio would be dominated by that padding and leave short utterances free to drift."""
    err = ((f1 - f0) ** 2).sum(-1) / (f0 ** 2).sum(-1).clamp(min=1e-3)
    if err.dim() == 1:                                 # packed (frames,) output: every frame is audio
        return err.mean()
    B, T = err.shape
    lens = torch.tensor([len(w) for w in wavs], dtype=torch.float32, device=err.device)
    full = 30.0 * 16000 if whisper else float(lens.max())
    n = (T * lens / full).ceil().clamp(1, T)
    mask = (torch.arange(T, device=err.device)[None, :] < n[:, None]).float()
    return ((err * mask).sum(1) / mask.sum(1)).mean()


def train_mode(model):
    """Training mode for the adapters; BatchNorm keeps the original model's statistics (Hojo's Conformer has one)."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=models.MODEL_NAMES)
    ap.add_argument("--variant", default="aae", choices=VARIANTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, help="optimizer steps (default: the release's, 8,500 or 9,250)")
    ap.add_argument("--units", type=int, help="units per micro-batch (default: the release's)")
    ap.add_argument("--accum", type=int, help="micro-batches per step (default: the release's)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, help="LoRA rank (default: the release's, 16; 9 for decoder-adapter)")
    ap.add_argument("--lam", type=float, help="weight of the preservation loss (decoder-kl: of the KL); default: the "
                                               "release's, 50")
    ap.add_argument("--clean-p", type=float, default=0.2)
    ap.add_argument("--snr", type=float, nargs=2, default=(-5.0, 20.0))
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--zh-share", type=float,
                    help="if > 0, draw each speech unit's language explicitly with this Mandarin probability "
                         "(default: the release's, 0.4 for hojo-asr-v1/aae and 0 otherwise)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=250)
    a = ap.parse_args()
    for k, v in recipe(a.model, a.variant).items():
        if getattr(a, k) is None:
            setattr(a, k, v)

    spec = models.spec(a.model)
    decoder = a.variant in ("decoder-adapter", "decoder-kl")
    speech_obj = {"transcript-ce": "ce", "decoder-adapter": "ce", "decoder-kl": "kl"}.get(a.variant, "pin")
    if speech_obj == "kl" and a.model != "whisper-large-v3":
        raise SystemExit("decoder-kl is implemented for whisper-large-v3 only")
    lam = 0.0 if a.variant == "no-preservation" else a.lam
    rank, units, accum = a.rank, a.units, a.accum
    out = Path(a.out)
    resume_path = out / "resume.pt"
    config = {"model": a.model, "variant": a.variant, "units": units, "accum": accum, "rank": rank, "lambda": lam,
              "lr": a.lr, "clean_p": a.clean_p, "snr_db": list(a.snr), "max_s": a.max_s, "zh_share": a.zh_share,
              "seed": a.seed}

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    ns_rows = read_jsonl(DATA / "manifests" / "train" / "nonspeech.jsonl")
    sp_rows = read_jsonl(DATA / "manifests" / "train" / "speech.jsonl")
    pseudo = ROOT / "assets" / "manifests" / "train" / "pseudo" / f"{a.model}.jsonl"
    labels = {r["id"]: r["text"] for r in read_jsonl(pseudo)}
    for r in sp_rows:
        r["text"] = labels[r["id"]]                    # the original model's own transcript: the transcript-CE target
    rng.shuffle(ns_rows)
    rng.shuffle(sp_rows)
    hold_ns, hold_sp = ns_rows[:HELD_OUT_NONSPEECH], sp_rows[:HELD_OUT_SPEECH]
    ns_rows, sp_rows = ns_rows[HELD_OUT_NONSPEECH:], sp_rows[HELD_OUT_SPEECH:]
    zh_rows = [r for r in sp_rows if r["lang"] == "zh"]
    en_rows = [r for r in sp_rows if r["lang"] != "zh"]

    ad = models.load(a.model, models.cuda())
    locus = spec.decoder_locus if decoder else spec.encoder_locus
    wrapped = lora.apply(ad.model, locus, r=rank, alpha=2.0 * rank)
    tap = FeatureTap(ad.model, spec.features)
    for m in ad.model.modules():                       # the preservation target must be a deterministic original pass
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    if spec.lm_checkpointing:
        lm = ad.model.decoder_model if a.model == "hojo-asr-v1" else ad.model
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(lm, "config"):
            lm.config.use_cache = False
    params = [p for m in wrapped.values() for p in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)

    start, loss_curve, curve, window, hist, before = 0, [], [], [], [], 0.0
    if resume_path.exists():
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        check_resume(st.get("config", {}), config)
        with torch.no_grad():
            for k, m in wrapped.items():
                m.A.copy_(st["lora"][k]["A"])
                m.B.copy_(st["lora"][k]["B"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start, loss_curve, curve, before = st["step"], st["loss_curve"], st["curve"], st["wallclock_s"]
        rng.setstate(st["rng"])
        print(json.dumps({"resumed_at": start}), flush=True)
    t0 = time.time()

    def record(done):
        return {"model": a.model, "variant": a.variant, "steps": done, "units_per_microbatch": units, "accum": accum,
                "lr": a.lr, "rank": rank, "alpha": 2.0 * rank, "lambda": lam if speech_obj in ("pin", "kl") else 0.0,
                "speech_objective": speech_obj,
                "independent_noise": a.variant == "independent-noise", "clean_p": a.clean_p, "snr_db": list(a.snr),
                "zh_share": a.zh_share, "locus": locus, "feature_module": spec.features, "seed": a.seed,
                "trained_params": sum(p.numel() for p in params), "nonspeech_pool": len(ns_rows),
                "speech_pool": len(sp_rows), "loss_curve": list(loss_curve), "held_out_curve": list(curve),
                "wallclock_s": round(before + time.time() - t0)}

    @torch.no_grad()
    def held_out():
        ad.model.eval()
        r2 = random.Random(1)
        hr = sum(bool(t.strip()) for i in range(0, HELD_OUT_NONSPEECH, 8)
                 for t in ad.transcribe([load_wav(r, r2, a.max_s) for r in hold_ns[i:i + 8]]))
        # the first HELD_OUT_SPEECH noise clips pair with the held-out speech (zip stops at the shorter list)
        mixed = [mix(load_wav(s, r2, a.max_s), load_wav(n, r2, a.max_s), r2, 0.0, 10.0)
                 for s, n in zip(hold_sp, hold_ns, strict=False)]
        dele = sum(not t.strip() for i in range(0, HELD_OUT_SPEECH, 8) for t in ad.transcribe(mixed[i:i + 8]))
        train_mode(ad.model)
        return 100.0 * hr / HELD_OUT_NONSPEECH, 100.0 * dele / HELD_OUT_SPEECH

    def save(done):
        weights.save(out, lora.merged_state(wrapped), record(done))
        torch.save({"lora": {k: {"A": m.A.detach().cpu(), "B": m.B.detach().cpu()} for k, m in wrapped.items()},
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": done, "config": config,
                    "loss_curve": loss_curve, "curve": curve, "rng": rng.getstate(),
                    "wallclock_s": before + time.time() - t0}, str(resume_path) + ".tmp")
        os.replace(str(resume_path) + ".tmp", resume_path)

    train_mode(ad.model)
    for step in range(start, a.steps):
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "ce_n": 0.0, "speech": 0.0}
        for _ in range(accum):
            ns = rng.sample(ns_rows, units)
            if a.zh_share > 0:
                sp = [rng.choice(zh_rows) if rng.random() < a.zh_share else rng.choice(en_rows) for _ in range(units)]
            else:
                sp = rng.sample(sp_rows, units)
            n_w = [load_wav(r, rng, a.max_s) for r in ns]
            m_w = []
            for s, n in zip(sp, n_w, strict=True):
                sw = load_wav(s, rng, a.max_s)
                if a.variant == "independent-noise":   # same noise statistics, a different recording than n
                    n = load_wav(rng.choice(ns_rows), rng, a.max_s)
                m_w.append(sw if rng.random() < a.clean_p else mix(sw, n, rng, a.snr[0], a.snr[1]))
            ce_n = ad.loss(n_w, [""] * units, ["en"] * units).float()
            scaler.scale(ce_n / accum).backward()
            if speech_obj == "ce":
                sp_loss = ad.loss(m_w, [x["text"] for x in sp], [x["lang"] for x in sp]).float()
                scaler.scale(sp_loss / accum).backward()
            elif speech_obj == "kl":
                texts, langs = [x["text"] for x in sp], [x["lang"] for x in sp]
                with torch.no_grad(), lora.disabled(wrapped):
                    z0, labels, feats = ad.decoder_logits(m_w, texts, langs)
                z1, _, _ = ad.decoder_logits(m_w, texts, langs, features=feats)
                keep = labels != -100
                lp0 = torch.log_softmax(z0.float()[keep], -1)
                lp1 = torch.log_softmax(z1.float()[keep], -1)
                sp_loss = (lp0.exp() * (lp0 - lp1)).sum(-1).mean()        # KL(original || edited), mean over steps
                scaler.scale(lam * sp_loss / accum).backward()
            elif lam > 0:
                with torch.no_grad(), lora.disabled(wrapped):
                    f0 = features(ad, tap, m_w).float()
                f1 = features(ad, tap, m_w).float()
                sp_loss = preservation(f1, f0, m_w, a.model == "whisper-large-v3")
                scaler.scale(lam * sp_loss / accum).backward()
            else:                                      # no-preservation: speech clips are drawn but unused
                sp_loss = torch.zeros(())
            acc["ce_n"] += float(ce_n) / accum
            acc["speech"] += float(sp_loss) / accum
        acc["loss"] = acc["ce_n"] + (acc["speech"] if speech_obj == "ce" else lam * acc["speech"])
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        done = step + 1
        window.append(acc["loss"])
        hist.append(acc)
        if done % a.loss_every == 0 or done == a.steps:
            loss_curve.append({"step": done, "loss": round(sum(window) / len(window), 5)})
            window = []
            print(json.dumps({"step": done, "loss": loss_curve[-1]["loss"],
                              "ce_n": round(sum(x["ce_n"] for x in hist) / len(hist), 5),
                              "speech": round(sum(x["speech"] for x in hist) / len(hist), 6),
                              "wallclock_s": round(before + time.time() - t0)}), flush=True)
            hist = []
        if a.eval_every and (done % a.eval_every == 0 or done == a.steps):
            hr, dele = held_out()
            curve.append({"step": done, "held_out_hr": hr, "held_out_deletion": dele})
            print(json.dumps(curve[-1]), flush=True)
        if a.save_every and done % a.save_every == 0 and done != a.steps:
            save(done)
    ad.model.eval()
    save(a.steps)
    print(json.dumps({"saved": str(out), "steps": a.steps}))


if __name__ == "__main__":
    main()
