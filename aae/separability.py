"""Speech versus non-speech in Whisper large-v3's encoder, before and after AAE: does the encoder separate them?

    python -m aae.separability --out runs/separability/whisper-large-v3.json

Extraction (GPU). For every clip of the training pools (``train/speech``, ``train/nonspeech``: 2,999 + 13,602 clips)
and of nine complete evaluation sets (clean speech: LibriSpeech test-clean and test-other, FLEURS Mandarin, AISHELL-1;
speech in noise: the tail and 5 dB sets; non-speech: UrbanSound8K, WHAM!, filtered FSD50K), the output of each of the 32
encoder blocks and the features after ``ln_post`` that the decoder reads (layer 32), each averaged over the frames that
cover the audio (the first 30 s, the window Whisper reads), for the original model and for AAE. AAE changes blocks 16
to 31 only, so its features are stored for layers 16 to 32. On the evaluation sets, the decision margin g of
``aae.locus`` for both models: the best text token over end-of-text at the step after
<|sot|><|lang|><|transcribe|><|0.00|>, with the language detected by the original model (g > 0: the model writes text).

Analysis (CPU). Per layer, a logistic-regression probe (standardized features, balanced classes, C = 1, fixed in
advance) is fitted on the training pools and scored on the evaluation sets: AUC of clean speech and of speech in noise
against non-speech, and the share of each set on the probe's side of 0.5. Per set, the share that the layer-32 probe
calls non-speech against the share on which the decoder prefers text (g > 0). A t-SNE map of the layer-32 features of
both models, fitted jointly on every evaluation clip, and a figure of it all.

Writes the features (``features_original.npy``, ``features_aae.npy``, float16) and ``extract.json`` next to ``--out``;
a rerun with these present skips the extraction. Runs in envs/main; about 4 hours on one Quadro RTX 8000 shared with
other jobs (55,427 clips), then a few minutes on the CPU.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from aae import audio, models, weights
from aae.evaluate import DATA, MANIFESTS
from aae.prepare import read_jsonl

MODEL = "whisper-large-v3"
EDIT = f"{MODEL}/aae"
FIRST_EDITED = 16                                    # AAE adapts encoder blocks 16 to 31
N_LAYERS = 33                                        # 32 block outputs and the features after ln_post
FRAMES_PER_S = 50                                    # Whisper's encoder: 1,500 frames per 30 s
POOLS = {"speech": "speech", "nonspeech": "nonspeech"}                         # training pool -> group
SETS = {"ls_clean": "speech", "ls_other": "speech", "fleurs_zh": "speech", "aishell": "speech",
        "tail": "noisy", "snr5": "noisy", "urban": "nonspeech", "wham": "nonspeech", "fsd50k": "nonspeech"}
GROUP_TITLE = {"speech": "clean speech", "noisy": "speech in noise", "nonspeech": "non-speech"}
COLOR = {"speech": "#2a78d6", "noisy": "#eb6834", "nonspeech": "#1baf7a"}     # categorical slots 1-3


def pool(h: torch.Tensor, nfr) -> torch.Tensor:
    """Mean of each clip's frames over the first ``nfr[i]`` frames, the frames that cover its audio (float32)."""
    n = torch.tensor(nfr, device=h.device)
    mask = (torch.arange(h.shape[1], device=h.device)[None] < n[:, None]).to(h.dtype)
    return ((h * mask[..., None]).float().sum(1) / n[:, None].float()).float()


def manifests() -> list[tuple[str, str, bool, list]]:
    """(name, group, evaluation set?, rows): the nine complete evaluation sets, then the two training pools."""
    out = [(s, g, True, read_jsonl(MANIFESTS / f"{s}.jsonl")) for s, g in SETS.items()]
    return out + [(f"train/{p}", g, False, read_jsonl(DATA / "manifests" / "train" / f"{p}.jsonl"))
                  for p, g in POOLS.items()]


def margins(z, seqs, filters, eot: int) -> list[list[float]]:
    """The decision margin g of ``aae.locus`` per clip and model: from the stacked last-step logits ``z`` (rows: models,
    then clips) for the prompts ``seqs``, after Whisper's decoding filters, the best text token minus end-of-text."""
    out = []
    for i, s in enumerate(seqs):
        zf = z[:, i].clone()
        t = torch.tensor(s)[None].expand(zf.shape[0], -1)
        for f in filters:
            f.apply(zf, t)
        zt = zf.clone()
        zt[:, eot] = -math.inf
        out.append((zt.amax(-1) - zf[:, eot]).tolist())
    return out


def extract(args, out_dir: Path) -> None:
    from safetensors.torch import load_file
    from whisper.decoding import DecodingOptions, DecodingTask, detect_language

    torch.manual_seed(0)
    ad = models.load(MODEL, models.cuda())
    model, dev, tk = ad.model, ad.device, ad.tok
    params = dict(model.named_parameters())
    loaded = load_file(str(weights.resolve(EDIT) / "model.safetensors"))
    edited = {k: v.to(dtype=params[k].dtype).pin_memory() for k, v in loaded.items()}
    original = {k: params[k].detach().cpu().pin_memory() for k in edited}
    assert all(k.startswith("encoder.blocks.") and int(k.split(".")[2]) >= FIRST_EDITED for k in edited)

    def use(tensors):
        with torch.no_grad():
            for k, v in tensors.items():
                params[k].copy_(v, non_blocking=True)
        torch.cuda.synchronize()

    task = DecodingTask(model, DecodingOptions(task="transcribe", language="en", temperature=0.0, fp16=True,
                                               without_timestamps=False))
    filters = task.logit_filters
    eot, ts0 = tk.eot, tk.timestamp_begin

    blocks = model.encoder.blocks
    hidden = {}
    for i, b in enumerate(blocks):
        b.register_forward_hook(lambda m, a, o, i=i: hidden.__setitem__(i, o))

    sets = manifests()
    n = sum(len(rows) for *_, rows in sets)
    feats_o = np.lib.format.open_memmap(out_dir / "features_original.npy", "w+", np.float16, (n, N_LAYERS, 1280))
    feats_a = np.lib.format.open_memmap(out_dir / "features_aae.npy", "w+", np.float16,
                                        (n, N_LAYERS - FIRST_EDITED, 1280))
    index, at, checked = [], 0, False
    for name, group, is_eval, rows in sets:
        for b in range(0, len(rows), args.batch):
            chunk = rows[b: b + args.batch]
            wavs = [audio.load(r["staged"]) for r in chunk]
            mel = ad.mels(wavs)
            nfr = [min(1500, int(math.ceil(len(w) / audio.SAMPLE_RATE * FRAMES_PER_S))) for w in wavs]
            with torch.no_grad():
                f0 = model.embed_audio(mel)
                po = torch.stack([pool(hidden[i], nfr) for i in range(len(blocks))] + [pool(f0, nfr)], 1)
                # AAE: blocks 0-15 are the original ones, so continue from the original output of block 15
                use(edited)
                x = hidden[FIRST_EDITED - 1]
                for blk in blocks[FIRST_EDITED:]:
                    x = blk(x)
                fe = model.encoder.ln_post(x)
                pa = torch.stack([pool(hidden[i], nfr) for i in range(FIRST_EDITED, len(blocks))] + [pool(fe, nfr)], 1)
                if not checked:                       # the continuation equals AAE's full forward pass
                    assert torch.equal(model.embed_audio(mel), fe)
                    checked = True
                if is_eval:
                    lang_tokens, _ = detect_language(model, f0, tk)       # decoder unchanged: either model's logits
                    langs = [tk.decode([int(t)]).strip("<|>") for t in lang_tokens]
                    seqs = [[tk.sot, tk.to_language_token(lang), tk.transcribe, ts0] for lang in langs]
                    inp = torch.tensor(seqs, device=dev)
                    z = torch.stack([model.decoder(inp, f0)[:, -1].float(), model.decoder(inp, fe)[:, -1].float()])
                    g = margins(z, seqs, filters, eot)
                use(original)
            feats_o[at: at + len(chunk)] = po.cpu().numpy().astype(np.float16)
            feats_a[at: at + len(chunk)] = pa.cpu().numpy().astype(np.float16)
            for i, r in enumerate(chunk):
                rec = {"id": r["id"], "set": name, "group": group, "eval": is_eval}
                if is_eval:
                    rec.update(lang=langs[i], g={"original": g[i][0], "aae": g[i][1]})
                index.append(rec)
            at += len(chunk)
            del f0, fe, x
            hidden.clear()
            if (b // args.batch) % 50 == 0:
                print(json.dumps({"set": name, "done": b + len(chunk), "of": len(rows)}), flush=True)
    assert at == n
    feats_o.flush()
    feats_a.flush()
    (out_dir / "extract.json").write_text(json.dumps({"model": MODEL, "edit": EDIT, "clips": index},
                                                     ensure_ascii=False) + "\n")


def analyze(out_dir: Path) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    clips = json.loads((out_dir / "extract.json").read_text())["clips"]
    feats = {"original": np.load(out_dir / "features_original.npy", mmap_mode="r"),
             "aae": np.load(out_dir / "features_aae.npy", mmap_mode="r")}
    first = {"original": 0, "aae": FIRST_EDITED}                 # layer of each file's first column
    train = np.array([not c["eval"] for c in clips])
    test = ~train
    is_speech = np.array([c["group"] != "nonspeech" for c in clips])
    group = np.array([c["group"] for c in clips])
    sets = np.array([c["set"] for c in clips])
    y_tr, y_te = is_speech[train], is_speech[test]
    clean, noisy = group[test] != "noisy", group[test] != "speech"     # each speech kind against non-speech

    probe, p32 = {}, {}
    for system, f in feats.items():
        probe[system] = []
        for col in range(f.shape[1]):
            x = np.asarray(f[:, col], dtype=np.float32)
            sc = StandardScaler().fit(x[train])
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(x[train]), y_tr)
            p = clf.predict_proba(sc.transform(x[test]))[:, 1]
            right = (p >= 0.5) == y_te
            probe[system].append({"layer": first[system] + col,
                                  "auc_clean": roc_auc_score(y_te[clean], p[clean]),
                                  "auc_noisy": roc_auc_score(y_te[noisy], p[noisy]),
                                  "correct": {s: float(right[sets[test] == s].mean()) for s in SETS}})
            print(json.dumps({"system": system, **probe[system][-1]}), flush=True)
        p32[system] = p

    g = {s: np.array([c["g"][s] for c in clips if c["eval"]]) for s in feats}
    decision = {}
    for s in SETS:
        m = sets[test] == s
        decision[s] = {"n": int(m.sum()), "group": SETS[s],
                       "probe_nonspeech": {k: float((p32[k][m] < 0.5).mean()) for k in feats},
                       "text": {k: float((g[k][m] > 0).mean()) for k in feats},
                       "g_median": {k: float(np.median(g[k][m])) for k in feats}}
    auc = {k: {"probe_clean": probe[k][-1]["auc_clean"], "probe_noisy": probe[k][-1]["auc_noisy"],
               "g_clean": roc_auc_score(y_te[clean], g[k][clean]), "g_noisy": roc_auc_score(y_te[noisy], g[k][noisy])}
           for k in feats}

    x = np.concatenate([np.asarray(feats[k][test, -1], dtype=np.float32) for k in feats])
    x = PCA(n_components=50, random_state=0).fit_transform(x)
    xy = TSNE(perplexity=30, init="pca", random_state=0, n_jobs=-1).fit_transform(x)
    xy = {"original": xy[: test.sum()], "aae": xy[test.sum():]}

    test_clips = [c for c in clips if c["eval"]]
    records = [{"id": c["id"], "set": c["set"], "g": c["g"],
                "p_speech": {k: float(p32[k][i]) for k in feats},
                "tsne": {k: [float(v) for v in xy[k][i]] for k in feats}} for i, c in enumerate(test_clips)]
    return {"model": MODEL, "edit": EDIT,
            "n": {"train": int(train.sum()), "eval": int(test.sum())},
            "probe": probe, "decision": decision, "auc_layer32": auc, "clips": records}


def figure(res: dict, path: Path, theme: str = "light") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from aae.states import COL as col
    from aae.states import THEMES

    t = THEMES[theme]
    ink, muted, rule, bg = t["ink"], t["muted"], t["rule"], t["bg"]
    dark = {"speech": t["speech"], "nonspeech": t["nonspeech"]}
    said = {"text": "#D55E00", "empty": "#0072B2" if theme == "light" else "#56B4E9"}   # Okabe-Ito: what is written
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": rule, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": ink, "axes.facecolor": bg})
    clips = res["clips"]
    kind = np.array(["nonspeech" if SETS[c["set"]] == "nonspeech" else "speech" for c in clips])
    xy = {k: np.array([c["tsne"][k] for c in clips]) for k in ("original", "aae")}
    g = {k: np.array([c["g"][k] for c in clips]) for k in ("original", "aae")}
    allxy = np.concatenate(list(xy.values()))
    lo, hi = allxy.min(0), allxy.max(0)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ns = kind == "nonspeech"

    fig = plt.figure(figsize=(15.5, 11.5), facecolor=bg)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.78], wspace=0.17, hspace=0.24, left=0.03, right=0.985,
                          top=0.835, bottom=0.04)

    def frame(a, letter, title, sub):
        a.set_xlim(lo[0], hi[0])
        a.set_ylim(lo[1], hi[1])
        a.set_xticks([])
        a.set_yticks([])
        for side in ("left", "bottom", "top", "right"):
            a.spines[side].set_visible(True)
            a.spines[side].set_color(rule)
        a.text(0, 1.075, f"({letter})  {title}", transform=a.transAxes, fontsize=14, fontweight="bold", color=ink,
               va="bottom")
        a.text(0, 1.02, sub, transform=a.transAxes, fontsize=11, color=muted, va="bottom")

    def callouts(a, lines):
        y = 0.035
        for text, color in reversed(lines):
            a.text(0.025, y, text, transform=a.transAxes, fontsize=11.5, color=color, va="bottom",
                   fontweight="bold" if color != muted else "normal",
                   bbox=dict(boxstyle="square,pad=0.15", fc=bg, ec="none", alpha=0.85))
            y += 0.065

    probe32 = res["probe"]["original"][-1]
    ns_found = min(100 * res["decision"][s]["probe_nonspeech"]["original"] for s in SETS if SETS[s] == "nonspeech")
    a = fig.add_subplot(gs[0, 0])
    for part in ("nonspeech", "speech"):
        a.scatter(*xy["original"][kind == part].T, s=2.2, c=col[part], lw=0, alpha=0.85, rasterized=True)
    frame(a, "a", "What the encoder hears", "original encoder features, by input")
    callouts(a, [(f"linear probe AUC: {probe32['auc_clean']:.4f} (clean), {probe32['auc_noisy']:.4f} (in noise)",
                  ink), (f"non-speech found by the probe: {ns_found:.2f}% to 100%", muted)])

    for k, (letter, system, title) in enumerate((("b", "original", "What the decoder does"),
                                                 ("c", "aae", "After AAE"))):
        a = fig.add_subplot(gs[(k + 1) // 2, (k + 1) % 2])
        text = g[system] > 0
        for m, color in ((~text, said["empty"]), (text, said["text"])):
            a.scatter(*xy[system][m].T, s=2.2, c=color, lw=0, alpha=0.85, rasterized=True)
        sub = "original features, by the decoder's choice" if system == "original" else \
            "AAE features, by the decoder's choice"
        frame(a, letter, title, sub)
        callouts(a, [(f"non-speech: text wins on {100 * text[ns].mean():.1f}%", said["text"] if text[ns].mean() > 0.5
                      else said["empty"]), (f"speech: text wins on {100 * text[~ns].mean():.2f}%", muted)])

    a = fig.add_subplot(gs[1, 1])
    a.axvspan(FIRST_EDITED - 0.5, 31.5, color=t["grid"], lw=0)
    a.text((FIRST_EDITED + 31) / 2, 1.0, "AAE adapters", ha="center", va="top", color=muted, fontsize=10,
           transform=a.get_xaxis_transform())
    rows = res["probe"]["original"]
    layers = [r["layer"] for r in rows]
    a.plot(layers, [r["auc_clean"] for r in rows], color=col["speech"], lw=2.2, label="clean speech vs non-speech")
    a.plot(layers, [r["auc_noisy"] for r in rows], color=dark["speech"], lw=2.2, ls="--",
           label="speech in noise vs non-speech")
    a.set_ylim(0.95, 1.002)
    a.set_xlim(-0.5, 32.5)
    a.set_xlabel("encoder layer (32: the features the decoder reads)")
    a.set_ylabel("linear-probe AUC")
    a.grid(axis="y", color=t["grid"], lw=0.8)
    a.legend(frameon=False, fontsize=10.5, loc="lower right", labelcolor=ink)
    a.text(0, 1.075, "(d)  At every layer", transform=a.transAxes, fontsize=14, fontweight="bold", color=ink,
           va="bottom")
    a.text(0, 1.02, f"probe fitted on the training pools; AUC at least {res_auc_min(res):.3f}", transform=a.transAxes,
           fontsize=11, color=muted, va="bottom")

    key = fig.add_subplot(gs[:, 2])
    key.axis("off")
    handles = [Line2D([], [], ls="", marker="o", ms=9, mfc=col["speech"], mec="none",
                      label=f"speech, clean and in noise ({(~ns).sum():,} clips)"),
               Line2D([], [], ls="", marker="o", ms=9, mfc=col["nonspeech"], mec="none",
                      label=f"non-speech ({ns.sum():,} clips)"),
               Line2D([], [], ls="", marker="o", ms=9, mfc=said["text"], mec="none",
                      label="decoder prefers text (g > 0)"),
               Line2D([], [], ls="", marker="o", ms=9, mfc=said["empty"], mec="none",
                      label="decoder prefers an empty transcript")]
    key.legend(handles=handles, loc="upper left", frameon=False, fontsize=11, borderaxespad=0, labelcolor=ink)
    key.text(0, 0.80, "Each point is one clip: its encoder features,\naveraged over the frames that cover its audio, "
                      "at\nthe output the decoder reads. One t-SNE map is\nfitted on the features of the original "
                      "model and\nof AAE, so (a), (b) and (c) share positions.\n\ng: the margin of the best text "
                      "token over\nend-of-text at the step after the prompt\n<|sot|><|lang|><|transcribe|><|0.00|>, "
                      "where an\nempty transcript is decided.\n\nThe original encoder separates the two kinds of\n"
                      "input, but the decoder writes text for all of\nthem. AAE does not teach the encoder to detect\n"
                      "non-speech; it moves non-speech to where the\nfrozen decoder ends the transcript.",
             transform=key.transAxes, va="top", fontsize=10.5, color=muted, linespacing=1.45)

    fig.text(0.03, 0.975, "Whisper's encoder already separates speech from non-speech", fontsize=19,
             fontweight="bold", color=ink, va="top")
    fig.text(0.03, 0.937, f"Whisper large-v3, {res['n']['eval']:,} clips of the nine complete evaluation sets. The "
                          "failure is not detection: the decoder writes text on every non-speech clip,\nalthough a "
                          "linear probe on the encoder features finds nearly all of them.", fontsize=12, color=muted,
             va="top", linespacing=1.4)
    fig.savefig(path, dpi=200, facecolor=bg)
    if theme == "light":
        fig.savefig(path.with_suffix(".pdf"), dpi=300, facecolor=bg)      # vector text, rasterized points
    plt.close(fig)


def res_auc_min(res: dict) -> float:
    """The smallest probe AUC of the original model over all layers, clean speech and speech in noise."""
    return min(min(r["auc_clean"], r["auc_noisy"]) for r in res["probe"]["original"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if (out.parent / "extract.json").exists():
        print("| have", out.parent / "extract.json")
    else:
        extract(args, out.parent)
    res = analyze(out.parent)
    out.write_text(json.dumps(res, ensure_ascii=False) + "\n")
    figure(res, out.with_suffix(".png"))
    figure(res, out.with_name(out.stem + "-dark.png"), theme="dark")
    print("wrote", out, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
