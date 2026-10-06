"""Encoder edit versus decoder edit on Whisper large-v3, seen where both act: the decoder state that decides silence.

    python -m aae.states --out runs/states/whisper-large-v3.json

The encoder features cannot show a decoder edit, which leaves them unchanged; every edit acts on the decoder's final
hidden state (after its last layer norm, before the output projection) at the step after
<|sot|><|lang|><|transcribe|><|0.00|>, where the logits of end-of-text and of every text token are read off. For every
clip of the nine complete evaluation sets of ``aae.separability`` (38,826 clips), this module records that state and
the decision margin g of ``aae.locus`` (best text token minus end-of-text after Whisper's filters; g > 0: the model
writes text) for the original model and the three Whisper edits of the ablation study:

  aae               encoder blocks 16 to 31, abstention + feature preservation (the released model)
  transcript-ce     encoder blocks 16 to 31, abstention + transcript cross-entropy on m
  decoder-adapter   self-attention of all 32 decoder layers, abstention + transcript cross-entropy on m

The language is detected once, by the original model, as in ``aae.locus``. Per edit and clip, the relative change of the
state, ||h_edit - h_original|| / ||h_original||; a t-SNE map fitted jointly on the states of all four models; and a
figure of both.

Writes the states (``states.npy``, float16, clips x 4 models x 1280) and ``extract.json`` next to ``--out``; a rerun
with these present skips the extraction. Runs in envs/main; about 4 hours on one Quadro RTX 8000 shared with other
jobs (two encoder passes per clip), then a few minutes on the CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aae import audio, models, weights
from aae.evaluate import MANIFESTS
from aae.prepare import read_jsonl
from aae.separability import FIRST_EDITED, GROUP_TITLE, MODEL, SETS, margins

SYSTEMS = ("original", "aae", "transcript-ce", "decoder-adapter")
ENCODER_EDITS = ("aae", "transcript-ce")
TITLE = {"original": "Original model", "aae": "AAE: encoder edit, feature preservation",
         "transcript-ce": "Encoder edit, transcript loss", "decoder-adapter": "Decoder edit, transcript loss"}


def extract(args, out_dir: Path) -> None:
    from safetensors.torch import load_file
    from whisper.decoding import DecodingOptions, DecodingTask, detect_language

    torch.manual_seed(0)
    ad = models.load(MODEL, models.cuda())
    model, dev, tk = ad.model, ad.device, ad.tok
    params = dict(model.named_parameters())
    tensors = {}                       # edited weights, kept in pinned CPU memory (the GPUs may be shared)
    for e in SYSTEMS[1:]:
        loaded = load_file(str(weights.resolve(f"{MODEL}/{e}") / "model.safetensors"))
        tensors[e] = {k: v.to(dtype=params[k].dtype).pin_memory() for k, v in loaded.items()}
    for e in ENCODER_EDITS:            # the continuation from block 15 below relies on this
        assert all(k.startswith("encoder.blocks.") and int(k.split(".")[2]) >= FIRST_EDITED for k in tensors[e])
    assert all(k.startswith("decoder.blocks.") for k in tensors["decoder-adapter"])
    original = {k: params[k].detach().cpu().pin_memory() for k in sorted({k for t in tensors.values() for k in t})}
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

    task = DecodingTask(model, DecodingOptions(task="transcribe", language="en", temperature=0.0, fp16=True,
                                               without_timestamps=False))
    filters = task.logit_filters
    eot, ts0 = tk.eot, tk.timestamp_begin
    blocks = model.encoder.blocks
    caught = {}
    blocks[FIRST_EDITED - 1].register_forward_hook(lambda m, a, o: caught.__setitem__("block", o))
    model.decoder.ln.register_forward_hook(lambda m, a, o: caught.__setitem__("ln", o))

    def decide(inp, feats):
        """Last-step logits and the final decoder state that produced them."""
        z = model.decoder(inp, feats)[:, -1].float()
        return z, caught["ln"][:, -1].float()

    sets = [(s, g, read_jsonl(MANIFESTS / f"{s}.jsonl")) for s, g in SETS.items()]
    n = sum(len(rows) for *_, rows in sets)
    states = np.lib.format.open_memmap(out_dir / "states.npy", "w+", np.float16, (n, len(SYSTEMS), 1280))
    index, at, checked = [], 0, False
    for name, group, rows in sets:
        for b in range(0, len(rows), args.batch):
            chunk = rows[b: b + args.batch]
            mel = ad.mels([audio.load(r["staged"]) for r in chunk])
            z, h = {}, {}
            with torch.no_grad():
                use(None)
                f0 = model.embed_audio(mel)
                h15 = caught["block"]
                lang_tokens, _ = detect_language(model, f0, tk)
                langs = [tk.decode([int(t)]).strip("<|>") for t in lang_tokens]
                seqs = [[tk.sot, tk.to_language_token(lang), tk.transcribe, ts0] for lang in langs]
                inp = torch.tensor(seqs, device=dev)
                z["original"], h["original"] = decide(inp, f0)
                for e in ENCODER_EDITS:        # blocks 0-15 are the original ones: continue from block 15's output
                    use(e)
                    x = h15
                    for blk in blocks[FIRST_EDITED:]:
                        x = blk(x)
                    fe = model.encoder.ln_post(x)
                    if not checked:            # the continuation equals the edit's full forward pass
                        assert torch.equal(model.embed_audio(mel), fe)
                    z[e], h[e] = decide(inp, fe)
                    del x, fe
                checked = True
                use("decoder-adapter")
                z["decoder-adapter"], h["decoder-adapter"] = decide(inp, f0)
                use(None)
            g = margins(torch.stack([z[s] for s in SYSTEMS]), seqs, filters, eot)
            states[at: at + len(chunk)] = torch.stack([h[s] for s in SYSTEMS], 1).cpu().numpy().astype(np.float16)
            for i, r in enumerate(chunk):
                index.append({"id": r["id"], "set": name, "group": group, "lang": langs[i],
                              "g": dict(zip(SYSTEMS, g[i], strict=True))})
            at += len(chunk)
            del f0, h15, z, h
            if (b // args.batch) % 50 == 0:
                print(json.dumps({"set": name, "done": b + len(chunk), "of": len(rows)}), flush=True)
    assert at == n
    states.flush()
    (out_dir / "extract.json").write_text(json.dumps({"model": MODEL, "systems": SYSTEMS, "clips": index},
                                                     ensure_ascii=False) + "\n")


def analyze(out_dir: Path) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    clips = json.loads((out_dir / "extract.json").read_text())["clips"]
    st = np.load(out_dir / "states.npy", mmap_mode="r")
    h = {s: np.asarray(st[:, k], dtype=np.float32) for k, s in enumerate(SYSTEMS)}
    group = np.array([c["group"] for c in clips])
    speech = group != "nonspeech"
    norm0 = np.linalg.norm(h["original"], axis=1)
    change = {e: np.linalg.norm(h[e] - h["original"], axis=1) / norm0 for e in SYSTEMS[1:]}
    g = {s: np.array([c["g"][s] for c in clips]) for s in SYSTEMS}
    summary = {s: {"text_nonspeech": float(100 * (g[s][~speech] > 0).mean()),
                   "empty_speech": float(100 * (g[s][speech] <= 0).mean()),
                   "g_median": {"speech": float(np.median(g[s][speech])), "nonspeech": float(np.median(g[s][~speech]))}}
               for s in SYSTEMS}
    for e in SYSTEMS[1:]:
        summary[e]["change_median"] = {k: float(100 * np.median(change[e][m])) for k, m in
                                       (("speech", speech), ("nonspeech", ~speech))}
        summary[e]["change_by_group"] = {k: float(100 * np.median(change[e][group == k])) for k in GROUP_TITLE}
        print(json.dumps({"edit": e, **summary[e]}), flush=True)

    x = PCA(n_components=50, random_state=0).fit_transform(np.concatenate([h[s] for s in SYSTEMS]))
    xy = TSNE(perplexity=30, init="pca", random_state=0, n_jobs=-1).fit_transform(x)
    n = len(clips)
    records = [{"id": c["id"], "set": c["set"], "g": c["g"],
                "change": {e: float(change[e][i]) for e in SYSTEMS[1:]},
                "tsne": {s: [float(v) for v in xy[k * n + i]] for k, s in enumerate(SYSTEMS)}}
               for i, c in enumerate(clips)]
    return {"model": MODEL, "systems": SYSTEMS, "n": n, "summary": summary, "clips": records}


COL = {"speech": "#009E73", "nonspeech": "#E69F00"}            # Okabe-Ito bluish green and orange, as in the paper
THEMES = {
    "light": {"bg": "#ffffff", "ink": "#1a1a1a", "muted": "#5a5a5a", "rule": "#d0d0d0", "grid": "#ececec",
              "speech": "#00694C", "nonspeech": "#9C6B00"},
    "dark": {"bg": "#0d1117", "ink": "#e6edf3", "muted": "#9da7b1", "rule": "#30363d", "grid": "#21262d",
             "speech": "#3fd0a4", "nonspeech": "#f2b84b"},
}
LOSS = {"original": "Original model", "aae": "Preservation loss", "transcript-ce": "Transcript CE",
        "decoder-adapter": "Transcript CE"}
WHERE = {"original": "no edit", "aae": "encoder (AAE)", "transcript-ce": "encoder", "decoder-adapter": "decoder"}


def layout(res: dict) -> dict:
    """What every drawing of the map shares: input kind per clip, positions, limits, and the outline of the region
    where each kind sat under the original model (the smoothed density region holding 95% of its clips)."""
    from scipy.ndimage import gaussian_filter

    clips = res["clips"]
    kind = np.array(["nonspeech" if SETS[c["set"]] == "nonspeech" else "speech" for c in clips])
    xy = {s: np.array([c["tsne"][s] for c in clips]) for s in SYSTEMS}
    allxy = np.concatenate(list(xy.values()))
    lo, hi = allxy.min(0), allxy.max(0)
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    bins = 300
    ex, ey = np.linspace(lo[0], hi[0], bins + 1), np.linspace(lo[1], hi[1], bins + 1)
    region = {}
    for part in COL:
        d, _, _ = np.histogram2d(*xy["original"][kind == part].T, bins=(ex, ey))
        d = gaussian_filter(d, 2.5)
        flat = np.sort(d.ravel())[::-1]
        region[part] = (d.T, flat[np.searchsorted(np.cumsum(flat), 0.95 * flat.sum())])
    return {"kind": kind, "sets": np.array([c["set"] for c in clips]), "xy": xy, "lo": lo, "hi": hi,
            "cx": (ex[:-1] + ex[1:]) / 2, "cy": (ey[:-1] + ey[1:]) / 2, "region": region}


def draw_map(a, L: dict, s: str, t: dict, size: float, outline: float, arrow: float) -> None:
    """One panel: the clips under system ``s``, the outlines of the original regions, and for an edit, where each
    speech set's mean position goes (an arrow) or a ring where it stays."""
    from matplotlib import patheffects

    for part in ("nonspeech", "speech"):
        a.scatter(*L["xy"][s][L["kind"] == part].T, s=size, c=COL[part], lw=0, alpha=0.85, rasterized=True)
    for part in COL:
        d, level = L["region"][part]
        a.contour(L["cx"], L["cy"], d, levels=[level], colors=[t[part]], linewidths=outline, linestyles="--")
    if s != "original":
        halo = [patheffects.withStroke(linewidth=arrow * 2.6, foreground=t["bg"])]
        span = float(np.linalg.norm(L["hi"] - L["lo"]))
        for st, grp in SETS.items():
            if grp == "nonspeech":
                continue
            m = L["sets"] == st
            c0, c1 = L["xy"]["original"][m].mean(0), L["xy"][s][m].mean(0)
            if np.linalg.norm(c1 - c0) < 0.02 * span:
                a.plot(*c0, marker="o", ms=arrow * 3.6, mfc=t["bg"], mec=t["speech"], mew=arrow, zorder=5,
                       path_effects=halo)
            else:
                arr = a.annotate("", xy=c1, xytext=c0, zorder=5,
                                 arrowprops=dict(arrowstyle="-|>", color=t["speech"], lw=arrow,
                                                 mutation_scale=arrow * 7, shrinkA=0, shrinkB=0))
                arr.arrow_patch.set_path_effects(halo)
    a.set_xlim(L["lo"][0], L["hi"][0])
    a.set_ylim(L["lo"][1], L["hi"][1])
    a.set_xticks([])
    a.set_yticks([])
    a.set_facecolor(t["bg"])
    for side in ("left", "bottom", "top", "right"):
        a.spines[side].set_visible(True)
        a.spines[side].set_color(t["rule"])


def figure(res: dict, path: Path, theme: str = "light") -> None:
    """The full figure of docs/RESULTS.md: all four models, change bars and a key."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    t = THEMES[theme]
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": t["rule"], "axes.labelcolor": t["ink"],
                         "xtick.color": t["muted"], "ytick.color": t["muted"], "axes.spines.top": False,
                         "axes.spines.right": False, "text.color": t["ink"]})
    S, L = res["summary"], layout(res)
    fig = plt.figure(figsize=(15.5, 11.5), facecolor=t["bg"])
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.78], wspace=0.08, hspace=0.24, left=0.03, right=0.985,
                          top=0.835, bottom=0.04)
    for k, s in enumerate(SYSTEMS):
        a = fig.add_subplot(gs[k // 2, k % 2])
        draw_map(a, L, s, t, size=2.2, outline=1.6, arrow=1.9)
        a.text(0, 1.075, f"({'abcd'[k]})  {LOSS[s]}", transform=a.transAxes, fontsize=14, fontweight="bold",
               color=t["ink"], va="bottom")
        a.text(0, 1.02, WHERE[s], transform=a.transAxes, fontsize=11, color=t["muted"], va="bottom")
        lines = [(f"non-speech with text: {S[s]['text_nonspeech']:.1f}%", t["muted"])]
        if s != "original":
            c = S[s]["change_median"]
            lines = [(f"speech change: {c['speech']:.1f}%", t["speech"]),
                     (f"non-speech change: {c['nonspeech']:.1f}%", t["nonspeech"])] + lines
        y = 0.035
        for text, color in reversed(lines):
            a.text(0.025, y, text, transform=a.transAxes, fontsize=11.5, color=color, va="bottom",
                   fontweight="normal" if color == t["muted"] else "bold",
                   bbox=dict(boxstyle="square,pad=0.15", fc=t["bg"], ec="none", alpha=0.85))
            y += 0.065

    b = fig.add_subplot(gs[0, 2])
    b.set_facecolor(t["bg"])
    edits = SYSTEMS[1:]
    yy = 1.3 * np.arange(len(edits))[::-1]
    h = 0.34
    for j, part in enumerate(("speech", "nonspeech")):
        vals = [S[e]["change_median"][part] for e in edits]
        bars = b.barh(yy - j * h, vals, h * 0.9, color=COL[part])
        for bar, v in zip(bars, vals, strict=True):
            b.text(v + 2, bar.get_y() + bar.get_height() / 2, f"{v:.0f}%" if v >= 10 else f"{v:.1f}%",
                   va="center", fontsize=11, color=t[part], fontweight="bold")
    for e, y0 in zip(edits, yy, strict=True):
        lbl = b.text(0, y0 + 0.3, LOSS[e], fontsize=11.5, fontweight="bold", color=t["ink"], va="bottom")
        b.annotate(f"  {WHERE[e]}", xy=(1, 0), xycoords=lbl, va="bottom", fontsize=10, color=t["muted"])
    b.set_yticks([])
    b.set_ylim(yy[-1] - h - 0.35, yy[0] + 0.75)
    b.set_xlim(0, 1.22 * max(S[e]["change_median"][p] for e in edits for p in ("speech", "nonspeech")))
    b.grid(axis="x", color=t["grid"], lw=0.8)
    b.set_axisbelow(True)
    b.spines["left"].set_visible(False)
    b.set_xlabel("median relative change of the state per clip (%)", fontsize=10.5, color=t["muted"])
    b.text(0, 1.075, "(e)  Change of the state", transform=b.transAxes, fontsize=14, fontweight="bold",
           color=t["ink"], va="bottom")
    b.text(0, 1.02, "||h_edit − h_original|| / ||h_original||", transform=b.transAxes, fontsize=11,
           color=t["muted"], va="bottom")

    key = fig.add_subplot(gs[1, 2])
    key.axis("off")
    handles = [Line2D([], [], ls="", marker="o", ms=9, mfc=COL["speech"], mec="none",
                      label=f"speech, clean and in noise ({(L['kind'] == 'speech').sum():,} clips)"),
               Line2D([], [], ls="", marker="o", ms=9, mfc=COL["nonspeech"], mec="none",
                      label=f"non-speech ({(L['kind'] == 'nonspeech').sum():,} clips)"),
               Line2D([], [], color=t["speech"], lw=1.6, ls="--", label="where speech sat in the original model"),
               Line2D([], [], color=t["nonspeech"], lw=1.6, ls="--",
                      label="where non-speech sat in the original model"),
               Line2D([], [], color=t["speech"], lw=1.9, marker=">", ms=6, label="a speech set shifts"),
               Line2D([], [], ls="", marker="o", ms=7, mfc=t["bg"], mec=t["speech"], mew=1.9,
                      label="a speech set stays")]
    key.legend(handles=handles, loc="upper left", frameon=False, fontsize=11, handlelength=2.2, borderaxespad=0,
               labelcolor=t["ink"])
    key.text(0, 0.46, "Each point is one clip: the decoder's final hidden\nstate at the step after the prompt "
                      "<|sot|><|lang|>\n<|transcribe|><|0.00|>, where Whisper chooses\nbetween text and an empty "
                      "transcript. One t-SNE\nmap is fitted on the states of all four models, so\npositions are "
                      "comparable across panels. Arrows join\neach speech set's mean position in the original\nmodel "
                      "to its position after the edit. The decoder\nedit cannot use a preservation loss: it never\n"
                      "changes the features.",
             transform=key.transAxes, va="top", fontsize=10.5, color=t["muted"], linespacing=1.45)

    fig.text(0.03, 0.975, "The preservation loss keeps speech in place; transcript CE shifts it", fontsize=19,
             fontweight="bold", color=t["ink"], va="top")
    fig.text(0.03, 0.937, "Whisper large-v3: the decoder state that decides between text and silence, for "
                          f"{res['n']:,} clips of the nine complete evaluation sets. Every edit silences\n"
                          "non-speech; edits trained with transcript cross-entropy, in the encoder or in the decoder, "
                          "also change the state of speech about as much.", fontsize=12, color=t["muted"], va="top",
             linespacing=1.4)
    fig.savefig(path, dpi=200, facecolor=t["bg"])
    if theme == "light":
        fig.savefig(path.with_suffix(".pdf"), dpi=300, facecolor=t["bg"])  # vector text, rasterized points
    plt.close(fig)


def paper_figure(res: dict, path: Path) -> None:
    """The paper's column figure: the three Whisper edits, set in Times at the size the paper prints them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.legend_handler import HandlerPatch, HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrow

    class ArrowKey(HandlerPatch):
        """The legend's arrow, drawn like the arrows on the maps."""

        def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
            arrow = FancyArrow(-xdescent, height / 2 - ydescent, width, 0, width=0.09 * height,
                               head_width=0.75 * height, head_length=0.5 * height, length_includes_head=True,
                               color=orig.get_facecolor(), lw=0)
            arrow.set_transform(trans)
            return [arrow]

    for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSerif-*.ttf"):
        font_manager.fontManager.addfont(str(f))        # Times metrics, as in the paper; TrueType embeds cleanly
    t = THEMES["light"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "STIXGeneral"],
                         "font.size": 7, "text.color": t["ink"], "pdf.fonttype": 42})
    S, L = res["summary"], layout(res)
    width = 3.35                                                    # \columnwidth of the ICASSP kit, in inches
    height, head, foot = 1.47, 0.284, 0.392                         # inches: whole figure, titles, labels + key
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(1, 3, wspace=0.05, left=0.005, right=0.995, top=1 - head / height, bottom=foot / height)
    for k, s in enumerate(SYSTEMS[1:]):
        a = fig.add_subplot(gs[0, k])
        draw_map(a, L, s, t, size=0.22, outline=0.55, arrow=0.75)
        a.text(0, 1.175, f"({'abc'[k]}) {LOSS[s]}", transform=a.transAxes, fontsize=7.6, fontweight="bold",
               va="bottom", color=t["ink"])
        a.text(0, 1.025, WHERE[s], transform=a.transAxes, fontsize=7, va="bottom", color=t["muted"])
        c = S[s]["change_median"]
        a.text(0.5, -0.035, f"speech change {c['speech']:.0f}%", transform=a.transAxes, fontsize=7.3, va="top",
               ha="center", color=t["speech"], fontweight="bold")
    dot = {part: Line2D([], [], ls="", marker="o", ms=3.4, mfc=COL[part], mec="none") for part in COL}
    dashed = tuple(Line2D([], [], color=t[part], lw=0.75, ls=(0, (2.4, 1.4))) for part in COL)
    moves = FancyArrow(0, 0, 1, 0, color=t["speech"])
    stays = Line2D([], [], ls="", marker="o", ms=3.4, mfc=t["bg"], mec=t["speech"], mew=0.8)
    # filled column by column: the first row reads speech, non-speech, outline; the second the two speech marks
    fig.legend([dot["speech"], moves, dot["nonspeech"], stays, dashed],
               ["speech", "speech set shifts", "non-speech", "speech set stays", "original positions"],
               handler_map={FancyArrow: ArrowKey(), tuple: HandlerTuple(ndivide=None, pad=0.2)}, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=6.8, handlelength=2.0, handletextpad=0.4,
               columnspacing=0.9, labelspacing=0.35, borderaxespad=0.15, borderpad=0.1)
    fig.savefig(path, dpi=600, facecolor=t["bg"])
    plt.close(fig)


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
    paper_figure(res, out.with_name(out.stem + "-paper.pdf"))
    print("wrote", out, out.with_suffix(".png"))


if __name__ == "__main__":
    main()
