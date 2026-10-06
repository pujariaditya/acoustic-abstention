"""Collect the evaluation runs into the results tables, and check each number against the paper's value.

    python -m aae.report --runs runs                              # prints the tables, writes runs/results.json

Expected layout (``scripts/reproduce.sh`` produces it):

    runs/complete/<model>/<system>/{metrics.json,transcripts.jsonl}
        <system>: original, aae, the ablation variants (no-preservation, transcript-ce, independent-noise,
        decoder-adapter), aae-run1 for Hojo-ASR-V1 and aae-8500 for Whisper large-v3
    runs/heldout/<model>/{original,aae}/{metrics.json,transcripts.jsonl}     the held-out-noise speech sets
    runs/vad/silero.jsonl                                                    Silero VAD decisions (aae.vad)
    runs/crosswalk/{urban,fsd50k}_{original,aae}.json
    runs/stress/<model>.json
    runs/locus/whisper-large-v3.json
    runs/separability/whisper-large-v3.json
    runs/states/whisper-large-v3.json

The tables are printed in Markdown, in the order and form of docs/RESULTS.md. Every number with an entry in
``assets/paper_results.json`` is compared with it as printed; a mismatch is flagged, not hidden. Sections whose runs
are missing are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from aae import metrics, models, vad

ROOT = Path(__file__).resolve().parents[1]
MODELS = models.MODEL_NAMES
SHORT = {"whisper-large-v3": "Whisper", "qwen3-asr-1.7b": "Qwen3", "ark-asr-3b": "ARK", "hojo-asr-v1": "Hojo"}
NONSPEECH = metrics.NONSPEECH_LEGS
SPEECH = ("ls_clean", "ls_other", "tail", "snr5", "fleurs_zh", "aishell")
HELDOUT = metrics.SPEECH_HELDOUT_LEGS
OOD_TITLE = {"silence": "digital silence", "white_noise": "white noise", "pink_noise": "pink noise",
             "nsynth": "NSynth notes"}
TITLE = {"urban": "UrbanSound8K", "wham": "WHAM! noise", "fsd50k": "FSD50K, filtered",
         "fsd50k_all": "FSD50K, all evaluation clips", "ls_clean": "clean", "ls_other": "other", "tail": "tail",
         "snr5": "5 dB", "fleurs_zh": "FLEURS", "aishell": "AISHELL-1",
         "tail_heldout": "tail, held-out noise", "snr5_heldout": "5 dB, held-out noise"}
VARIANTS = {"aae": "AAE (full recipe)", "no-preservation": "no preservation, λ = 0",
            "transcript-ce": "transcript CE on m", "independent-noise": "independent noise in m",
            "decoder-adapter": "adapters in the decoder"}
ENCODER_LAYERS = {"whisper-large-v3": "16 to 31 of 32", "qwen3-asr-1.7b": "12 to 23 of 24",
                  "ark-asr-3b": "16 to 31 of 32", "hojo-asr-v1": "16 to 31 of 32"}
# UrbanSound8K clip ids are fsID-classID-occurrence-slice
URBAN_CLASSES = ("air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling",
                 "gun_shot", "jackhammer", "siren", "street_music")
EDITS = ("aae", "transcript-ce", "decoder-adapter")                     # the Whisper edits of aae.locus
EDGES = list(range(-30, 31))                                            # 1-logit bins of the decision histograms


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _transcripts(run_dir):
    p = Path(run_dir) / "transcripts.jsonl"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return {(r["leg"], r["id"]): r for r in map(json.loads, f)}


def reference(model: str) -> str:
    """The full-recipe row of the ablation, trained like the variants: Hojo-ASR-V1's first run, whose sampling matches
    theirs, and Whisper large-v3's run at 8,500 steps, their length."""
    return {"hojo-asr-v1": "aae-run1", "whisper-large-v3": "aae-8500"}.get(model, "aae")


def interval_text(lo: float, hi: float) -> str:
    return f"{signed(lo)} to {signed(hi)}"


def signed(v: float) -> str:
    """A change in points as printed: explicit sign, two decimals, and 0.00 without a sign."""
    s = f"{abs(v):.2f}"
    return "0.00" if s == "0.00" else ("+" if v > 0 else "-") + s


def pct(flags) -> float:
    flags = [bool(x) for x in flags]
    return 100.0 * sum(flags) / len(flags)


class Checker:
    """Formats a number and compares it with the paper's value for the same key, as printed."""

    def __init__(self, expected):
        self.expected, self.checked, self.differ = expected, 0, []

    def __call__(self, key, value, fmt="{:.2f}"):
        shown = fmt(value) if callable(fmt) else fmt.format(value)
        want = self.expected.get(key)
        if want is None:
            return shown
        self.checked += 1
        if shown != want:
            self.differ.append((key, shown, want))
            return f"{shown} (paper {want})"
        return shown


def report(runs: Path, expected: dict) -> dict:
    chk = Checker(expected)
    res, out = {}, []
    comp = {m: {s: _load(runs / "complete" / m / s / "metrics.json") for s in ("original", "aae")} for m in MODELS}
    have = [m for m in MODELS if comp[m]["original"] and comp[m]["aae"]]
    tx_cache = {}

    def tx(m, system, part="complete"):
        if (part, m, system) not in tx_cache:
            tx_cache[(part, m, system)] = _transcripts(runs / part / m / system)
        return tx_cache[(part, m, system)]

    def same(m, system, legs, part="complete"):
        """{leg: (identical, n)}: transcripts equal to the original model's after normalization."""
        a, b = tx(m, "original", part), tx(m, system, part)
        if not (a and b):
            return None
        return {leg: (sum(metrics.normalize(a[k]["text"], leg) == metrics.normalize(b[k]["text"], leg)
                          for k in a if k[0] == leg), sum(1 for k in a if k[0] == leg)) for leg in legs}

    def empty(run, leg):
        x = run["legs"][leg]
        return round(x.get("empty_rate", 0.0) * x["n"] / 100)

    def interval(m, system, leg, part="complete"):
        """The paired bootstrap 95% interval of the change in WER or CER against the original model (points)."""
        a, b = tx(m, "original", part), tx(m, system, part)
        keys = [k for k in a if k[0] == leg]
        ea, n = metrics.edits(leg, [a[k] for k in keys])
        eb, _ = metrics.edits(leg, [b[k] for k in keys])
        return metrics.change_interval(ea, eb, n)

    # ------------------------------------------------------------------ 1. hallucination on the complete sets
    if have:
        out += ["## 1. Hallucination on the complete non-speech sets", "",
                "Clips that receive any text, with the rate (%) in parentheses, and the 95% Wilson interval of the "
                "rate with AAE.", "",
                "| | original model | AAE | AAE, 95% interval (%) |", "|---|---|---|---|"]
        for leg in (*NONSPEECH, "fsd50k_all"):
            rows = [m for m in have if leg in comp[m]["aae"]["legs"] and leg in comp[m]["original"]["legs"]]
            if not rows:
                continue
            out.append(f"| ***{TITLE[leg]}, {comp[rows[0]]['aae']['legs'][leg]['n']:,} clips*** | | | |")
            for m in rows:
                cells = []
                for s in ("original", "aae"):
                    x = comp[m][s]["legs"][leg]
                    k = round(x["hr"] * x["n"] / 100)
                    res.setdefault("hallucination", {}).setdefault(m, {}).setdefault(leg, {})[s] = \
                        {"hr": x["hr"], "n": x["n"], "count": k, "errors": x.get("n_errors", 0)}
                    cells.append(f"{chk(f'count/{m}/{leg}/{s}', k, '{:,}')} ({chk(f'hr/{m}/{leg}/{s}', x['hr'])})")
                lo, hi = metrics.wilson(res["hallucination"][m][leg]["aae"]["count"], comp[m]["aae"]["legs"][leg]["n"])
                cells.append(chk(f"wilson/{m}/{leg}", f"{100 * lo:.2f}–{100 * hi:.2f}", "{}"))
                out.append(f"| {models.spec(m).title} | " + " | ".join(cells) + " |")

        overlap = ROOT / "assets" / "manifests" / "musan_overlap.csv"   # evaluation clips cut from MUSAN noise
        with open(overlap, encoding="utf-8") as f:
            ov = list(csv.DictReader(f))
        excl = {"urban": {r["clip"] for r in ov if r["set"] == "urbansound8k"},
                "fsd50k": {r["clip"] for r in ov if r["set"] == "fsd50k" and r["in_filtered_fsd50k"] == "yes"}}
        changes, texted = [], 0
        for m in have:
            t = tx(m, "aae") or {}
            for leg in ("urban", "fsd50k"):
                recs = [r for k, r in t.items() if k[0] == leg]
                if not recs:
                    continue
                has = [bool(r.get("error")) or metrics.nonempty(r.get("text")) for r in recs]
                keep = [h for r, h in zip(recs, has, strict=True) if r["id"] not in excl[leg]]
                texted += sum(h for r, h in zip(recs, has, strict=True) if r["id"] in excl[leg])
                changes.append(100 * sum(keep) / len(keep) - 100 * sum(has) / len(has))
        if changes:
            res["musan_overlap"] = {"clips": len(ov), "aae_with_text": texted,
                                    "largest_change": max(changes, key=abs)}
            out += ["", f"Without the {chk('musan/clips', len(ov), '{:d}')} evaluation clips cut from MUSAN noise "
                        f"recordings used in training (`assets/manifests/musan_overlap.csv`), each AAE rate changes by at "
                        f"most {chk('musan/largest_change', max(abs(c) for c in changes), '{:.4f}')} points; "
                        f"{chk('musan/aae_with_text', texted, '{:d}')} of those clips receive text with AAE."]

        # out-of-domain non-speech (aae.ood): runs/ood/<model>/<original|aae>
        ood = {m: {x: _load(runs / "ood" / m / x / "metrics.json") for x in ("original", "aae")} for m in have}
        if all(ood[m]["original"] and ood[m]["aae"] for m in have):
            legs = metrics.NONSPEECH_OOD_LEGS
            res["ood"] = {m: {x: {leg: ood[m][x]["legs"][leg]["hr"] for leg in legs} for x in ood[m]} for m in have}
            out += ["", "### Out-of-domain non-speech", "",
                    "Sets from outside the sources of the training pools (`python -m aae.ood`): 1,000 clips each of "
                    "digital silence, white noise and pink noise (1 to 10 s, -60 to -20 dBFS), and the "
                    f"{ood[have[0]]['aae']['legs']['nsynth']['n']:,} NSynth test notes outside its vocal family. HR (%), "
                    "original model / AAE:", "",
                    "| model | " + " | ".join(OOD_TITLE[leg] for leg in legs) + " |", "|---|" + "---|" * len(legs)]
            for m in have:
                out.append(f"| {models.spec(m).title} | " + " | ".join(
                    f"{chk(f'ood/{m}/{leg}/original', res['ood'][m]['original'][leg])} / "
                    f"{chk(f'ood/{m}/{leg}/aae', res['ood'][m]['aae'][leg])}" for leg in legs) + " |")
            n = {leg: ood[have[0]]["aae"]["legs"][leg]["n"] for leg in legs}
            texted = sum(round(res["ood"][m]["aae"][leg] * n[leg] / 100) for m in have for leg in legs)
            hi_set = 100 * metrics.wilson(0, n["silence"])[1]
            hi_nsynth = 100 * metrics.wilson(0, n["nsynth"])[1]
            out += ["", f"With AAE, the models return text for {chk('ood/aae_texted', texted, '{:d}')} of their "
                        f"{len(have) * sum(n.values()):,} clips ({sum(n.values()):,} per model). For a rate of 0, the "
                        f"95% interval ends at {chk('ood/wilson_1000', hi_set)}% on a 1,000-clip set and at "
                        f"{chk('ood/wilson_nsynth', hi_nsynth)}% on NSynth."]

    # ------------------------------------------------------------------ 2. speech on the complete test sets
    if have:
        out += ["", "## 2. Speech accuracy on the complete test sets", "",
                "WER (CER for the Mandarin sets); *same*: transcripts identical to the original model's after "
                "normalization (%); *empty*: utterances returned empty, original model / AAE.", "",
                "| model | | " + " | ".join(TITLE[leg] for leg in SPEECH) + " |", "|---|---|" + "---|" * len(SPEECH)]
        for m in have:
            for s, label in (("original", "original model"), ("aae", "AAE")):
                vals = {leg: comp[m][s]["legs"][leg][metrics.metric_of(leg)] for leg in SPEECH}
                res.setdefault("speech", {}).setdefault(m, {})[s] = vals
                out.append(f"| {models.spec(m).title + ' ' if s == 'original' else ''}| {label} | "
                           + " | ".join(chk(f"speech/{m}/{leg}/{s}", vals[leg]) for leg in SPEECH) + " |")
            sm = same(m, "aae", SPEECH)
            if sm:
                res.setdefault("identical", {})[m] = {leg: {"identical": k, "n": n} for leg, (k, n) in sm.items()}
                out.append("| | same (%) | " + " | ".join(chk(f"same/{m}/{leg}", 100 * k / n, "{:.1f}")
                                                         for leg, (k, n) in sm.items()) + " |")
            em = {leg: [empty(comp[m][s], leg) for s in ("original", "aae")] for leg in SPEECH}
            res.setdefault("empty", {})[m] = em
            out.append("| | empty | " + " | ".join(
                f"{chk(f'empty/{m}/{leg}/original', em[leg][0], '{:d}')} / "
                f"{chk(f'empty/{m}/{leg}/aae', em[leg][1], '{:d}')}" for leg in SPEECH) + " |")
        sp = res["speech"]
        deltas = [(round(round(sp[m]["aae"][leg], 2) - round(sp[m]["original"][leg], 2), 2), m, leg)
                  for m in have for leg in SPEECH]
        top = max(deltas)
        res["speech_change"] = {"pairs": len(deltas), "within_0.25": sum(d <= 0.25 for d, _, _ in deltas),
                                "largest_increase": top[0], "largest_increase_at": top[1:],
                                "largest_decrease": -min(deltas)[0]}
        out += ["", f"Changes against the original model, from the printed rates: largest increase "
                    f"{chk('speech/largest_increase', top[0])} ({models.spec(top[1]).title}, {TITLE[top[2]]}); "
                    f"{sum(d <= 0.25 for d, _, _ in deltas)} of {len(deltas)} pairs change by at most 0.25 points."]
        ci = {m: {leg: interval(m, "aae", leg) for leg in SPEECH} for m in have if tx(m, "original") and tx(m, "aae")}
        if ci:
            res["speech_interval"] = ci
            out += ["", "Paired bootstrap 95% interval of each change (points): 10,000 draws of the utterances with "
                        "replacement, both systems scored on every draw (`aae.metrics.change_interval`).", "",
                    "| model | " + " | ".join(TITLE[leg] for leg in SPEECH) + " |", "|---|" + "---|" * len(SPEECH)]
            for m, row in ci.items():
                out.append(f"| {models.spec(m).title} | " + " | ".join(
                    chk(f"interval/{m}/{leg}", interval_text(*row[leg]), "{}") for leg in SPEECH) + " |")
            above = [(m, leg) for m, row in ci.items() for leg in SPEECH if row[leg][1] > metrics.TOLERANCE[leg]]
            pairs = sum(len(row) for row in ci.values())
            res["speech_interval_above"] = above
            out += ["", f"{chk('interval/below_tolerance', pairs - len(above), '{:d}')} of {pairs} upper limits lie "
                        f"below the tolerance of their set; above it: "
                        + ("; ".join(f"{models.spec(m).title}, {TITLE[leg]}" for m, leg in above) or "none") + "."]

        ho = {m: {s: _load(runs / "heldout" / m / s / "metrics.json") for s in ("original", "aae")} for m in MODELS}
        have_ho = [m for m in MODELS if ho[m]["original"] and ho[m]["aae"]]
        if have_ho:
            out += ["", "### Speech with held-out noise", "",
                    "The 1,000 utterances of *tail* and *5 dB*, mixed the same way with WHAM! noise from its "
                    "evaluation splits (cv, tt), which no model saw in training "
                    "(`assets/manifests/complete/*_heldout.jsonl`): WER, the change with its paired bootstrap 95% "
                    "interval (points), and transcripts identical to the original model's (*same*, %).", "",
                    "| model | set | original model | AAE | change (95% interval) | same |",
                    "|---|---|---|---|---|---|"]
            for m in have_ho:
                for leg in HELDOUT:
                    w = {sy: ho[m][sy]["legs"][leg]["wer"] for sy in ("original", "aae")}
                    lo, hi = interval(m, "aae", leg, part="heldout")
                    k, n = same(m, "aae", [leg], part="heldout")[leg]
                    change = round(round(w["aae"], 2) - round(w["original"], 2), 2)
                    res.setdefault("heldout", {}).setdefault(m, {})[leg] = {**w, "change": change, "interval": [lo, hi],
                                                                            "same": 100 * k / n}
                    out.append(f"| {models.spec(m).title if leg == HELDOUT[0] else ''} | {TITLE[leg]} | "
                               f"{chk(f'heldout/{m}/{leg}/original', w['original'])} | "
                               f"{chk(f'heldout/{m}/{leg}/aae', w['aae'])} | "
                               f"{chk(f'heldout/{m}/{leg}/change', change, signed)} "
                               f"({chk(f'heldout/{m}/{leg}/interval', interval_text(lo, hi), '{}')}) | "
                               f"{chk(f'heldout/{m}/{leg}/same', 100 * k / n, '{:.1f}')} |")
            h = res["heldout"]
            top = max((v["change"], m, leg) for m in h for leg, v in h[m].items())
            sames = [v["same"] for m in h for v in h[m].values()]
            his = [v["interval"][1] for m in h for v in h[m].values()]
            out += ["", f"Largest increase {chk('heldout/largest_increase', top[0], signed)} "
                        f"({models.spec(top[1]).title}, {TITLE[top[2]]}); every upper limit at most "
                        f"{chk('heldout/interval_max', max(his), signed)}; identical transcripts "
                        f"{chk('heldout/same_min', min(sames), '{:.1f}')}% to "
                        f"{chk('heldout/same_max', max(sames), '{:.1f}')}%."]

    # ------------------------------------------------------------------ 2b. the models' own default decoding
    # runs/default/<model>/<original|aae>: Whisper's transcribe() and Hojo-ASR-V1's released beam search (aae.evaluate
    # --decoding default), on every complete set
    dflt = {m: {x: _load(runs / "default" / m / x / "metrics.json") for x in ("original", "aae")}
            for m in ("whisper-large-v3", "hojo-asr-v1") if m in have}
    dflt = {m: v for m, v in dflt.items() if v["original"] and v["aae"]
            and all(leg in v[x]["legs"] for x in v for leg in (*NONSPEECH, *SPEECH))}
    if dflt:
        res["default"] = {}
        out += ["", "### Default decoding", "",
                "Whisper large-v3 with `whisper.transcribe` and its defaults (temperature fallback, the skip rule, "
                "conditioning on previous text), and Hojo-ASR-V1 with its released 4-beam search (repetition penalty "
                "2.0); original model and AAE. HR (%) on the non-speech sets, WER or CER (%) on the speech sets, and "
                "*same*: transcripts identical to the original model's under the same decoding (%):", "",
                "| model | | " + " | ".join(TITLE[leg].split(",")[0] for leg in NONSPEECH) + " | "
                + " | ".join(TITLE[leg] for leg in SPEECH) + " | same |", "|---|---|" + "---|" * 10]
        notes = []
        for m, v in dflt.items():
            hr = {x: {leg: v[x]["legs"][leg]["hr"] for leg in NONSPEECH} for x in v}
            err = {x: {leg: v[x]["legs"][leg][metrics.metric_of(leg)] for leg in SPEECH} for x in v}
            change = {leg: round(round(err["aae"][leg], 2) - round(err["original"][leg], 2), 2) for leg in SPEECH}
            beyond = [leg for leg in SPEECH if metrics.beyond_tolerance(leg, err["aae"][leg] - err["original"][leg])]
            sm = same(m, "aae", SPEECH, part="default")
            pooled = None if sm is None else 100 * sum(k for k, _ in sm.values()) / sum(n for _, n in sm.values())
            res["default"][m] = {"hr": hr, "err": err, "change": change, "beyond": beyond, "same": pooled}
            for x, label in (("original", "original model"), ("aae", "AAE")):
                cells = [chk(f"default/{m}/{leg}/{x}", hr[x][leg]) for leg in NONSPEECH] + \
                        [chk(f"default/{m}/{leg}/{x}", err[x][leg]) for leg in SPEECH]
                tail = chk(f"default/{m}/same", pooled, "{:.1f}") if x == "aae" and pooled is not None else ""
                out.append(f"| {models.spec(m).title + ' ' if x == 'original' else ''}| {label} | "
                           + " | ".join(cells) + f" | {tail} |")
            notes.append(f"{models.spec(m).title}: with AAE the largest speech increase is "
                         f"{chk(f'default/{m}/largest_increase', max(change.values()), signed)} points, "
                         f"{chk(f'default/{m}/beyond', len(beyond), '{:d}')} of {len(SPEECH)} sets beyond their "
                         f"tolerance; HR at most {chk(f'default/{m}/hr_max', max(hr['aae'].values()))}% (original "
                         f"model: {chk(f'default/{m}/hr_original_min', min(hr['original'].values()))}% to "
                         f"{chk(f'default/{m}/hr_original_max', max(hr['original'].values()))}%).")
            for leg in beyond:              # where a set crosses its tolerance: how many utterances carry the change
                a, b = tx(m, "original", "default"), tx(m, "aae", "default")
                keys = [k for k in a if k[0] == leg]
                ea, n = metrics.edits(leg, [a[k] for k in keys])
                eb, _ = metrics.edits(leg, [b[k] for k in keys])
                big = {i for i in range(len(keys)) if eb[i] - ea[i] > 20}
                rest = [i for i in range(len(keys)) if i not in big]
                without = 100 * (sum(eb[i] for i in rest) - sum(ea[i] for i in rest)) / sum(n[i] for i in rest)
                res["default"][m].setdefault("outliers", {})[leg] = {"utterances": len(big), "change_without": without}
                notes.append(f"{TITLE[leg]}: {chk(f'default/{m}/{leg}/outliers', len(big), '{:d}')} of {len(keys):,} "
                             f"utterances gain more than 20 {'characters' if metrics.metric_of(leg) == 'cer' else 'words'} "
                             f"of error each, {sum(eb[i] - ea[i] for i in big):,} of the "
                             f"{sum(eb) - sum(ea):,} added; without them the change is "
                             f"{chk(f'default/{m}/{leg}/change_without', without, signed)} points.")
        en = [leg for leg in SPEECH if leg not in metrics.SPEECH_ZH_LEGS]
        for m in dflt:                      # words written beyond the reference: text over the noise after the speech
            ins = {x: {leg: metrics.insertions(leg, [r for k, r in tx(m, x, "default").items() if k[0] == leg])
                       for leg in en} for x in ("original", "aae")}
            res["default"][m]["insertions"] = ins
            notes.append(f"{models.spec(m).title}: words inserted, original model / AAE (after the last reference "
                         f"word): " + "; ".join(
                f"{TITLE[leg]} {chk(f'default/{m}/{leg}/insertions/original', ins['original'][leg][0], '{:d}')} / "
                f"{chk(f'default/{m}/{leg}/insertions/aae', ins['aae'][leg][0], '{:d}')} "
                f"({chk(f'default/{m}/{leg}/insertions_after/original', ins['original'][leg][1], '{:d}')} / "
                f"{chk(f'default/{m}/{leg}/insertions_after/aae', ins['aae'][leg][1], '{:d}')})" for leg in en) + ".")
        out += [""] + notes

    # ------------------------------------------------------------------ 2c. the language of the speech transcripts
    parts = [("complete", "greedy decoding", have)] + ([("default", "default decoding", list(dflt))] if dflt else [])
    out += ["", "### Output language", "",
            "Speech transcripts whose script (Chinese characters or Latin letters, `aae.metrics.script`) differs "
            "between the original model and AAE, out of all six speech sets; and utterances of the two Mandarin sets "
            "answered in Latin script, original model / AAE.", "",
            "| model | decoding | script changed | Mandarin answered in Latin script |", "|---|---|---|---|"]
    res["script"] = {}
    for part, label, ms in parts:
        for m in ms:
            a, b = tx(m, "original", part), tx(m, "aae", part)
            keys = [k for k in a if k[0] in SPEECH]
            changed = sum(1 for k in keys if "" not in (metrics.script(a[k]["text"]), metrics.script(b[k]["text"]))
                          and metrics.script(a[k]["text"]) != metrics.script(b[k]["text"]))
            latin = {x: sum(1 for k in keys if k[0] in metrics.SPEECH_ZH_LEGS and metrics.script(t[k]["text"]) == "en")
                     for x, t in (("original", a), ("aae", b))}
            res["script"][f"{part}/{m}"] = {"changed": changed, "n": len(keys), "latin_zh": latin}
            out.append(f"| {models.spec(m).title} | {label} | {chk(f'script/{part}/{m}/changed', changed, '{:d}')} "
                       f"of {len(keys):,} | {chk(f'script/{part}/{m}/latin_zh/original', latin['original'], '{:d}')} / "
                       f"{chk(f'script/{part}/{m}/latin_zh/aae', latin['aae'], '{:d}')} |")
    greedy = [v for k, v in res["script"].items() if k.startswith("complete/")]
    out += ["", f"With greedy decoding, AAE changes the script of "
                f"{chk('script/complete/changed_total', sum(v['changed'] for v in greedy), '{:d}')} of "
                f"{sum(v['n'] for v in greedy):,} speech transcripts."]

    # ------------------------------------------------------------------ 3. three definitions on Whisper large-v3
    cw = {f"{leg}_{s}": _load(runs / "crosswalk" / f"{leg}_{s}.json")
          for leg in ("urban", "fsd50k") for s in ("original", "aae")}
    if any(cw.values()):
        out += ["", "## 3. Whisper large-v3 under the three definitions", "",
                "One decoding pass per system and set, each clip scored under every definition (%).", "",
                "| set | system | any text | text kept by the skip rule | text with p_ns ≤ 0.6 | clips the skip rule keeps, "
                "with or without text (the count of Aparin et al.) |",
                "|---|---|---|---|---|---|"]
        for key, d in cw.items():
            if d:
                cells = [chk(f"crosswalk/{key}/{h}", d[h]) for h in ("HR_raw", "HR_whisper", "HR_nsp")] + \
                        [chk(f"crosswalk/{key}/kept", d["filter_keeps"])]
                res.setdefault("crosswalk", {})[key] = {h: d[h] for h in ("HR_raw", "HR_whisper", "HR_nsp",
                                                                           "filter_keeps", "nsp_below_0.6")}
                leg, s = key.split("_")
                out.append(f"| {TITLE[leg]} | {'original model' if s == 'original' else 'AAE'} | "
                           + " | ".join(cells) + " |")

    # ------------------------------------------------------------------ 3b. gating with a voice activity detector
    vad_path = runs / "vad" / "silero.jsonl"
    if vad_path.exists() and have and all(tx(m, "original") for m in have):
        with open(vad_path, encoding="utf-8") as f:
            gate = {(r["leg"], r["id"]): vad.passes(r) for r in map(json.loads, f)}
        legs = (*NONSPEECH, *SPEECH)
        passed = {leg: pct(v for k, v in gate.items() if k[0] == leg) for leg in legs}
        rejected = {leg: sum(not v for k, v in gate.items() if k[0] == leg) for leg in SPEECH}
        res["vad"] = {"passed": passed, "rejected_speech": rejected, "models": {}}
        out += ["", "### Gating with a voice activity detector", "",
                "Silero VAD (`silero-vad` 6.2.3, default settings) in front of each original model: a clip with "
                "at least one speech segment goes to the model unchanged, any other clip gets an empty transcript "
                "(`python -m aae.vad`). HR (%) on the non-speech sets and the change in WER or CER against the "
                "original model (points) on the speech sets, for the gated original model and for AAE.", "",
                "| model | system | " + " | ".join(TITLE[leg] for leg in legs) + " |", "|---|---|" + "---|" * len(legs)]
        cell = lambda leg, v: f"{v:.2f}" if leg in NONSPEECH else signed(v)              # noqa: E731
        for m in have:
            orig = tx(m, "original")
            gated = {k: (r if gate[k] else {**r, "text": "", "error": None}) for k, r in orig.items() if k in gate}
            row = {}
            for leg in legs:
                recs = [r for k, r in gated.items() if k[0] == leg]
                x = metrics.leg_metrics(leg, recs)
                if leg in NONSPEECH:
                    row[leg] = x["hr"]
                else:
                    base = comp[m]["original"]["legs"][leg][metrics.metric_of(leg)]
                    row[leg] = round(round(x[metrics.metric_of(leg)], 2) - round(base, 2), 2)
            res["vad"]["models"][m] = row
            aae_row = {leg: comp[m]["aae"]["legs"][leg]["hr"] if leg in NONSPEECH else
                       round(round(comp[m]["aae"]["legs"][leg][metrics.metric_of(leg)], 2)
                             - round(comp[m]["original"]["legs"][leg][metrics.metric_of(leg)], 2), 2) for leg in legs}
            out.append(f"| {models.spec(m).title} | VAD gate | " + " | ".join(
                chk(f"vad/{m}/{leg}", row[leg], lambda v, leg=leg: cell(leg, v)) for leg in legs) + " |")
            out.append("| | AAE | " + " | ".join(cell(leg, aae_row[leg]) for leg in legs) + " |")
        hrs = [res["vad"]["models"][m][leg] for m in have for leg in NONSPEECH]
        lower = sum(comp[m]["aae"]["legs"][leg]["hr"] < res["vad"]["models"][m][leg] for m in have for leg in NONSPEECH)
        out += ["", "The gate passes " + ", ".join(
                    f"{chk(f'vad/passed/{leg}', passed[leg], '{:.1f}')}% of {TITLE[leg]}" for leg in NONSPEECH)
                + f" clips to the recognizer, so the gated models return text for "
                  f"{chk('vad/hr_min', min(hrs))}% to {chk('vad/hr_max', max(hrs))}% of the non-speech clips; it "
                  f"rejects {chk('vad/speech_rejected', sum(rejected.values()), '{:d}')} of "
                  f"{sum(comp[have[0]]['original']['legs'][leg]['n'] for leg in SPEECH):,} speech utterances ("
                + ", ".join(f"{TITLE[leg]} {rejected[leg]}" for leg in SPEECH) + "). AAE returns text for fewer "
                  f"non-speech clips than the gated model on {chk('vad/aae_lower', lower, '{:d}')} of {len(hrs)} model "
                  "and set pairs."]

    # ------------------------------------------------------------------ 4. ablation study on the complete sets
    def ablation_row(m, system):
        run, orig = _load(runs / "complete" / m / system / "metrics.json"), comp[m]["original"]
        if not (run and orig) or not all(leg in run["legs"] for leg in (*NONSPEECH, *SPEECH)):
            return None
        hr = {leg: run["legs"][leg]["hr"] for leg in NONSPEECH}
        err = {leg: run["legs"][leg][metrics.metric_of(leg)] for leg in SPEECH}
        base = {leg: orig["legs"][leg][metrics.metric_of(leg)] for leg in SPEECH}
        beyond = [leg for leg in SPEECH if metrics.beyond_tolerance(leg, err[leg] - base[leg])]
        sm = same(m, system, SPEECH)
        return {"hr": hr, "hr_mean": sum(hr.values()) / len(hr), "err": err,
                "change": {leg: round(round(err[leg], 2) - round(base[leg], 2), 2) for leg in SPEECH},
                "increase": {leg: err[leg] - base[leg] for leg in SPEECH}, "beyond": beyond,
                "same": None if sm is None else 100 * sum(k for k, _ in sm.values()) / sum(n for _, n in sm.values()),
                "empty": sum(empty(run, leg) for leg in SPEECH),
                "speech_n": sum(run["legs"][leg]["n"] for leg in SPEECH)}

    # rows keyed by the checkpoint they evaluate: the full-recipe row is aae (aae-run1 for Hojo-ASR-V1, whose second
    # run, aae, is listed as well)
    systems = {m: [reference(m), *list(VARIANTS)[1:]] + (["aae"] if reference(m) != "aae" else []) for m in have}
    abl = {m: {s: ablation_row(m, s) for s in systems[m]} for m in have}
    if any(r for rows in abl.values() for r in rows.values()):
        res["ablation"] = abl
        out += ["", "## 4. Ablation study on the complete sets", "",
                "Mean HR over the three non-speech sets (%), and in parentheses the number of the six speech sets "
                "beyond their tolerance; bold: every speech set within its tolerance.", "",
                "| variant | " + " | ".join(models.spec(m).title for m in have) + " |", "|---|" + "---|" * len(have)]
        for v, label in VARIANTS.items():
            cells = []
            for m in have:
                s = reference(m) if v == "aae" else v
                r = abl[m][s]
                if r is None:
                    cells.append("")
                    continue
                c = (f"{chk(f'ablation/{m}/{s}/hr', r['hr_mean'])} "
                     f"({chk(f'ablation/{m}/{s}/beyond', len(r['beyond']), '{:d}')})")
                cells.append(f"**{c}**" if not r["beyond"] else c)
            out.append(f"| {label} | " + " | ".join(cells) + " |")

        out += ["", "Per set: HR (%), change in WER or CER against the original model (points; bold: beyond the "
                    "tolerance), transcripts identical to the original model's (*same*, %, over all speech "
                    "utterances) and speech utterances returned empty.", "",
                "| variant | " + " | ".join(TITLE[leg].split(",")[0] for leg in NONSPEECH) + " | "
                + " | ".join(TITLE[leg] for leg in SPEECH) + " | same | empty |", "|---|" + "---|" * 11]
        for m in have:
            out.append(f"| ***{models.spec(m).title}*** |" + " |" * 11)
            runs_of = {"hojo-asr-v1": {"aae-run1": "AAE, first run", "aae": "AAE, second run"},
                       "whisper-large-v3": {"aae-8500": "AAE, 8,500 steps", "aae": "AAE, 9,250 steps"},
                       }.get(m, {"aae": "AAE"})
            labels = {**{v: label for v, label in VARIANTS.items() if v != "aae"}, **runs_of}
            order = systems[m][:1] + systems[m][len(VARIANTS):] + systems[m][1:len(VARIANTS)]
            for s in order:
                r = abl[m][s]
                if r is None:
                    continue
                hr = " | ".join(chk(f"ablation/{m}/{s}/hr/{leg}", r["hr"][leg]) for leg in NONSPEECH)
                ch = []
                for leg in SPEECH:
                    c = chk(f"ablation/{m}/{s}/change/{leg}", r["change"][leg], signed)
                    ch.append(f"**{c}**" if leg in r["beyond"] else c)
                sm = "" if r["same"] is None else chk(f"ablation/{m}/{s}/same", r["same"], "{:.1f}")
                out.append(f"| {labels[s].split(' (')[0]} | {hr} | " + " | ".join(ch)
                           + f" | {sm} | {chk(f'ablation/{m}/{s}/empty', r['empty'], '{:,}')} |")

        crossings = [(m, s, leg, *interval(m, s, leg)) for m in have for s in ("transcript-ce", "independent-noise",
                                                                                   "decoder-adapter")
                     if abl[m].get(s) for leg in abl[m][s]["beyond"]]
        if crossings:
            clear = [c for c in crossings if c[3] > metrics.TOLERANCE[c[2]]]
            res["ablation_crossings"] = [{"model": m, "variant": s, "set": leg, "interval": [lo, hi],
                                          "clear": (m, s, leg, lo, hi) in clear} for m, s, leg, lo, hi in crossings]
            out += ["", "Speech sets beyond their tolerance with transcript CE, independent noise or decoder adapters, "
                        "with the paired bootstrap 95% interval of the change (points); *clear*: the whole interval "
                        "lies beyond the tolerance.", "",
                    "| model | variant | set | change | 95% interval | tolerance | clear |",
                    "|---|---|---|---|---|---|---|"]
            for m, s, leg, lo, hi in crossings:
                out.append(f"| {models.spec(m).title} | {VARIANTS[s]} | {TITLE[leg]} | "
                           f"{signed(abl[m][s]['change'][leg])} | "
                           f"{chk(f'ablation/{m}/{s}/interval/{leg}', interval_text(lo, hi), '{}')} | "
                           f"+{metrics.TOLERANCE[leg]} | {'yes' if (m, s, leg, lo, hi) in clear else 'no'} |")
            ce_clear = sorted({m for m, s, *_ in clear if s == "transcript-ce"})
            out += ["", f"{chk('ablation/crossings/clear', len(clear), '{:d}')} of "
                        f"{chk('ablation/crossings', len(crossings), '{:d}')} crossings are clear; transcript CE has a "
                        f"clear crossing on {chk('ablation/crossings/ce_clear_models', len(ce_clear), '{:d}')} of "
                        f"{len(have)} models."]

    # ------------------------------------------------------------------ 4b. a decoder edit with output preservation
    # runs/complete/whisper-large-v3/decoder-kl-<lambda>: decoder adapters as many as AAE's, trained with the
    # abstention loss and lambda * KL(original || edited) of the next-token distributions on m (aae.train --variant
    # decoder-kl)
    m = "whisper-large-v3"
    kl_runs = {f"decoder-kl-{lam}": lam for lam in (1, 50)}
    kl = {v: ablation_row(m, v) for v in kl_runs} if m in have else {}
    kl = {v: r for v, r in kl.items() if r}
    if kl:
        res["decoder_kl"] = kl
        ref = abl[m][reference(m)]
        out += ["", "### A decoder edit with output preservation (Whisper large-v3)", "",
                "Decoder adapters with as many parameters as AAE's (rank 18 on the self-attention of every decoder "
                "layer, 5.90M), trained like the variants above with the abstention loss plus lambda times the KL "
                "divergence of the edited decoder's next-token distributions from the original model's on m, along "
                "the original model's transcript of s. HR (%), change in WER or CER (points; bold: beyond the "
                "tolerance), *same* (%) and speech utterances returned empty; AAE at 8,500 steps for reference:", "",
                "| system | " + " | ".join(TITLE[leg].split(",")[0] for leg in NONSPEECH) + " | "
                + " | ".join(TITLE[leg] for leg in SPEECH) + " | same | empty |", "|---|" + "---|" * 11]
        for v, r in [("aae-8500", ref)] + list(kl.items()):
            label = "AAE, 8,500 steps" if v == "aae-8500" else f"decoder, KL, lambda = {kl_runs[v]}"
            hr = " | ".join(chk(f"ablation/{m}/{v}/hr/{leg}", r["hr"][leg]) for leg in NONSPEECH)
            ch = []
            for leg in SPEECH:
                c = chk(f"ablation/{m}/{v}/change/{leg}", r["change"][leg], signed)
                ch.append(f"**{c}**" if leg in r["beyond"] else c)
            sm = "" if r["same"] is None else chk(f"ablation/{m}/{v}/same", r["same"], "{:.1f}")
            out.append(f"| {label} | {hr} | " + " | ".join(ch)
                       + f" | {sm} | {chk(f'ablation/{m}/{v}/empty', r['empty'], '{:,}')} |")
        for v, r in kl.items():
            out += ["", f"lambda = {kl_runs[v]}: mean HR {chk(f'ablation/{m}/{v}/hr', r['hr_mean'])}%, "
                        f"{chk(f'ablation/{m}/{v}/beyond', len(r['beyond']), '{:d}')} of {len(SPEECH)} speech sets "
                        "beyond their tolerance."]
            for leg in r["beyond"]:
                lo, hi = interval(m, v, leg)
                out.append(f"{TITLE[leg]}: change {signed(r['change'][leg])}, 95% interval "
                           f"{chk(f'ablation/{m}/{v}/interval/{leg}', interval_text(lo, hi), '{}')}.")

    # ------------------------------------------------------------------ 5. encoder edit versus decoder edit (Whisper)
    loc = _load(runs / "locus" / "whisper-large-v3.json")
    if loc and all(tx("whisper-large-v3", s) for s in ("original", *EDITS)):
        assert set(loc["edits"]) == set(EDITS), loc["edits"]
        sp = loc["speech"]["ls_clean"] + loc["speech"]["ls_other"]
        ns = loc["nonspeech"]["urban"]
        dsp = loc["decision_speech"]["ls_clean"] + loc["decision_speech"]["ls_other"]
        assert [r["id"] for r in dsp] == [r["id"] for r in sp]
        med = statistics.median
        texts = {s: {k[1]: r["text"] or "" for k, r in tx("whisper-large-v3", s).items()
                     if k[0] in ("ls_clean", "ls_other")} for s in ("original", *EDITS)}
        norm = metrics.normalize_en
        q = statistics.quantiles([r["gamma_min"] for r in sp], n=4)
        L = {"n_speech": len(sp), "n_nonspeech": len(ns), "gamma_quartiles": q,
             "gamma_median": med(r["gamma_min"] for r in sp),
             "redecode_same": pct(norm(r["text"]) == norm(texts["original"][r["id"]]) for r in sp),
             "teacher_forcing_reproduces": pct(r["tf_ok"] for r in sp),
             "g_original": med(r["text_margin"]["original"] for r in ns),
             "text_original": pct(r["text_margin"]["original"] > 0 for r in ns), "edits": {}}
        for e in EDITS:
            d = {"delta": med(r["delta_max_word"][e] for r in sp), "kl": med(r["kl_mean_word"][e] for r in sp),
                 "words": pct(r["kept_word"][e] for r in sp), "tokens": pct(r["kept"][e] for r in sp),
                 "same": pct(norm(texts[e][r["id"]]) == norm(texts["original"][r["id"]]) for r in sp),
                 "changed_by_quartile": [pct(not r["kept"][e] for r in sp if sum(r["gamma_min"] > b for b in q) == i)
                                         for i in range(4)],
                 "g": med(r["text_margin"][e] for r in ns), "text": pct(r["text_margin"][e] > 0 for r in ns)}
            if e != "decoder-adapter":
                d.update({"drift_speech": med(r["drift"][e]["audio"] for r in sp),
                          "drift_speech_window": med(r["drift"][e]["window"] for r in sp),
                          "drift_nonspeech": med(r["drift"][e]["audio"] for r in ns)})
            L["edits"][e] = d
        L["decision"] = {}
        for part, rows in (("nonspeech", ns), ("speech", dsp)):
            for e in ("original", *EDITS):
                g = [r["text_margin"][e] for r in rows]
                hist = [0] * (len(EDGES) - 1)
                for v in g:
                    hist[min(max(int(v - EDGES[0]), 0), len(hist) - 1)] += 1     # values beyond the range: end bins
                L["decision"].setdefault(part, {})[e] = {"text_pct": pct(x > 0 for x in g), "median": med(g),
                                                         "hist": hist}
        res["locus"] = L
        sg = lambda v: f"{v:+.1f}"            # noqa: E731
        out += ["", "## 5. Encoder edit versus decoder edit on Whisper large-v3", "",
                f"Speech: the complete LibriSpeech test-clean and test-other sets ({len(sp):,} utterances), each edit "
                f"teacher-forced along the original greedy path. Non-speech: the complete UrbanSound8K set "
                f"({len(ns):,} clips).", "",
                "| edit | speech drift (%) | δ | KL (nats) | words (%) | tokens (%) | same (%) | g | text (%) "
                "| non-speech drift (%) |", "|---|---|---|---|---|---|---|---|---|---|",
                f"| original model | | | | | | | {chk('locus/original/g', L['g_original'], sg)} | "
                f"{chk('locus/original/text', L['text_original'], '{:.1f}')} | |"]
        names = {"aae": "AAE: encoder, preservation loss", "transcript-ce": "encoder, transcript CE",
                 "decoder-adapter": "decoder, transcript CE"}
        for e in EDITS:
            d = L["edits"][e]
            enc = e != "decoder-adapter"
            cells = [chk(f"locus/{e}/drift_speech", 100 * d["drift_speech"], "{:.1f}") if enc else "",
                     chk(f"locus/{e}/delta", d["delta"]), chk(f"locus/{e}/kl", d["kl"], "{:.4f}"),
                     chk(f"locus/{e}/words", d["words"], "{:.1f}"), chk(f"locus/{e}/tokens", d["tokens"], "{:.1f}"),
                     chk(f"locus/{e}/same", d["same"], "{:.1f}"), chk(f"locus/{e}/g", d["g"], sg),
                     chk(f"locus/{e}/text", d["text"], "{:.1f}"),
                     chk(f"locus/{e}/drift_nonspeech", 100 * d["drift_nonspeech"], "{:.1f}") if enc else ""]
            out.append(f"| {names[e]} | " + " | ".join(cells) + " |")
        a = L["edits"]["aae"]
        dc = L["decision"]["speech"]
        out += ["", f"Re-decoding reproduces {chk('locus/redecode_same', L['redecode_same'], '{:.1f}')}% of the "
                    f"original transcripts. Median smallest margin along a transcript: "
                    f"γ = {chk('locus/gamma_median', L['gamma_median'])}; "
                    f"quartiles {q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}. "
                    f"AAE changes the token sequence of "
                    f"{chk('locus/aae/changed_q1', a['changed_by_quartile'][0], '{:.1f}')}% of the utterances in the "
                    f"lowest quartile of margins and "
                    f"{chk('locus/aae/changed_q4', a['changed_by_quartile'][3], '{:.1f}')}% in the highest; its speech "
                    f"features drift by {chk('locus/aae/drift_speech_window', 100 * a['drift_speech_window'], '{:.1f}')}% over the "
                    f"whole window. "
                    f"Decision margin g on speech, median: original model "
                    f"{chk('locus/decision/speech/original', dc['original']['median'], sg)}, AAE "
                    f"{chk('locus/decision/speech/aae', dc['aae']['median'], sg)}, decoder adapters "
                    f"{chk('locus/decision/speech/decoder-adapter', dc['decoder-adapter']['median'], sg)}."]

    # ------------------------------------------------------------------ 5b. speech and non-speech in the encoder
    sep = _load(runs / "separability" / "whisper-large-v3.json")
    if sep:
        from aae.separability import SETS as SEP_SETS

        med = statistics.median
        sg = lambda v: f"{v:+.1f}"            # noqa: E731
        g = {s: {k: [c["g"][k] for c in sep["clips"] if c["set"] == s] for k in ("original", "aae")} for s in SEP_SETS}
        ns_sets = [s for s, grp in SEP_SETS.items() if grp == "nonspeech"]
        sp_sets = [s for s, grp in SEP_SETS.items() if grp != "nonspeech"]
        pooled = {part: {k: [v for s in sets for v in g[s][k]] for k in ("original", "aae")}
                  for part, sets in (("nonspeech", ns_sets), ("speech", sp_sets))}
        # the smallest constant raise of end-of-text at the decision step that leaves no more UrbanSound8K clips with
        # text (g > b) than AAE does, and the speech utterances that the same raise would end there (g <= b)
        keep = sum(v > 0 for v in g["urban"]["aae"])
        b = sorted(g["urban"]["original"], reverse=True)[keep]
        bias = {"logits": b, "urban_text": pct(v > b for v in g["urban"]["original"]),
                "speech_empty": {s: pct(v <= b for v in g[s]["original"]) for s in sp_sets},
                "aae_speech_empty": {s: pct(v <= 0 for v in g[s]["aae"]) for s in sp_sets}}
        auc_min = min(min(r["auc_clean"], r["auc_noisy"]) for r in sep["probe"]["original"])
        res["separability"] = {"probe": sep["probe"], "decision": sep["decision"], "bias": bias, "auc_min": auc_min,
                               "g_median": {part: {k: med(v) for k, v in d.items()} for part, d in pooled.items()}}
        out += ["", "### Speech and non-speech in the original encoder", "",
                f"A logistic-regression probe per encoder layer of Whisper large-v3, fitted on the training pools "
                f"({sep['n']['train']:,} clips) and scored on the complete sets ({sep['n']['eval']:,} clips): AUC of "
                "clean speech and of speech in noise against non-speech. Layer k is the output of encoder block k, "
                "layer 32 the features after the final layer norm; AAE adapts blocks 16 to 31, so its features "
                "differ from layer 16 on.", "",
                "| layer | original: clean speech | original: speech in noise | AAE: clean speech | "
                "AAE: speech in noise |", "|---|---|---|---|---|"]
        aae_rows = {r["layer"]: r for r in sep["probe"]["aae"]}
        for r in sep["probe"]["original"]:
            n = r["layer"]
            cells = [chk(f"separability/probe/original/{n}/{a}", r[f"auc_{a}"], "{:.4f}") for a in ("clean", "noisy")]
            cells += [chk(f"separability/probe/aae/{n}/{a}", aae_rows[n][f"auc_{a}"], "{:.4f}") if n in aae_rows
                      else "" for a in ("clean", "noisy")]
            out.append(f"| {n}{' (decoder input)' if n == 32 else ''} | " + " | ".join(cells) + " |")
        out += ["", "Per set, at the decoder input (layer 32): *non-speech*: share the probe calls non-speech (%); "
                    "*text*: share on which a text token beats end-of-text at the decision step of `aae.locus` (g > 0, "
                    "%); *g*: median margin.", "",
                "| set | clips | non-speech: original | AAE | text: original | AAE | g: original | AAE |",
                "|---|---|---|---|---|---|---|---|"]
        for s in SEP_SETS:
            d = sep["decision"][s]
            cells = [chk(f"separability/{s}/{k}/probe_nonspeech", 100 * d["probe_nonspeech"][k]) for k in g[s]]
            cells += [chk(f"separability/{s}/{k}/text", 100 * d["text"][k]) for k in g[s]]
            cells += [chk(f"separability/{s}/{k}/g", d["g_median"][k], sg) for k in g[s]]
            out.append(f"| {TITLE[s]} | {d['n']:,} | " + " | ".join(cells) + " |")
        ns_probe = min(100 * sep["decision"][s]["probe_nonspeech"]["original"] for s in ns_sets)
        gm = res["separability"]["g_median"]
        out += ["", f"The probe's AUC is at least {chk('separability/auc_min', auc_min, '{:.3f}')} at every layer of "
                    f"the original model, and at the decoder input it calls "
                    f"{chk('separability/probe_nonspeech_min', ns_probe)}% to 100% of the non-speech clips non-speech. "
                    f"The decoder still prefers text on "
                    f"{chk('separability/original/text', pct(v > 0 for v in pooled['nonspeech']['original']))}% of "
                    f"the {chk('separability/nonspeech_n', len(pooled['nonspeech']['original']), '{:,}')} non-speech "
                    f"clips. Median g, non-speech and speech: original model "
                    f"{chk('separability/g/nonspeech/original', gm['nonspeech']['original'], sg)} and "
                    f"{chk('separability/g/speech/original', gm['speech']['original'], sg)}, AAE "
                    f"{chk('separability/g/nonspeech/aae', gm['nonspeech']['aae'], sg)} and "
                    f"{chk('separability/g/speech/aae', gm['speech']['aae'], sg)}. Raising end-of-text at the "
                    f"decision step by a constant {chk('separability/bias/logits', b)} logits leaves text on "
                    f"{chk('separability/bias/urban_text', bias['urban_text'])}% of UrbanSound8K clips, as many as "
                    f"AAE, and ends "
                    f"{chk('separability/bias/speech_empty_min', min(bias['speech_empty'].values()))}% to "
                    f"{chk('separability/bias/speech_empty_max', max(bias['speech_empty'].values()))}% of the speech "
                    f"utterances of a set there, against at most "
                    f"{chk('separability/aae/speech_empty_max', max(bias['aae_speech_empty'].values()))}% with AAE."]

    # ------------------------------------------------------------------ 5c. the decision state under each edit
    sta = _load(runs / "states" / "whisper-large-v3.json")
    if sta:
        S = sta["summary"]
        names = {"original": "original model", "aae": "AAE: encoder, preservation loss",
                 "transcript-ce": "encoder, transcript CE", "decoder-adapter": "decoder, transcript CE"}
        ce = [S[e]["change_median"]["speech"] / S["aae"]["change_median"]["speech"]
              for e in ("transcript-ce", "decoder-adapter")]
        res["states"] = {"n": sta["n"], "summary": S, "speech_ratio_min": min(ce)}
        out += ["", "### Where each edit acts: the decoder's decision state", "",
                f"The decoder's final hidden state h at the step after <|sot|><|lang|><|transcribe|><|0.00|>, where "
                f"an empty transcript is decided, for every clip of the nine complete sets "
                f"({chk('states/n', sta['n'], '{:,}')} clips) under the original model and the three Whisper edits "
                f"of section 5. *Moves*: median relative change ||h_edit - h_original|| / ||h_original|| over the "
                f"clips (%); *text*: non-speech clips on which a text token beats end-of-text (g > 0, %).", "",
                "| model | speech moves | clean speech | speech in noise | non-speech moves | text |",
                "|---|---|---|---|---|---|"]
        for k in ("original", "aae", "transcript-ce", "decoder-adapter"):
            text = chk(f"states/{k}/text", S[k]["text_nonspeech"], "{:.1f}")
            if k == "original":
                out.append(f"| {names[k]} | | | | | {text} |")
                continue
            c, by = S[k]["change_median"], S[k]["change_by_group"]
            out.append(f"| {names[k]} | {chk(f'states/{k}/speech', c['speech'], '{:.1f}')} | "
                       f"{chk(f'states/{k}/clean', by['speech'], '{:.1f}')} | "
                       f"{chk(f'states/{k}/noisy', by['noisy'], '{:.1f}')} | "
                       f"{chk(f'states/{k}/nonspeech', c['nonspeech'], '{:.1f}')} | {text} |")
        out += ["", f"Each edit trained with transcript CE moves the state of speech about "
                    f"{chk('states/speech_ratio_min', min(ce), '{:.0f}')} times as far as AAE does, or more."]

    # ------------------------------------------------------------------ 6. low-SNR stress test
    stress = {m: _load(runs / "stress" / f"{m}.json") for m in MODELS}
    if any(stress.values()):
        out += ["", "## 6. Low-SNR stress test on the complete sets", "",
                "Every utterance of LibriSpeech test-other (WER) and FLEURS Mandarin test (CER) mixed with a "
                "training-pool noise clip. *empty*: utterances returned empty; *error*: WER or CER (%); *same*: AAE "
                "transcripts identical to the original model's transcript of the same mixture (%).", "",
                "| model | set | original, -5 dB: empty | error | AAE, -5 dB: empty | error | same | "
                "AAE, 5 dB: empty | error | AAE, 15 dB: empty | error |", "|---|---|" + "---|" * 9]
        for m, st in stress.items():
            if not st:
                continue
            t = {"original": st["table"]["original"], "aae": st["table"]["checkpoint"]}
            res.setdefault("stress", {})[m] = t
            for lang, name in (("en", "test-other"), ("zh", "FLEURS")):
                o, a = t["original"][f"-5dB_{lang}"], {b: t["aae"][f"{b}dB_{lang}"] for b in (-5, 5, 15)}
                cells = [chk(f"stress/{m}/original/-5dB_{lang}/empty", o["empty"], "{:,}"),
                         chk(f"stress/{m}/original/-5dB_{lang}/err", o["err"]),
                         chk(f"stress/{m}/aae/-5dB_{lang}/empty", a[-5]["empty"], "{:,}"),
                         chk(f"stress/{m}/aae/-5dB_{lang}/err", a[-5]["err"]),
                         chk(f"stress/{m}/aae/-5dB_{lang}/same", 100 * a[-5]["same"] / a[-5]["n"], "{:.1f}")]
                for b in (5, 15):
                    cells += [chk(f"stress/{m}/aae/{b}dB_{lang}/empty", a[b]["empty"], "{:,}"),
                              chk(f"stress/{m}/aae/{b}dB_{lang}/err", a[b]["err"])]
                out.append(f"| {models.spec(m).title if lang == 'en' else ''} | {name} | " + " | ".join(cells) + " |")

        def pooled(m, system, snr):
            cells = [res["stress"][m][system][f"{snr}dB_{lang}"] for lang in ("en", "zh")]
            return 100 * sum(c["empty"] for c in cells) / sum(c["n"] for c in cells)

        ms = list(res["stress"])
        aae5 = [pooled(m, "aae", -5) for m in ms]
        res["stress_pooled"] = {"aae_-5dB": dict(zip(ms, aae5, strict=True)),
                                "original_-5dB": {m: pooled(m, "original", -5) for m in ms},
                                "aae_5_15dB_max": max(pooled(m, "aae", b) for m in ms for b in (5, 15))}
        orig5 = max(res["stress_pooled"]["original_-5dB"].values())
        err5 = [round(round(res["stress"][m]["aae"][f"-5dB_{lang}"]["err"], 2)                 # from the printed rates
                      - round(res["stress"][m]["original"][f"-5dB_{lang}"]["err"], 2), 2)
                for m in ms for lang in ("en", "zh")]
        res["stress_pooled"].update({"aae_-5dB_err_rise_max": max(err5),
                                     "aae_-5dB_err_falls": sum(d < 0 for d in err5)})
        out += ["", f"Pooled over both sets at -5 dB, {chk('stress/aae/-5dB/empty_min', min(aae5), '{:.1f}')}% to "
                    f"{chk('stress/aae/-5dB/empty_max', max(aae5), '{:.1f}')}% of utterances are returned empty with "
                    f"AAE, against at most {chk('stress/original/-5dB/empty_max', orig5, '{:.1f}')}% for the original "
                    f"models; at 5 and 15 dB, at most "
                    f"{chk('stress/aae/5_15dB/empty_max', res['stress_pooled']['aae_5_15dB_max'])}%. At -5 dB, the "
                    f"error rate with AAE rises by at most {chk('stress/aae/-5dB/err_rise_max', max(err5))} points "
                    f"over the original model's on the same mixtures and falls on "
                    f"{chk('stress/aae/-5dB/err_falls', sum(d < 0 for d in err5), '{:d}')} of {len(err5)} model and "
                    f"set pairs."]

    # ------------------------------------------------------------------ 7. remaining outputs on UrbanSound8K
    outs = {m: sorted(k[1] for k, r in (tx(m, "aae") or {}).items()
                      if k[0] == "urban" and (r.get("error") or metrics.nonempty(r.get("text")))) for m in have}
    if have and all(tx(m, "aae") for m in have):
        per = {m: Counter(URBAN_CLASSES[int(i.split("-")[1])] for i in outs[m]) for m in have}
        anyc = Counter(URBAN_CLASSES[int(i.split("-")[1])] for i in set().union(*map(set, outs.values())))
        res["residual"] = {"by_model": outs, "by_class": dict(anyc),
                           "all_models": sorted(set.intersection(*map(set, outs.values())))}
        out += ["", "## 7. Remaining outputs on UrbanSound8K", "",
                "UrbanSound8K clips that receive text with AAE, per sound class; *any*: clips with text from at least "
                "one model.", "",
                "| class | " + " | ".join(SHORT[m] for m in have) + " | any |", "|---|" + "---|" * (len(have) + 1)]
        for c in sorted(anyc, key=lambda c: (-anyc[c], c)):
            cells = [chk(f"residual/{c}/{m}", per[m][c], "{:d}") for m in have] + \
                [chk(f"residual/{c}/any", anyc[c], "{:d}")]
            out.append(f"| {c.replace('_', ' ')} | " + " | ".join(cells) + " |")
        out.append("| all classes | " + " | ".join(chk(f"residual/all/{m}", len(outs[m]), "{:d}") for m in have)
                   + f" | {chk('residual/all/any', sum(anyc.values()), '{:d}')} |")
        voice = ("children_playing", "street_music")                   # the classes that can contain voices
        ids = [k[1] for k in tx(have[0], "aae") if k[0] == "urban"]
        n_rest = sum(URBAN_CLASSES[int(i.split("-")[1])] not in voice for i in ids)
        rest = {m: 100 * (len(outs[m]) - sum(per[m][c] for c in voice)) / n_rest for m in have}
        res["residual"]["without_voice_classes"] = rest
        out += ["", f"Without the children playing and street music clips "
                    f"({chk('residual/voice_classes/clips', len(ids) - n_rest, '{:,}')} of {len(ids):,}), the AAE rates "
                    f"on UrbanSound8K are " + ", ".join(f"{chk(f'residual/without_voice/{m}', rest[m])}% "
                                                        f"({models.spec(m).title})" for m in have)
                    + f"; at most {chk('residual/without_voice/max', max(rest.values()))}%."]

    # ------------------------------------------------------------------ 8. training
    trained = {m: (comp[m]["aae"] or {}).get("training") for m in have}
    dec = {m: ((_load(runs / "complete" / m / "decoder-adapter" / "metrics.json") or {}).get("training")) for m in have}
    if have and all(trained.values()):
        res["training"] = {m: {"steps": t["steps"], "trained_params": t["trained_params"],
                               "wallclock_s": t["wallclock_s"],
                               "decoder_adapter_params": (dec[m] or {}).get("trained_params")}
                           for m, t in trained.items()}
        out += ["", "## 8. Training", "",
                "| model | adapted encoder layers | AAE adapters (M) | decoder adapters (M) | steps | hours |",
                "|---|---|---|---|---|---|"]
        for m, t in res["training"].items():
            d = "" if t["decoder_adapter_params"] is None else \
                chk(f"training/{m}/decoder_params", t["decoder_adapter_params"] / 1e6)
            out.append(f"| {models.spec(m).title} | {ENCODER_LAYERS[m]} | "
                       f"{chk(f'training/{m}/aae_params', t['trained_params'] / 1e6)} | {d} | "
                       f"{chk(f'training/{m}/steps', t['steps'], '{:,}')} | "
                       f"{chk(f'training/{m}/hours', t['wallclock_s'] / 3600, '{:.1f}')} |")

    # ------------------------------------------------------------------ problems the numbers depend on
    for path in sorted(runs.glob("complete/*/*/metrics.json")):          # decode errors or batch splits
        legs = json.loads(path.read_text()).get("legs", {})
        bad = {leg: (x.get("n_errors", 0), x.get("oom_splits", 0)) for leg, x in legs.items()
               if x.get("n_errors", 0) or x.get("oom_splits", 0)}
        if bad:
            out.append(f"\nwarning: {path.parent.relative_to(runs)}: "
                       + ", ".join(f"{leg} {e} failed clip(s), {s} split batch(es)" for leg, (e, s) in bad.items()))
    print("\n".join(out))
    print(f"\n{chk.checked} numbers compared with the paper: "
          + ("all identical." if not chk.differ else f"{len(chk.differ)} differ."))
    for key, got, want in chk.differ:
        print(f"  {key}: {got} here, {want} in the paper")
    res["compared"], res["differ"] = chk.checked, [list(x) for x in chk.differ]
    (runs / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--expected", default=str(ROOT / "assets" / "paper_results.json"))
    args = ap.parse_args()
    report(Path(args.runs), json.loads(Path(args.expected).read_text()))


if __name__ == "__main__":
    main()
