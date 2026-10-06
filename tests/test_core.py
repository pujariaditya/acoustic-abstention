"""CPU-only checks of the pieces every number depends on. Run with ``envs/main/bin/python -m pytest``."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aae import audio, lora, metrics, models, noisy, weights

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "assets" / "manifests"


def read(p):
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def test_int16_conventions():
    y = np.array([-1.5, -1.0, -0.25, 0.0, 0.5, 1.0], dtype=np.float32)
    a = audio.to_int16(y)
    assert a.tolist() == [-32767, -32767, -8191, 0, 16383, 32767]      # scale by 32767, truncate toward zero
    assert audio.digest(a) == audio.digest(a.copy()) and len(audio.digest(a)) == 16


def test_noise_segment_tiles_short_clips():
    noise = np.arange(5, dtype=np.float32)
    assert noisy.noise_segment(noise, 12, 3).tolist() == [3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert noisy.noise_segment(np.zeros(0, np.float32), 4, 0).tolist() == [0, 0, 0, 0]


def test_snr5_mixture_level():
    rng = np.random.default_rng(0)
    speech, noise = rng.standard_normal(16000).astype(np.float32) * 0.1, rng.standard_normal(40000).astype(np.float32)
    y = noisy.snr5(speech, noise, 1000)
    n = y - speech
    assert abs(10 * np.log10(np.mean(speech.astype(np.float64) ** 2) / np.mean(n.astype(np.float64) ** 2)) - 5.0) < 1e-3


def test_noisy_mixers_are_bit_stable():
    # digests of this exact arithmetic under NumPy 2.2 and 2.5; a change in dtype promotion would change the noisy
    # test sets, which aae.prepare would then report as digest mismatches
    rng = np.random.default_rng(0)
    speech = (rng.standard_normal(16000) * 0.1).astype(np.float32)
    noise = rng.standard_normal(40000).astype(np.float32) * 0.3
    got = {k: audio.digest(audio.to_int16(noisy.MIXERS[k](speech, noise, 1234))) for k in ("tail", "snr5")}
    assert got == {"tail": "793d3c0bbfec4dfc", "snr5": "a915f6e833c3ff47"}


def test_tail_layout():
    speech = np.full(8000, 0.1, dtype=np.float32)
    y = noisy.tail(speech, np.full(100000, 0.2, dtype=np.float32), 0)
    assert len(y) == 8000 + 16000 + 48000
    assert np.all(y[8000:24000] == 0) and np.allclose(y[24000:], 0.1)       # silence, then noise at the speech RMS


def test_lora_merge_equals_adapted_forward():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
    wrapped = lora.apply(net, r"^2$", r=2, alpha=4.0)
    with torch.no_grad():
        wrapped["2"].B.normal_()
    x = torch.randn(3, 8)
    adapted = net(x)
    merged = lora.merged_state(wrapped)["2.weight"]
    with lora.disabled(wrapped):
        assert not torch.allclose(net(x), adapted)                        # adapters off: the original network
    plain = torch.nn.functional.linear(torch.relu(net[0](x)), merged, net[2].bias)
    assert torch.allclose(plain, adapted, atol=1e-5)


def test_decode_splits_a_batch_on_out_of_memory(tmp_path):
    from aae import evaluate

    rows = []
    for i, d in enumerate([1.0, 0.5, 2.0, 1.5, 0.8]):
        np.save(tmp_path / f"{i}.npy", np.zeros(1600, dtype=np.int16))
        rows.append({"id": str(i), "staged": str(tmp_path / f"{i}.npy"), "duration": d})

    class FitsTwo:                                     # a recognizer that runs out of memory above two clips
        calls = []

        def transcribe(self, wavs):
            self.calls.append(len(wavs))
            if len(wavs) > 2:
                raise torch.OutOfMemoryError("fake")
            return ["x"] * len(wavs)

    out, _, splits = evaluate.decode(FitsTwo(), rows, batch=4)
    assert [r["id"] for r in out] == ["0", "1", "2", "3", "4"]        # manifest order
    assert all(r["text"] == "x" and r["error"] is None for r in out)
    assert FitsTwo.calls == [4, 2, 2, 1] and splits == 1             # duration order, halved once


def test_metrics_definitions():
    recs = [{"text": "", "error": None, "ref": ""}, {"text": " . ", "error": None, "ref": ""},
            {"text": "hello", "error": None, "ref": ""}, {"text": "", "error": "boom", "ref": ""}]
    assert metrics.leg_metrics("urban", recs)["hr"] == 75.0                 # punctuation and failures count
    m = metrics.leg_metrics("ls_clean", [{"text": "Hello, world!", "error": None, "ref": "HELLO WORLD"}])
    assert m["wer"] == 0.0 and m["empty_rate"] == 0.0
    assert metrics.normalize_zh("這 是 測試") == "这是测试"
    lo, hi = metrics.wilson(30, 8732)
    assert (round(100 * lo, 2), round(100 * hi, 2)) == (0.24, 0.49)


def test_tolerance_rule():
    assert not metrics.beyond_tolerance("ls_clean", 0.3)                 # +0.3 WER on test-clean is within
    assert not metrics.beyond_tolerance("ls_clean", 0.3004)              # crossings below 0.0005 points are not counted
    assert metrics.beyond_tolerance("ls_clean", 0.3006)
    assert metrics.beyond_tolerance("fleurs_zh", 1.42) and not metrics.beyond_tolerance("fleurs_zh", -6.0)


def test_model_specs():
    assert set(models.MODEL_NAMES) == {"whisper-large-v3", "qwen3-asr-1.7b", "ark-asr-3b", "hojo-asr-v1"}
    s = models.spec("whisper-large-v3")
    import re
    hit = [n for n in ("encoder.blocks.16.attn.query", "encoder.blocks.31.mlp.2", "encoder.blocks.15.attn.query",
                       "decoder.blocks.3.attn.query") if re.search(s.encoder_locus, n)]
    assert hit == ["encoder.blocks.16.attn.query", "encoder.blocks.31.mlp.2"]


def test_every_release_is_pinned():
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs" / "checkpoints.yaml"))
    assert set(cfg) == set(weights.CHECKPOINTS)
    assert len(weights.CHECKPOINTS) == 24 and all(len(v) == 64 for v in weights.CHECKPOINTS.values())
    assert {k.split("/")[1] for k in weights.CHECKPOINTS} <= set(weights.VARIANTS)


@pytest.mark.parametrize("path,count", [("complete/urban.jsonl", 8732), ("complete/wham.jsonl", 8000),
                                        ("complete/fsd50k.jsonl", 6414), ("complete/fsd50k_all.jsonl", 10231),
                                        ("complete/ls_clean.jsonl", 2620), ("complete/ls_other.jsonl", 2939),
                                        ("complete/tail.jsonl", 1000), ("complete/snr5.jsonl", 1000),
                                        ("complete/tail_heldout.jsonl", 1000), ("complete/snr5_heldout.jsonl", 1000),
                                        ("complete/fleurs_zh.jsonl", 945), ("complete/aishell.jsonl", 7176),
                                        ("train/nonspeech.jsonl", 13602), ("train/speech.jsonl", 2999),
                                        ("stress.jsonl", 11652)])
def test_manifest_sizes(path, count):
    rows = read(MANIFESTS / path)
    assert len(rows) == count
    ids = [r["id"] for r in rows] if "id" in rows[0] else None
    if ids is not None:
        assert len(set(ids)) == len(ids)


def test_stress_pairs_every_test_utterance_with_training_noise():
    rows = read(MANIFESTS / "stress.jsonl")
    speech = [(r["source"], r["id"], r["digest"]) for leg in ("ls_other", "fleurs_zh")
              for r in read(MANIFESTS / "complete" / f"{leg}.jsonl")]
    # every utterance of both complete sets, in manifest order, at -5, 5 and 15 dB
    assert [(r["speech"]["source"], r["speech"]["id"], r["speech"]["digest"]) for r in rows[::3]] == speech
    assert [r["snr"] for r in rows] == [-5.0, 5.0, 15.0] * len(speech)
    assert all(r["lang"] == ("zh" if r["speech"]["source"].startswith("fleurs") else "en") for r in rows)
    noise = {(r["source"], r["id"]): r["digest"] for r in read(MANIFESTS / "train" / "nonspeech.jsonl")}
    assert all(noise[(r["noise"]["source"], r["noise"]["id"])] == r["noise"]["digest"] for r in rows)
    assert len({(r["noise"]["source"], r["noise"]["id"]) for r in rows}) == len(rows)      # no clip used twice


def test_heldout_noise_sets_reuse_the_utterances_with_evaluation_noise():
    base = read(MANIFESTS / "complete" / "tail.jsonl")
    train_noise = {(r["source"], r["id"]) for r in read(MANIFESTS / "train" / "nonspeech.jsonl")}
    for leg in ("tail_heldout", "snr5_heldout"):
        rows = read(MANIFESTS / "complete" / f"{leg}.jsonl")
        assert [r["speech"] for r in rows] == [r["speech"] for r in base]        # the same 1,000 utterances, in order
        assert all(r["noise"]["source"] in ("wham:cv", "wham:tt") for r in rows)
        assert not {(r["noise"]["source"], r["noise"]["id"]) for r in rows} & train_noise
        assert len({r["noise"]["id"] for r in rows}) == len(rows)                 # no clip used twice
    assert [r["noise"] for r in read(MANIFESTS / "complete" / "tail_heldout.jsonl")] == \
        [r["noise"] for r in read(MANIFESTS / "complete" / "snr5_heldout.jsonl")]


def test_edits_reproduce_the_error_rate_and_the_interval_is_paired():
    recs = [{"ref": "the cat sat on the mat", "text": "the cat sat on a mat"},
            {"ref": "hello world", "text": ""}, {"ref": "red green blue", "text": "red green blue sky"}]
    e, n = metrics.edits("ls_clean", recs)
    assert (e, n) == ([1, 2, 1], [6, 2, 3])
    assert abs(100 * sum(e) / sum(n) - metrics.leg_metrics("ls_clean", recs)["wer"]) < 1e-9
    assert metrics.change_interval(e, e, n) == (0.0, 0.0)                         # identical systems: no change
    lo, hi = metrics.change_interval([0, 0, 0], e, n, resamples=2000)
    assert 0 < lo <= 100 * sum(e) / sum(n) <= hi


def test_pseudo_labels_cover_the_speech_pool():
    ids = [r["id"] for r in read(MANIFESTS / "train" / "speech.jsonl")]
    for m in models.MODEL_NAMES:
        assert [r["id"] for r in read(MANIFESTS / "train" / "pseudo" / f"{m}.jsonl")] == ids


def test_transcribe_input_equals_the_staged_clip(tmp_path):
    import soundfile as sf

    from aae import transcribe

    y = (0.3 * np.sin(np.linspace(0, 2000, 44100 * 2))).astype(np.float32)
    sf.write(tmp_path / "a.wav", y, 44100)
    audio.stage(tmp_path / "a.wav", tmp_path / "a.npy")
    rows, wavs = transcribe.read_files([tmp_path / "a.wav"])
    assert np.array_equal(wavs[0], audio.load(tmp_path / "a.npy"))    # the same samples the evaluation decodes
    assert rows[0]["duration"] == len(wavs[0]) / audio.SAMPLE_RATE


def test_checkpoint_round_trip(tmp_path):
    new = {"weight": torch.randn(3, 4)}
    weights.save(tmp_path, new, {"model": "m", "variant": "aae"})
    lin = torch.nn.Linear(4, 3)
    record = weights.apply(lin, tmp_path)
    assert torch.equal(lin.weight.data, new["weight"].half().float()) and record["variant"] == "aae"
    assert not list(tmp_path.glob("*.tmp"))
    weights.save(tmp_path, {"nope": torch.zeros(1)}, {})
    with pytest.raises(KeyError):
        weights.apply(lin, tmp_path)


def test_training_defaults_are_the_released_arguments():
    from aae import train

    assert train.recipe("whisper-large-v3", "aae") == {"steps": 9250, "units": 2, "accum": 4, "rank": 16,
                                                       "zh_share": 0.0, "lam": 50.0}
    assert train.recipe("whisper-large-v3", "decoder-kl-1")["lam"] == 1.0
    assert train.recipe("hojo-asr-v1", "aae")["zh_share"] == 0.4
    assert train.recipe("qwen3-asr-1.7b", "transcript-ce")["units"] == 4
    assert train.recipe("ark-asr-3b", "decoder-adapter")["rank"] == 9


def test_resume_refuses_a_different_recipe():
    from aae import train

    saved = {"model": "m", "variant": "aae", "units": 2, "accum": 4, "rank": 16, "zh_share": 0.0}
    train.check_resume(saved, dict(saved))
    with pytest.raises(SystemExit):
        train.check_resume(saved, {**saved, "units": 4})


def test_preservation_counts_only_frames_with_audio():
    from aae.train import preservation

    f0 = torch.ones(2, 1500, 4)
    f1 = f0.clone()
    f1[:, 600:] += 1.0                                # relative error 1 on every frame from 12 s on
    wavs = [np.zeros(16000 * 10, np.float32), np.zeros(16000 * 30, np.float32)]
    # 10 s clip: its 500 audio frames are all unchanged; 30 s clip: 900 of 1500 changed
    assert float(preservation(f1, f0, wavs, whisper=True)) == pytest.approx((0.0 + 0.6) / 2)


def test_separability_pools_only_frames_with_audio():
    from aae.separability import pool

    h = torch.zeros(2, 1500, 3, dtype=torch.float16)
    h[:, :500] = 1.0
    h[:, 500:] = 7.0                                  # frames past the audio of a 10 s clip
    out = pool(h, [500, 1500])                        # 10 s clip, 30 s clip
    assert out.dtype == torch.float32
    assert torch.allclose(out[0], torch.ones(3))
    assert torch.allclose(out[1], torch.full((3,), (500 * 1.0 + 1000 * 7.0) / 1500))


def _runs(tmp_path):
    """A tiny runs folder: one model, the original model, AAE and one ablation variant, ten clips per set."""
    from aae import report

    def put(system, hr, err, texts):
        legs = {leg: {"n": 10, "n_errors": 0, "hr": hr[leg]} for leg in report.NONSPEECH}
        legs.update({leg: {"n": 10, "n_errors": 0, "hr": 100.0 - 10 * texts.count(""), metrics.metric_of(leg): err[leg],
                           "empty_rate": 10.0 * texts.count("")} for leg in report.SPEECH})
        d = tmp_path / "complete" / "whisper-large-v3" / system
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"legs": legs}))
        with open(d / "transcripts.jsonl", "w") as f:
            for leg in report.SPEECH:
                for i, t in enumerate(texts):
                    f.write(json.dumps({"leg": leg, "id": str(i), "text": t, "ref": "a"}) + "\n")
            for leg in report.NONSPEECH:
                for i in range(10):
                    f.write(json.dumps({"leg": leg, "id": f"1-{i}-0-0.wav", "text": "x" if i < hr[leg] / 10 else ""})
                            + "\n")

    base = {leg: 2.0 for leg in report.SPEECH}
    put("original", {leg: 100.0 for leg in report.NONSPEECH}, base, ["a"] * 10)
    put("aae", {"urban": 10.0, "wham": 0.0, "fsd50k": 0.0}, {**base, "ls_clean": 2.3004}, ["a"] * 9 + ["b"])
    put("transcript-ce", {"urban": 0.0, "wham": 0.0, "fsd50k": 0.0}, {**base, "fleurs_zh": 3.5}, ["a"] * 8 + ["", "c"])
    return tmp_path


def test_report_computes_and_checks(tmp_path):
    from aae import report

    runs = _runs(tmp_path)
    res = report.report(runs, {"hr/whisper-large-v3/urban/aae": "10.00", "speech/whisper-large-v3/ls_clean/aae": "2.31",
                               "ablation/whisper-large-v3/transcript-ce/beyond": "1"})
    assert res["compared"] == 3 and res["differ"] == [["speech/whisper-large-v3/ls_clean/aae", "2.30", "2.31"]]
    assert res["hallucination"]["whisper-large-v3"]["urban"]["aae"]["count"] == 1
    assert res["identical"]["whisper-large-v3"]["ls_clean"] == {"identical": 9, "n": 10}
    aae, ce = res["ablation"]["whisper-large-v3"]["aae"], res["ablation"]["whisper-large-v3"]["transcript-ce"]
    assert aae["beyond"] == [] and aae["change"]["ls_clean"] == 0.3          # +0.3004: within the tolerance
    assert ce["beyond"] == ["fleurs_zh"] and ce["empty"] == 6 and ce["same"] == 80.0 and ce["hr_mean"] == 0.0
    assert res["residual"]["by_class"] == {"air_conditioner": 1}            # ids 1-0-0-0.wav: class 0
    crossing, = res["ablation_crossings"]                                   # transcript CE on FLEURS, with its interval
    assert (crossing["variant"], crossing["set"]) == ("transcript-ce", "fleurs_zh")
    assert 0 <= crossing["interval"][0] < crossing["interval"][1]           # two of ten changed: never below zero
    assert json.loads((runs / "results.json").read_text())["compared"] == 3


def test_report_scores_the_vad_gate(tmp_path):
    from aae import report

    runs = _runs(tmp_path)
    (runs / "vad").mkdir()
    with open(runs / "vad" / "silero.jsonl", "w") as f:             # the gate passes one UrbanSound8K clip and all
        for leg in report.NONSPEECH:                                 # speech except the first test-clean utterance
            for i in range(10):
                f.write(json.dumps({"leg": leg, "id": f"1-{i}-0-0.wav",
                                    "segments": [[0, 160]] if (leg, i) == ("urban", 0) else []}) + "\n")
        for leg in report.SPEECH:
            for i in range(10):
                f.write(json.dumps({"leg": leg, "id": str(i),
                                    "segments": [] if (leg, i) == ("ls_clean", 0) else [[0, 160]]}) + "\n")
    res = report.report(runs, {})
    g = res["vad"]["models"]["whisper-large-v3"]
    assert g["urban"] == 10.0 and g["wham"] == 0.0 and res["vad"]["passed"]["urban"] == 10.0
    assert res["vad"]["rejected_speech"]["ls_clean"] == 1 and g["ls_clean"] == 8.0     # 1 of 10 words lost vs 2.00


def test_report_matches_the_constant_end_of_text_raise_to_aae(tmp_path):
    from aae import report
    from aae.separability import SETS

    runs = _runs(tmp_path)
    (runs / "separability").mkdir()
    clips = []
    for s, grp in SETS.items():                       # ten clips per set; AAE leaves text on one non-speech clip
        for i in range(10):
            go, ga = (5.0 + i, 1.0 if i == 9 else -10.0) if grp == "nonspeech" else (10.0 + i, 10.0 + i)
            clips.append({"id": str(i), "set": s, "g": {"original": go, "aae": ga}})
    probe = {k: [{"layer": n, "auc_clean": 1.0, "auc_noisy": 0.99} for n in range(first, 33)]
             for k, first in (("original", 0), ("aae", 16))}
    decision = {s: {"n": 10, "probe_nonspeech": {"original": 1.0, "aae": 1.0}, "text": {"original": 1.0, "aae": 0.1},
                    "g_median": {"original": 0.0, "aae": 0.0}} for s in SETS}
    (runs / "separability" / "whisper-large-v3.json").write_text(json.dumps(
        {"n": {"train": 1, "eval": 90}, "probe": probe, "decision": decision, "clips": clips}))
    b = report.report(runs, {})["separability"]["bias"]
    # original UrbanSound8K margins 5..14: a raise of 13 leaves text on one clip, as AAE does,
    # and ends the speech margins 10..13 of each set
    assert b["logits"] == 13.0 and b["urban_text"] == 10.0
    assert set(b["speech_empty"].values()) == {40.0} and max(b["aae_speech_empty"].values()) == 0.0


def test_expected_numbers_are_all_reported():
    """Every key of assets/paper_results.json has one of the forms aae.report prints."""
    expected = json.loads((ROOT / "assets" / "paper_results.json").read_text())
    prefixes = ("hr/", "count/", "wilson/", "musan/", "speech/", "same/", "empty/", "interval/", "heldout/",
                "vad/", "crosswalk/", "ablation/", "locus/", "separability/", "states/", "stress/", "residual/",
                "training/", "ood/", "default/", "script/")
    assert len(expected) > 600 and all(k == "_about" or k.startswith(prefixes) for k in expected)
    assert all(isinstance(v, str) for v in expected.values())
