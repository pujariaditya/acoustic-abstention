"""Acoustic abstention encoding (AAE): the audio encoder learns to stay silent on non-speech; the decoder never changes.

Low-rank adapters on the upper encoder layers and the audio projector learn to map non-speech to features that the
frozen decoder reads as an empty transcript, while a per-frame loss keeps the features of speech, mixed with the same
non-speech clip, close to their original values. The adapters are merged, so a released checkpoint has the original
architecture and decodes with the same procedure as the original model.

Entry points, all through ``python -m``:

``aae.transcribe``   your own audio, original model and checkpoint side by side
``aae.prepare``      stage the evaluation and training audio, checking every clip against the paper's digest
``aae.evaluate``     decode the complete evaluation sets with an original model or a checkpoint, and score them
``aae.crosswalk``    Whisper large-v3 under the three hallucination-rate definitions in use
``aae.stress``       the low-SNR stress test on the complete LibriSpeech test-other and FLEURS Mandarin test sets
``aae.locus``        encoder edit versus decoder edit on Whisper large-v3 (the margin argument, measured)
``aae.separability`` speech versus non-speech in Whisper large-v3's encoder: probes, decision margin, t-SNE
``aae.states``       encoder edit versus decoder edit in the decoder state that decides silence: t-SNE map
``aae.train``        train AAE or one of the four variants of the ablation study
``aae.report``       every results table from a set of runs, each number checked against the paper's value

``scripts/reproduce.sh`` runs every evaluation behind the paper and docs/RESULTS.md, all on complete evaluation sets;
``scripts/train.sh`` retrains any released checkpoint.
"""

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("aae")
except (ImportError, PackageNotFoundError):  # a source checkout that is not installed
    __version__ = "unknown"
