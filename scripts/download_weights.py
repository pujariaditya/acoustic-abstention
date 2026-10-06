#!/usr/bin/env python3
"""Fetch the released checkpoints from the Hugging Face hub and verify each by sha256.

    python scripts/download_weights.py                          # all 22 (~14.8 GB)
    python scripts/download_weights.py --variant aae            # the four released models (~2.4 GB)
    python scripts/download_weights.py --only whisper-large-v3/aae,hojo-asr-v1/aae-run1

Each checkpoint lands in ``checkpoints/<model>/<variant>/`` (``$AAE_WEIGHTS`` moves it) as ``model.safetensors`` (the
merged weights of every tensor the adapter changed) and ``training.json`` (the exact training arguments and curves).
The digests are compiled into ``aae/weights.py``: a digest served from the same place as the file it describes would
attest to nothing. The repository is public; no token is needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aae import weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", default="", help="only this variant, e.g. aae")
    ap.add_argument("--model", default="", help="only this model, e.g. whisper-large-v3")
    ap.add_argument("--only", default="", help="comma-separated <model>/<variant> list")
    ap.add_argument("--repo", default=weights.HF_REPO)
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download

    wanted = [k for k in weights.CHECKPOINTS
              if (not args.only or k in args.only.split(","))
              and (not args.variant or k.split("/")[1] == args.variant)
              and (not args.model or k.split("/")[0] == args.model)]
    if not wanted:
        raise SystemExit("nothing matches; checkpoints are: " + ", ".join(weights.CHECKPOINTS))
    from huggingface_hub.errors import HfHubHTTPError

    print(f"| {args.repo}: {len(wanted)} checkpoint(s) -> {weights.WEIGHTS_ROOT}")
    for key in wanted:
        dest = weights.local_dir(*key.split("/"))
        try:
            for name in ("model.safetensors", "training.json"):         # real files under checkpoints/<key>/
                hf_hub_download(args.repo, f"{key}/{name}", local_dir=weights.WEIGHTS_ROOT)
        except HfHubHTTPError as e:                                     # includes "repository not found" (401/404)
            raise SystemExit(f"cannot download {key} from {args.repo} ({type(e).__name__}: {e}). Check the network "
                             "connection and the repository name (AAE_HF_REPO).") from None
        got = weights.sha256_of(dest / "model.safetensors")
        if got != weights.CHECKPOINTS[key]:
            for name in ("model.safetensors", "training.json"):
                (dest / name).unlink(missing_ok=True)
            raise SystemExit(f"sha256 mismatch for {key}: expected {weights.CHECKPOINTS[key]}, got {got}. "
                             "Removed it: a checkpoint that is not the released one is never kept.")
        print(f"|   {key}  sha256 ok  -> {dest}")
    print("| done")


if __name__ == "__main__":
    main()
