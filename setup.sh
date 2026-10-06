#!/usr/bin/env bash
# Create the two Python environments and download the four original recognizers at their pinned revisions.
#
#   envs/main     Python 3.12, torch 2.9.1 (CUDA 12.8), transformers 5.17  -> Whisper large-v3, Qwen3-ASR-1.7B
#   envs/legacy   Python 3.10, torch 2.5.1 (CUDA 12.1), transformers 4.57  -> ARK-ASR-3B, Hojo-ASR-V1
#
# Two environments because the models' own code needs incompatible transformers releases: Qwen3-ASR is native in
# transformers 5, while ARK-ASR's remote code and the hojo-asr package are written for transformers 4. Both install
# the same requirements.txt; its Python-version markers pick each environment's pinned versions.
# Weights go to models/ (about 27 GB); nothing is redistributed by this repository.
set -euo pipefail
cd "$(dirname "$0")"
UV=${UV:-uv}
command -v "$UV" >/dev/null || { echo "need uv (https://docs.astral.sh/uv/) or set UV" >&2; exit 1; }

for env in main legacy; do
  py=$([ "$env" = main ] && echo 3.12 || echo 3.10)
  echo "==> envs/$env (Python $py)"
  [ -x "envs/$env/bin/python" ] || "$UV" venv "envs/$env" --python "$py"
  "$UV" pip install --python "envs/$env/bin/python" --index-strategy unsafe-best-match -r requirements.txt
  "$UV" pip install --python "envs/$env/bin/python" --no-deps -e .
done

echo "==> original recognizers (pinned revisions, see aae/models/__init__.py)"
envs/main/bin/python - <<'PYEOF'
from pathlib import Path

import whisper
from huggingface_hub import snapshot_download

from aae import models

for s in models.SPECS.values():
    dest = models.MODELS_ROOT / s.local_dir
    print(f"|   {s.name}: {s.source} @ {s.revision[:12]}")
    if s.source.startswith("openai-whisper:"):
        # whisper._download checks the sha256 that is part of the URL (the pinned revision above).
        url = whisper._MODELS["large-v3"]
        assert s.revision in url, "openai-whisper no longer points at the pinned large-v3 weights"
        whisper._download(url, str(dest), in_memory=False)
    else:
        snapshot_download(s.source, revision=s.revision, local_dir=str(dest))
PYEOF

echo
echo "Setup complete. Next:"
echo "  envs/main/bin/python scripts/fetch_data.py --data-root ~/corpora  # the public corpora (~95 GB of archives)"
echo "  envs/main/bin/python -m aae.prepare --data-root ~/corpora         # stage and verify 59,244 clips (23 GB)"
echo "  envs/main/bin/python scripts/download_weights.py                 # the 22 released checkpoints (~15 GB)"
echo "  ./scripts/reproduce.sh                                           # every evaluation, on the complete sets"
