#!/usr/bin/env bash
# Retrain one released checkpoint with its exact arguments (configs/checkpoints.yaml).
#
#   ./scripts/train.sh whisper-large-v3 aae                 # -> checkpoints/retrained/whisper-large-v3/aae
#   ./scripts/train.sh hojo-asr-v1 decoder-adapter
#
# Runs in the model's environment and resumes if interrupted. Needs `python -m aae.prepare --sets train` first.
# Budget: 8,500 or 9,250 steps; 2.5 to 26.2 hours of wall-clock time per run in our runs (10.6 to 26.2 for the four
# released models) on one Quadro RTX 8000 shared with other jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:?usage: train.sh <model> <variant>}; VARIANT=${2:?usage: train.sh <model> <variant>}
OUT=${OUT:-checkpoints/retrained/$MODEL/$VARIANT}

ENV=$(envs/main/bin/python -c "from aae import models; print(models.spec('$MODEL').env)")
ARGS=$(envs/main/bin/python - "$MODEL/$VARIANT" <<'PYEOF'
import sys
import yaml
cfg = yaml.safe_load(open("configs/checkpoints.yaml"))
key = sys.argv[1]
if key not in cfg:
    sys.exit(f"no released checkpoint {key}; choose one of: {', '.join(cfg)}")
c = cfg[key]
args = [f"--variant {c['variant']}", f"--steps {c['steps']}", f"--units {c['units']}", f"--accum {c['accum']}",
        f"--rank {c['rank']}", f"--zh-share {c.get('zh_share', 0)}", f"--lam {c.get('lam', 50)}"]
print(" ".join(args))
PYEOF
)
echo "==> envs/$ENV: python -m aae.train --model $MODEL $ARGS --out $OUT"
# shellcheck disable=SC2086  # $ARGS is a list of flags
exec "envs/$ENV/bin/python" -m aae.train --model "$MODEL" $ARGS --out "$OUT"
