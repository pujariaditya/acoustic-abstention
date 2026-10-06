#!/usr/bin/env bash
# Run every evaluation behind the paper and docs/RESULTS.md with the released checkpoints, on the complete evaluation
# sets, then compare each number with the paper's value.
#
#   ./scripts/reproduce.sh                       # everything (about 2.5 GPU-days on one Quadro RTX 8000)
#   STAGES=ablation ./scripts/reproduce.sh       # one stage: complete | ablation | heldout | vad | crosswalk | stress | locus | separability | states
#
# Needs setup.sh, `python -m aae.prepare` and scripts/download_weights.py first. Each model runs in its own
# environment. Finished runs are skipped, so an interrupted reproduction resumes. Results land in runs/, and
# `python -m aae.report` prints the tables with every number checked against assets/paper_results.json.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=${RUNS:-runs}
STAGES=${STAGES:-complete ablation heldout vad crosswalk stress locus separability states}
MODELS="whisper-large-v3 qwen3-asr-1.7b ark-asr-3b hojo-asr-v1"
VARIANTS="no-preservation transcript-ce independent-noise decoder-adapter"
LEGS=urban,wham,fsd50k,ls_clean,ls_other,tail,snr5,fleurs_zh,aishell      # the ablation's sets

py() {  # the interpreter of the environment that runs model $1
  local env
  env=$(envs/main/bin/python -c "from aae import models; print(models.spec('$1').env)")
  echo "envs/$env/bin/python"
}
evaluate() {  # model system checkpoint-or-original [legs] [part]
  local m=$1 name=$2 ck=$3 out="$RUNS/${5:-complete}/$1/$2" p
  [ -f "$out/metrics.json" ] && { echo "| have $out"; return; }
  echo "==> $m / $name"
  p=$(py "$m")
  if [ "$ck" = original ]; then "$p" -m aae.evaluate --model "$m" --original ${4:+--legs "$4"} --out "$out"
  else "$p" -m aae.evaluate --model "$m" --checkpoint "$ck" ${4:+--legs "$4"} --out "$out"; fi
}

for stage in $STAGES; do
  case $stage in
    complete)   # Table 1, Figure 2, the hallucination counts and identical transcripts, unfiltered FSD50K
      for m in $MODELS; do
        evaluate "$m" original original
        evaluate "$m" aae "$m/aae"
      done ;;
    ablation)   # Table 3 and its per-set results: the four variants, and the full-recipe rows trained like them
      for m in $MODELS; do       # (Hojo-ASR-V1's first run; Whisper large-v3 at 8,500 steps), on the complete sets
        [ "$m" = hojo-asr-v1 ] && evaluate "$m" aae-run1 "$m/aae-run1" "$LEGS"
        [ "$m" = whisper-large-v3 ] && evaluate "$m" aae-8500 "$m/aae-8500" "$LEGS"
        for v in $VARIANTS; do evaluate "$m" "$v" "$m/$v" "$LEGS"; done
      done ;;
    heldout)    # speech with held-out noise: the noisy sets' utterances mixed with WHAM! evaluation noise
      for m in $MODELS; do
        evaluate "$m" original original tail_heldout,snr5_heldout heldout
        evaluate "$m" aae "$m/aae" tail_heldout,snr5_heldout heldout
      done ;;
    vad)        # the voice-activity-detector baseline: Silero VAD decisions on the complete sets (CPU)
      out="$RUNS/vad/silero.jsonl"
      if [ -f "$out" ]; then echo "| have $out"; else envs/main/bin/python -m aae.vad --out "$out"; fi ;;
    crosswalk)  # Table 2: Whisper large-v3 under the three definitions
      for leg in urban fsd50k; do
        for sys in original aae; do
          out="$RUNS/crosswalk/${leg}_$sys.json"
          [ -f "$out" ] && { echo "| have $out"; continue; }
          ck=$([ "$sys" = aae ] && echo "--checkpoint whisper-large-v3/aae" || true)
          # shellcheck disable=SC2086  # $ck is empty or two words
          envs/main/bin/python -m aae.crosswalk --leg "$leg" $ck --out "$out"
        done
      done ;;
    stress)     # the low-SNR stress test on the complete LibriSpeech test-other and FLEURS Mandarin test sets
      for m in $MODELS; do
        out="$RUNS/stress/$m.json"
        [ -f "$out" ] && { echo "| have $out"; continue; }
        p=$(py "$m")
        "$p" -m aae.stress --model "$m" --checkpoint "$m/aae" --out "$out"
      done ;;
    locus)      # encoder edit versus decoder edit on Whisper large-v3
      out="$RUNS/locus/whisper-large-v3.json"
      if [ -f "$out" ]; then echo "| have $out"; else envs/main/bin/python -m aae.locus --out "$out"; fi ;;
    separability)  # speech versus non-speech in Whisper large-v3's encoder: probes, decision margins, t-SNE
      out="$RUNS/separability/whisper-large-v3.json"
      if [ -f "$out" ]; then echo "| have $out"; else envs/main/bin/python -m aae.separability --out "$out"; fi ;;
    states)     # encoder edit versus decoder edit in the decoder state that decides silence (Whisper large-v3)
      out="$RUNS/states/whisper-large-v3.json"
      if [ -f "$out" ]; then echo "| have $out"; else envs/main/bin/python -m aae.states --out "$out"; fi ;;
    *) echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done

envs/main/bin/python -m aae.report --runs "$RUNS"
