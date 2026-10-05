#!/usr/bin/env bash
# One-command reproduction. Every step is idempotent (skips when its output already exists), so re-running resumes after a failure.
# Single-GPU by design: CUDA_VISIBLE_DEVICES defaults to 0.   Usage: bash scripts/run_all.sh   [SKIP_REAL=1] [EPOCHS=3]
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
bash scripts/setup.sh
. .venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a
export HF_HOME="${HF_HOME:-$PWD/.hf}"; case "$HF_HOME" in /*) ;; *) export HF_HOME="$PWD/$HF_HOME";; esac
mkdir -p logs artifacts/{model,datasets,evaluation,demo,report}
step() { echo; echo "=== $* ==="; }

step "1 dataset discovery";           [ -f data/dataset_candidates.json ] || python scripts/discover_datasets.py || echo "[warn] discovery failed, continuing"
step "2 synthetic expressions";       python scripts/generate_math.py
step "3 spoken realisations";         python scripts/generate_spoken.py
step "4 TTS (Kokoro, multi-voice)";   python scripts/generate_tts.py 2>&1 | grep -vE "Warning|WeightNorm" | tail -3
if [ "${SKIP_REAL:-0}" != "1" ]; then
  step "5 real audio corpus";         python scripts/build_real_corpus.py || echo "[warn] real corpus failed; continuing synthetic-only"
fi
step "6 build + validate dataset";    [ -d data/processed/mathspeech ] || python scripts/build_dataset.py; python scripts/validate_dataset.py
step "7 baseline evaluation";         python scripts/evaluate.py --systems zeroshot
step "8 fine-tune Qwen3-ASR (canonical LaTeX targets)"; [ -f artifacts/model/qwen3asr-canonical/training_summary.json ] || python scripts/train.py --target_mode canonical --output_dir artifacts/model/qwen3asr-canonical --epochs "${EPOCHS:-3}"
if [ "${SPOKEN_ABLATION:-0}" = "1" ]; then step "8b ablation (spoken targets)"; [ -f artifacts/model/qwen3asr-spoken/training_summary.json ] || python scripts/train.py --target_mode spoken --output_dir artifacts/model/qwen3asr-spoken --epochs 1 --run_name qwen3asr-spoken-ablation; fi
step "9 evaluation (test, test_hard, test_comp, test_ood)"; python scripts/evaluate.py --systems zeroshot,canonical
step "9b real-speech probe (eval-only)"; python scripts/eval_external.py || echo "[warn] external probe skipped"
step "10 demo examples";              python scripts/generate_demo.py
step "11 report";                     python scripts/generate_report.py
echo; echo "Done. Report: artifacts/evaluation/report.html   Demo: python app/app.py"
