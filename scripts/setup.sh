#!/usr/bin/env bash
# Environment setup: venv (reusing system torch if present), deps, spaCy model for Kokoro's G2P.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || python3 -m venv --system-site-packages .venv
. .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf}" PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PWD/.pipcache}"
python -c "import torch" 2>/dev/null || pip install -q torch torchaudio
pip install -q -e . 
[ -f .env ] || cp .env.example .env
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
