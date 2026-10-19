#!/usr/bin/env bash
# On the GPU box:  bash run.sh            (setup, then the pilot detached; follow with: tail -f run.log)
# Hard cap: the pilot is killed after MAX_MIN minutes (default 115); results so far are kept.
set -euo pipefail
cd "$(dirname "$0")"
MAX_MIN=${MAX_MIN:-115}

# Reuse the image's CUDA torch (it matches the host driver); pip only adds what is missing.
BASE="" ; SITE=""
for p in /venv/main/bin/python /opt/conda/bin/python "$(command -v python3 || true)"; do
  if [ -x "$p" ] && "$p" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    BASE="$p"; SITE="--system-site-packages"; break
  fi
done
BASE="${BASE:-python3}"
echo "base python: $BASE ${SITE:+(reusing its CUDA torch)}"
if [ ! -x .venv/bin/python ]; then
  "$BASE" -m venv $SITE .venv || { apt-get update -qq && apt-get install -y -qq python3-venv && "$BASE" -m venv $SITE .venv; }
fi
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(), '| torch', torch.__version__)"
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-8B-Base', allow_patterns=['*.json','*.safetensors','*.txt'])"

nohup bash -c "timeout ${MAX_MIN}m .venv/bin/python pilot.py $*; .venv/bin/python report.py; \
  tar czf results.tar.gz results; echo '== PILOT FINISHED'" > run.log 2>&1 &
echo "started: tail -f run.log"
