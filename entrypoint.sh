#!/usr/bin/env bash
set -euo pipefail

echo "[4danyone-webui] container start"

# Non-fatal GPU probe: prints nvidia-smi + PyTorch device capabilities
# (expect the unlocked CMP 170HX to report capability 8.0 / GA100 / 64 GiB).
python -m app.health > /tmp/health.txt 2>&1 || true
cat /tmp/health.txt

if [[ "${AUTO_DOWNLOAD_MODELS:-false}" == "true" ]]; then
  echo "[4danyone-webui] AUTO_DOWNLOAD_MODELS=true: downloading models + SMPL-X..."
  python -m app.init_models --models --smplx || echo "[4danyone-webui] asset init reported an error"
fi

exec python /app/app/webui.py
