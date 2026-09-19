#!/usr/bin/env bash
set -euo pipefail

echo "[4danyone-webui] container start"

# Non-fatal GPU probe (expect the unlocked CMP 170HX to report
# capability 8.0 / GA100 / 64 GiB per card).
nvidia-smi || true
python - <<'PY' || true
import torch
if torch.cuda.is_available():
    print(f"torch {torch.__version__} / cuda {torch.version.cuda} / {torch.cuda.device_count()} devices")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name} capability {p.major}.{p.minor} {p.total_memory / 2**30:.0f} GiB")
else:
    print("CUDA not available to torch")
PY

if [[ "${AUTO_DOWNLOAD_MODELS:-false}" == "true" ]]; then
  echo "[4danyone-webui] AUTO_DOWNLOAD_MODELS=true: downloading models + SMPL-X + SMPLer-X..."
  python /app/scripts/download_model.py \
      --model_dir "${MODEL_DIR:-/app/models}" \
      --gvhmr_root "${GVHMR_ROOT:-/app/third_party/GVHMR}" \
      || echo "[4danyone-webui] model download reported an error"
  python /app/scripts/provision_smplx.py \
      || echo "[4danyone-webui] SMPL-X provisioning reported an error"
  python /app/scripts/download_smplerx.py \
      || echo "[4danyone-webui] SMPLer-X download reported an error"
  python /app/scripts/download_prompthmr.py \
      || echo "[4danyone-webui] PromptHMR download reported an error"
fi

exec python /app/launcher.py
