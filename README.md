# 4DAnyone WebUI (Docker)

Gradio WebUI for [ant-research/4DAnyone](https://github.com/ant-research/4DAnyone): upload a casual monocular video, generate synchronized multi-view videos, preview and download every generated view.

## Features

- **Generate** tab: upload a video (or pick a bundled example), choose view layout (views/per-layer, pitch layers, yaw span), Turbo on by default.
- **Jobs** tab: persistent SQLite queue, live log, inline dense/sparse/skeleton previews, **Download all views** zip (videos + cameras.json + metadata.json + GVHMR motion).
- **Settings** tab: GPU health report, one-click model downloads, SMPL-X provisioning.
- All models cached in a persistent Docker volume; downloads resume safely.
- Multi-GPU: one job is spread across all visible GPUs by 4DAnyone's built-in NCCL distributed denoiser.

## Target platform

- **GPU**: unlocked NVIDIA CMP 170HX — GA100 die, **compute capability 8.0 (sm_80)**, 64 GB HBM2e (3 cards recommended).
- Base image is PyTorch 2.8 / CUDA 12.8 (sm_80 included); FlashAttention-3 is Hopper-only, so **SageAttention 1.0.6** (pure-Python Triton wheel, no compile) is preinstalled. `auto` resolves Fa3 → SageAttention → SDPA; override per job in the UI.
- ~22–24 GB peak VRAM per run, so a single 64 GB card suffices; 3 cards split view-groups for speed.

## Quick start

```bash
docker compose up -d --build
# open http://<host>:7860
```

Requirements on the host: Docker + NVIDIA Container Toolkit, driver exposing the cards.

First run, in **Settings**:
1. **Download public models (init)** — ~16–20 GB into the `models` volume.
2. **Provision SMPL-X** — auto-installs from a public mirror (or `SMPLX_SOURCE` env / drop zip in `data/smplx/`).
3. **Run GPU health check** — expect capability `8.0`, 64 GB per card.

## Configuration (env)

| Variable | Default | Purpose |
| --- | --- | --- |
| `AUTO_DOWNLOAD_MODELS` | `false` | download models + SMPL-X at container start |
| `MAX_CONCURRENCY` | `1` | simultaneous jobs |
| `DEFAULT_ATTENTION_BACKEND` | `auto` | `auto` / `sageattention` / `sdpa` |
| `MODEL_DIR` / `DATA_DIR` | `/app/models` / `/app/data` | model cache / outputs (Docker volumes) |
| `SMPLX_SOURCE` | (none) | path to `models_smplx_v1_1.zip` or `SMPLX_NEUTRAL.npz` |
| `WEBUI_USERNAME` / `WEBUI_PASSWORD` | (none) | enable HTTP auth |

Manual CLI (pre-seed the model volume during image build/CI):

```bash
python -m app.init_models --models --smplx   # or --all / --status
python scripts/provision_smplx.py
```

## Notes

- SMPL-X is separately licensed; the bundled mirror may disappear — keep the official zip (`smpl-x.is.tue.mpg.de`) as a fallback.
- Outputs are per-job Dirs under `data/jobs/<id>/out`; completed runs are never overwritten and can be re-run to resume.
- Run container with `ipc: host` (compose does this) for NCCL/torch shared memory across the 3 cards.

## Prepare Video


```sh
ffmpeg -i input.mp4 -frames:v 121 -vf "pad=w='max(iw,ih*9/16)':h='max(ih,iw*16/9)':x='(ow-iw)/2':y='(oh-ih)/2':color=black" -c:v libx264 -pix_fmt yuv420p -c:a aac output.mp4
```
