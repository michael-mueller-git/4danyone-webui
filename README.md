# 4DAnyone WebUI (Docker)

Docker packaging for the **official** [ant-research/4DAnyone](https://github.com/ant-research/4DAnyone)
Gradio Space, with a small upload supervisor in front so you can start a new run
from the browser without restarting the container.

Two WebUIs are served by one container:

| Port | What | Purpose |
| --- | --- | --- |
| **7860** | Control / upload page | upload a video (or pick a bundled example), then get redirected to the viewer |
| **7861** | Official 4DAnyone Space | the upstream Gradio GUI: run inference, preview, browse the Rerun 3D viewer, resume/delete |

Uploading a new video stops the current official viewer process and starts a
fresh one for the new clip — no container restart needed. One run at a time
(upstream's single-task design).

## Target platform

- **GPU**: unlocked NVIDIA CMP 170HX — GA100 die, **compute capability 8.0 (sm_80)**, 64 GB HBM2e (multi-GPU recommended).
- Base image is PyTorch 2.8 / CUDA 12.8 (sm_80 included); FlashAttention-3 is Hopper-only, so **SageAttention 1.0.6** (pure-Python Triton wheel, no compile) is preinstalled. The pipeline's `auto` attention backend therefore resolves to SageAttention (`flash_attn_3 → sageattention → sdpa` by availability).
- ~22–24 GB peak VRAM per run, so a single 64 GB card suffices; multiple GPUs split view-groups for speed.

## Quick start

```bash
docker compose up -d --build
# open http://<host>:7860   (upload page)
# you will be redirected to http://<host>:7861   (official 4DAnyone viewer)
```

Requirements on the host: Docker + NVIDIA Container Toolkit, driver exposing the cards.

First run:
1. **Download models** — set `AUTO_DOWNLOAD_MODELS=true` in `docker-compose.yml` once, or run
   `docker compose exec webui python /app/scripts/download_model.py --model_dir /app/models --gvhmr_root /app/third_party/GVHMR` (~16–20 GB into the `models` volume).
2. **Provision SMPL-X** — auto-installs from a public mirror (or `SMPLX_SOURCE` env / drop `models_smplx_v1_1.zip` into `data/smplx/`):
   `docker compose exec webui python /app/scripts/provision_smplx.py`.
3. **Download SMPLer-X** (only if you select `smplerx`) — `docker compose exec webui python /app/scripts/download_smplerx.py` (~2.6 GB).
4. **Download PromptHMR** (only if you select `prompthmr`) — `docker compose exec webui python /app/scripts/download_prompthmr.py`.
5. **GPU health** — check the *GPU health* accordion on the control page (expect capability `8.0`, 64 GB per card).

Then upload a video (≥121 frames, 1080p+, 9:16 ideal) on the control page, pick a **pose model**
(`gvhmr` or `smplerx`), and run it in the viewer. By default each new upload deletes all previous
runs/uploads/cache/logs first (uncheck *Delete previous runs before starting*, or set
`CLEAR_ON_START=false`, to keep them for the *Previous runs* / reopen flow).

## Configuration (env)

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOTION_BACKEND` | `gvhmr` | default pose model for the control-page dropdown: `gvhmr`, `prompthmr` (SOTA pose + stable world), or `smplerx` |
| `CLEAR_ON_START` | `true` | delete previous runs/uploads/cache/logs before each new video (control-page checkbox overrides per run) |
| `SMPLERX_BATCH` | `16` | frames per SMPLer-X batch (VRAM vs speed) |
| `AUTO_DOWNLOAD_MODELS` | `false` | download models + SMPL-X + SMPLer-X at container start |
| `DEFAULT_ATTENTION_BACKEND` | `auto` | `auto` / `sageattention` / `sdpa` |
| `CONTROL_PORT` / `OFFICIAL_PORT` | `7860` / `7861` | ports of the two WebUIs |
| `PUBLIC_VIEWER_URL` | `http://127.0.0.1:7861` | browser URL of the official Space; set to `http://<host>:7861` on a remote host |
| `VIEWER_REDIRECT_DELAY` | `15` | seconds to wait after the viewer port opens before redirecting the browser (raise if the Rerun scene fails to load on first open) |
| `VIDEO_PATH` / `OUTPUT_DIR` | (none) | pre-start the official Space for a video / saved run at boot (automation) |
| `MODEL_DIR` / `DATA_DIR` | `/app/models` / `/app/data` | model cache / outputs (Docker volumes) |
| `SMPLX_SOURCE` | (none) | path to `models_smplx_v1_1.zip` or `SMPLX_NEUTRAL.npz` |
| `SMPLX_MIRROR_REPO` / `SMPLX_MIRROR_FILE` | `liujiting/models_smplx_v1_1` / `models_smplx_v1_1.zip` | public mirror fallback for SMPL-X |
| `SMPL_SOURCE` | (none) | path to a licensed `SMPL_NEUTRAL.pkl` for the PromptHMR backend |
| `SMPL_MIRROR_REPO` / `SMPL_MIRROR_FILE` | `lithiumice/motion_imitation` / `smpl/SMPL_NEUTRAL.pkl` | public mirror fallback for SMPL |

Manual CLI (pre-seed the model volume during image build/CI):

```bash
python /app/scripts/download_model.py --model_dir /app/models --gvhmr_root /app/third_party/GVHMR
python /app/scripts/provision_smplx.py
python /app/scripts/download_smplerx.py   # SMPLer-X motion backend
```

## Pose model (motion stage)

The pose stage recovers SMPL-X body motion from the video and feeds the
skeleton/diffusion stages. Two backends are bundled and selectable **per run**
from the control-page *Pose model* dropdown (`MOTION_BACKEND` only sets the
default):

| Backend | Model | Tradeoff |
| --- | --- | --- |
| `gvhmr` (default) | **GVHMR** (upstream, video/temporal) | stable depth and global trajectory, no jitter; per-frame pose can be softer |
| `prompthmr` | **PromptHMR-Vid** (CVPR 2025, video/temporal) | best pose accuracy + stable world trajectory; heavier motion stage |
| `smplerx` | **SMPLer-X-H32** (ViT-H, NeurIPS 2023, per-frame) | sharper per-frame pose than GVHMR; jitter and unstable depth because frames are regressed independently |

The PromptHMR backend vendors PromptHMR into `/opt/prompthmr` and uses only the
single-person, static-camera path — none of its compiled world-video extras
(detectron2 / SAM2 / DROID-SLAM / Metric3D / pytorch3d) are installed. Run
`python /app/scripts/download_prompthmr.py` (or `AUTO_DOWNLOAD_MODELS=true`) to
fetch its checkpoints into `models/prompthmr/`. With internet egress to
`drive.google.com` and `dl.fbaipublicfiles.com` this is fully automatic: the
script pulls the `phmr`/`phmr_vid` Google Drive folders, the standalone ViTPose
checkpoint (published as a lone Drive file, *not* inside a folder), the BEDLAM2
video head, the YOLO detector (reused from GVHMR) and the MetaCLIP backbone.
TLS verification is skipped for these public, read-only downloads by default
(set `PROMPTHMR_INSECURE_SSL=0` to enforce it), so clusters behind a
TLS-inspecting proxy work without mounting its CA.

PromptHMR's weights are **not published on HuggingFace**, so internal-CA /
air-gapped clusters that cannot reach those hosts must mirror these files once
(into their own HF repo or a PVC dir):

```
phmr/checkpoint.ckpt
phmr/config.yaml
phmr_vid/prhmr_release_002.ckpt
phmr_vid/prhmr_release_002.yaml
vitpose-h-coco_25.pth
smplx2smpl_joints.npy
smplx2smpl.pkl
smpl/SMPL_NEUTRAL.pkl      # separately licensed (see SMPL_SOURCE below)
l14_fullcc2.5b.pt          # https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt
```

Then either:
- `PROMPTHMR_CHECKPOINTS_SOURCE=/app/models/prompthmr-src` (PVC dir), or
- `PROMPTHMR_HF_REPO=<your-org>/prompthmr-weights` (pulled via `HF_ENDPOINT`).

The YOLO detector is reused from GVHMR's `yolov8x.pt` and SMPL-X from the model
cache, so neither needs mirroring. SMPL is licensed separately (like SMPL-X):
provide `SMPL_SOURCE=/path/to/SMPL_NEUTRAL.pkl`, or let the script fall back to
the public mirror (`SMPL_MIRROR_REPO` / `SMPL_MIRROR_FILE`).

The SMPLer-X checkpoint (`smpler_x_h32_correct.pth.tar`, ~2.6 GB from Hugging
Face `caizhongang/SMPLer-X`) is downloaded on first use or with
`AUTO_DOWNLOAD_MODELS=true`; run `python /app/scripts/download_smplerx.py` to
pre-seed the `models` volume. GVHMR assets are part of the standard model
download.

Because the two backends produce different motion for the same clip, give a
swapped run a fresh output directory (the control page already does this per
upload).

Notes:
- The swap is isolated to the motion stage; the skeleton/geometry stage still
  uses the GVHMR checkout's body-model utilities (unchanged).
- SMPLer-X predicts per-frame camera-frame SMPL-X; the launcher converts it to
  the pipeline's gravity-aligned world frame (GVHMR convention), so cached
  motion from one backend is not reused by the other — give swapped runs new
  output directories.
- `SMPLERX_BATCH` trades VRAM for speed on the 170HX (default 16 is fine on
  64 GB).

## How the supervisor works

`launcher.py` (the only custom code) serves the control page and owns the
lifecycle of the official Space subprocess:

1. Upload a video → validated (≥121 frames via PyAV), saved to `data/uploads/`.
2. With *Delete previous runs before starting* checked (`CLEAR_ON_START`), the previous official
   process is stopped (`SIGTERM`) and `data/uploads/`, `data/fdanyone/`, `data/space-cache/`,
   `data/logs/` are wiped. Bundled examples, the model cache, and the SMPL-X source are kept.
3. A fresh `python app.py --video_path <video> --output_dir <fresh> ... --server_port 7861` starts
   with the dropdown's `MOTION_BACKEND`.
4. Once port 7861 accepts connections, your browser is redirected there.
5. Finished runs stay under `data/fdanyone/<clip>-<timestamp>/` on the volume; reopen any of them from the *Previous runs* accordion.

## Notes

- SMPL-X is separately licensed; the bundled mirror may disappear — keep the official zip (`smpl-x.is.tue.mpg.de`) as a fallback via `SMPLX_SOURCE`.
- Run the container with `ipc: host` (compose does this) for NCCL/torch shared memory across the GPUs.
- The official Space is upstream's `app.py`, used unmodified.

## Kubernetes

The same image works in k8s, but mind the health-check semantics:

- **Readiness/liveness probe port 7860 only** (the always-on control page). Port 7861 is bound only while a viewer session is active — probing it makes the pod flap between runs and blackhole both ports from the Service.
- Expose both ports through one Service (named ports `control: 7860`, `viewer: 7861`); set `PUBLIC_VIEWER_URL` to the Service/Ingress URL for 7861 so the redirect points somewhere the browser can reach.
- Single replica (`replicas: 1`, `strategy: Recreate`) — the supervisor holds the viewer subprocess in memory.
- Replicate compose GPU settings: `hostIPC: true`, an `emptyDir` with `medium: Memory` + `sizeLimit: 1Gi` at `/dev/shm`, `nvidia.com/gpu` resource limits, and PVCs for `/app/models` and `/app/data`.

## Prepare Video

```sh
ffmpeg -i input.mp4 -frames:v 121 -vf "fps=30,scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black" -c:v libx264 -pix_fmt yuv420p -c:a aac -r 30 output.mp4
```
