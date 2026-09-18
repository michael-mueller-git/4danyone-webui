# syntax=docker/dockerfile:1
#
# 4DAnyone WebUI — official Gradio Space + upload supervisor.
#
# Base: PyTorch 2.8 / CUDA 12.8 with cuDNN. sm_80 (GA100) is included, so the
# unlocked CMP 170HX (compute capability 8.0, 64 GB) works out of the box.
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# System dependencies: ffmpeg CLI + codecs, git for the submodule, GL for OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libx264-dev \
        git \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Vendor 4DAnyone + its GVHMR submodule.
ARG FDANYONE_REPO=https://github.com/ant-research/4DAnyone.git
ARG FDANYONE_REV=61e0ccd5c85c01c612b4c4aecb3de148e6f5bae5
ARG GVHMR_DEPTH=1
RUN git init /app \
    && git -C /app remote add origin ${FDANYONE_REPO} \
    && git -C /app fetch --depth 1 origin ${FDANYONE_REV} \
    && git -C /app checkout --detach FETCH_HEAD \
    && git -C /app submodule update --init --depth ${GVHMR_DEPTH} third_party/GVHMR

# Core 4DAnyone requirements (torch is inherited from the base image).
RUN pip install --no-cache-dir -r /app/requirements.txt

# Official GUI dependencies (gradio 6.20.0, gradio-rerun, rerun-sdk).
RUN pip install --no-cache-dir -r /app/requirements-gui.txt \
    && python -c "import gradio, gradio_rerun, rerun_sdk; print('gui deps OK')"

# SageAttention 1.0.6 is the pure-Python (Triton) wheel: no CUDA compile, and it
# runs the only code path available on sm_80. With it installed (and
# FlashAttention-3 absent, which is Hopper-only), the pipeline's `auto` backend
# resolves to SageAttention on the 170HX.
RUN pip install --no-cache-dir sageattention==1.0.6 \
    && python -c "import sageattention; print('sageattention OK')"

# Best-effort extras for the GVHMR submodule runtime imports.
RUN pip install --no-cache-dir rich matplotlib scikit-image joblib trimesh chumpy hydra_colorlog || true

# WebUI launcher + headless SMPL-X provisioning.
COPY entrypoint.sh /app/entrypoint.sh
COPY launcher.py /app/launcher.py
COPY scripts/provision_smplx.py /app/scripts/provision_smplx.py
COPY scripts/download_smplerx.py /app/scripts/download_smplerx.py
RUN chmod +x /app/entrypoint.sh

# SMPLer-X motion stage (optional body-pose backend). The patched worker
# dispatches on MOTION_BACKEND; the vendored package is a pure-PyTorch port of
# the SMPLer-X-H32 body regressor (no mmcv/mmdet/mmpose).
COPY src/fdanyone/motion/worker.py /app/fdanyone/motion/worker.py
COPY src/fdanyone_smplerx/ /app/fdanyone_smplerx/
RUN python -c "import fdanyone_smplerx; print('smplerx package OK')"

ENV PYTHONPATH=/app \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    GVHMR_ROOT=/app/third_party/GVHMR \
    GRADIO_TEMP_DIR=/app/data/.gradio-tmp \
    MOTION_BACKEND=smplerx

EXPOSE 7860 7861

ENTRYPOINT ["/app/entrypoint.sh"]
