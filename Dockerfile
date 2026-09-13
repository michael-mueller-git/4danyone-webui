# syntax=docker/dockerfile:1
#
# 4DAnyone Gradio WebUI
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
ARG FDANYONE_REV=main
ARG GVHMR_DEPTH=1
RUN git clone --depth 1 --branch ${FDANYONE_REV} ${FDANYONE_REPO} /app \
    && git -C /app submodule update --init --depth ${GVHMR_DEPTH} third_party/GVHMR

# Core 4DAnyone requirements (torch is inherited from the base image).
RUN pip install --no-cache-dir -r /app/requirements.txt

# WebUI dependencies. SageAttention 1.0.6 is the pure-Python (Triton) wheel:
# no CUDA compile, and it runs the only code path available on sm_80.
COPY requirements-webui.txt /tmp/requirements-webui.txt
RUN pip install --no-cache-dir -r /tmp/requirements-webui.txt \
    && python -c "import gradio, sageattention; print('webui deps OK')"

# Best-effort extras for the GVHMR submodule runtime imports.
RUN pip install --no-cache-dir rich matplotlib scikit-image joblib trimesh chumpy hydra_colorlog || true

# WebUI application files.
COPY entrypoint.sh /app/entrypoint.sh
COPY app/ /app/app/
COPY scripts/ /app/scripts/
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    GVHMR_ROOT=/app/third_party/GVHMR

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
