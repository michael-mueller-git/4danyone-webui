"""4DAnyone Gradio WebUI.

Tabs:
  * Generate  - upload a video, choose a view layout, enqueue a job.
  * Jobs      - track progress, preview dense/sparse/skeleton views, download.
  * Settings  - GPU health, model/SMPL-X provisioning.

Jobs run through a persistent SQLite queue in a separate subprocess, keeping
the Gradio process free of CUDA state. One job is distributed across all
visible GPUs by the 4DAnyone pipeline itself.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

import gradio as gr

from app import config, gallery, health, init_models, jobs

MAX_DENSE_VIDEOS = 24
MAX_SPARSE_VIDEOS = 12
MAX_SKELETON_VIDEOS = 12

backend = jobs.JobBackend()

_INIT_STATE = {"message": "idle"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- background model provisioning ---------------------------------------
def _init_task(banner: str, fn) -> None:
    def _worker() -> None:
        _INIT_STATE["message"] = f"running: {banner}"
        _append_init_log(f"=== {_now()} :: {banner} ===")

        def log(line: str) -> None:
            _append_init_log(line)

        try:
            fn(log)
            _INIT_STATE["message"] = f"done: {banner}"
            log(f"[ok] {banner}")
        except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
            _INIT_STATE["message"] = f"error: {banner}: {type(exc).__name__}: {exc}"
            log(f"[error] {banner}: {type(exc).__name__}: {exc}")

    threading.Thread(target=_worker, name=f"init-{banner}", daemon=True).start()


def _append_init_log(line: str) -> None:
    config.INIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.INIT_LOG, "a") as handle:
        handle.write(line + "\n")


def _init_log_tail(chars: int = 8000) -> str:
    if not config.INIT_LOG.is_file():
        return ""
    data = config.INIT_LOG.read_text(errors="replace")
    return data[-chars:] if len(data) > chars else data


def _button_init(banner: str, fn):
    def handler():
        activity = threading.active_count()
        if _INIT_STATE["message"].startswith("running:") and banner.split(":", 1)[0] in _INIT_STATE["message"]:
            return f"already {_INIT_STATE['message']}"
        _init_task(banner, fn)
        return f"started: {banner}"

    return handler


# -- Generate tab ---------------------------------------------------------
def _list_examples() -> list[str]:
    root = config.DATA_DIR / "source" / "pexels"
    if not root.is_dir():
        return []
    return [str(path) for path in sorted(root.glob("*.mp4"))]


def _parse_layer_pitches(raw: str) -> list[int]:
    try:
        return json.loads(raw) if raw.strip() else [15]
    except (json.JSONDecodeError, TypeError):
        return [int(part) for part in raw.replace(" ", "").strip("[]").split(",") if part]


def _parse_gpu_ids(raw: str) -> list[int] | None:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        return None
    return [int(value) for value in values]


def _submit(
    video_file,
    example_path,
    views_per_layer,
    layer_pitches,
    start_yaw,
    yaw_span,
    views_per_group,
    enable_rcp,
    enable_tcr,
    enable_turbo,
    seed,
    attention_backend,
    gpu_ids,
    target_fps,
):
    source = video_file
    if source is None:
        source = example_path
    if source is None:
        return "Choose an uploaded video or an example video."

    readiness = init_models.is_ready()
    if not readiness["gvhmr"]:
        return "GVHMR checkout is missing; rebuild the image."
    if not readiness["models"]:
        return "Public models are not downloaded yet. Open Settings -> Download public models (init)."
    if not readiness["smplx"]:
        return "SMPL-X is not installed. Open Settings -> Provision SMPL-X."

    params = {
        "views_per_layer": int(views_per_layer),
        "layer_pitches": _parse_layer_pitches(layer_pitches),
        "start_yaw": int(start_yaw),
        "yaw_span": int(yaw_span),
        "views_per_group": views_per_group,
        "enable_rcp": bool(enable_rcp),
        "enable_tcr": bool(enable_tcr),
        "enable_turbo": bool(enable_turbo),
        "seed": int(seed),
        "attention_backend": attention_backend,
        "gpu_ids": _parse_gpu_ids(gpu_ids or ""),
        "target_fps": "auto" if not target_fps else target_fps,
    }
    job_id = backend.create_job(video_path=source, params=params)
    return f"Job #{job_id} queued. Go to the Jobs tab."


# -- Jobs tab -------------------------------------------------------------
def _job_rows() -> list[dict]:
    rows = []
    for job in backend.list(60):
        params = json.loads(job.get("params") or "{}")
        rows.append(
            {
                "id": job["id"],
                "status": job["status"],
                "video": job["video_name"],
                "views": params.get("views_per_layer"),
                "pitches": params.get("layer_pitches"),
                "turbo": params.get("enable_turbo"),
                "attention": params.get("attention_backend"),
                "gpu_ids": ",".join(map(str, params["gpu_ids"])) if params.get("gpu_ids") else "all",
                "created": job["created_at"],
                "error": (job.get("error") or "")[:120],
            }
        )
    return rows


def _job_rows_table() -> list[list]:
    columns = ["id", "status", "video", "views", "pitches", "turbo", "attention", "gpu_ids", "created", "error"]
    return [[row[column] for column in columns] for row in _job_rows()]


def _load_job(job_id, *comps):
    n_dense = MAX_DENSE_VIDEOS
    n_sparse = MAX_SPARSE_VIDEOS
    n_skeleton = MAX_SKELETON_VIDEOS
    dense_out = list(comps[0:n_dense])
    sparse_out = list(comps[n_dense : n_dense + n_sparse])
    skeleton_out = list(comps[n_dense + n_sparse : n_dense + n_sparse + n_skeleton])

    def _noop_views(components, visible=False):
        return [gr.update(value=None, visible=visible) for _ in components]

    if job_id is None:
        return (
            "no job selected",
            "",
            gr.update(value=None, visible=False),
            *_noop_views(dense_out),
            *_noop_views(sparse_out),
            *_noop_views(skeleton_out),
        )

    job = backend.get(int(job_id))
    if job is None:
        return (
            "job not found",
            "",
            gr.update(value=None, visible=False),
            *_noop_views(dense_out),
            *_noop_views(sparse_out),
            *_noop_views(skeleton_out),
        )

    status_md = (
        f"**job #{job['id']}** · `{job['status']}` · video: `{job['video_name']}`  \n"
        f"created: `{job['created_at']}`  \n"
        f"out: `{job['out_dir']}`"
    )
    if job.get("error"):
        status_md += f"\nerror: `{job['error']}`"
    log = jobs.tail_log(int(job_id))
    outputs = gallery.job_outputs(job["out_dir"] or "")
    zip_path = gallery.build_zip(int(job_id), job["out_dir"]) if outputs.get("exists") else None

    def _fill(components, sources):
        return [
            (gr.update(value=sources[i], visible=True) if i < len(sources) else gr.update(value=None, visible=False))
            for i in range(len(components))
        ]

    return (
        status_md,
        log,
        gr.update(value=zip_path, visible=zip_path is not None),
        *_fill(dense_out, outputs["dense"]),
        *_fill(sparse_out, outputs["sparse"]),
        *_fill(skeleton_out, outputs["skeletons"]),
    )


# -- Settings tab ---------------------------------------------------------
def _gpu_report() -> str:
    return health.summary_text()


def _models_status() -> str:
    return init_models.status_text()


def _auto_init() -> None:
    if not config.AUTO_DOWNLOAD_MODELS:
        return
    _init_task(
        "auto-download (models)",
        lambda log: (init_models.download_public_models(log), init_models.download_examples(log)),
    )
    _init_task("auto-download (smplx)", lambda log: init_models.provision_smplx(log))


# -- App assembly ---------------------------------------------------------
def build() -> gr.Blocks:
    _auto_init()
    with gr.Blocks(title="4DAnyone WebUI", theme=gr.themes.Soft()) as demo:
        with gr.Tabs():
            # ---------------- Generate ----------------
            with gr.Tab("Generate"):
                gr.Markdown(
                    "Upload a monocular video of a single person (≥121 frames, 1080p+, 9:16 ideal). "
                    "Each job is distributed across every visible GPU by the 4DAnyone pipeline."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        video_file = gr.File(label="Input video", file_types=["video"])
                        gr.Markdown("or pick a bundled example (after *Download example videos* in Settings):")
                        examples = _list_examples()
                        example_path = gr.Dropdown(
                            choices=[{"value": path, "label": str(path).split("/")[-1]} for path in examples],
                            label="Example video",
                            value=None,
                        )
                    with gr.Column(scale=1):
                        views_per_layer = gr.Number(label="Views per layer (divisible by 4 or 6)", value=6, precision=0)
                        layer_pitches = gr.Textbox(label="Layer pitches (degrees, JSON list)", value="[15]")
                        with gr.Row():
                            start_yaw = gr.Number(label="Start yaw", value=0, precision=0)
                            yaw_span = gr.Slider(label="Yaw span (°)", minimum=1, maximum=360, value=360, step=1)
                        with gr.Row():
                            views_per_group = gr.Dropdown(label="Views per group", choices=["auto", "4", "6"], value="auto")
                            target_fps = gr.Dropdown(
                                label="Target FPS", choices=["auto", "24", "25", "30"], value="auto"
                            )
                        with gr.Row():
                            enable_turbo = gr.Checkbox(label="Turbo (4-step)", value=config.DEFAULT_ENABLE_TURBO)
                            enable_rcp = gr.Checkbox(label="RCP", value=True)
                            enable_tcr = gr.Checkbox(label="TCR", value=True)
                        with gr.Row():
                            attention_backend = gr.Dropdown(
                                label="Attention backend", choices=["auto", "sageattention", "sdpa"],
                                value=config.DEFAULT_ATTENTION,
                            )
                            gpu_ids = gr.Textbox(
                                label="GPU IDs (0,1,2 or empty for all)", value="", placeholder="all visible GPUs"
                            )
                        seed = gr.Number(label="Seed", value=42, precision=0)
                submit = gr.Button("Generate", variant="primary")
                submit_msg = gr.Textbox(label="Result", interactive=False)
                submit.click(
                    _submit,
                    inputs=[
                        video_file, example_path, views_per_layer, layer_pitches, start_yaw, yaw_span,
                        views_per_group, enable_rcp, enable_tcr, enable_turbo, seed, attention_backend,
                        gpu_ids, target_fps,
                    ],
                    outputs=[submit_msg],
                )

            # ---------------- Jobs ----------------
            with gr.Tab("Jobs"):
                gr.Markdown("Select a job to preview its generated views and download the archive.")
                with gr.Row():
                    refresh = gr.Button("Refresh list")
                    job_dropdown = gr.Dropdown(label="Job", choices=[], interactive=True)
                    cancel = gr.Button("Cancel running job")
                jobs_table = gr.Dataframe(
                    headers=["id", "status", "video", "views", "pitches", "turbo", "attention", "gpu_ids", "created", "error"],
                    label="Jobs",
                    interactive=False,
                    wrap=True,
                )
                status_md = gr.Markdown("no job selected")
                log_box = gr.Textbox(label="Job log (tail)", lines=14, interactive=False)
                download_all = gr.File(label="Download all views (zip)")

                dense_out = [gr.Video(label=f"dense {i:02d}", visible=False, interactive=False) for i in range(MAX_DENSE_VIDEOS)]
                sparse_out = [gr.Video(label=f"sparse {i:02d}", visible=False, interactive=False) for i in range(MAX_SPARSE_VIDEOS)]
                skeleton_out = [gr.Video(label=f"skeleton {i:02d}", visible=False, interactive=False) for i in range(MAX_SKELETON_VIDEOS)]

                def _refresh_rows():
                    return (
                        gr.update(choices=[str(job["id"]) for job in backend.list()]),
                        gr.update(value=_job_rows_table()),
                    )

                refresh.click(_refresh_rows, outputs=[job_dropdown, jobs_table])

                loaded_outputs = [
                    status_md, log_box, download_all,
                    *dense_out, *sparse_out, *skeleton_out,
                ]
                job_dropdown.select(
                    _load_job,
                    inputs=[job_dropdown, *dense_out, *sparse_out, *skeleton_out],
                    outputs=loaded_outputs,
                ).then(_refresh_rows, outputs=[job_dropdown, jobs_table])

                def _cancel(job_id):
                    if job_id:
                        return f"cancel {job_id}: {backend.cancel(int(job_id))}"
                    return "choose a job id to cancel"

                cancel.click(_cancel, inputs=[job_dropdown], outputs=[status_md])

            # ---------------- Settings ----------------
            with gr.Tab("Settings"):
                with gr.Accordion("GPU health", open=True):
                    gpu_button = gr.Button("Run GPU health check")
                    gpu_text = gr.Textbox(label="Health report", lines=18, interactive=False)
                    gpu_button.click(_gpu_report, outputs=[gpu_text])

                with gr.Accordion("Models & assets", open=True):
                    status_btn = gr.Button("Refresh status")
                    status_text = gr.Textbox(label="Model status", lines=14, interactive=False)
                    status_btn.click(_models_status, outputs=[status_text])

                    with gr.Row():
                        models_btn = gr.Button("Download public models (init)")
                        smplx_btn = gr.Button("Provision SMPL-X")
                        examples_btn = gr.Button("Download example videos")
                    init_status = gr.Textbox(label="Init state", value=_INIT_STATE["message"], interactive=False)
                    init_log = gr.Textbox(label="Init log (tail)", lines=12, interactive=False)

                    models_btn.click(_button_init("public models", lambda log: init_models.download_public_models(log)), outputs=[init_status])
                    smplx_btn.click(_button_init("smplx", lambda log: init_models.provision_smplx(log)), outputs=[init_status])
                    examples_btn.click(_button_init("examples", lambda log: init_models.download_examples(log)), outputs=[init_status])

                gr.Markdown(
                    "SMPL-X is a separately licensed body model. The WebUI first looks for a dropped "
                    f"archive in `{config.SMPLX_DROP_DIR}` or `SMPLX_SOURCE`, then falls back to a public "
                    "mirror of the official `models_smplx_v1_1.zip`."
                )

        # Periodic refresh of init state + log and models status.
        def _init_state_refresh():
            return _INIT_STATE["message"], _init_log_tail(), init_models.status_text()

        gr.Timer(2.0).tick(
            _init_state_refresh,
            outputs=[init_status, init_log, status_text],
        )

        # Keep the jobs table live while jobs are queued or running.
        gr.Timer(3.0).tick(lambda: gr.update(value=_job_rows_table()), outputs=[jobs_table])

        demo.queue(default_concurrency_limit=16)
    return demo


def launch() -> None:
    demo = build()
    auth = None
    if config.WEBUI_USERNAME and config.WEBUI_PASSWORD:
        auth = (config.WEBUI_USERNAME, config.WEBUI_PASSWORD)
    demo.launch(server_name=config.HOST, server_port=config.PORT, auth=auth)


if __name__ == "__main__":
    launch()
