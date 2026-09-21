"""4DAnyone Docker launcher.

Serves a small control page on CONTROL_PORT (default 7860) that accepts a
video upload, then supervises the *official* 4DAnyone Gradio Space running on
OFFICIAL_PORT (default 7861). Uploading a new video stops the previous
official process and starts a fresh one, so no container restart is needed
between runs.

The official Space is launched unmodified via upstream ``app.py``:

    python app.py --video_path <video> --output_dir <run> \
        --model_dir <models> --gvhmr_root <gvhmr> --cache_dir <cache> \
        --server_name 0.0.0.0 --server_port <OFFICIAL_PORT> \
        --attention_backend <auto>
"""

from __future__ import annotations

import atexit
import datetime as _dt
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Must be set before Gradio is imported (it initializes its upload cache).
os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(os.environ.get("DATA_DIR", "/app/data")) / ".gradio-tmp"))

import gradio as gr  # noqa: E402

REPO_ROOT = Path("/app")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models")).resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data")).resolve()
GVHMR_ROOT = Path(os.environ.get("GVHMR_ROOT", "/app/third_party/GVHMR")).resolve()

CONTROL_PORT = int(os.environ.get("CONTROL_PORT", "7860"))
OFFICIAL_PORT = int(os.environ.get("OFFICIAL_PORT", "7861"))
PUBLIC_VIEWER_URL = os.environ.get("PUBLIC_VIEWER_URL") or f"http://127.0.0.1:{OFFICIAL_PORT}"
ATTENTION_BACKEND = os.environ.get("DEFAULT_ATTENTION_BACKEND", "auto")

DEFAULT_MOTION_BACKEND = os.environ.get("MOTION_BACKEND", "gvhmr").strip().lower()
CLEAR_ON_START = os.environ.get("CLEAR_ON_START", "true").strip().lower() not in {"0", "false", "no", "off"}
POSE_MODELS = (
    ("GVHMR — stable depth/trajectory", "gvhmr"),
    ("PromptHMR-Vid — SOTA pose + world", "prompthmr"),
    ("SMPLer-X — sharper per-frame pose", "smplerx"),
)

UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "fdanyone"
CACHE_DIR = DATA_DIR / "space-cache"
LOGS_DIR = DATA_DIR / "logs"
EXAMPLES_DIR = DATA_DIR / "source" / "pexels"

# Disposable artifacts wiped before a fresh video task. Never touches the
# bundled examples (EXAMPLES_DIR), the licensed SMPL-X source, or MODEL_DIR.
CLEAN_DIRS = (UPLOADS_DIR, OUTPUTS_DIR, CACHE_DIR, LOGS_DIR)

MIN_FRAMES = 121
READY_TIMEOUT = 240
STOP_GRACE = 10
# Port-open != UI-ready; the Rerun scene can fail to load if the browser is
# redirected too early, so wait this long after readiness before redirecting.
VIEWER_REDIRECT_DELAY = max(0.0, float(os.environ.get("VIEWER_REDIRECT_DELAY", "1")))


def _now() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^\w.-]", "_", name)
    return cleaned or "video.mp4"


def _frame_count(path: Path) -> int | None:
    try:
        import av
    except Exception:
        return None
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.frames:
                return stream.frames
            rate = stream.average_rate or stream.guessed_rate
            duration = stream.duration * stream.time_base if stream.duration else container.duration / av.time_base
            if rate and duration:
                return int(float(duration) * float(rate))
            return None
    except Exception:
        return None


def _wait_ready(proc: subprocess.Popen, timeout: float) -> bool:
    """Return True once the official Space accepts connections, or if it died."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", OFFICIAL_PORT), timeout=2):
                return True
        except OSError:
            time.sleep(1.0)
    return False


def _settle_viewer() -> None:
    """Give the official Space's UI/assets time to finish booting.

    The port opens before the Gradio frontend and Rerun scene are usable, so a
    redirect sent immediately after readiness can fail to load until a reload.
    """

    if VIEWER_REDIRECT_DELAY > 0:
        time.sleep(VIEWER_REDIRECT_DELAY)


def _redirect_html(url: str) -> str:
    # A <meta http-equiv="refresh"> is honored even when injected via innerHTML
    # (Gradio drops <script> tags in gr.HTML values), so auto-redirect works.
    # The link is a fallback if the browser refuses the meta refresh.
    return (
        '<div style="padding:1rem;font-size:1.1rem">'
        f'<meta http-equiv="refresh" content="0; url={url}">'
        f'Viewer ready — <a href="{url}">Open the 4DAnyone viewer</a>'
        "</div>"
    )


class Supervisor:
    """Owns the official Space subprocess and the current-run metadata."""

    def __init__(self) -> None:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.proc: subprocess.Popen | None = None
        self.current: dict | None = None

    # -- process control -------------------------------------------------
    def _official_command(self, *, video: str | None, output: str) -> list[str]:
        command = [
            sys.executable,
            str(REPO_ROOT / "app.py"),
            "--output_dir", str(output),
            "--model_dir", str(MODEL_DIR),
            "--gvhmr_root", str(GVHMR_ROOT),
            "--cache_dir", str(CACHE_DIR),
            "--server_name", "0.0.0.0",
            "--server_port", str(OFFICIAL_PORT),
            "--attention_backend", ATTENTION_BACKEND,
        ]
        if video:
            command[2:2] = ["--video_path", str(video)]
        return command

    def stop_official(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = None
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            self.proc.wait(timeout=STOP_GRACE)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            self.proc.wait()
        self.proc = None

    def start_official(self, *, video: str | None, output: str, backend: str) -> None:
        self.stop_official()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / f"official-{_now()}.log"
        command = self._official_command(video=video, output=output)
        environment = os.environ.copy()
        environment["MOTION_BACKEND"] = backend
        with log_path.open("wb") as log:
            self.proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.current = {"video": video or "", "output": output, "log": str(log_path), "backend": backend}

    def clear_runs(self) -> None:
        """Delete disposable run artifacts before a fresh video task.

        Stops the viewer first so nothing is deleted while in use, then wipes
        uploads, published/scratch outputs, cache, and old logs. Bundled
        examples, the model cache, and the licensed SMPL-X source are untouched.
        """

        self.stop_official()
        for directory in CLEAN_DIRS:
            try:
                if directory.is_dir():
                    for child in directory.iterdir():
                        if child.is_dir() and not child.is_symlink():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            child.unlink(missing_ok=True)
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                print(f"[launcher] could not clear {directory}: {exc}", flush=True)
        self.current = None

    # -- discovery helpers ------------------------------------------------
    def list_examples(self) -> list[str]:
        if not EXAMPLES_DIR.is_dir():
            return []
        return sorted(str(p) for p in EXAMPLES_DIR.glob("*.mp4"))

    def list_runs(self) -> list[str]:
        if not OUTPUTS_DIR.is_dir():
            return []
        dirs = sorted(
            (p for p in OUTPUTS_DIR.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [str(p) for p in dirs]

    def log_tail(self, path: str | None, chars: int = 8000) -> str:
        if not path or not Path(path).is_file():
            return ""
        data = Path(path).read_text(errors="replace")
        return data[-chars:] if len(data) > chars else data


supervisor = Supervisor()


# -- handlers -------------------------------------------------------------
def _resolve_source(video_file: str | None, example: str | None) -> str:
    if video_file and Path(video_file).is_file():
        return str(Path(video_file).resolve())
    if example and Path(example).is_file():
        return str(Path(example).resolve())
    raise gr.Error("Choose an uploaded video or a bundled example.")


def _stage_upload(source: str) -> str:
    path = Path(source)
    if path.is_relative_to(UPLOADS_DIR) or path.is_relative_to(EXAMPLES_DIR):
        return str(path)
    target = UPLOADS_DIR / f"{_now()}-{_sanitize(path.name)}"
    shutil.copy2(path, target)
    return str(target)


def start_run(video_file, example, backend, clear_old):
    if clear_old:
        yield gr.update(value="Clearing previous runs…"), gr.update(visible=False)
        supervisor.clear_runs()
    yield gr.update(value="Saving video…"), gr.update(visible=False)
    try:
        source = _stage_upload(_resolve_source(video_file, example))
    except gr.Error as exc:
        yield gr.update(value=str(exc)), gr.update(visible=False)
        return
    frames = _frame_count(Path(source))
    if frames is not None and frames < MIN_FRAMES:
        yield (
            gr.update(value=f"Video has {frames} frames; at least {MIN_FRAMES} are required."),
            gr.update(visible=False),
        )
        return
    output = str(OUTPUTS_DIR / f"{Path(source).stem}-{_now()}")
    yield (
        gr.update(value=f"Starting the 4DAnyone viewer (video: {Path(source).name}, pose model: {backend}). "
                        "The first start can take a minute…"),
        gr.update(visible=False),
    )
    supervisor.start_official(video=source, output=output, backend=backend)
    if not _wait_ready(supervisor.proc, READY_TIMEOUT):
        log = supervisor.log_tail(supervisor.current["log"])
        yield (
            gr.update(value=f"Viewer failed to start.\nLog tail:\n{log}"),
            gr.update(visible=False),
        )
        return
    _settle_viewer()
    yield (
        gr.update(value=f"Viewer ready: {PUBLIC_VIEWER_URL}"),
        gr.update(value=_redirect_html(PUBLIC_VIEWER_URL), visible=True),
    )


def reopen_run(run_dir, backend):
    yield gr.update(value="Opening the saved run…"), gr.update(visible=False)
    if not run_dir or not Path(run_dir).is_dir():
        yield gr.update(value="Select a run to reopen."), gr.update(visible=False)
        return
    supervisor.start_official(video=None, output=str(Path(run_dir).resolve()), backend=backend)
    if not _wait_ready(supervisor.proc, READY_TIMEOUT):
        log = supervisor.log_tail(supervisor.current["log"])
        yield (
            gr.update(value=f"Failed to open the run.\nLog tail:\n{log}"),
            gr.update(visible=False),
        )
        return
    _settle_viewer()
    yield (
        gr.update(value=f"Viewer ready: {PUBLIC_VIEWER_URL}"),
        gr.update(value=_redirect_html(PUBLIC_VIEWER_URL), visible=True),
    )


def stop_viewer():
    supervisor.stop_official()
    return "Viewer stopped. Upload a video to start a new run."


def refresh_current():
    info = "no active run"
    if supervisor.current:
        info = (
            f"video:  {supervisor.current['video'] or '(saved task)'}\n"
            f"model:  {supervisor.current.get('backend', DEFAULT_MOTION_BACKEND)}\n"
            f"output: {supervisor.current['output']}\n"
            f"log:    {supervisor.current['log']}"
        )
    return (
        gr.update(value=info),
        gr.update(value=supervisor.log_tail(supervisor.current["log"] if supervisor.current else None)),
        gr.update(choices=[str(p) for p in supervisor.list_runs()]),
    )


def gpu_status():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        return result.stdout or "nvidia-smi returned no output."
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def assets_status():
    gvhmr = (GVHMR_ROOT / "hmr4d" / "__init__.py").is_file()
    smplx = (MODEL_DIR / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz").is_file()
    try:
        from fdanyone import assets

        missing = [rel for rel in assets.MODEL_FILES if not (MODEL_DIR / rel).is_file()]
    except Exception:
        missing = ["(fdanyone not importable yet)"]
    lines = [
        f"model dir:            {MODEL_DIR}",
        f"gvhmr checkout:       {'ok' if gvhmr else 'MISSING'}",
        f"SMPL-X neutral:       {'present' if smplx else 'missing (licensed)'}",
        f"public assets missing: {len(missing)}",
    ]
    lines += [f"  - {name}" for name in missing[:15]]
    return "\n".join(lines)


def _default_pose_model() -> str:
    values = {value for _, value in POSE_MODELS}
    return DEFAULT_MOTION_BACKEND if DEFAULT_MOTION_BACKEND in values else POSE_MODELS[0][1]


# -- app assembly ---------------------------------------------------------
def build() -> gr.Blocks:
    examples = supervisor.list_examples()
    with gr.Blocks(title="4DAnyone WebUI") as demo:
        gr.Markdown(
            "# 4DAnyone WebUI\n\n"
            f"Upload a monocular video of a single person (≥121 frames, 1080p+, 9:16 ideal). "
            f"After it is ready your browser is sent to the official 4DAnyone viewer at "
            f"[{PUBLIC_VIEWER_URL}]({PUBLIC_VIEWER_URL}); run inference there. "
            "Uploading a new video replaces the current viewer session — no container restart needed."
        )
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### New run")
                video_file = gr.Video(label="Input video", sources=["upload"], height=280)
                example = gr.Dropdown(
                    choices=[{"value": path, "label": Path(path).name} for path in examples],
                    label="…or a bundled example",
                    value=None,
                )
                pose_model = gr.Dropdown(
                    choices=list(POSE_MODELS),
                    value=_default_pose_model(),
                    label="Pose model",
                )
                clear_old = gr.Checkbox(
                    label="Delete previous runs before starting",
                    value=CLEAR_ON_START,
                )
                start_btn = gr.Button("Start with this video", variant="primary")
                status = gr.Textbox(label="Status", lines=3, interactive=False)
                redirect = gr.HTML(visible=False)
            with gr.Column(scale=1):
                with gr.Accordion("Current viewer", open=True):
                    current_info = gr.Textbox(label="Current run", lines=4, interactive=False)
                    log_box = gr.Textbox(label="Viewer log (tail)", lines=10, interactive=False)
                    with gr.Row():
                        refresh_btn = gr.Button("Refresh")
                        stop_btn = gr.Button("Stop viewer", variant="stop")
                with gr.Accordion("Previous runs", open=False):
                    run_dropdown = gr.Dropdown(label="Run (output dir)", choices=[], value=None)
                    reopen_btn = gr.Button("Reopen selected run")
                with gr.Accordion("GPU health", open=False):
                    gpu_btn = gr.Button("Run GPU health check")
                    gpu_text = gr.Textbox(label="Health report", lines=12, interactive=False)
                with gr.Accordion("Models & assets", open=False):
                    assets_btn = gr.Button("Refresh asset status")
                    assets_text = gr.Textbox(label="Assets", lines=8, interactive=False)

        start_btn.click(start_run, [video_file, example, pose_model, clear_old], [status, redirect])
        reopen_btn.click(reopen_run, [run_dropdown, pose_model], [status, redirect])
        stop_btn.click(stop_viewer, outputs=[status])
        refresh_btn.click(refresh_current, outputs=[current_info, log_box, run_dropdown])
        gpu_btn.click(gpu_status, outputs=[gpu_text])
        assets_btn.click(assets_status, outputs=[assets_text])
    return demo


def _preseed() -> None:
    video = os.environ.get("VIDEO_PATH")
    if not video:
        return
    path = Path(video).expanduser().resolve()
    if not path.is_file():
        # Bundled example clips are auto-downloaded by the official app on
        # first use, so pass those names through and let it fetch them.
        try:
            from fdanyone.assets import EXAMPLE_FILES
        except Exception:
            EXAMPLE_FILES = ()
        bundled = any(Path(relative).name == path.name for relative in EXAMPLE_FILES)
        if not bundled:
            print(f"[launcher] VIDEO_PATH does not exist and is not a bundled example: {path}", flush=True)
            return
        print(f"[launcher] VIDEO_PATH is a bundled example; the official app will download {path.name}", flush=True)
    output = os.environ.get("OUTPUT_DIR")
    if output:
        output = str(Path(output).expanduser().resolve())
    else:
        output = str(OUTPUTS_DIR / f"{path.stem}-{_now()}")
    if CLEAR_ON_START:
        print("[launcher] clearing previous runs before pre-starting the official Space", flush=True)
        supervisor.clear_runs()
    print(f"[launcher] VIDEO_PATH set: starting official Space for {path.name}", flush=True)
    try:
        supervisor.start_official(video=str(path), output=output, backend=DEFAULT_MOTION_BACKEND)
    except Exception as exc:  # noqa: BLE001 - surface, do not crash the control page
        print(f"[launcher] failed to pre-start official Space: {exc}", flush=True)


def launch() -> None:
    demo = build()
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0",
        server_port=CONTROL_PORT,
        theme=gr.themes.Soft(),
        show_error=True,
    )


def _shutdown(_signum, _frame) -> None:
    supervisor.stop_official()
    sys.exit(0)


if __name__ == "__main__":
    atexit.register(supervisor.stop_official)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    _preseed()
    launch()
