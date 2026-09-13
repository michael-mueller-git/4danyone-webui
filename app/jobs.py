"""Persistent job queue for the 4DAnyone WebUI.

Jobs are recorded in a small SQLite database and executed by a subprocess
(``python -m app.runner_job``) so the Gradio process never blocks or holds
CUDA context. A scheduler thread keeps at most ``MAX_CONCURRENCY`` jobs
running; the pipeline itself distributes one job across every visible GPU.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import config

LOGGER = logging.getLogger("jobs")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    status      TEXT    NOT NULL DEFAULT 'queued',
    video_name  TEXT,
    params      TEXT,
    out_dir     TEXT,
    manifest    TEXT,
    error       TEXT,
    created_at  TEXT,
    started_at  TEXT,
    finished_at TEXT,
    pid         INTEGER
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(str(config.DB_PATH), timeout=30)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA busy_timeout=30000;")
    connection.row_factory = sqlite3.Row
    return connection


def update_job(job_id: int, **fields: Any) -> None:
    """Update arbitrary columns; safe to call from the runner subprocess."""

    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [job_id]
    with _connect() as connection:
        connection.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)


def tail_log(job_id: int, max_chars: int = 12_000) -> str:
    log_file = log_path(job_id)
    if not log_file.is_file():
        return ""
    data = log_file.read_text(errors="replace")
    return data[-max_chars:] if len(data) > max_chars else data


def log_path(job_id: int) -> Path:
    return config.LOGS_DIR / f"job-{job_id}.log"


class JobBackend:
    """Owns the DB and the scheduler thread for one WebUI process."""

    def __init__(self, max_concurrency: int | None = None) -> None:
        self.max_concurrency = max_concurrency if max_concurrency else config.MAX_CONCURRENCY
        config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        with _connect() as connection:
            connection.execute(_SCHEMA)
        self._workdir = str(Path(__file__).resolve().parent)
        self._running: dict[int, subprocess.Popen] = {}
        self._condition = threading.Condition()
        self._scheduler = threading.Thread(target=self._schedule_loop, name="job-scheduler", daemon=True)
        self._scheduler.start()

    # -- query API ---------------------------------------------------------
    def create_job(self, *, video_path: str | Path, params: dict[str, Any]) -> int:
        job_dir = config.JOBS_DIR
        job_dir.mkdir(parents=True, exist_ok=True)

        video_path = Path(video_path)
        video_name = video_path.name
        # Use an existing id slot for a stable directory layout.
        with _connect() as connection:
            cursor = connection.execute(
                "INSERT INTO jobs (status, video_name, created_at) VALUES ('queued', ?, ?)",
                (video_name, _now()),
            )
            job_id = int(cursor.lastrowid)

        job_dir = config.JOBS_DIR / str(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        out_dir = job_dir / "out"

        extension = video_path.suffix or ".mp4"
        stored_video = job_dir / f"input{extension}"
        import shutil

        shutil.copy2(video_path, stored_video)

        manifest = {
            "job_id": job_id,
            "video_path": str(stored_video),
            "video_name": video_name,
            "out_dir": str(out_dir),
            "model_dir": str(config.MODEL_DIR),
            "gvhmr_root": str(config.GVHMR_ROOT),
            "params": params,
            "python": sys.executable,
        }
        manifest_path = job_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        with _connect() as connection:
            connection.execute(
                "UPDATE jobs SET out_dir = ?, manifest = ? WHERE id = ?",
                (str(out_dir), str(manifest_path), job_id),
            )
        with self._condition:
            self._condition.notify_all()
        return job_id

    def list(self, limit: int = 60) -> list[dict[str, Any]]:
        with _connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, job_id: int) -> dict[str, Any] | None:
        with _connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    # -- control API -------------------------------------------------------
    def _spawn(self, job_id: int, manifest_path: str) -> subprocess.Popen:
        environment = os.environ.copy()
        environment.setdefault("PYTHONUNBUFFERED", "1")
        log_file = open(log_path(job_id), "ab", buffering=0)
        proc = subprocess.Popen(
            [sys.executable, "-m", "app.runner_job", str(job_id), manifest_path],
            cwd=self._workdir,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._running[job_id] = proc
        log_file.close()
        return proc

    def cancel(self, job_id: int) -> str:
        proc = self._running.get(job_id)
        if proc is None:
            return "job is not currently running"
        import signal

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        return "cancellation requested"

    def _schedule_loop(self) -> None:
        while True:
            with _connect() as connection:
                rows = connection.execute(
                    "SELECT id, manifest FROM jobs WHERE status = 'queued' ORDER BY id ASC"
                ).fetchall()
            running = [job_id for job_id in self._running if self._running[job_id].poll() is None]
            self._running = {job_id: proc for job_id, proc in self._running.items() if proc.poll() is None}
            for row in rows:
                if len(running) >= self.max_concurrency:
                    break
                job_id = int(row["id"])
                manifest_path = row["manifest"]
                if not manifest_path:
                    update_job(job_id, status="failed", error="missing manifest", finished_at=_now())
                    continue
                update_job(job_id, status="running", started_at=_now())
                try:
                    self._spawn(job_id, manifest_path)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.exception("spawn failed for job %s", job_id)
                    update_job(job_id, status="failed", error=str(exc), finished_at=_now())
                    continue
                running.append(job_id)

            # Reap finished subprocess crawls: the runner marks status, so if a
            # worker died without finalizing, fail the job.
            for job_id, proc in list(self._running.items()):
                code = proc.poll()
                if code is None:
                    continue
                self._running.pop(job_id, None)
                job = self.get(job_id)
                if job and job["status"] == "running":
                    update_job(
                        job_id,
                        status="failed",
                        error=f"worker exited unexpectedly with code {code}",
                        finished_at=_now(),
                    )

            with self._condition:
                self._condition.wait(timeout=2.0)
