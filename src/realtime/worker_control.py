"""HTTP control API for a worker Jetson."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from aiohttp import web

from src.runtime_config import load_realtime_config, redact_url


class RealtimeWorkerControl:
    """Start, stop, restart, and inspect local camera worker processes."""

    def __init__(self, config: dict[str, Any], repo_dir: Path):
        self.config = config
        self.repo_dir = repo_dir.resolve()
        self.control = config.get("control", {})
        self.worker_control = self.control.get("worker", {})
        self.host = self.worker_control.get("bind_host", "0.0.0.0")
        self.port = int(self.worker_control.get("port", 8787))
        self.auto_scan = bool(self.worker_control.get("auto_scan", True))
        self.auto_start_enabled = bool(self.worker_control.get("start_on_launch", False))
        self.auto_start_retry_seconds = max(
            1.0,
            float(self.worker_control.get("auto_start_retry_seconds", 5.0)),
        )
        self.sources = [str(s) for s in self.worker_control.get("sources", [0, 1, 2, 3])]
        self.configured_camera_ids = [str(s) for s in self.worker_control.get("camera_ids", [])]
        if len(set(self.configured_camera_ids)) != len(self.configured_camera_ids):
            raise ValueError("control.worker.camera_ids must be unique")
        log_root = Path(os.environ.get("RUNTIME_LOG_DIR", "outputs")).expanduser()
        if not log_root.is_absolute():
            log_root = self.repo_dir / log_root
        self.log_dir = log_root / "realtime_worker_logs"

    async def run(self) -> None:
        app = web.Application()
        app.add_routes([web.get("/status", self.handle_status), web.post("/control", self.handle_control)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host=self.host, port=self.port)
        await site.start()
        print(f"Worker control API listening on {self.host}:{self.port}")
        auto_start_task = asyncio.create_task(self.auto_start_loop())
        try:
            await asyncio.Event().wait()
        finally:
            auto_start_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await auto_start_task

    async def auto_start_loop(self) -> None:
        """Start cameras after boot and recover workers that exit unexpectedly."""
        while True:
            if self.auto_start_enabled:
                try:
                    discovered = self.discover_sources() if self.auto_scan else self.sources
                    workers = self.worker_processes()
                    mapping_error = self.source_mapping_error(discovered)
                    if mapping_error:
                        if workers:
                            self.stop_workers()
                        print(f"Worker auto-start waiting: {mapping_error}")
                    elif not workers:
                        result = self.start_workers(force_scan=self.auto_scan)
                        print(f"Worker auto-start: {result.get('message', result)}")
                    elif not self.workers_match_sources(workers, discovered):
                        self.stop_workers()
                        result = self.start_workers(force_scan=self.auto_scan)
                        print(
                            "Worker auto-recovery: "
                            f"expected={len(discovered)} running={len(workers)} "
                            f"result={result.get('message', result)}"
                        )
                except Exception as exc:
                    print(f"Worker auto-start failed: {exc}")
            await asyncio.sleep(self.auto_start_retry_seconds)

    async def handle_status(self, request: web.Request) -> web.Response:
        return web.json_response(self.status())

    async def handle_control(self, request: web.Request) -> web.Response:
        payload = await request.json()
        action = payload.get("action")
        if action == "start":
            self.auto_start_enabled = True
            result = self.start_workers()
        elif action == "stop":
            self.auto_start_enabled = False
            result = self.stop_workers()
        elif action == "restart":
            self.auto_start_enabled = True
            self.stop_workers()
            await asyncio.sleep(1)
            result = self.start_workers()
        elif action == "scan":
            result = {"sources": self.discover_sources()}
        elif action == "restart_scan":
            self.auto_start_enabled = True
            self.stop_workers()
            await asyncio.sleep(1)
            result = self.start_workers(force_scan=True)
        else:
            return web.json_response({"ok": False, "error": f"Unsupported action: {action}"}, status=400)
        ok = bool(result.get("ok", True))
        return web.json_response(
            {"ok": ok, "action": action, "result": result, "status": self.status()},
            status=200 if ok else 409,
        )

    def start_workers(self, force_scan: bool = False) -> dict[str, Any]:
        if self.worker_processes():
            return {"ok": True, "message": "workers already running"}

        if self.auto_scan or force_scan:
            self.sources = self.discover_sources()

        if not self.sources:
            return {"ok": False, "message": "no cameras discovered", "sources": []}

        mapping_error = self.source_mapping_error(self.sources)
        if mapping_error:
            return {
                "ok": False,
                "message": mapping_error,
                "sources": self.public_sources(self.sources),
                "configured_camera_ids": self.configured_camera_ids,
            }

        camera_ids = self.camera_ids_for_sources()
        source_specs = [f"{camera_id}={source}" for camera_id, source in zip(camera_ids, self.sources)]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        command = ["setsid", "-f", "bash", "scripts/start_4_realtime_workers.sh", *source_specs]
        with (self.log_dir / "launcher.log").open("ab") as log:
            subprocess.Popen(
                command,
                cwd=self.repo_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return {
            "ok": True,
            "message": "workers starting",
            "sources": self.public_sources(self.sources),
            "camera_ids": camera_ids,
        }

    def stop_workers(self) -> dict[str, Any]:
        before = self.worker_processes(include_launcher=True)
        self.run_shell(["pkill", "-TERM", "-f", "[r]ealtime_worker.py"])
        self.run_shell(["pkill", "-TERM", "-f", "[s]tart_4_realtime_workers"])
        time.sleep(0.5)
        self.run_shell(["pkill", "-KILL", "-f", "[r]ealtime_worker.py"])
        self.run_shell(["pkill", "-KILL", "-f", "[s]tart_4_realtime_workers"])
        return {
            "ok": True,
            "message": "workers stopped",
            "previous": self.public_workers(before),
        }

    def status(self) -> dict[str, Any]:
        workers = self.worker_processes()
        discovered_sources = self.discover_sources()
        return {
            "running": len(workers) > 0,
            "auto_start_enabled": self.auto_start_enabled,
            "worker_count": len(workers),
            "workers": self.public_workers(workers),
            "sources": self.public_sources(self.sources),
            "camera_ids": self.camera_ids_for_sources(),
            "configured_camera_ids": self.configured_camera_ids,
            "source_mapping_error": self.source_mapping_error(discovered_sources),
            "discovered_sources": self.public_sources(discovered_sources),
            "logs": self.tail_logs(),
            "metrics": self.worker_metrics(),
        }

    @staticmethod
    def public_sources(sources: list[str]) -> list[str]:
        return [redact_url(str(source)) for source in sources]

    @staticmethod
    def public_workers(workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **worker,
                "cmd": redact_url(str(worker.get("cmd", ""))),
            }
            for worker in workers
        ]

    def camera_ids_for_sources(self, sources: list[str] | None = None) -> list[str]:
        """Return stable camera IDs for the sources on this Jetson."""
        active_sources = self.sources if sources is None else sources
        if self.configured_camera_ids:
            if len(self.configured_camera_ids) != len(active_sources):
                return []
            return self.configured_camera_ids

        default_camera_id = self.worker_control.get("camera_id") or self.config.get("worker", {}).get("camera_id")
        if len(active_sources) == 1 and default_camera_id:
            return [str(default_camera_id)]

        return [f"cam{idx + 1}" for idx in range(len(active_sources))]

    def source_mapping_error(self, sources: list[str]) -> str | None:
        """Reject ambiguous source-to-global-camera-ID mappings."""
        if self.configured_camera_ids and len(sources) != len(self.configured_camera_ids):
            return (
                f"discovered {len(sources)} camera(s), but "
                f"{len(self.configured_camera_ids)} camera ID(s) are configured"
            )
        return None

    def workers_match_sources(
        self,
        workers: list[dict[str, Any]],
        sources: list[str],
    ) -> bool:
        """Check both worker count and active camera-ID/source arguments."""
        if self.source_mapping_error(sources) or len(workers) != len(sources):
            return False
        expected = dict(zip(self.camera_ids_for_sources(sources), sources))

        active = {}
        for worker in workers:
            try:
                tokens = shlex.split(str(worker.get("cmd", "")))
                camera_index = tokens.index("--camera-id") + 1
                source_index = tokens.index("--source") + 1
                active[tokens[camera_index]] = tokens[source_index]
            except (ValueError, IndexError):
                return False
        return active == expected

    def discover_sources(self) -> list[str]:
        """Discover capture interfaces for currently connected USB cameras.

        Prefer /dev/v4l/by-path/*video-index0 because it is stable per USB port.
        Fall back to /dev/video even-numbered nodes, which is the common UVC
        capture-node layout for C920 cameras.
        """
        by_path = sorted(Path("/dev/v4l/by-path").glob("*video-index0"))
        # Keep the by-path symlink itself. Resolving it back to /dev/videoN
        # defeats stable camera identity after unplug/replug or device renumbering.
        sources = [str(path) for path in by_path if path.exists()]
        if sources:
            return sources

        video_nodes = sorted(
            Path("/dev").glob("video*"),
            key=lambda path: int(path.name.replace("video", "")) if path.name.replace("video", "").isdigit() else 9999,
        )
        return [str(path) for path in video_nodes if path.name.replace("video", "").isdigit() and int(path.name.replace("video", "")) % 2 == 0]

    def worker_processes(self, include_launcher: bool = False) -> list[dict[str, Any]]:
        needles = ["[r]ealtime_worker.py"]
        if include_launcher:
            needles.append("[s]tart_4_realtime_workers")
        workers = []
        for needle in needles:
            result = subprocess.run(["pgrep", "-af", needle], text=True, capture_output=True, check=False)
            for line in result.stdout.splitlines():
                pid, _, cmd = line.partition(" ")
                if pid.isdigit():
                    workers.append({"pid": int(pid), "cmd": cmd})
        return workers

    def tail_logs(self) -> dict[str, list[str]]:
        logs = {}
        for path in self.camera_log_paths():
            try:
                logs[path.name] = path.read_text(errors="replace").splitlines()[-5:]
            except OSError:
                logs[path.name] = []
        return logs

    def worker_metrics(self) -> dict[str, dict[str, Any]]:
        metrics = {}
        pattern = re.compile(r"camera=(\S+) frame=(\d+) detections=(\d+) sent_fps=([0-9.]+)")
        for path in self.camera_log_paths():
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                match = pattern.search(line)
                if match:
                    camera_id, frame, detections, fps = match.groups()
                    metrics[camera_id] = {
                        "frame_id": int(frame),
                        "detections": int(detections),
                        "sent_fps": float(fps),
                        "log": path.name,
                    }
                    break
        return metrics

    def camera_log_paths(self) -> list[Path]:
        return [
            path
            for path in sorted(self.log_dir.glob("*.log"))
            if path.name not in {"control.log", "launcher.log"}
        ]

    @staticmethod
    def run_shell(command: list[str]) -> None:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def load_control_config(path: Path) -> dict[str, Any]:
    return load_realtime_config(path)
