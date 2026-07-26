"""Jetson camera worker for realtime distributed ReID."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import aiohttp
import cv2

from src.detector import YOLOPersonDetector
from src.realtime.protocol import encode_jpeg, pack_frame
from src.runtime_config import (
    load_realtime_config,
    load_yolo_config as load_runtime_yolo_config,
)


class RealtimeWorker:
    """Run YOLO on one camera and stream detections/crops to the prime Jetson."""

    def __init__(self, config: dict[str, Any], yolo_config: dict[str, Any]):
        worker_config = config["worker"]
        network_config = config["network"]

        self.camera_id = str(worker_config["camera_id"])
        self.camera_source = worker_config.get("source", 0)
        self.target_fps = float(worker_config.get("target_fps", 10))
        self.capture_width = int(worker_config.get("capture_width", 640))
        self.capture_height = int(worker_config.get("capture_height", 480))
        self.capture_fps = float(worker_config.get("capture_fps", max(self.target_fps, 10)))
        self.frame_quality = int(worker_config.get("frame_jpeg_quality", 70))
        self.crop_quality = int(worker_config.get("crop_jpeg_quality", 85))
        self.reconnect_seconds = float(worker_config.get("reconnect_seconds", 3))
        self.print_every = int(worker_config.get("print_every_frames", 100))

        self.prime_url = network_config["prime_url"].rstrip("/")
        self.ingest_path = network_config.get("ingest_path", "/ws/ingest")
        self.ingest_url = f"{self.prime_url}{self.ingest_path}"

        self.detector = YOLOPersonDetector(yolo_config)
        self.frame_id = 0

    async def run_forever(self) -> None:
        """Connect to the prime server and stream frames, reconnecting on failure."""
        while True:
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"Worker connection failed: {exc}")
                await asyncio.sleep(self.reconnect_seconds)

    async def _run_once(self) -> None:
        cap = self.open_capture()
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open camera source: {self.camera_source}")

        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        next_send_at = None
        sent = 0
        started = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(self.ingest_url, max_msg_size=0) as ws:
                    print(f"Connected worker camera={self.camera_id} to {self.ingest_url}")

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            raise RuntimeError("Camera returned no frame")

                        captured_at = time.time()
                        monotonic_now = time.monotonic()
                        if (
                            frame_interval > 0
                            and next_send_at is not None
                            and monotonic_now < next_send_at
                        ):
                            await asyncio.sleep(next_send_at - monotonic_now)
                            monotonic_now = time.monotonic()
                        if frame_interval > 0:
                            next_send_at = self.advance_send_deadline(
                                next_send_at,
                                monotonic_now,
                                frame_interval,
                            )

                        detections, crops = self.detector.detect(frame)
                        frame_jpeg = encode_jpeg(frame, self.frame_quality)
                        crop_jpegs = [encode_jpeg(crop, self.crop_quality) for crop in crops]

                        packet = pack_frame(
                            camera_id=self.camera_id,
                            frame_id=self.frame_id,
                            detections=detections,
                            frame_jpeg=frame_jpeg,
                            crop_jpegs=crop_jpegs,
                            width=frame.shape[1],
                            height=frame.shape[0],
                            timestamp=captured_at,
                        )
                        await ws.send_bytes(packet)

                        sent += 1
                        if self.print_every > 0 and sent % self.print_every == 0:
                            fps = sent / max(time.time() - started, 1e-6)
                            print(
                                f"camera={self.camera_id} frame={self.frame_id} "
                                f"detections={len(detections)} sent_fps={fps:.2f}"
                            )

                        self.frame_id += 1
        finally:
            cap.release()

    @staticmethod
    def advance_send_deadline(
        previous_deadline: float | None,
        now: float,
        interval: float,
    ) -> float:
        """Advance a fixed-rate deadline without accumulating processing delay."""
        if interval <= 0:
            return now
        if previous_deadline is None:
            return now + interval
        overdue_intervals = max(
            1,
            int((now - previous_deadline) / interval) + 1,
        )
        return previous_deadline + overdue_intervals * interval

    def open_capture(self) -> cv2.VideoCapture:
        """Open a camera source with settings that work better for multi-USB capture."""
        if isinstance(self.camera_source, str) and self.camera_source.startswith("/dev/"):
            cap = cv2.VideoCapture(self.camera_source, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self.camera_source)

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            cap.set(cv2.CAP_PROP_FPS, self.capture_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return cap


def load_worker_config(
    config_path: Path,
    camera_id: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    config = load_realtime_config(config_path)

    if camera_id is not None:
        config["worker"]["camera_id"] = camera_id
    if source is not None:
        config["worker"]["source"] = int(source) if source.isdigit() else source
    return config


def load_yolo_config(config_path: Path) -> dict[str, Any]:
    if config_path.is_dir():
        config_path = config_path / "yolo_config.yaml"
    return load_runtime_yolo_config(config_path)
