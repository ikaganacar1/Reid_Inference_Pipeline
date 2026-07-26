"""Prime Jetson realtime server for centralized ReID, tracking, viewing, and recording."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web
import cv2
import numpy as np

from src.realtime.identity_assignment import GlobalIdentityAssigner
from src.realtime.protocol import FramePacket, decode_jpeg, encode_jpeg, unpack_frame
from src.reid_client import create_reid_client
from src.runtime_config import (
    load_prime_pipeline_configs as load_runtime_pipeline_configs,
    load_realtime_config as load_runtime_realtime_config,
)
from src.tracker import ReIDTracker


class RealtimePrimeServer:
    """Receive Jetson worker frames, run ReID/tracking, save video, and stream viewers."""

    def __init__(self, realtime_config: dict[str, Any], pipeline_configs: dict[str, Any]):
        self.config = realtime_config
        self.pipeline_configs = pipeline_configs
        self.network = realtime_config["network"]
        self.prime = realtime_config["prime"]
        self.control = realtime_config.get("control", {})
        self.worker_nodes = self.control.get("worker_nodes", [])
        self.started_at = time.time()
        self.shutdown_event: asyncio.Event | None = None
        self.state_lock = asyncio.Lock()

        self.queue: asyncio.Queue[FramePacket] = asyncio.Queue(
            maxsize=int(self.prime.get("max_queue_size", 16))
        )
        self.viewers: set[web.WebSocketResponse] = set()
        self.camera_ingest_connections: dict[str, web.WebSocketResponse] = {}
        self.camera_ingest_peers: dict[str, str | None] = {}
        self.duplicate_camera_connections = 0
        self.camera_stats: dict[str, dict[str, Any]] = defaultdict(dict)
        self.camera_runtime: dict[str, dict[str, Any]] = defaultdict(dict)
        self.camera_input_state: dict[str, dict[str, Any]] = defaultdict(dict)
        self.dropped_packets_total = 0
        self.dropped_packets_by_camera: dict[str, int] = defaultdict(int)

        self.save_video = bool(self.prime.get("save_video", True))
        self.output_dir = Path(
            os.environ.get(
                "REALTIME_OUTPUT_DIR",
                self.prime.get("output_dir", "outputs/realtime"),
            )
        ).expanduser()
        recording_mountpoint = os.environ.get("RECORDING_MOUNTPOINT")
        self.recording_mountpoint: Path | None = None
        if self.save_video and recording_mountpoint:
            mountpoint = Path(recording_mountpoint).expanduser().resolve()
            resolved_output = self.output_dir.resolve()
            output_on_mount = resolved_output == mountpoint or mountpoint in resolved_output.parents
            if not mountpoint.is_mount() or not output_on_mount:
                raise RuntimeError(
                    "Recording mountpoint is unavailable or does not contain the output "
                    f"directory: mountpoint={mountpoint} output={resolved_output}"
                )
            self.recording_mountpoint = mountpoint
        self.reid_client = create_reid_client(pipeline_configs["reid"])
        self.trackers: dict[str, ReIDTracker] = {}
        self.output_fps = float(self.prime.get("output_fps", 15))
        self.recording_segment_seconds = max(
            0.0,
            float(self.prime.get("recording_segment_seconds", 900)),
        )
        self.recording_min_free_bytes = max(
            0,
            int(
                float(
                    os.environ.get(
                        "RECORDING_MIN_FREE_GB",
                        self.prime.get("recording_min_free_gb", 5),
                    )
                )
                * 1024**3
            ),
        )
        self.recording_disk_check_seconds = max(
            1.0,
            float(self.prime.get("recording_disk_check_seconds", 10)),
        )
        self.recording_last_disk_check = 0.0
        self.recording_free_bytes: int | None = None
        self.recording_paused_reason: str | None = None
        self.viewer_quality = int(self.prime.get("viewer_jpeg_quality", 75))
        self.viewer_send_timeout_seconds = max(
            0.01,
            float(self.prime.get("viewer_send_timeout_seconds", 0.2)),
        )
        self.camera_offline_seconds = max(
            1.0,
            float(self.prime.get("camera_offline_seconds", 5.0)),
        )
        self.reid_batch_size = int(self.prime.get("reid_batch_size", 16))
        self.sync_batch_window_seconds = max(
            0.0,
            float(self.prime.get("sync_batch_window_ms", 25)) / 1000.0,
        )
        self.sync_batch_max_packets = max(1, int(self.prime.get("sync_batch_max_packets", 4)))
        self.max_capture_clock_skew_seconds = max(
            0.0,
            float(self.prime.get("max_capture_clock_skew_seconds", 5.0)),
        )
        self.camera_tracker_reset_seconds = max(
            0.0,
            float(self.prime.get("camera_tracker_reset_seconds", 3.0)),
        )
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.writer_started_at: dict[str, float] = {}
        self.writer_segment_index: dict[str, int] = defaultdict(int)
        self.recording_session = time.strftime("%Y%m%d_%H%M%S")
        self.recording_dir = self.output_dir / "recordings" / self.recording_session
        self.identity_assigner = GlobalIdentityAssigner(self.prime, self.output_dir)
        self.gallery = self.identity_assigner.gallery
        self.pending_identity_tracks = self.identity_assigner.pending_identity_tracks
        self.debug_reid = bool(self.prime.get("debug_reid", True))
        self.save_debug_crops = bool(self.prime.get("save_debug_crops", True))
        self.debug_crop_interval_frames = max(1, int(self.prime.get("debug_crop_interval_frames", 10)))
        self.debug_dir = self.output_dir / "reid_debug"
        self.debug_events_path = self.debug_dir / "events.jsonl"
        self.filter_edge_false_positives = bool(self.prime.get("filter_edge_false_positives", True))
        self.edge_filter_margin_ratio = float(self.prime.get("edge_filter_margin_ratio", 0.025))
        self.edge_filter_max_width_ratio = float(self.prime.get("edge_filter_max_width_ratio", 0.22))
        self.edge_filter_min_height_ratio = float(self.prime.get("edge_filter_min_height_ratio", 0.45))
        self.new_identity_min_frames = max(1, int(self.prime.get("new_identity_min_frames", 5)))
        self.pending_identity_ttl_seconds = float(self.prime.get("pending_identity_ttl_seconds", 2.0))
        self.recheck_existing_tracks = bool(self.prime.get("recheck_existing_tracks", True))
        self.existing_track_recheck_threshold = float(self.prime.get("existing_track_recheck_threshold", 0.55))
        self.existing_track_remap_margin = float(self.prime.get("existing_track_remap_margin", 0.08))
        self.existing_track_max_distance = float(self.prime.get("existing_track_max_distance", 0.5))
        self.new_track_match_threshold = float(self.prime.get("new_track_match_threshold", 0.45))
        self.new_track_match_margin = float(self.prime.get("new_track_match_margin", 0.08))

        if self.save_video:
            self.recording_dir.mkdir(parents=True, exist_ok=True)
        if self.debug_reid:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> None:
        """Start the HTTP/WebSocket server and processing task."""
        self.shutdown_event = asyncio.Event()
        app = web.Application(client_max_size=int(self.prime.get("client_max_size_mb", 32)) * 1024 * 1024)
        app.add_routes(
            [
                web.get("/", self.handle_index),
                web.get("/status", self.handle_status),
                web.get("/api/workers", self.handle_workers_status),
                web.post("/api/control", self.handle_control),
                web.get(self.network.get("ingest_path", "/ws/ingest"), self.handle_ingest),
                web.get(self.network.get("viewer_path", "/ws/view"), self.handle_view),
            ]
        )

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            host=self.network.get("prime_bind_host", "0.0.0.0"),
            port=int(self.network.get("prime_port", 8765)),
        )
        await site.start()

        print(
            "Realtime prime server listening on "
            f"{self.network.get('prime_bind_host', '0.0.0.0')}:{self.network.get('prime_port', 8765)}"
        )
        print(f"Viewer URL: http://<prime-ip>:{self.network.get('prime_port', 8765)}/")

        worker_task = asyncio.create_task(self.process_loop())
        try:
            await self.shutdown_event.wait()
        finally:
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task
            for writer in self.writers.values():
                writer.release()
            self.reid_client.close()
            await runner.cleanup()

    async def handle_ingest(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(max_msg_size=0, heartbeat=10)
        await ws.prepare(request)
        peer = request.remote
        print(f"Worker connected: {peer}")

        owned_camera_id = None
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    try:
                        packet = unpack_frame(msg.data, received_at=time.time())
                        if owned_camera_id is None:
                            owner = self.camera_ingest_connections.get(packet.camera_id)
                            if owner is not None and owner is not ws and not owner.closed:
                                self.duplicate_camera_connections += 1
                                await ws.close(
                                    code=4009,
                                    message=f"camera_id already connected: {packet.camera_id}".encode(),
                                )
                                break
                            owned_camera_id = packet.camera_id
                            self.camera_ingest_connections[packet.camera_id] = ws
                            self.camera_ingest_peers[packet.camera_id] = peer
                        elif packet.camera_id != owned_camera_id:
                            await ws.close(code=4008, message=b"one camera_id per connection")
                            break

                        if self.queue.full():
                            dropped = self.queue.get_nowait()
                            self.queue.task_done()
                            self.dropped_packets_total += 1
                            self.dropped_packets_by_camera[dropped.camera_id] += 1
                        await self.queue.put(packet)
                    except Exception as exc:
                        await ws.send_str(json.dumps({"type": "error", "error": str(exc)}))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"Worker websocket error: {ws.exception()}")
        finally:
            if (
                owned_camera_id is not None
                and self.camera_ingest_connections.get(owned_camera_id) is ws
            ):
                self.camera_ingest_connections.pop(owned_camera_id, None)
                self.camera_ingest_peers.pop(owned_camera_id, None)

        print(f"Worker disconnected: {peer}")
        return ws

    async def handle_view(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        self.viewers.add(ws)
        print(f"Viewer connected: {request.remote}")
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"Viewer websocket error: {ws.exception()}")
        finally:
            self.viewers.discard(ws)
            print(f"Viewer disconnected: {request.remote}")
        return ws

    async def handle_status(self, request: web.Request) -> web.Response:
        async with self.state_lock:
            payload = {
                "queue_size": self.queue.qsize(),
                "queue_max": int(self.prime.get("max_queue_size", 16)),
                "dropped_packets": self.dropped_packets_total,
                "dropped_packets_by_camera": dict(self.dropped_packets_by_camera),
                "duplicate_camera_connections": self.duplicate_camera_connections,
                "ingest_connections": dict(self.camera_ingest_peers),
                "viewers": len(self.viewers),
                "cameras": self.camera_stats,
                "gallery": self.gallery.snapshot(),
                "uptime_seconds": time.time() - self.started_at,
                "recording_dir": str(self.recording_dir) if self.save_video else None,
                "recording": {
                    "enabled": self.save_video,
                    "paused": self.recording_paused_reason is not None,
                    "reason": self.recording_paused_reason,
                    "mountpoint": (
                        str(self.recording_mountpoint)
                        if self.recording_mountpoint is not None
                        else None
                    ),
                    "free_bytes": self.recording_free_bytes,
                    "minimum_free_bytes": self.recording_min_free_bytes,
                },
            }
        return web.json_response(payload)

    async def handle_workers_status(self, request: web.Request) -> web.Response:
        return web.json_response({"nodes": await self.worker_statuses()})

    async def handle_control(self, request: web.Request) -> web.Response:
        payload = await request.json()
        action = payload.get("action")

        if action in {"worker_start", "worker_stop", "worker_restart", "worker_restart_scan"}:
            default_node = self.worker_nodes[0]["name"] if self.worker_nodes else "worker-jetson"
            node_name = payload.get("node", default_node)
            worker_action = action.replace("worker_", "")
            if worker_action == "restart_scan":
                worker_action = "restart_scan"
            result = await self.proxy_worker_control(node_name, worker_action)
            return web.json_response(result)

        if action == "reset_gallery":
            async with self.state_lock:
                self.identity_assigner.reset()
            return web.json_response({"ok": True, "message": "gallery reset"})

        if action == "prime_stop":
            if self.shutdown_event is not None:
                self.shutdown_event.set()
            return web.json_response({"ok": True, "message": "prime stopping"})

        return web.json_response({"ok": False, "error": f"Unsupported action: {action}"}, status=400)

    async def handle_index(self, request: web.Request) -> web.Response:
        viewer_path = self.network.get("viewer_path", "/ws/view")
        return web.Response(
            text=(
                VIEWER_HTML.replace("__VIEWER_PATH__", viewer_path)
                .replace("__CAMERA_OFFLINE_SECONDS__", str(self.camera_offline_seconds))
            ),
            content_type="text/html",
        )

    async def worker_statuses(self) -> list[dict[str, Any]]:
        results = []
        for node in self.worker_nodes:
            url = node["url"].rstrip("/")
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as session:
                    async with session.get(f"{url}/status") as response:
                        data = await response.json()
                results.append({"name": node["name"], "url": url, "ok": True, **data})
            except Exception as exc:
                results.append({"name": node.get("name", url), "url": url, "ok": False, "error": str(exc)})
        return results

    async def proxy_worker_control(self, node_name: str, action: str) -> dict[str, Any]:
        node = next((item for item in self.worker_nodes if item.get("name") == node_name), None)
        if node is None:
            return {"ok": False, "error": f"Unknown worker node: {node_name}"}

        url = node["url"].rstrip("/")
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.post(f"{url}/control", json={"action": action}) as response:
                    data = await response.json()
            return {"ok": response.status < 400, "node": node_name, "response": data}
        except Exception as exc:
            return {"ok": False, "node": node_name, "error": str(exc)}

    async def process_loop(self) -> None:
        while True:
            packets = await self.collect_packet_batch()
            try:
                async with self.state_lock:
                    results = await asyncio.to_thread(self.process_packets, packets)
                for result in results:
                    await self.broadcast(result)
            except Exception as exc:
                packet_labels = ",".join(f"{p.camera_id}:{p.frame_id}" for p in packets)
                print(f"Failed to process packet batch [{packet_labels}]: {exc}")
            finally:
                for _ in packets:
                    self.queue.task_done()

    async def collect_packet_batch(self) -> list[FramePacket]:
        """Collect a short arrival-time microbatch for efficient GPU ReID."""
        packets = [await self.queue.get()]
        if self.sync_batch_window_seconds <= 0 or self.sync_batch_max_packets <= 1:
            return packets

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.sync_batch_window_seconds
        while len(packets) < self.sync_batch_max_packets:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                packets.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return packets

    def process_packet(self, packet: FramePacket) -> dict[str, Any]:
        """Compatibility wrapper for callers that process one packet directly."""
        return self.process_packets([packet])[0]

    def process_packets(self, packets: list[FramePacket]) -> list[dict[str, Any]]:
        """Decode a packet microbatch, run one batched ReID call, then track per camera."""
        if not packets:
            return []

        total_start = time.perf_counter()
        prepared = []
        all_crops: list[np.ndarray] = []
        for packet in packets:
            frame = decode_jpeg(packet.frame_jpeg)
            if frame.shape[1] != packet.width or frame.shape[0] != packet.height:
                raise ValueError(
                    f"Decoded frame size mismatch from camera={packet.camera_id}: "
                    f"decoded={frame.shape[1]}x{frame.shape[0]} header={packet.width}x{packet.height}"
                )
            crops = [decode_jpeg(crop_jpeg) for crop_jpeg in packet.crop_jpegs]
            if len(crops) != len(packet.detections):
                raise ValueError(
                    f"Crop/detection mismatch from camera={packet.camera_id}: "
                    f"{len(crops)} crops, {len(packet.detections)} detections"
                )
            detections, crops, rejected = self.filter_detections(packet, crops)
            start = len(all_crops)
            all_crops.extend(crops)
            event_timestamp = self.effective_packet_timestamp(packet)
            prepared.append(
                (packet, frame, detections, crops, rejected, start, len(all_crops), event_timestamp)
            )

        reid_start = time.perf_counter()
        all_embeddings = self.reid_client.infer(all_crops, max_batch_size=self.reid_batch_size)
        reid_ms = (time.perf_counter() - reid_start) * 1000
        if len(all_embeddings) != len(all_crops):
            raise ValueError(
                f"ReID output/crop mismatch: {len(all_embeddings)} embeddings for {len(all_crops)} crops"
            )

        prepared.sort(key=lambda item: item[-1])
        results = []
        for packet, frame, detections, crops, rejected, start, end, event_timestamp in prepared:
            packet_start = time.perf_counter()
            tracker, reset_reason = self.prepare_camera_tracker(packet, event_timestamp)
            embeddings = all_embeddings[start:end]

            track_start = time.perf_counter()
            tracks = tracker.update(detections, frame, embeddings)
            track_ms = (time.perf_counter() - track_start) * 1000
            track_records = self.identity_assigner.assign_tracks(
                packet.camera_id,
                packet.frame_id,
                packet.width,
                packet.height,
                tracks,
                embeddings,
                crops,
                timestamp=event_timestamp,
            )
            annotated = frame.copy()
            self.draw_tracks(annotated, track_records)
            self.write_video(packet.camera_id, annotated)

            viewer_jpeg = encode_jpeg(annotated, self.viewer_quality)
            now = time.time()
            process_ms = (time.perf_counter() - packet_start) * 1000 + reid_ms
            receive_latency_ms = max(0.0, (now - event_timestamp) * 1000)
            self.camera_stats[packet.camera_id] = {
                "frame_id": packet.frame_id,
                "last_seen": now,
                "capture_timestamp": event_timestamp,
                "receive_latency_ms": round(receive_latency_ms, 1),
                "detections": len(detections),
                "raw_detections": len(packet.detections),
                "rejected_detections": rejected,
                "tracks": len(track_records),
                "tracker_reset_reason": reset_reason,
                "reid_batch_crops": len(all_crops),
                **self.update_camera_metrics(
                    packet.camera_id,
                    packet.frame_id,
                    now,
                    reid_ms,
                    track_ms,
                    process_ms,
                ),
            }

            results.append(
                {
                    "type": "frame",
                    "camera_id": packet.camera_id,
                    "frame_id": packet.frame_id,
                    "timestamp": event_timestamp,
                    "detections": len(detections),
                    "raw_detections": len(packet.detections),
                    "rejected_detections": rejected,
                    "fps": self.camera_stats[packet.camera_id].get("fps", 0.0),
                    "process_ms": self.camera_stats[packet.camera_id].get("process_ms", 0.0),
                    "tracks": track_records,
                    "jpeg_b64": base64.b64encode(viewer_jpeg).decode("ascii"),
                }
            )

        self.camera_runtime["_prime"]["last_batch_ms"] = (
            time.perf_counter() - total_start
        ) * 1000
        return results

    def effective_packet_timestamp(self, packet: FramePacket) -> float:
        """Use capture time when worker clocks are sane, otherwise use receive time."""
        received_at = float(packet.received_at if packet.received_at is not None else time.time())
        if abs(float(packet.timestamp) - received_at) > self.max_capture_clock_skew_seconds:
            return received_at
        return float(packet.timestamp)

    def prepare_camera_tracker(
        self,
        packet: FramePacket,
        event_timestamp: float,
    ) -> tuple[ReIDTracker, str | None]:
        """Reset camera-local state on restart, rollback, or a long capture gap."""
        state = self.camera_input_state[packet.camera_id]
        previous_frame_id = state.get("last_frame_id")
        previous_timestamp = state.get("last_timestamp")
        reset_reason = None
        if previous_frame_id is not None and packet.frame_id <= int(previous_frame_id):
            reset_reason = "frame_id_rollback"
        elif (
            previous_timestamp is not None
            and self.camera_tracker_reset_seconds > 0
            and event_timestamp - float(previous_timestamp) > self.camera_tracker_reset_seconds
        ):
            reset_reason = "capture_gap"

        tracker = self.trackers.get(packet.camera_id)
        if tracker is None:
            tracker = ReIDTracker(self.pipeline_configs["tracker"])
            self.trackers[packet.camera_id] = tracker
        elif reset_reason is not None:
            tracker.reset()
            self.identity_assigner.reset_camera(packet.camera_id)
            self.camera_runtime.pop(packet.camera_id, None)
            state["reset_count"] = int(state.get("reset_count", 0)) + 1

        state["last_frame_id"] = int(packet.frame_id)
        state["last_timestamp"] = max(
            event_timestamp,
            float(previous_timestamp) if previous_timestamp is not None else event_timestamp,
        )
        return tracker, reset_reason

    def filter_detections(
        self,
        packet: FramePacket,
        crops: list[np.ndarray],
    ) -> tuple[np.ndarray, list[np.ndarray], int]:
        """Reject detector boxes that are likely static edge false positives."""
        detections = packet.detections
        if len(detections) == 0 or not self.filter_edge_false_positives:
            return detections, crops, 0

        frame_width = float(packet.width)
        frame_height = float(packet.height)
        edge_margin = frame_width * self.edge_filter_margin_ratio
        max_edge_width = frame_width * self.edge_filter_max_width_ratio
        min_edge_height = frame_height * self.edge_filter_min_height_ratio

        keep_indices = []
        for index, det in enumerate(detections):
            x1, y1, x2, y2 = [float(value) for value in det[:4]]
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            near_side_edge = x1 <= edge_margin or x2 >= frame_width - edge_margin
            narrow = box_width <= max_edge_width
            tall_enough = box_height >= min_edge_height

            if near_side_edge and narrow and tall_enough:
                continue
            keep_indices.append(index)

        if len(keep_indices) == len(detections):
            return detections, crops, 0

        filtered_detections = detections[keep_indices] if keep_indices else np.empty((0, 6), dtype=np.float32)
        filtered_crops = [crops[index] for index in keep_indices]
        return filtered_detections, filtered_crops, len(detections) - len(keep_indices)

    def update_camera_metrics(
        self,
        camera_id: str,
        frame_id: int,
        now: float,
        reid_ms: float,
        track_ms: float,
        process_ms: float,
    ) -> dict[str, float]:
        runtime = self.camera_runtime[camera_id]
        previous_seen = runtime.get("last_seen")
        previous_frame = runtime.get("last_frame_id")
        fps = runtime.get("fps", 0.0)
        if previous_seen is not None and previous_frame is not None:
            elapsed = max(now - previous_seen, 1e-6)
            frame_delta = max(frame_id - previous_frame, 1)
            instant_fps = frame_delta / elapsed
            fps = instant_fps if fps <= 0 else (0.85 * fps + 0.15 * instant_fps)

        runtime["last_seen"] = now
        runtime["last_frame_id"] = frame_id
        runtime["fps"] = fps
        runtime["reid_ms"] = reid_ms
        runtime["track_ms"] = track_ms
        runtime["process_ms"] = process_ms
        return {
            "fps": round(fps, 2),
            "reid_ms": round(reid_ms, 1),
            "track_ms": round(track_ms, 1),
            "process_ms": round(process_ms, 1),
        }

    def assign_global_ids(
        self,
        packet: FramePacket,
        tracks: np.ndarray,
        embeddings: np.ndarray,
        crops: list[np.ndarray] | None = None,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        return self.identity_assigner.assign_tracks(
            packet.camera_id,
            packet.frame_id,
            packet.width,
            packet.height,
            tracks,
            embeddings,
            crops,
            timestamp=now,
        )

    def draw_tracks(self, frame: np.ndarray, tracks: list[dict[str, Any]]) -> None:
        for record in tracks:
            x1, y1, x2, y2 = [int(v) for v in record["bbox"]]
            global_id = record["global_id"]
            local_id = record["local_track_id"]
            color = self.color_for_id(global_id or local_id)
            label = f"ID:{global_id if global_id is not None else '-'}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def write_video(self, camera_id: str, frame: np.ndarray) -> None:
        if not self.save_video:
            return

        now = time.monotonic()
        if not self.recording_space_available(now):
            return
        writer = self.writers.get(camera_id)
        started_at = self.writer_started_at.get(camera_id, now)
        if (
            writer is not None
            and self.recording_segment_seconds > 0
            and now - started_at >= self.recording_segment_seconds
        ):
            writer.release()
            self.writers.pop(camera_id, None)
            self.writer_started_at.pop(camera_id, None)
            writer = None

        if writer is None:
            h, w = frame.shape[:2]
            self.writer_segment_index[camera_id] += 1
            segment_index = self.writer_segment_index[camera_id]
            out_path = self.recording_dir / f"{camera_id}_{segment_index:06d}_processed.mp4"
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.output_fps,
                (w, h),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open output video writer: {out_path}")
            self.writers[camera_id] = writer
            self.writer_started_at[camera_id] = now
            print(f"Recording camera={camera_id} to {out_path}")
        writer.write(frame)

    def recording_space_available(self, now: float) -> bool:
        """Pause all writers when the configured free-space reserve is reached."""
        mountpoint = getattr(self, "recording_mountpoint", None)
        if mountpoint is not None and not mountpoint.is_mount():
            return self.pause_recording("recording_mountpoint_unavailable")
        if getattr(self, "recording_min_free_bytes", 0) <= 0:
            if getattr(self, "recording_paused_reason", None) is not None:
                print("Recording resumed: required mountpoint restored")
                self.recording_paused_reason = None
            return True
        if (
            self.recording_free_bytes is not None
            and now - self.recording_last_disk_check < self.recording_disk_check_seconds
        ):
            return self.recording_paused_reason is None

        self.recording_last_disk_check = now
        try:
            self.recording_free_bytes = shutil.disk_usage(self.recording_dir).free
            reason = (
                "low_disk_space"
                if self.recording_free_bytes < self.recording_min_free_bytes
                else None
            )
        except OSError as exc:
            reason = f"disk_check_failed:{exc}"

        if reason is not None:
            return self.pause_recording(reason)

        if self.recording_paused_reason is not None:
            print("Recording resumed: free-space reserve restored")
        self.recording_paused_reason = None
        return True

    def pause_recording(self, reason: str) -> bool:
        if self.recording_paused_reason != reason:
            print(f"Recording paused: {reason}")
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
        self.writer_started_at.clear()
        self.recording_paused_reason = reason
        return False

    async def broadcast(self, payload: dict[str, Any]) -> None:
        if not self.viewers:
            return
        message = json.dumps(payload, separators=(",", ":"))
        sockets = list(self.viewers)
        sends = [
            asyncio.wait_for(ws.send_str(message), timeout=self.viewer_send_timeout_seconds)
            for ws in sockets
        ]
        outcomes = await asyncio.gather(*sends, return_exceptions=True)
        dead = [ws for ws, outcome in zip(sockets, outcomes) if isinstance(outcome, Exception)]
        for ws in dead:
            self.viewers.discard(ws)

    @staticmethod
    def color_for_id(identity: int) -> tuple[int, int, int]:
        rng = np.random.default_rng(int(identity) * 9973)
        color = rng.integers(60, 235, size=3)
        return int(color[0]), int(color[1]), int(color[2])


def load_realtime_config(path: Path) -> dict[str, Any]:
    return load_runtime_realtime_config(path)


def load_prime_pipeline_configs(config_dir: Path) -> dict[str, Any]:
    return load_runtime_pipeline_configs(config_dir)


VIEWER_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Realtime ReID Dashboard</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: Arial, sans-serif; background: #0f172a; color: #f9fafb; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 12px 18px; background: #020617; border-bottom: 1px solid #334155; }
    button { cursor: pointer; border: 1px solid #475569; background: #1e293b; color: #f8fafc; border-radius: 6px; padding: 7px 10px; }
    button:hover { background: #334155; }
    button.danger { border-color: #991b1b; background: #7f1d1d; }
    button.good { border-color: #166534; background: #14532d; }
    select { background: #111827; color: #f8fafc; border: 1px solid #475569; border-radius: 6px; padding: 6px; }
    #layout { display: grid; grid-template-columns: minmax(0, 1fr) 380px; height: calc(100vh - 53px); overflow: hidden; }
    #grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); grid-auto-rows: minmax(260px, 1fr); gap: 10px; padding: 10px; overflow: auto; }
    #grid[data-count="1"] { grid-template-columns: 1fr; grid-template-rows: 1fr; }
    #grid[data-count="2"] { grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-rows: 1fr; }
    #grid[data-count="3"], #grid[data-count="4"] { grid-template-columns: repeat(2, minmax(0, 1fr)); grid-template-rows: repeat(2, minmax(0, 1fr)); }
    .card, aside section { background: #1f2937; border: 1px solid #374151; border-radius: 8px; overflow: hidden; }
    .card { display: flex; flex-direction: column; min-height: 0; }
    .meta { padding: 8px 10px; font-size: 14px; display: flex; justify-content: space-between; gap: 8px; }
    img { display: block; width: 100%; min-height: 0; flex: 1; object-fit: contain; background: #020617; }
    aside { border-left: 1px solid #334155; padding: 12px; display: grid; gap: 12px; align-content: start; background: #111827; overflow: auto; }
    aside section { padding: 12px; }
    h2 { font-size: 15px; margin: 0 0 10px; color: #cbd5e1; }
    .metric { display: flex; justify-content: space-between; margin: 6px 0; font-size: 14px; }
    .small { color: #94a3b8; font-size: 12px; }
    .row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 8px 0 0; color: #cbd5e1; font-size: 12px; max-height: 160px; overflow: auto; }
    .ok { color: #86efac; }
    .bad { color: #fca5a5; }
    @media (max-width: 1100px) { #layout { grid-template-columns: 1fr; height: auto; overflow: visible; } #grid { overflow: visible; } aside { border-left: 0; border-top: 1px solid #334155; } }
  </style>
</head>
<body>
  <header>
    <div><strong>Realtime ReID Dashboard</strong> <span id="wsStatus" class="small">connecting...</span></div>
    <div class="small">Viewer: <span id="clock"></span></div>
  </header>
  <div id="layout">
    <main id="grid"></main>
    <aside>
      <section>
        <h2>Prime Metrics</h2>
        <div class="metric"><span>Queue</span><b id="queueMetric">-</b></div>
        <div class="metric"><span>Viewers</span><b id="viewerMetric">-</b></div>
        <div class="metric"><span>Cameras</span><b id="cameraMetric">-</b></div>
        <div class="metric"><span>Gallery IDs</span><b id="galleryMetric">-</b></div>
        <div class="metric"><span>Recording</span><b id="recordingMetric">-</b></div>
        <div class="row">
          <button onclick="control('reset_gallery')">Reset Gallery</button>
          <button class="danger" onclick="stopPrime()">Stop Prime</button>
        </div>
      </section>
      <section>
        <h2>Worker Controls</h2>
        <label class="small">Target node <select id="workerNodeSelect"></select></label>
        <div id="workers" class="small">loading...</div>
        <div class="row">
          <button class="good" onclick="workerControl('worker_start')">Start Workers</button>
          <button onclick="workerControl('worker_restart_scan')">Scan + Restart</button>
          <button class="danger" onclick="workerControl('worker_stop')">Stop Workers</button>
        </div>
        <pre id="controlResult"></pre>
      </section>
      <section>
        <h2>Camera Metrics</h2>
        <div id="cameraStats" class="small">waiting...</div>
      </section>
    </aside>
  </div>
  <script>
    const grid = document.getElementById("grid");
    const wsStatus = document.getElementById("wsStatus");
    const cards = new Map();
    const lastFrames = new Map();
    const workerNodeSelect = document.getElementById("workerNodeSelect");
    const cameraOfflineSeconds = Number("__CAMERA_OFFLINE_SECONDS__");

    function cardFor(cameraId) {
      if (cards.has(cameraId)) return cards.get(cameraId);
      const card = document.createElement("section");
      card.className = "card";
      card.innerHTML = `<div class="meta"><b>${cameraId}</b><span></span></div><img>`;
      grid.appendChild(card);
      cards.set(cameraId, card);
      grid.dataset.count = cards.size;
      return card;
    }

    function removeOfflineCards(activeCameraIds) {
      for (const [cameraId, card] of cards.entries()) {
        if (!activeCameraIds.has(cameraId)) {
          card.remove();
          cards.delete(cameraId);
          lastFrames.delete(cameraId);
        }
      }
      grid.dataset.count = cards.size;
    }

    function connect() {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}__VIEWER_PATH__`);
      ws.onopen = () => wsStatus.textContent = "connected";
      ws.onclose = () => {
        wsStatus.textContent = "disconnected, retrying...";
        setTimeout(connect, 1500);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type !== "frame") return;
        const card = cardFor(data.camera_id);
        card.querySelector("img").src = `data:image/jpeg;base64,${data.jpeg_b64}`;
        card.querySelector("span").textContent =
          `f ${data.frame_id} | ${Number(data.fps || 0).toFixed(1)} FPS | ${Number(data.process_ms || 0).toFixed(1)} ms | det ${data.detections} | tr ${data.tracks.length}`;
        lastFrames.set(data.camera_id, { frame_id: data.frame_id, t: Date.now() });
      };
    }

    async function refreshStatus() {
      document.getElementById("clock").textContent = new Date().toLocaleTimeString();
      try {
        const s = await fetch("/status").then(r => r.json());
        document.getElementById("queueMetric").textContent = `${s.queue_size}/${s.queue_max || "?"}`;
        document.getElementById("viewerMetric").textContent = s.viewers;
        const nowSeconds = Date.now() / 1000;
        const activeCameraIds = new Set(
          Object.entries(s.cameras || {})
            .filter(([, camera]) => nowSeconds - camera.last_seen <= cameraOfflineSeconds)
            .map(([cameraId]) => cameraId)
        );
        removeOfflineCards(activeCameraIds);
        document.getElementById("cameraMetric").textContent = activeCameraIds.size;
        document.getElementById("galleryMetric").textContent = (s.gallery || []).length;
        const recording = s.recording || {};
        const freeGiB = recording.free_bytes == null
          ? "?"
          : (recording.free_bytes / (1024 ** 3)).toFixed(1);
        document.getElementById("recordingMetric").textContent = !recording.enabled
          ? "disabled"
          : recording.paused
            ? `paused (${recording.reason})`
            : `${freeGiB} GiB free`;
        document.getElementById("cameraStats").innerHTML = Object.entries(s.cameras || {}).map(([id, c]) => {
          const ageSeconds = nowSeconds - c.last_seen;
          const age = ageSeconds.toFixed(1);
          const stateClass = ageSeconds <= cameraOfflineSeconds ? "ok" : "bad";
          return `<div class="metric"><span class="${stateClass}">${id}</span><b>${Number(c.fps || 0).toFixed(1)} FPS | ${Number(c.process_ms || 0).toFixed(1)} ms</b></div>` +
                 `<div class="small">frame ${c.frame_id} | det ${c.detections} | tracks ${c.tracks} | reid ${Number(c.reid_ms || 0).toFixed(1)} ms | track ${Number(c.track_ms || 0).toFixed(1)} ms | age ${age}s</div>`;
        }).join("") || "no cameras";
      } catch (err) {
        document.getElementById("queueMetric").textContent = "error";
      }
      try {
        const w = await fetch("/api/workers").then(r => r.json());
        const nodes = w.nodes || [];
        syncWorkerNodeSelect(nodes);
        document.getElementById("workers").innerHTML = nodes.map(n => {
          const cls = n.ok && n.running ? "ok" : "bad";
          const state = n.ok ? `${n.worker_count || 0} processes` : n.error;
          const discovered = (n.discovered_sources || []).join(", ");
          const metrics = Object.entries(n.metrics || {}).map(([id, m]) =>
            `<div class="metric"><span>${id}</span><b>${Number(m.sent_fps || 0).toFixed(1)} sent FPS | det ${m.detections}</b></div>`
          ).join("");
          return `<div><b>${n.name}</b>: <span class="${cls}">${state}</span><br>${metrics}<div class="small">active: ${(n.sources || []).join(", ")}</div><div class="small">scan: ${discovered}</div></div>`;
        }).join("") || "no worker nodes configured";
      } catch (err) {
        document.getElementById("workers").innerHTML = `<span class="bad">${err}</span>`;
      }
    }

    function syncWorkerNodeSelect(nodes) {
      const previous = workerNodeSelect.value;
      workerNodeSelect.innerHTML = "";
      for (const n of nodes) {
        const option = document.createElement("option");
        option.value = n.name;
        option.textContent = n.name;
        workerNodeSelect.appendChild(option);
      }
      if (previous && nodes.some(n => n.name === previous)) {
        workerNodeSelect.value = previous;
      }
    }

    async function control(action, body = {}) {
      const result = await fetch("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, ...body })
      }).then(r => r.json());
      document.getElementById("controlResult").textContent = JSON.stringify(result, null, 2);
      setTimeout(refreshStatus, 800);
    }

    function workerControl(action) {
      const node = workerNodeSelect.value;
      if (!node) {
        document.getElementById("controlResult").textContent = "No worker node configured.";
        return;
      }
      control(action, { node });
    }

    function stopPrime() {
      if (confirm("Stop the prime server? The dashboard will disconnect.")) {
        control("prime_stop");
      }
    }

    connect();
    refreshStatus();
    setInterval(refreshStatus, 2000);
  </script>
</body>
</html>
"""
