"""
Pipeline Worker Service
Processes jobs from Redis queue and executes ReID pipeline
Minimal modifications to existing pipeline code
"""
import sys
import os
import json
import time
import logging
import redis
import psycopg2
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reid_pipeline.pipeline.production_pipeline import ProductionReIDPipeline
from reid_pipeline.multi_camera_pipeline import MultiCameraReIDPipeline
from reid_pipeline.utils.config import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "reid_pipeline")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def resolve_model_path(model_path: Optional[str]) -> Optional[str]:
    """
    Resolve model path to absolute path.
    If path is relative, prepend MODEL_DIR.
    If path is already absolute, return as-is.
    """
    if not model_path:
        return None

    path = Path(model_path)
    if path.is_absolute():
        return str(path)

    # Relative path - prepend MODEL_DIR
    return str(MODEL_DIR / model_path)

# Redis connection
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Database connection
def get_db_connection():
    """Get PostgreSQL database connection"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


class PipelineWorker:
    """Worker that processes pipeline jobs from Redis queue"""

    def __init__(self):
        self.redis_client = redis_client
        self.running = True
        logger.info("Pipeline worker initialized")

    def _serialize_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert stats dictionary to JSON-serializable format.
        Converts deque objects to lists.
        """
        from collections import deque

        def convert_value(v):
            if isinstance(v, deque):
                return list(v)
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            else:
                return v

        return {k: convert_value(v) for k, v in stats.items()}

    def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status in Redis and PostgreSQL"""
        try:
            # Update Redis
            update_data = {"status": status}
            update_data.update(kwargs)

            for key, value in update_data.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                self.redis_client.hset(f"job:{job_id}", key, str(value))

            # Update database
            conn = get_db_connection()
            cursor = conn.cursor()

            # Build dynamic UPDATE query
            set_clauses = ["status = %s"]
            values = [status]

            if "progress" in kwargs:
                set_clauses.append("progress = %s")
                values.append(kwargs["progress"])

            if "stats" in kwargs:
                set_clauses.append("stats = %s")
                values.append(json.dumps(kwargs["stats"]))

            if "output_video" in kwargs:
                set_clauses.append("output_video = %s")
                values.append(kwargs["output_video"])

            if "error_message" in kwargs:
                set_clauses.append("error_message = %s")
                values.append(kwargs["error_message"])

            if status == "running":
                set_clauses.append("started_at = %s")
                values.append(datetime.now())

            if status in ["completed", "failed", "cancelled"]:
                set_clauses.append("completed_at = %s")
                values.append(datetime.now())

            values.append(job_id)

            query = f"""
                UPDATE pipeline_jobs
                SET {', '.join(set_clauses)}
                WHERE job_id = %s
            """

            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Job {job_id} status updated to {status}")

        except Exception as e:
            logger.error(f"Error updating job status: {e}")

    def process_single_camera_job(self, job_data: Dict[str, Any]):
        """Process single-camera pipeline job"""
        job_id = job_data["job_id"]
        video_path = job_data["video_path"]
        config = job_data["config"]

        try:
            logger.info(f"Starting single-camera job: {job_id}")
            self.update_job_status(job_id, "running", progress=0.0)

            # Prepare output path
            output_filename = f"{job_id}_output.mp4"
            output_path = OUTPUT_DIR / output_filename

            # Create pipeline with configuration
            pipeline = ProductionReIDPipeline(
                yolo_model_path=resolve_model_path(config.get("yolo_model", "yolo11n.pt")),
                reid_model_path=resolve_model_path(config.get("reid_model")),
                device=config.get("device", "cuda"),
                detection_conf=config.get("detection_conf", 0.3),
                reid_threshold_match=config.get("reid_threshold_match", 0.70),
                reid_threshold_new=config.get("reid_threshold_new", 0.50),
                gallery_max_size=config.get("gallery_max_size", 500),
                reid_batch_size=config.get("reid_batch_size", 16),
                use_tensorrt=config.get("use_tensorrt", False),
                tensorrt_precision=config.get("tensorrt_precision", "fp16"),
                enable_display=False,  # Always disable display in worker
                logger=logger
            )

            # Wrap pipeline to report progress
            original_display_thread = pipeline._display_thread

            def wrapped_display_thread(output_path_arg):
                """Wrapped display thread that reports progress"""
                try:
                    frame_count = 0
                    last_update = time.time()

                    # Call original display thread logic
                    # We need to intercept frame processing
                    original_display_thread(output_path_arg)

                except Exception as e:
                    logger.error(f"Error in wrapped display thread: {e}")

            # Monkey patch to track progress
            def update_progress():
                """Update progress periodically"""
                if hasattr(pipeline, 'stats'):
                    frames_processed = pipeline.stats.get('frames_processed', 0)
                    frames_captured = pipeline.stats.get('frames_captured', 1)
                    progress = min(100.0, (frames_processed / frames_captured) * 100) if frames_captured > 0 else 0.0

                    # Convert stats to JSON-serializable format
                    serializable_stats = self._serialize_stats(pipeline.stats)

                    self.update_job_status(
                        job_id,
                        "running",
                        progress=progress,
                        stats=serializable_stats
                    )

            # Start a background thread to update progress
            import threading

            def progress_updater():
                while pipeline.running:
                    update_progress()
                    time.sleep(2)

            progress_thread = threading.Thread(target=progress_updater, daemon=True)
            progress_thread.start()

            # Run pipeline
            pipeline.run(
                video_source=video_path,
                output_path=str(output_path)
            )

            # Final progress update
            update_progress()

            # Get final statistics
            gallery_stats = pipeline.gallery_manager.get_statistics()
            final_stats = {
                'frames_captured': pipeline.stats['frames_captured'],
                'frames_processed': pipeline.stats['frames_processed'],
                'total_detections': pipeline.stats['total_detections'],
                'total_persons_tracked': gallery_stats['gallery_size']  # Actual unique persons in gallery
            }

            # Verify output file exists
            if not output_path.exists():
                logger.error(f"Output file does not exist: {output_path}")
                raise FileNotFoundError(f"Output video not created: {output_path}")

            output_size = output_path.stat().st_size
            logger.info(f"Output video created: {output_path} (size: {output_size} bytes)")

            self.update_job_status(
                job_id,
                "completed",
                progress=100.0,
                output_video=str(output_path),
                stats=final_stats
            )

            logger.info(f"Job {job_id} completed successfully with output: {output_path}")

        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
            self.update_job_status(
                job_id,
                "failed",
                error_message=str(e)
            )

    def process_multi_camera_job(self, job_data: Dict[str, Any]):
        """Process multi-camera pipeline job"""
        job_id = job_data["job_id"]
        video_paths = job_data["video_paths"]
        config = job_data["config"]

        try:
            logger.info(f"Starting multi-camera job: {job_id}")
            self.update_job_status(job_id, "running", progress=0.0)

            # Prepare output path
            output_filename = f"{job_id}_multi_camera_output.avi"
            output_path = OUTPUT_DIR / output_filename

            # Create multi-camera pipeline
            pipeline = MultiCameraReIDPipeline(
                yolo_model_path=resolve_model_path(config.get("yolo_model", "yolo11n.pt")),
                reid_model_path=resolve_model_path(config.get("reid_model")),
                device=config.get("device", "cuda"),
                detection_conf=config.get("detection_conf", 0.3),
                reid_threshold_match=config.get("reid_threshold_match", 0.50),
                reid_threshold_new=config.get("reid_threshold_new", 0.70),
                gallery_max_size=config.get("gallery_max_size", 1000),
                reid_batch_size=config.get("reid_batch_size", 16),
                use_tensorrt=config.get("use_tensorrt", False),
                tensorrt_precision=config.get("tensorrt_precision", "fp16"),
                display_scale=config.get("display_scale", 0.5),
                logger=logger
            )

            # Progress tracking
            import threading

            def progress_updater():
                while pipeline.running:
                    if hasattr(pipeline, 'stats'):
                        total_processed = sum(pipeline.stats['frames_processed'])
                        avg_progress = (total_processed / 4) if total_processed > 0 else 0.0

                        # Convert stats to JSON-serializable format
                        serializable_stats = self._serialize_stats(pipeline.stats)

                        self.update_job_status(
                            job_id,
                            "running",
                            progress=min(100.0, avg_progress / 10.0),  # Rough estimate
                            stats=serializable_stats
                        )

                    time.sleep(2)

            progress_thread = threading.Thread(target=progress_updater, daemon=True)
            progress_thread.start()

            # Run pipeline
            pipeline.run(
                video_paths=video_paths,
                output_path=str(output_path)
            )

            # Final statistics
            gallery_stats = pipeline.gallery_manager.get_statistics()
            final_stats = {
                'total_persons_tracked': gallery_stats['gallery_size'],  # Actual unique persons in gallery
                'total_detections': pipeline.stats['total_detections'],
                'frames_written': pipeline.stats.get('frames_written', 0),
                'frames_processed': pipeline.stats.get('frames_processed', 0),
                'frames_captured': pipeline.stats.get('frames_captured', 0)
            }

            # Verify output file exists
            if not output_path.exists():
                logger.error(f"Multi-camera output file does not exist: {output_path}")
                raise FileNotFoundError(f"Output video not created: {output_path}")

            output_size = output_path.stat().st_size
            logger.info(f"Multi-camera output video created: {output_path} (size: {output_size} bytes)")

            self.update_job_status(
                job_id,
                "completed",
                progress=100.0,
                output_video=str(output_path),
                stats=final_stats
            )

            logger.info(f"Multi-camera job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Error processing multi-camera job {job_id}: {e}", exc_info=True)
            self.update_job_status(
                job_id,
                "failed",
                error_message=str(e)
            )

    def process_job(self, job_data: Dict[str, Any]):
        """Process a job based on its type"""
        job_type = job_data.get("type", "single_camera")

        if job_type == "single_camera":
            self.process_single_camera_job(job_data)
        elif job_type == "multi_camera":
            self.process_multi_camera_job(job_data)
        else:
            logger.error(f"Unknown job type: {job_type}")

    def run(self):
        """Main worker loop"""
        logger.info("Worker started, waiting for jobs...")

        while self.running:
            try:
                # Blocking pop from Redis queue (timeout 1 second)
                result = self.redis_client.brpop("pipeline_jobs", timeout=1)

                if result:
                    queue_name, job_json = result
                    job_data = json.loads(job_json)
                    logger.info(f"Received job: {job_data['job_id']}")

                    # Check if job is cancelled
                    job_status = self.redis_client.hget(f"job:{job_data['job_id']}", "status")
                    if job_status == "cancelled":
                        logger.info(f"Job {job_data['job_id']} is cancelled, skipping")
                        continue

                    # Process the job
                    self.process_job(job_data)

            except KeyboardInterrupt:
                logger.info("Worker shutting down...")
                self.running = False
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(1)

        logger.info("Worker stopped")


if __name__ == "__main__":
    worker = PipelineWorker()
    worker.run()
