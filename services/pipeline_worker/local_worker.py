"""
Local Pipeline Worker - No Docker Dependencies
Processes jobs from SQLite and runs pipelines locally
"""
import sys
import json
import time
import os
import sqlite3
from pathlib import Path
from datetime import datetime
import logging
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DB_PATH = PROJECT_ROOT / "local_api.db"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Default models
DEFAULT_YOLO = str(MODELS_DIR / "yolo11n.pt")
DEFAULT_REID = str(MODELS_DIR / "lttc_0.1.4.49.onnx")

def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def update_job_status(job_id: str, status: str, progress: float = None,
                      output_video: str = None, stats: dict = None, error: str = None):
    """Update job status in database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    updates = ["status = ?"]
    params = [status]

    if progress is not None:
        updates.append("progress = ?")
        params.append(progress)

    if output_video:
        updates.append("output_video = ?")
        params.append(output_video)

    if stats:
        updates.append("stats = ?")
        params.append(json.dumps(stats))

    if error:
        updates.append("error_message = ?")
        params.append(error)

    if status == "running":
        updates.append("started_at = ?")
        params.append(datetime.now().isoformat())
    elif status in ["completed", "failed"]:
        updates.append("completed_at = ?")
        params.append(datetime.now().isoformat())

    params.append(job_id)
    query = f"UPDATE pipeline_jobs SET {', '.join(updates)} WHERE job_id = ?"
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def update_eval_job_status(job_id: str, status: str, results: dict = None, error: str = None):
    """Update evaluation job status"""
    conn = get_db_connection()
    cursor = conn.cursor()

    updates = ["status = ?"]
    params = [status]

    if results:
        updates.append("results = ?")
        params.append(json.dumps(results))

    if error:
        updates.append("error_message = ?")
        params.append(error)

    if status in ["completed", "failed"]:
        updates.append("completed_at = ?")
        params.append(datetime.now().isoformat())

    params.append(job_id)
    query = f"UPDATE evaluation_jobs SET {', '.join(updates)} WHERE eval_job_id = ?"
    cursor.execute(query, params)
    conn.commit()
    conn.close()

def get_pending_job():
    """Get next pending job from database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check pipeline jobs
    cursor.execute("SELECT * FROM pipeline_jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1")
    job = cursor.fetchone()
    if job:
        conn.close()
        return {"type": "pipeline", "job": dict(job)}

    # Check evaluation jobs
    cursor.execute("SELECT * FROM evaluation_jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1")
    job = cursor.fetchone()
    if job:
        conn.close()
        return {"type": "evaluation", "job": dict(job)}

    conn.close()
    return None

def run_single_camera_pipeline(job_id: str, video_path: str, config: dict):
    """Run single camera pipeline"""
    from reid_pipeline.pipeline.production_pipeline import ProductionReIDPipeline

    logger.info(f"Starting single camera pipeline for job {job_id}")
    update_job_status(job_id, "running", progress=0.0)

    try:
        # Prepare output path
        output_path = str(OUTPUT_DIR / f"{job_id}.mp4")

        # Get model paths
        reid_model = config.get('reid_model') or DEFAULT_REID
        yolo_model = config.get('yolo_model') or DEFAULT_YOLO

        # Create pipeline
        pipeline = ProductionReIDPipeline(
            yolo_model_path=yolo_model,
            reid_model_path=reid_model,
            device=config.get('device', 'cuda'),
            detection_conf=config.get('detection_conf', 0.3),
            reid_threshold_match=config.get('reid_threshold_match', 0.70),
            reid_threshold_new=config.get('reid_threshold_new', 0.50),
            gallery_max_size=config.get('gallery_max_size', 500),
            reid_batch_size=config.get('reid_batch_size', 16),
            enable_display=False,
            use_tensorrt=config.get('use_tensorrt', False)
        )

        # Run pipeline
        pipeline.run(video_source=video_path, output_path=output_path)

        # Get stats directly from pipeline.stats
        stats = {
            'frames_captured': pipeline.stats.get('frames_captured', 0),
            'frames_processed': pipeline.stats.get('frames_processed', 0),
            'total_detections': pipeline.stats.get('total_detections', 0),
            'total_persons_tracked': pipeline.stats.get('total_persons_tracked', 0),
        }

        update_job_status(job_id, "completed", progress=100.0,
                         output_video=output_path, stats=stats)
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        traceback.print_exc()
        update_job_status(job_id, "failed", error=str(e))

def run_multi_camera_pipeline(job_id: str, video_paths: list, config: dict):
    """Run multi camera pipeline"""
    from reid_pipeline.multi_camera_pipeline import MultiCameraReIDPipeline

    logger.info(f"Starting multi camera pipeline for job {job_id}")
    update_job_status(job_id, "running", progress=0.0)

    try:
        # Prepare output path
        output_path = str(OUTPUT_DIR / f"{job_id}.avi")

        # Get model paths
        reid_model = config.get('reid_model') or DEFAULT_REID
        yolo_model = config.get('yolo_model') or DEFAULT_YOLO

        # Create pipeline
        pipeline = MultiCameraReIDPipeline(
            yolo_model_path=yolo_model,
            reid_model_path=reid_model,
            device=config.get('device', 'cuda'),
            detection_conf=config.get('detection_conf', 0.3),
            reid_threshold_match=config.get('reid_threshold_match', 0.50),
            reid_threshold_new=config.get('reid_threshold_new', 0.70),
            gallery_max_size=config.get('gallery_max_size', 1000),
            reid_batch_size=config.get('reid_batch_size', 16),
            use_tensorrt=config.get('use_tensorrt', False),
            enable_display=False
        )

        # Run pipeline
        pipeline.run(video_paths, output_path)

        update_job_status(job_id, "completed", progress=100.0, output_video=output_path)
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        traceback.print_exc()
        update_job_status(job_id, "failed", error=str(e))

def run_evaluation(job_id: str, dataset_id: int, config: dict):
    """Run evaluation pipeline"""
    from reid_pipeline.evaluation.evaluation_pipeline import Market1501EvaluationPipeline

    logger.info(f"Starting evaluation for job {job_id}")
    update_eval_job_status(job_id, "running")

    try:
        # Get dataset path
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT upload_path FROM datasets WHERE dataset_id = ?", (dataset_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise ValueError(f"Dataset {dataset_id} not found")

        dataset_path = row['upload_path']

        # Find actual dataset directory (may be nested)
        dataset_dir = Path(dataset_path)
        if (dataset_dir / 'query').exists():
            pass  # dataset_path is correct
        else:
            for subdir in dataset_dir.iterdir():
                if subdir.is_dir() and (subdir / 'query').exists():
                    dataset_path = str(subdir)
                    break

        # Get model path
        reid_model = config.get('reid_model') or DEFAULT_REID

        # Run evaluation
        eval_config = {
            'reid_batch_size': config.get('reid_batch_size', 16),
            'gallery_max_size': config.get('gallery_max_size', 1500),
            'reid_threshold_match': config.get('reid_threshold_match', 0.70),
            'reid_threshold_new': config.get('reid_threshold_new', 0.50),
            'device': 'cuda',
            'embedding_dim': 256
        }

        pipeline = Market1501EvaluationPipeline(
            dataset_path=dataset_path,
            reid_model_path=reid_model,
            config=eval_config
        )

        results = pipeline.run_evaluation(subset_size=config.get('subset_size'))

        update_eval_job_status(job_id, "completed", results=results)
        logger.info(f"Evaluation {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Evaluation {job_id} failed: {e}")
        traceback.print_exc()
        update_eval_job_status(job_id, "failed", error=str(e))

def process_job(job_info: dict):
    """Process a single job"""
    job_type = job_info["type"]
    job = job_info["job"]

    if job_type == "pipeline":
        job_id = job['job_id']
        config = json.loads(job['config']) if isinstance(job['config'], str) else job['config']
        input_video = job['input_video']

        # Check if multi-camera
        try:
            video_paths = json.loads(input_video)
            if isinstance(video_paths, list) and len(video_paths) == 4:
                run_multi_camera_pipeline(job_id, video_paths, config)
            else:
                run_single_camera_pipeline(job_id, input_video, config)
        except json.JSONDecodeError:
            run_single_camera_pipeline(job_id, input_video, config)

    elif job_type == "evaluation":
        job_id = job['eval_job_id']
        dataset_id = job['dataset_id']
        config = json.loads(job['config']) if isinstance(job['config'], str) else job['config']
        run_evaluation(job_id, dataset_id, config)

def main():
    """Main worker loop"""
    logger.info("="*60)
    logger.info("Local Pipeline Worker Started")
    logger.info(f"Database: {DB_PATH}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Models dir: {MODELS_DIR}")
    logger.info("="*60)

    while True:
        try:
            job_info = get_pending_job()

            if job_info:
                logger.info(f"Processing job: {job_info}")
                process_job(job_info)
            else:
                time.sleep(2)  # Poll every 2 seconds

        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        except Exception as e:
            logger.error(f"Worker error: {e}")
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()
