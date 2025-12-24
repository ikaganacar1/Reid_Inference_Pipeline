"""
Local FastAPI Backend for ReID Pipeline - No Docker Dependencies
Uses SQLite and in-memory job queue instead of PostgreSQL and Redis
"""
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
import uuid
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import sqlite3
import shutil
import threading
from collections import deque

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ReID Pipeline API (Local)",
    description="Local API for Person Re-Identification Pipeline",
    version="1.0.0-local"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - use local paths
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
DB_PATH = PROJECT_ROOT / "local_api.db"

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job queue (replaces Redis)
class LocalJobQueue:
    def __init__(self):
        self.jobs = deque()
        self.job_status = {}
        self.lock = threading.Lock()

    def push(self, job_data):
        with self.lock:
            self.jobs.append(job_data)
            job_id = job_data.get('job_id')
            if job_id:
                self.job_status[job_id] = {'status': 'pending', 'progress': 0.0}

    def pop(self):
        with self.lock:
            if self.jobs:
                return self.jobs.popleft()
            return None

    def get_status(self, job_id):
        with self.lock:
            return self.job_status.get(job_id, {})

    def set_status(self, job_id, status_data):
        with self.lock:
            if job_id not in self.job_status:
                self.job_status[job_id] = {}
            self.job_status[job_id].update(status_data)

job_queue = LocalJobQueue()

# SQLite database connection
def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize SQLite database tables"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create jobs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            config TEXT NOT NULL,
            input_video TEXT NOT NULL,
            output_video TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            progress REAL DEFAULT 0.0,
            stats TEXT,
            error_message TEXT
        )
    """)

    # Create configurations table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_configs (
            config_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            config TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Create datasets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            dataset_type TEXT NOT NULL,
            upload_path TEXT NOT NULL,
            num_query INTEGER DEFAULT 0,
            num_gallery INTEGER DEFAULT 0,
            status TEXT DEFAULT 'uploaded',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create evaluation jobs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluation_jobs (
            eval_job_id TEXT PRIMARY KEY,
            dataset_id INTEGER,
            config TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            results TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            error_message TEXT
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialized")

# Pydantic models
class PipelineConfig(BaseModel):
    preset: Optional[str] = "development"
    device: str = "cuda"
    yolo_model: str = "yolo11n.pt"
    reid_model: Optional[str] = None
    detection_conf: float = 0.3
    reid_threshold_match: float = 0.70
    reid_threshold_new: float = 0.50
    gallery_max_size: int = 500
    reid_batch_size: int = 16
    use_tensorrt: bool = False
    enable_display: bool = False

class MultiCameraConfig(BaseModel):
    preset: Optional[str] = "development"
    device: str = "cuda"
    yolo_model: str = "yolo11n.pt"
    reid_model: Optional[str] = None
    detection_conf: float = 0.3
    reid_threshold_match: float = 0.50
    reid_threshold_new: float = 0.70
    gallery_max_size: int = 1000
    reid_batch_size: int = 16
    use_tensorrt: bool = False
    display_scale: float = 0.5

class EvaluationConfig(BaseModel):
    dataset_id: int
    yolo_model: str = "yolo11n.pt"
    reid_model: Optional[str] = None
    reid_threshold_match: float = 0.70
    reid_threshold_new: float = 0.50
    gallery_max_size: int = 1500
    reid_batch_size: int = 16
    use_tensorrt: bool = False
    subset_size: Optional[int] = None

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_database()

# API Endpoints
@app.get("/")
async def root():
    return {"status": "healthy", "service": "ReID Pipeline API (Local)", "version": "1.0.0-local"}

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "redis": "local-queue",
        "database": "sqlite",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    try:
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "path": str(file_path),
            "size": file_path.stat().st_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/start")
async def start_pipeline(video_path: str, config: PipelineConfig):
    try:
        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pipeline_jobs (job_id, status, config, input_video, created_at, progress)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (job_id, "pending", json.dumps(config.dict()), video_path, now, 0.0))
        conn.commit()
        conn.close()

        job_queue.push({
            "job_id": job_id,
            "type": "single_camera",
            "video_path": video_path,
            "config": config.dict()
        })

        return {"success": True, "job_id": job_id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/multi-camera/start")
async def start_multi_camera_pipeline(video_paths: List[str], config: MultiCameraConfig):
    try:
        if len(video_paths) != 4:
            raise HTTPException(status_code=400, detail="Exactly 4 video paths required")

        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO pipeline_jobs (job_id, status, config, input_video, created_at, progress)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (job_id, "pending", json.dumps(config.dict()), json.dumps(video_paths), now, 0.0))
        conn.commit()
        conn.close()

        job_queue.push({
            "job_id": job_id,
            "type": "multi_camera",
            "video_paths": video_paths,
            "config": config.dict()
        })

        return {"success": True, "job_id": job_id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs")
async def list_jobs(limit: int = 20, offset: int = 0, sort_by: str = "created_at", sort_order: str = "desc"):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        allowed_fields = ["created_at", "status", "progress", "job_id"]
        sort_by = sort_by if sort_by in allowed_fields else "created_at"
        sort_order = "DESC" if sort_order.lower() == "desc" else "ASC"

        cursor.execute(f"SELECT * FROM pipeline_jobs ORDER BY {sort_by} {sort_order} LIMIT ? OFFSET ?", (limit, offset))
        jobs = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT COUNT(*) as total FROM pipeline_jobs")
        total = cursor.fetchone()['total']

        conn.close()
        return {"success": True, "jobs": jobs, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Job not found")

        job = dict(row)
        # Parse JSON fields
        if job.get('config'):
            job['config'] = json.loads(job['config'])
        if job.get('stats'):
            job['stats'] = json.loads(job['stats'])

        # Merge with real-time status from queue
        status = job_queue.get_status(job_id)
        if status:
            job['status'] = status.get('status', job['status'])
            job['progress'] = status.get('progress', job.get('progress', 0.0))

        return {"success": True, "job": job}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    try:
        job_queue.set_status(job_id, {'status': 'cancelled'})

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE pipeline_jobs SET status = 'cancelled' WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()

        return {"success": True, "message": f"Job {job_id} cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs/bulk-delete")
async def bulk_delete_jobs(job_ids: List[str]):
    try:
        deleted_count = 0
        conn = get_db_connection()
        cursor = conn.cursor()

        for job_id in job_ids:
            cursor.execute("SELECT output_video FROM pipeline_jobs WHERE job_id = ?", (job_id,))
            result = cursor.fetchone()
            if result and result['output_video']:
                output_path = Path(result['output_video'])
                if output_path.exists():
                    output_path.unlink()

            cursor.execute("DELETE FROM pipeline_jobs WHERE job_id = ?", (job_id,))
            deleted_count += 1

        conn.commit()
        conn.close()

        return {"success": True, "deleted_count": deleted_count, "total_requested": len(job_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/output/{job_id}")
async def get_output_video(job_id: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT output_video FROM pipeline_jobs WHERE job_id = ?", (job_id,))
        result = cursor.fetchone()
        conn.close()

        if not result or not result['output_video']:
            raise HTTPException(status_code=404, detail="Output video not found")

        output_path = Path(result['output_video'])
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found on disk")

        mime_types = {'.mp4': 'video/mp4', '.avi': 'video/x-msvideo', '.mkv': 'video/x-matroska'}
        media_type = mime_types.get(output_path.suffix.lower(), 'video/mp4')

        return FileResponse(output_path, media_type=media_type, filename=output_path.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/configs")
async def list_configs():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_configs ORDER BY updated_at DESC")
        configs = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return {"success": True, "configs": configs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/configs")
async def save_config(name: str, config: PipelineConfig):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        cursor.execute("SELECT config_id FROM pipeline_configs WHERE name = ?", (name,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute("UPDATE pipeline_configs SET config = ?, updated_at = ? WHERE name = ?",
                          (json.dumps(config.dict()), now, name))
        else:
            cursor.execute("INSERT INTO pipeline_configs (name, config, created_at, updated_at) VALUES (?, ?, ?, ?)",
                          (name, json.dumps(config.dict()), now, now))

        conn.commit()
        conn.close()
        return {"success": True, "message": f"Configuration '{name}' saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/configs/{config_id}")
async def get_config(config_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_configs WHERE config_id = ?", (config_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Configuration not found")

        config = dict(row)
        if config.get('config'):
            config['config'] = json.loads(config['config'])
        return {"success": True, "config": config}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/configs/{config_id}")
async def delete_config(config_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM pipeline_configs WHERE config_id = ?", (config_id,))
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="Configuration not found")
        conn.commit()
        conn.close()
        return {"success": True, "message": "Configuration deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    try:
        yolo_models = []
        reid_models = []

        if MODELS_DIR.exists():
            for f in MODELS_DIR.iterdir():
                if f.is_file():
                    model_info = {'filename': f.name, 'path': str(f), 'size': f.stat().st_size, 'type': f.suffix[1:]}
                    if 'yolo' in f.name.lower():
                        yolo_models.append(model_info)
                    elif any(p in f.name.lower() for p in ['reid', 'lttc', 'resnet']):
                        reid_models.append(model_info)

        return {"success": True, "yolo_models": yolo_models, "reid_models": reid_models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Dataset endpoints
@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        dataset_id = str(uuid.uuid4())[:8]
        dataset_dir = DATASETS_DIR / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        zip_path = dataset_dir / file.filename
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract zip
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO datasets (name, dataset_type, upload_path, status) VALUES (?, ?, ?, ?)
        """, (file.filename, 'market1501', str(dataset_dir), 'uploaded'))
        dataset_id_db = cursor.lastrowid
        conn.commit()
        conn.close()

        return {"dataset_id": dataset_id_db, "status": "uploaded", "path": str(dataset_dir)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets")
async def list_datasets():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC")
    datasets = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"datasets": datasets}

@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"success": True, "dataset": dict(row)}

@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT upload_path FROM datasets WHERE dataset_id = ?", (dataset_id,))
    row = cursor.fetchone()
    if row and row['upload_path']:
        path = Path(row['upload_path'])
        if path.exists():
            shutil.rmtree(path)
    cursor.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
    conn.commit()
    conn.close()
    return {"success": True, "message": "Dataset deleted"}

# Evaluation endpoints
@app.post("/api/evaluation/start")
async def start_evaluation(config: EvaluationConfig):
    try:
        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO evaluation_jobs (eval_job_id, dataset_id, config, status, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (job_id, config.dataset_id, json.dumps(config.dict()), 'pending', now))
        conn.commit()
        conn.close()

        job_queue.push({
            "job_id": job_id,
            "type": "evaluation",
            "dataset_id": config.dataset_id,
            "config": config.dict()
        })

        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/evaluation/jobs")
async def list_evaluation_jobs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluation_jobs ORDER BY created_at DESC LIMIT 50")
    jobs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return {"jobs": jobs}

@app.get("/api/evaluation/jobs/{job_id}")
async def get_evaluation_job(job_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluation_jobs WHERE eval_job_id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job = dict(row)
    if job.get('config'):
        job['config'] = json.loads(job['config'])
    if job.get('results'):
        job['results'] = json.loads(job['results'])
    return job

# WebSocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except:
        pass
    finally:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
