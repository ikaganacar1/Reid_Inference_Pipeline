"""
FastAPI Backend for ReID Pipeline Web Interface
Handles pipeline control, video upload, and real-time status updates
"""
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
import uuid
import os
from pathlib import Path
from datetime import datetime
import logging
import redis
import psycopg2
from psycopg2.extras import RealDictCursor
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ReID Pipeline API",
    description="API for Person Re-Identification Pipeline",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "reid_pipeline")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Redis connection
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    redis_client.ping()
    logger.info("Connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# Database connection
def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            cursor_factory=RealDictCursor
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

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

class PipelineJob(BaseModel):
    job_id: str
    status: str
    config: Dict[str, Any]
    input_video: str
    output_video: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    progress: float
    stats: Optional[Dict[str, Any]]

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

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

# Initialize database tables
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()

            # Create jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_jobs (
                    job_id VARCHAR(255) PRIMARY KEY,
                    status VARCHAR(50) NOT NULL,
                    config JSONB NOT NULL,
                    input_video TEXT NOT NULL,
                    output_video TEXT,
                    created_at TIMESTAMP NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    progress FLOAT DEFAULT 0.0,
                    stats JSONB,
                    error_message TEXT
                )
            """)

            # Create configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_configs (
                    config_id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    config JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)

            conn.commit()
            cursor.close()
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
        finally:
            conn.close()

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ReID Pipeline API",
        "version": "1.0.0"
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    redis_status = "connected" if redis_client and redis_client.ping() else "disconnected"
    db_conn = get_db_connection()
    db_status = "connected" if db_conn else "disconnected"
    if db_conn:
        db_conn.close()

    return {
        "status": "healthy",
        "redis": redis_status,
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/upload/video")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file for processing"""
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = UPLOAD_DIR / f"{file_id}{file_extension}"

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Video uploaded: {file_path}")

        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "path": str(file_path),
            "size": file_path.stat().st_size
        }

    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/start")
async def start_pipeline(
    background_tasks: BackgroundTasks,
    video_path: str,
    config: PipelineConfig
):
    """Start single-camera pipeline job"""
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create job record in database
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pipeline_jobs
                (job_id, status, config, input_video, created_at, progress)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                job_id,
                "pending",
                json.dumps(config.dict()),
                video_path,
                datetime.now(),
                0.0
            ))
            conn.commit()
            cursor.close()
            conn.close()

        # Add job to Redis queue
        if redis_client:
            job_data = {
                "job_id": job_id,
                "type": "single_camera",
                "video_path": video_path,
                "config": config.dict(),
                "created_at": datetime.now().isoformat()
            }
            redis_client.lpush("pipeline_jobs", json.dumps(job_data))
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "pending",
                "progress": "0.0"
            })

        logger.info(f"Pipeline job created: {job_id}")

        return {
            "success": True,
            "job_id": job_id,
            "status": "pending",
            "message": "Pipeline job queued successfully"
        }

    except Exception as e:
        logger.error(f"Error starting pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pipeline/multi-camera/start")
async def start_multi_camera_pipeline(
    background_tasks: BackgroundTasks,
    video_paths: List[str],
    config: MultiCameraConfig
):
    """Start multi-camera pipeline job"""
    try:
        if len(video_paths) != 4:
            raise HTTPException(status_code=400, detail="Exactly 4 video paths required")

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create job record
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pipeline_jobs
                (job_id, status, config, input_video, created_at, progress)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                job_id,
                "pending",
                json.dumps(config.dict()),
                json.dumps(video_paths),
                datetime.now(),
                0.0
            ))
            conn.commit()
            cursor.close()
            conn.close()

        # Add job to Redis queue
        if redis_client:
            job_data = {
                "job_id": job_id,
                "type": "multi_camera",
                "video_paths": video_paths,
                "config": config.dict(),
                "created_at": datetime.now().isoformat()
            }
            redis_client.lpush("pipeline_jobs", json.dumps(job_data))
            redis_client.hset(f"job:{job_id}", mapping={
                "status": "pending",
                "progress": "0.0"
            })

        logger.info(f"Multi-camera pipeline job created: {job_id}")

        return {
            "success": True,
            "job_id": job_id,
            "status": "pending",
            "message": "Multi-camera pipeline job queued successfully"
        }

    except Exception as e:
        logger.error(f"Error starting multi-camera pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs")
async def list_jobs(limit: int = 20, offset: int = 0):
    """List all pipeline jobs"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM pipeline_jobs
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (limit, offset))

        jobs = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) as total FROM pipeline_jobs")
        total = cursor.fetchone()['total']

        cursor.close()
        conn.close()

        return {
            "success": True,
            "jobs": jobs,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and details"""
    try:
        # Always get data from database for complete information
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_jobs WHERE job_id = %s", (job_id,))
        job = cursor.fetchone()
        cursor.close()
        conn.close()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Optionally merge with Redis real-time data if available
        if redis_client:
            job_data = redis_client.hgetall(f"job:{job_id}")
            if job_data:
                # Update with real-time Redis data
                job['status'] = job_data.get("status", job.get('status'))
                job['progress'] = float(job_data.get("progress", job.get('progress', 0.0)))
                if job_data.get("stats"):
                    job['stats'] = json.loads(job_data.get("stats"))

        return {
            "success": True,
            "job": job
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    try:
        if redis_client:
            redis_client.hset(f"job:{job_id}", "status", "cancelled")

        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE pipeline_jobs
                SET status = 'cancelled'
                WHERE job_id = %s
            """, (job_id,))
            conn.commit()
            cursor.close()
            conn.close()

        logger.info(f"Job cancelled: {job_id}")

        return {
            "success": True,
            "message": f"Job {job_id} cancelled"
        }

    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/output/{job_id}")
async def get_output_video(job_id: str):
    """Download output video"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        cursor.execute("SELECT output_video FROM pipeline_jobs WHERE job_id = %s", (job_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not result or not result['output_video']:
            raise HTTPException(status_code=404, detail="Output video not found")

        output_path = Path(result['output_video'])
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Output file not found on disk")

        # Detect MIME type based on file extension
        file_extension = output_path.suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.mov': 'video/quicktime',
            '.webm': 'video/webm'
        }
        media_type = mime_types.get(file_extension, 'video/mp4')

        logger.info(f"Serving video: {output_path} (type: {media_type}, size: {output_path.stat().st_size} bytes)")

        return FileResponse(
            output_path,
            media_type=media_type,
            filename=output_path.name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting output video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/configs")
async def list_configs():
    """List saved configurations"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_configs ORDER BY updated_at DESC")
        configs = cursor.fetchall()
        cursor.close()
        conn.close()

        return {
            "success": True,
            "configs": configs
        }

    except Exception as e:
        logger.error(f"Error listing configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/configs")
async def save_config(name: str, config: PipelineConfig):
    """Save pipeline configuration"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        now = datetime.now()

        cursor.execute("""
            INSERT INTO pipeline_configs (name, config, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name)
            DO UPDATE SET config = EXCLUDED.config, updated_at = EXCLUDED.updated_at
        """, (name, json.dumps(config.dict()), now, now))

        conn.commit()
        cursor.close()
        conn.close()

        return {
            "success": True,
            "message": f"Configuration '{name}' saved successfully"
        }

    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/configs/{config_id}")
async def get_config(config_id: int):
    """Get a specific configuration by ID"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        cursor.execute("SELECT * FROM pipeline_configs WHERE config_id = %s", (config_id,))
        config = cursor.fetchone()
        cursor.close()
        conn.close()

        if not config:
            raise HTTPException(status_code=404, detail="Configuration not found")

        return {
            "success": True,
            "config": config
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/configs/{config_id}")
async def update_config(config_id: int, name: str, config: PipelineConfig):
    """Update an existing configuration"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        now = datetime.now()

        cursor.execute("""
            UPDATE pipeline_configs
            SET name = %s, config = %s, updated_at = %s
            WHERE config_id = %s
        """, (name, json.dumps(config.dict()), now, config_id))

        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Configuration not found")

        conn.commit()
        cursor.close()
        conn.close()

        return {
            "success": True,
            "message": f"Configuration '{name}' updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/configs/{config_id}")
async def delete_config(config_id: int):
    """Delete a configuration"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        cursor.execute("DELETE FROM pipeline_configs WHERE config_id = %s", (config_id,))

        if cursor.rowcount == 0:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Configuration not found")

        conn.commit()
        cursor.close()
        conn.close()

        return {
            "success": True,
            "message": "Configuration deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()

            # Echo back (for ping/pong)
            if data == "ping":
                await websocket.send_text("pong")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

# Background task to broadcast job updates
async def broadcast_job_updates():
    """Background task to broadcast job status updates to WebSocket clients"""
    while True:
        try:
            if redis_client:
                # Get all active jobs
                job_keys = redis_client.keys("job:*")
                updates = []

                for key in job_keys:
                    job_data = redis_client.hgetall(key)
                    if job_data:
                        job_id = key.split(":")[1]
                        updates.append({
                            "job_id": job_id,
                            "status": job_data.get("status"),
                            "progress": float(job_data.get("progress", 0.0)),
                            "stats": json.loads(job_data.get("stats", "{}"))
                        })

                if updates:
                    await manager.broadcast({
                        "type": "job_updates",
                        "data": updates,
                        "timestamp": datetime.now().isoformat()
                    })

        except Exception as e:
            logger.error(f"Error broadcasting updates: {e}")

        await asyncio.sleep(5)  # Update every 5 seconds

@app.on_event("startup")
async def start_broadcast_task():
    """Start background task for broadcasting updates"""
    asyncio.create_task(broadcast_job_updates())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
