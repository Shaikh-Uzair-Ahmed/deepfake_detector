import threading
import uuid
from datetime import datetime, timezone


class JobStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}

    def create(self, filename: str, job_dir: str) -> dict:
        job = {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "status": "queued",
            "message": "Queued for analysis.",
            "label": None,
            "confidence": None,
            "real_probability": None,
            "fake_probability": None,
            "num_frames": 0,
            "frame_count": 0,
            "used_face_crop": None,
            "model": "VideoViT v2",
            "error": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "job_dir": job_dir,
        }
        with self._lock:
            self._jobs[job["id"]] = job
        return dict(job)

    def update(self, job_id: str, **fields) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.update(fields)
            return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None


store = JobStore()
