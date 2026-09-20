from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.inference import detector
from api.jobs import store

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_ROOT = Path(__file__).resolve().parent / "_jobs"
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"

ALLOWED_EXT = {
    ".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp",
}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@asynccontextmanager
async def lifespan(_app: FastAPI):
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        detector.load()
    except Exception:
        # Health endpoint reports the error; API still starts so the UI can show it.
        pass
    yield


app = FastAPI(
    title="Deepfake Detector API",
    description="Thin wrapper around VideoViT v2 (train_vit2).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def public_job(job: dict) -> dict:
    job_id = job["id"]
    n = int(job.get("num_frames") or 0)
    return {
        "id": job_id,
        "filename": job.get("filename"),
        "status": job.get("status"),
        "message": job.get("message"),
        "label": job.get("label"),
        "confidence": job.get("confidence"),
        "real_probability": job.get("real_probability"),
        "fake_probability": job.get("fake_probability"),
        "num_frames": n,
        "used_face_crop": job.get("used_face_crop"),
        "model": job.get("model"),
        "error": job.get("error"),
        "frames": [
            {"index": i, "url": f"/api/jobs/{job_id}/frames/{i}"}
            for i in range(n)
        ] if job.get("status") == "done" else [],
    }


def run_job(job_id: str, media_path: str, preview_dir: str) -> None:
    try:
        def on_status(status: str, message: str) -> None:
            store.update(job_id, status=status, message=message)

        result = detector.predict(media_path, preview_dir, on_status=on_status)
        store.update(
            job_id,
            status="done",
            message="Analysis complete.",
            **result,
        )
    except Exception as exc:
        store.update(
            job_id,
            status="error",
            message="Analysis failed.",
            error=str(exc),
        )


@app.get("/api/health")
def health():
    return detector.info()


@app.post("/api/detect")
async def detect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not detector.ready:
        raise HTTPException(
            status_code=503,
            detail=detector.error or "VideoViT v2 checkpoint is not loaded.",
        )

    original = file.filename or "upload.bin"
    ext = os.path.splitext(original)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use a video or image.",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 200 MB limit.")

    job = store.create(original, str(JOBS_ROOT))
    job_dir = JOBS_ROOT / job["id"]
    preview_dir = job_dir / "frames"
    job_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    store.update(job["id"], job_dir=str(job_dir))

    media_path = job_dir / f"source{ext}"
    media_path.write_bytes(data)

    background_tasks.add_task(run_job, job["id"], str(media_path), str(preview_dir))
    return {"job_id": job["id"], **public_job(store.get(job["id"]))}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return public_job(job)


@app.get("/api/jobs/{job_id}/frames/{index}")
def get_frame(job_id: str, index: int):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "done":
        raise HTTPException(status_code=409, detail="Frames are not ready.")
    frame_path = Path(job["job_dir"]) / "frames" / f"{index}.jpg"
    if not frame_path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found.")
    return FileResponse(frame_path, media_type="image/jpeg")


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="ui")
else:
    @app.get("/")
    def root_hint():
        return {
            "service": "Deepfake Detector API",
            "ui": "Run the Vite frontend (npm run dev in frontend/) or build it so dist/ exists.",
            "health": "/api/health",
        }
