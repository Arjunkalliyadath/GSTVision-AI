# routers/extract.py  (v2.0)
# Adds model_version field to completed job status so the frontend
# can display which model version processed each document.

import os
import uuid
import aiofiles
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from typing import List
from pathlib import Path
from services.pipeline import process_file, get_model_version

router = APIRouter()

MEDIA_ROOT = os.getenv("MEDIA_ROOT", "../../media")

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tiff", ".tif",
    ".webp", ".gif",
    ".heic", ".heif",
}

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("/upload")
async def upload_and_extract(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload one or multiple GST 2A files and extract to Excel."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    jobs = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            supported = ", ".join(sorted(e.upper().lstrip(".") for e in ALLOWED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}': unsupported format '{ext}'. "
                       f"Supported: {supported}"
            )

        job_id     = str(uuid.uuid4())
        upload_dir = Path(MEDIA_ROOT) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path  = upload_dir / f"{job_id}_{file.filename}"

        async with aiofiles.open(str(file_path), "wb") as f:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{file.filename}' too large "
                           f"({len(content)/1024/1024:.1f} MB). Max 100 MB."
                )
            await f.write(content)

        background_tasks.add_task(
            run_extraction_job,
            str(file_path),
            job_id,
            file.filename
        )

        jobs.append({
            "job_id":   job_id,
            "filename": file.filename,
            "status":   "queued",
            "message":  "Processing started",
        })

    return {
        "success":     True,
        "total_files": len(jobs),
        "jobs":        jobs,
        "message":     f"{len(jobs)} file(s) queued for processing",
    }


# ── In-memory job status store ────────────────────────────────────────────────
job_statuses: dict = {}


async def run_extraction_job(file_path: str, job_id: str, original_filename: str):
    """Background task — runs full ML pipeline then records job status."""
    job_statuses[job_id] = {"status": "processing", "percent": 0, "step": "starting"}

    async def progress_callback(update: dict):
        job_statuses[job_id].update(update)
        job_statuses[job_id]["status"] = "processing"

    result = await process_file(file_path, job_id, progress_callback)

    if result["success"]:
        mv = get_model_version()
        job_statuses[job_id] = {
            "status":            "complete",
            "percent":           100,
            "excel_path":        result["excel_path"],
            "excel_filename":    result["excel_filename"],
            "records_extracted": result["records_extracted"],
            "confidence":        result["confidence"],
            "processing_time":   result["processing_time"],
            "method_used":       result.get("method_used", ""),
            "model_version":     mv.get("version", "base"),
        }
        if result.get("warning"):
            job_statuses[job_id]["warning"] = result["warning"]
    else:
        job_statuses[job_id] = {
            "status": "failed",
            "error":  result.get("error", "Unknown error"),
        }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get processing status of a job."""
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_statuses[job_id]


@router.get("/download/{excel_filename}")
async def download_excel(excel_filename: str):
    """Download generated Excel file."""
    excel_path = Path(MEDIA_ROOT) / "outputs" / excel_filename
    if not excel_path.exists():
        raise HTTPException(status_code=404, detail="Excel file not found")
    return FileResponse(
        path=str(excel_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=excel_filename,
    )
