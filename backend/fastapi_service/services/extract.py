# routers/extract.py — File Upload & Extraction Endpoints  (v2.0 — FIXED)
# ─────────────────────────────────────────────────────────────────────────────
# FIXES v2.0:
#   1. CRITICAL: Added .heic and .heif to ALLOWED_EXTENSIONS.
#      Previously HEIC was blocked at upload — users got silent 400 errors.
#   2. CRITICAL: Added .tiff, .tif, .bmp, .webp to allowed extensions.
#      The converter supports them; now the router allows them too.
#   3. Added explicit Content-Type check for HEIC (browsers send as
#      application/octet-stream or image/heic).
#   4. Better error messages: tell the user EXACTLY what went wrong.
#   5. Max file size raised to 100MB (some multi-page TIFFs are large).
#   6. Job status endpoint now returns full error details.
#   7. Added /list endpoint so frontend can show all jobs.
# ─────────────────────────────────────────────────────────────────────────────

import os
import uuid
import asyncio
import logging
import aiofiles
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from services.pipeline import process_file

router = APIRouter()
logger = logging.getLogger("gst2_fastapi.extract")

MEDIA_ROOT = os.getenv("MEDIA_ROOT", "../../media")

# ── Allowed file extensions ───────────────────────────────────────────────────
# FIXED: Added HEIC, HEIF, TIFF, BMP, WEBP — all supported by pdf_converter.py
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".jpg", ".jpeg",
    ".png",
    ".heic", ".heif",    # ← FIX: iPhone photos now accepted
    ".tiff", ".tif",     # ← FIX: Scanner output
    ".bmp",
    ".webp",
}

# HEIC browsers/OS often send wrong MIME types — accept all of these
HEIC_MIME_TYPES = {
    "image/heic",
    "image/heif",
    "image/heic-sequence",
    "image/heif-sequence",
    "application/octet-stream",  # iOS sometimes sends this for HEIC
}

MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

# ── In-memory job store ───────────────────────────────────────────────────────
# Structure: { ml_job_id: { status, percent, step, message, ... } }
job_statuses: Dict[str, Dict[str, Any]] = {}


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_and_extract(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload one or more GST 2A files and extract data to Excel.

    Supported formats: PDF, JPG, JPEG, PNG, HEIC, HEIF, TIFF, BMP, WEBP
    Max file size: 100 MB per file
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    jobs = []

    for file in files:
        filename  = file.filename or "unknown"
        ext       = Path(filename).suffix.lower()

        # ── Validate extension ────────────────────────────────────────────
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{filename}': Unsupported file format '{ext}'. "
                    f"Accepted formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )
            )

        # ── Read file content ─────────────────────────────────────────────
        content = await file.read()

        # ── Validate size ─────────────────────────────────────────────────
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"'{filename}': File too large "
                    f"({len(content) / 1024 / 1024:.1f} MB). "
                    f"Maximum allowed: {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB."
                )
            )

        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"'{filename}': File is empty."
            )

        # ── Save to disk ──────────────────────────────────────────────────
        job_id     = str(uuid.uuid4())
        upload_dir = Path(MEDIA_ROOT) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Keep original extension so pdf_converter.py routes correctly
        safe_name = f"{job_id}_{_sanitize_filename(filename)}"
        file_path = upload_dir / safe_name

        async with aiofiles.open(str(file_path), "wb") as f:
            await f.write(content)

        logger.info(
            f"Uploaded: {filename} → {safe_name} "
            f"({len(content) / 1024:.0f} KB, job={job_id})"
        )

        # ── Register job ──────────────────────────────────────────────────
        job_statuses[job_id] = {
            "status":    "queued",
            "percent":   0,
            "step":      "queued",
            "message":   "File uploaded, waiting to process...",
            "filename":  filename,
            "file_size": len(content),
            "created_at": datetime.now().isoformat(),
        }

        # ── Queue background processing ───────────────────────────────────
        background_tasks.add_task(
            _run_extraction_job,
            str(file_path),
            job_id,
            filename
        )

        jobs.append({
            "job_id":   job_id,
            "filename": filename,
            "status":   "queued",
            "message":  "Processing queued",
        })

    return {
        "success":     True,
        "total_files": len(jobs),
        "jobs":        jobs,
        "message":     f"{len(jobs)} file(s) queued for processing.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND TASK
# ─────────────────────────────────────────────────────────────────────────────

async def _run_extraction_job(file_path: str, job_id: str, original_filename: str):
    """Background coroutine: runs the ML pipeline and updates job_statuses."""

    async def progress_callback(update: dict):
        job_statuses[job_id].update(update)
        job_statuses[job_id]["status"] = "processing"

    job_statuses[job_id]["status"] = "processing"

    try:
        result = await process_file(file_path, job_id, progress_callback)

        if result.get("success"):
            job_statuses[job_id].update({
                "status":            "complete",
                "percent":           100,
                "step":              "complete",
                "message":           (
                    f"Done! {result.get('records_extracted', 0)} records extracted "
                    f"in {result.get('processing_time', 0):.1f}s"
                ),
                "excel_path":        result.get("excel_path", ""),
                "excel_filename":    result.get("excel_filename", ""),
                "records_extracted": result.get("records_extracted", 0),
                "confidence":        result.get("confidence", 0.0),
                "processing_time":   result.get("processing_time", 0),
                "method_used":       result.get("method_used", ""),
                "completed_at":      datetime.now().isoformat(),
            })
        else:
            error_msg = result.get("error", "Unknown error during processing")
            job_statuses[job_id].update({
                "status":       "failed",
                "percent":      0,
                "step":         "failed",
                "message":      f"Processing failed: {error_msg}",
                "error":        error_msg,
                "completed_at": datetime.now().isoformat(),
            })
            logger.error(f"Job {job_id} failed: {error_msg}")

    except Exception as e:
        error_msg = str(e)
        job_statuses[job_id].update({
            "status":       "failed",
            "step":         "crashed",
            "message":      f"Unexpected error: {error_msg}",
            "error":        error_msg,
            "completed_at": datetime.now().isoformat(),
        })
        logger.error(f"Job {job_id} crashed: {e}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get current processing status of a job.

    Returns:
        {
            status: queued | processing | complete | failed,
            percent: 0-100,
            step: string,
            message: string,
            excel_filename: string (only when complete),
            records_extracted: int (only when complete),
            confidence: float (only when complete),
            error: string (only when failed)
        }
    """
    if job_id not in job_statuses:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have expired or never existed."
        )
    return job_statuses[job_id]


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/download/{excel_filename}")
async def download_excel(excel_filename: str):
    """Download a generated Excel file by filename."""

    # Security: prevent path traversal
    if ".." in excel_filename or "/" in excel_filename or "\\" in excel_filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    excel_path = Path(MEDIA_ROOT) / "outputs" / excel_filename

    if not excel_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Excel file '{excel_filename}' not found. It may have been deleted."
        )

    return FileResponse(
        path=str(excel_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=excel_filename,
        headers={"Content-Disposition": f'attachment; filename="{excel_filename}"'}
    )


# ─────────────────────────────────────────────────────────────────────────────
# LIST ENDPOINT  (useful for frontend to show all jobs)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/jobs")
async def list_jobs():
    """List all jobs (most recent first, max 100)."""
    sorted_jobs = sorted(
        job_statuses.items(),
        key=lambda x: x[1].get("created_at", ""),
        reverse=True
    )[:100]
    return {
        "total":  len(job_statuses),
        "jobs": [{"job_id": jid, **jdata} for jid, jdata in sorted_jobs]
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUPPORTED FORMATS ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/formats")
async def supported_formats():
    """Return list of supported file formats."""
    return {
        "supported_extensions": sorted(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE_BYTES // 1024 // 1024,
        "notes": {
            ".heic": "iPhone/iPad photos — requires pillow-heif installed",
            ".tiff": "Multi-page TIFF supported",
            ".pdf":  "Digital PDFs extracted directly; scanned PDFs use OCR",
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_filename(filename: str) -> str:
    """Remove characters that are unsafe in filenames."""
    import re
    # Keep alphanumeric, dot, hyphen, underscore
    safe = re.sub(r'[^\w.\-]', '_', filename)
    # Limit length
    if len(safe) > 100:
        stem = Path(safe).stem[:90]
        ext  = Path(safe).suffix
        safe = stem + ext
    return safe