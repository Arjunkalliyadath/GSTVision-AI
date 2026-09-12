# documents/views.py  (v3.0 — BULK UPLOAD FIXED)
# ─────────────────────────────────────────────────────────────────────────────
# FIX: Multi-file upload no longer times out.
#
# OLD (broken): Loop → send file 1 to FastAPI (wait 10s) → send file 2 (wait 10s)
#               For 3+ files this easily exceeds the frontend's 30s axios timeout.
#
# NEW (fixed):  Read ALL files from Django request → send them ALL in ONE
#               multipart request to FastAPI → FastAPI queues all jobs at once
#               → return all job IDs immediately.
#
# This makes multi-file upload as fast as single-file upload regardless of
# how many files are selected.
# ─────────────────────────────────────────────────────────────────────────────

import os
import requests
import json
from django.conf import settings
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status as drf_status
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import FileResponse
from pathlib import Path
import uuid

from .models import UploadJob, ExtractedRecord, UserCorrection

FASTAPI_TIMEOUT_UPLOAD = 60   # seconds — enough for large batches
FASTAPI_TIMEOUT_STATUS = 10   # seconds


class UploadView(APIView):
    """
    Handle file uploads from frontend → forward ALL files at once to FastAPI.

    Key change: ONE batch HTTP request to FastAPI instead of N sequential ones.
    """
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        files = request.FILES.getlist("files")

        if not files:
            return Response({"error": "No files provided"}, status=400)

        # ── Step 1: Create Django DB records for every file ───────────────
        jobs_meta = []   # [(django_job, file_obj), ...]
        for file in files:
            job = UploadJob.objects.create(
                original_filename=file.name,
                uploaded_file=file,
                status="queued"
            )
            jobs_meta.append((job, file))

        # ── Step 2: Build ONE multipart request with ALL files ────────────
        # Reset file pointers (Django may have consumed them during DB save)
        multipart_files = []
        for job, file in jobs_meta:
            file.seek(0)
            # FastAPI expects the field name "files" repeated for each file
            multipart_files.append(
                ("files", (file.name, file.read(), file.content_type or "application/octet-stream"))
            )

        # ── Step 3: Send batch to FastAPI ──────────────────────────────────
        try:
            ml_response = requests.post(
                f"{settings.FASTAPI_ML_URL}/api/extract/upload",
                files=multipart_files,
                timeout=FASTAPI_TIMEOUT_UPLOAD,
            )

            if ml_response.status_code == 200:
                ml_data    = ml_response.json()
                ml_jobs    = ml_data.get("jobs", [])

                # Map FastAPI jobs back to Django jobs by order
                results = []
                for i, (job, file) in enumerate(jobs_meta):
                    ml_job_id = ml_jobs[i]["job_id"] if i < len(ml_jobs) else None
                    job.status = "processing"
                    job.save()
                    results.append({
                        "django_job_id": str(job.id),
                        "ml_job_id":     ml_job_id,
                        "filename":      file.name,
                        "status":        "processing",
                    })

                return Response({"jobs": results, "total": len(results)})

            else:
                # FastAPI returned an error — mark all Django jobs failed
                error_detail = ""
                try:
                    error_detail = ml_response.json().get("detail", ml_response.text[:200])
                except Exception:
                    error_detail = ml_response.text[:200]

                for job, file in jobs_meta:
                    job.status = "failed"
                    job.error_message = f"ML service error {ml_response.status_code}: {error_detail}"
                    job.save()

                return Response(
                    {"error": f"ML service returned {ml_response.status_code}: {error_detail}"},
                    status=502
                )

        except requests.exceptions.ConnectionError:
            for job, _ in jobs_meta:
                job.status = "failed"
                job.error_message = "Cannot connect to FastAPI ML service."
                job.save()
            return Response(
                {"error": "ML service is not running. Start the FastAPI service first (python main.py)."},
                status=503
            )

        except requests.exceptions.Timeout:
            for job, _ in jobs_meta:
                job.status = "failed"
                job.error_message = "FastAPI service timed out during upload."
                job.save()
            return Response(
                {"error": f"Upload timed out after {FASTAPI_TIMEOUT_UPLOAD}s. "
                          "Try uploading fewer files at once."},
                status=504
            )

        except Exception as e:
            for job, _ in jobs_meta:
                job.status = "failed"
                job.error_message = str(e)
                job.save()
            return Response({"error": str(e)}, status=500)


class JobStatusView(APIView):
    """Get status of a job by polling FastAPI ML service."""

    def get(self, request, ml_job_id):
        try:
            response = requests.get(
                f"{settings.FASTAPI_ML_URL}/api/extract/status/{ml_job_id}",
                timeout=FASTAPI_TIMEOUT_STATUS,
            )
            if response.status_code == 200:
                return Response(response.json())
            return Response({"error": "Job not found"}, status=404)
        except requests.exceptions.ConnectionError:
            return Response({"error": "ML service unavailable"}, status=503)
        except requests.exceptions.Timeout:
            return Response({"error": "ML service timeout"}, status=503)
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class DownloadExcelView(APIView):
    """Download Excel file by filename."""

    def get(self, request, excel_filename):
        excel_path = Path(os.getenv("MEDIA_ROOT", "media")) / "outputs" / excel_filename

        if not excel_path.exists():
            return Response({"error": "File not found"}, status=404)

        response = FileResponse(
            open(str(excel_path), "rb"),
            as_attachment=True,
            filename=excel_filename,
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        return response


class SaveCorrectionView(APIView):
    """Save user corrections for continuous learning."""

    def post(self, request):
        job_id        = request.data.get("job_id")
        corrections   = request.data.get("corrections", {})
        original_data = request.data.get("original_data", {})

        if not job_id or not corrections:
            return Response({"error": "job_id and corrections are required"}, status=400)

        training_path = Path(os.getenv("TRAINING_DATA_PATH", "./training_data")) / "annotated"
        training_path.mkdir(parents=True, exist_ok=True)

        correction_entry = {
            "job_id":        job_id,
            "corrections":   corrections,
            "original_data": original_data,
            "timestamp":     timezone.now().isoformat(),
        }

        save_path = training_path / f"correction_{uuid.uuid4()}.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(correction_entry, f, ensure_ascii=False, indent=2)

        return Response({
            "success":    True,
            "message":    "Correction saved. This will improve future extractions.",
            "file_saved": save_path.name,
        })


# NOTE: dataset-stats proxying now lives solely in training/views.py
# (TrainingStatsView), reachable at /api/documents/training/stats/.
# A duplicate of it used to live here too, silently shadowed by this
# same URL prefix — removed to avoid two views doing the same job.


class StartTrainingView(APIView):
    """Trigger bulk training / fine-tuning on FastAPI."""

    def post(self, request):
        try:
            response = requests.post(
                f"{settings.FASTAPI_ML_URL}/api/train/start",
                json={
                    "epochs":         request.data.get("epochs", 3),
                    "batch_size":     request.data.get("batch_size", 4),
                    "annotated_only": request.data.get("annotated_only", False),
                },
                timeout=30,
            )
            return Response(response.json(), status=response.status_code)
        except requests.exceptions.ConnectionError:
            return Response({"error": "ML service not running"}, status=503)
        except Exception as e:
            return Response({"error": str(e)}, status=500)