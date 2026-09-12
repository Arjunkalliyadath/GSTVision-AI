# 📄 GST2A Document AI — Automated GST 2A / E-Way Bill Extraction

**An end-to-end document intelligence system that turns messy, real-world GST 2A reconciliation statements and E-Way Bills — scanned photos, phone pictures, or digital PDFs — into clean, structured Excel reports, with a continuous-learning loop that improves the model from every human correction.**

---

## Why this project exists

Every business registered under India's GST regime has to reconcile GST 2A statements and E-Way Bills against its own books, every filing period. In practice, these documents show up as:

- Digital PDFs downloaded from the GSTN portal (clean text, but still tedious to re-type)
- Scanned or photographed paper copies with skewed angles, shadows, and inconsistent table layouts
- A mix of both, often in a single bulk upload

Manually transcribing GSTIN numbers, invoice values, and tax splits (IGST/CGST/SGST) from hundreds of these documents into a spreadsheet is slow and error-prone — a single mistyped GSTIN can throw off an entire reconciliation.

I built this project to automate that pipeline end-to-end: **upload a document (or a batch of them), get back a structured Excel sheet**, without needing a human to babysit the OCR. The system is also designed to get *better* the more it's used — every correction a user makes to a wrong field is captured and eventually folds back into fine-tuning the extraction model.

This was built as a personal/internship project to explore practical document-AI: routing logic between "digital vs. scanned" documents, multi-engine OCR consensus, and a lightweight continuous-learning loop — all applied to a real, high-friction compliance task rather than a toy dataset.

## How it works, in one paragraph

A file comes in through the React dashboard, Django stores a job record and forwards it to a separate FastAPI ML microservice. The ML service first checks whether the file is a **digital PDF** (has a real text layer) or a **photo/scan** — digital PDFs are parsed directly with `pdfplumber` for near-perfect accuracy in under a second. Anything else goes through a dedicated image pipeline: auto-rotation, adaptive contrast/shadow correction, and then the *same* image is run through **three OCR engines in parallel** (PaddleOCR, Tesseract, EasyOCR); whichever result scores highest on a confidence × line-count heuristic wins. LayoutLMv3 can optionally post-process the result for structural field recognition. The final records are written to Excel with `openpyxl`. Every correction a user makes in the dashboard is logged, and every N uploads (configurable) the system kicks off a LayoutLMv3 fine-tuning pass on the accumulated corrections automatically, in the background.

## Architecture

```
┌──────────────┐      ┌────────────────────┐      ┌─────────────────────────┐
│   Frontend   │─────▶│   Django REST API   │      │   FastAPI ML Service    │
│ React + Vite │      │  (jobs, records,    │◀────▶│  (OCR + LayoutLM +      │
│  Dashboard   │      │  user corrections)  │      │   auto-retraining)      │
└──────────────┘      └─────────┬──────────┘      └─────────────────────────┘
                                 │
                          PostgreSQL (job
                          tracking, correction
                              logging)
```

- **`frontend/`** — React 19 + Vite dashboard for uploading documents, tracking extraction jobs live, and viewing model version / auto-training status.
- **`backend/django_app/`** — Django REST API. Owns every `UploadJob`, proxies requests to the ML service, and persists every `UserCorrection` a human makes — the raw material for the next fine-tuning pass.
- **`backend/fastapi_service/`** — The ML microservice. Classifies each upload, routes it down the digital-PDF or image-OCR path, runs the triple-OCR pipeline, and owns the auto-retraining loop.

## The extraction pipeline, step by step

1. **File classification** — `file_classifier.py` detects whether an upload is a digital PDF (has a text layer) or a photographed/scanned document.
2. **Digital PDFs** → extracted directly via `pdfplumber` (fastest, highest accuracy, no OCR needed). If a "digital" PDF actually yields too few records (e.g. it turns out to be an image-only PDF misdetected), it automatically falls back to the image pipeline instead of failing.
3. **Photos/scans** → routed through a dedicated image pipeline:
   - Auto-rotation and adaptive preprocessing (shadow correction, contrast normalization) via OpenCV — up to 7 enhanced variants per page depending on image quality
   - **Triple OCR** — every image variant is run through **PaddleOCR**, **Tesseract**, and **EasyOCR**, and the result with the best composite confidence/line-count score is kept
   - Optional **LayoutLMv3** post-processing for structural field recognition on complex layouts
4. **Excel generation** — extracted records are written to a structured `.xlsx` report via `openpyxl`.
5. **Continuous learning** — every time a user corrects an extracted field in the dashboard, that correction is logged. After every 5 uploads (configurable via `AUTO_TRAIN_EVERY_N`), the system automatically kicks off a LayoutLMv3 fine-tuning pass on the accumulated corrections in the background — the model improves the more it's used.

## Features

- Upload single or multiple files (PDF, JPG, PNG, BMP, TIFF, WEBP, HEIC) up to 100MB each
- Async background processing with live job status polling (`queued → processing → complete/failed`)
- Dual-pipeline routing — never runs slow OCR on a PDF that already has clean text
- Triple-OCR with automatic best-result selection on image-based documents
- Auto-retraining loop driven by real user corrections, with a live "documents until next retrain" counter
- Model version tracking, exposed via `/model/version`
- Structured Excel export per job, downloadable straight from the dashboard

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite, React Router, Axios |
| API / persistence | Django 5, Django REST Framework, PostgreSQL |
| Async tasks | Celery, Redis |
| ML service | FastAPI, Uvicorn |
| OCR | PaddleOCR, Tesseract (pytesseract), EasyOCR |
| Document understanding | LayoutLMv3 (Transformers), fine-tuned + auto-retrained |
| Computer vision | OpenCV, scikit-image, Pillow |
| ML infra | PyTorch, Hugging Face Transformers/Datasets |
| Reporting | openpyxl / XlsxWriter |

## Project structure

```text
GST2A-Document-AI/
├── backend/
│   ├── django_app/
│   │   ├── documents/          # UploadJob, ExtractedRecord, UserCorrection models + views
│   │   ├── training/           # ModelVersion / AutoTrainingLog tracking + views
│   │   └── config/              # Django settings, URLs
│   └── fastapi_service/
│       ├── main.py              # FastAPI app entrypoint (/health, /model/version)
│       ├── routers/
│       │   ├── extract.py       # /api/extract — upload, status, download
│       │   └── train.py         # /api/train — dataset stats, auto-train status, calibration
│       ├── services/
│       │   ├── pipeline.py      # Dual-pipeline orchestration + triple-OCR scoring
│       │   ├── image_pipeline.py
│       │   ├── paddle_ocr.py / tesseract_ocr.py / easyocr_service.py
│       │   ├── layout_lm.py
│       │   ├── gst2a_pdf_parser.py / image_gst_parser.py
│       │   └── excel_generator.py
│       ├── training/
│       │   └── layoutlm_trainer.py
│       └── *.py (root-level)    # Internal dev/calibration scripts used during development
│                                 # (test_*.py, debug_match.py, run_*calibration*.py, show_results.py) —
│                                 # not required to run the app; some reference local paths from
│                                 # development and are kept for reference rather than reuse.
├── frontend/
│   ├── src/pages/Dashboard.jsx        # Upload + job tracking UI
│   ├── src/pages/ModelInfoPage.jsx    # Model version / dataset stats view
│   └── src/services/api.js
├── .env.example
└── README.md
```

## Getting started

### Prerequisites

- Python 3.10+ 
- Node.js 18+ and npm
- PostgreSQL (running locally, or update `.env` to point elsewhere)
- Redis (only needed if you actually run Celery workers; the core upload → extract → download flow works without it)
- **Disk space:** the FastAPI service's dependencies (PyTorch, PaddleOCR, Transformers, etc.) take several GB once installed — make sure you have at least 10–15 GB free before installing.
- A GPU is **optional**. Every ML component in this codebase checks `torch.cuda.is_available()` and falls back to CPU automatically (`USE_GPU=False` in `.env` also disables GPU explicitly). It'll be slower on CPU, but it works.

### 1. Clone

```bash
git clone https://github.com/Arjunkalliyadath/GST2A-Document-AI.git
cd GST2A-Document-AI
```

### 2. Configure environment

```bash
cp .env.example .env
# then edit .env with your own DB credentials and local model paths
```

### 3. Django API

```bash
cd backend/django_app
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver    # runs on http://127.0.0.1:8000
```

This is the lightweight service — Django, DRF, Celery, Postgres driver — and installs in seconds. It was verified end-to-end while preparing this README: migrations apply cleanly against a fresh PostgreSQL database, and the server boots and serves `/api/documents/...` correctly, including graceful "ML service offline" responses if the FastAPI service isn't running yet.

### 4. FastAPI ML service

```bash
cd backend/fastapi_service
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py    # runs on http://127.0.0.1:8001
```

**If you're on a CPU-only / low-spec machine (no NVIDIA GPU):** run this first, *before* `pip install -r requirements.txt`, to avoid pip pulling several extra gigabytes of CUDA libraries you won't use:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

Then run `pip install -r requirements.txt` as normal — pip will see torch is already satisfied and skip re-downloading it.

**If you do have an NVIDIA GPU** and want CUDA acceleration, install the accelerated builds first instead:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu121/
```
...then run `pip install -r requirements.txt` and set `USE_GPU=True` in `.env`.

> **Note:** `requirements.txt` pins the plain CPU-installable builds of `torch` and `paddlepaddle` by default, since the original `+cu121` / `-gpu` build tags aren't published on PyPI and would make a fresh `pip install -r requirements.txt` fail immediately on any machine that doesn't already have those exact wheels cached. If you followed the GPU steps above first, pip will keep your GPU build instead of downgrading it.

### 5. Frontend

```bash
cd frontend
npm install
npm run dev    # runs on http://127.0.0.1:5173
```

Verified: installs cleanly, `npm run build` produces a working production bundle, and the dev server serves the dashboard correctly.

## Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | PostgreSQL connection for the Django API |
| `DJANGO_SECRET_KEY`, `DJANGO_DEBUG`, `DJANGO_ALLOWED_HOSTS` | Standard Django settings |
| `FASTAPI_HOST`, `FASTAPI_PORT` | Where Django looks for the ML service |
| `LAYOUTLMV3_MODEL_PATH` | Local path to LayoutLMv3 weights (fine-tuned versions are saved here after auto-training) |
| `LLAMA_MODEL_PATH` | Optional local GGUF model used for LLM-based record validation, if present |
| `GOT_OCR_MODEL_PATH` | Legacy path, unused by the current pipeline (GOT-OCR was replaced by EasyOCR — see `services/pipeline.py` header comments) |
| `USE_GPU`, `GPU_DEVICE_ID` | Toggle GPU usage for OCR/LayoutLM/training |
| `MEDIA_ROOT` | Where uploaded files, generated Excel reports, and temp OCR images are stored |
| `TRAINING_DATA_PATH` | Where corrections and raw training data accumulate for auto-retraining |
| `AUTO_TRAIN_EVERY_N`, `AUTO_TRAIN_EPOCHS` | Auto-retraining schedule |

## API reference

### FastAPI ML service (`http://127.0.0.1:8001`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status, GPU availability, PaddleOCR readiness |
| `GET` | `/model/version` | Currently active fine-tuned model version |
| `POST` | `/api/extract/upload` | Upload one or more documents for extraction |
| `GET` | `/api/extract/status/{job_id}` | Poll extraction job progress |
| `GET` | `/api/extract/download/{excel_filename}` | Download the generated Excel report |
| `GET` | `/api/train/dataset/stats` | Training dataset size + next auto-train countdown |
| `GET` | `/api/train/status/current` | Whether a training run is in progress |

### Django API (`http://127.0.0.1:8000`) — consumed by the frontend

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/documents/upload/` | Forwards files to the ML service, creates `UploadJob` records |
| `GET` | `/api/documents/status/{ml_job_id}/` | Proxies job status from the ML service |
| `GET` | `/api/documents/download/{excel_filename}/` | Downloads the generated Excel file |
| `POST` | `/api/documents/correction/` | Saves a user's field correction (written to `training_data/annotated/` as JSON) |
| `GET` | `/api/documents/training/version/` | Proxies current model version |
| `GET` | `/api/documents/training/stats/` | Proxies dataset stats + auto-train schedule |
| `GET` | `/api/documents/training/status/` | Proxies whether auto-training is currently running |
| `POST` | `/api/documents/training/start/` | Manually triggers a training run on the ML service |

## Known limitations

Documented here deliberately, so nobody mistakes these for bugs later:

- **Job status is in-memory** on the FastAPI side (`job_statuses` dict in `routers/extract.py`) — restarting the FastAPI service clears in-flight job history. Fine for a single-instance dev/demo deployment; a production deployment would want this in Redis or Postgres instead.
- **`ExtractedRecord` / `UserCorrection` / `ModelVersion` / `AutoTrainingLog`** exist as Django models (and are migrated into Postgres) but the current data flow doesn't populate them yet — corrections are written to flat JSON files under `training_data/annotated/`, and the active model version is tracked via a `version.json` file the FastAPI service writes, not the database. Wiring these together (so the Django admin becomes a real dashboard over this data) is a natural next step.
- **GOT-OCR** was tried early on and dropped (4GB model, CUDA-only, too slow to be practical) in favor of EasyOCR — `GOT_OCR_MODEL_PATH` in `.env.example` is a leftover from that and isn't read by the current pipeline.
- The root-level dev/calibration scripts in `backend/fastapi_service/` (`test_*.py`, `debug_match.py`, `run_*calibration*.py`) were used locally during development against local data folders and hardcoded local paths — they aren't part of the running app and won't work out of the box on another machine.

## Roadmap ideas

- Persist job status and extracted records fully in Postgres instead of a mix of in-memory state and JSON files
- Wire up the existing `ModelVersion` / `AutoTrainingLog` Django models to the actual training runs
- Add authentication so multiple users/organizations can use their own upload history
- Containerize the three services (Docker Compose) for a true one-command local setup

## Author

**Arjun K**
- GitHub: [@Arjunkalliyadath](https://github.com/Arjunkalliyadath)
- Email: arjunkalliyadath2001@gmail.com

This project was built and is maintained by me as a hands-on exploration of practical document-AI applied to a real compliance workflow (GST reconciliation in India), rather than a toy/tutorial dataset.
