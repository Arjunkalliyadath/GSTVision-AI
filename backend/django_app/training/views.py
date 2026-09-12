# training/views.py  (v2.0 — AUTO-TRAIN STATUS + MODEL VERSION)
# ─────────────────────────────────────────────────────────────────────────────
# Endpoints consumed by the React ModelInfoPage:
#   GET /api/documents/training/stats/   → dataset stats + auto-train schedule
#   GET /api/documents/training/version/ → current model version
#   GET /api/documents/training/status/  → is auto-training running right now?
# ─────────────────────────────────────────────────────────────────────────────

import requests
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response

FASTAPI_URL     = getattr(settings, "FASTAPI_ML_URL", "http://127.0.0.1:8001")
FASTAPI_TIMEOUT = 8


class TrainingStatsView(APIView):
    """Dataset stats + auto-training schedule (proxies FastAPI)."""

    def get(self, request):
        try:
            r = requests.get(
                f"{FASTAPI_URL}/api/train/dataset/stats",
                timeout=FASTAPI_TIMEOUT
            )
            if r.status_code == 200:
                return Response(r.json())
            return Response({"error": f"FastAPI returned {r.status_code}"}, status=502)
        except requests.exceptions.ConnectionError:
            return Response(
                {"error": "ML service is not running. Start FastAPI first."},
                status=503
            )
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class ModelVersionView(APIView):
    """Current fine-tuned model version (proxies FastAPI /model/version)."""

    def get(self, request):
        try:
            r = requests.get(
                f"{FASTAPI_URL}/model/version",
                timeout=FASTAPI_TIMEOUT
            )
            if r.status_code == 200:
                return Response(r.json())
            return Response({"error": f"FastAPI returned {r.status_code}"}, status=502)
        except requests.exceptions.ConnectionError:
            return Response(
                {"version": "unknown", "note": "ML service offline"},
                status=200   # Return 200 so the frontend still renders
            )
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class TrainingCurrentStatusView(APIView):
    """Is auto-training running right now? (proxies FastAPI)."""

    def get(self, request):
        try:
            r = requests.get(
                f"{FASTAPI_URL}/api/train/status/current",
                timeout=FASTAPI_TIMEOUT
            )
            if r.status_code == 200:
                return Response(r.json())
            return Response({"is_training": False, "message": "Unknown"}, status=200)
        except Exception:
            return Response({"is_training": False, "message": "ML service offline"}, status=200)
