from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.UploadView.as_view(), name="upload"),
    path("status/<str:ml_job_id>/", views.JobStatusView.as_view(), name="job-status"),
    path("download/<str:excel_filename>/", views.DownloadExcelView.as_view(), name="download"),
    path("correction/", views.SaveCorrectionView.as_view(), name="save-correction"),
    # NOTE: "training/stats/" used to be defined here too, which silently shadowed
    # training.urls's identical "stats/" route (both resolve to the same URL and
    # did the same thing). Removed to keep one source of truth — see training/urls.py.
    path("training/start/", views.StartTrainingView.as_view(), name="start-training"),
]