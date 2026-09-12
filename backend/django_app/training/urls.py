from django.urls import path
from . import views

urlpatterns = [
    path("stats/",   views.TrainingStatsView.as_view(),          name="training-stats"),
    path("version/", views.ModelVersionView.as_view(),           name="model-version"),
    path("status/",  views.TrainingCurrentStatusView.as_view(),  name="training-status"),
]
