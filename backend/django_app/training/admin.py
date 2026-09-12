from django.contrib import admin
from .models import ModelVersion, AutoTrainingLog

@admin.register(ModelVersion)
class ModelVersionAdmin(admin.ModelAdmin):
    list_display = ("version", "trained_at", "epochs", "training_files", "accuracy", "is_active")
    list_filter  = ("is_active",)

@admin.register(AutoTrainingLog)
class AutoTrainingLogAdmin(admin.ModelAdmin):
    list_display = ("triggered_at", "status", "trigger_reason", "model_version")
    list_filter  = ("status",)
