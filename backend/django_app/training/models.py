from django.db import models


class ModelVersion(models.Model):
    """
    Tracks each auto-training run result.
    Written by the training pipeline; read by the frontend /model/version API.
    """
    version      = models.CharField(max_length=50)          # e.g. "v20250401_1430"
    trained_at   = models.DateTimeField(auto_now_add=True)
    epochs       = models.IntegerField(default=10)
    training_files = models.IntegerField(default=0)
    final_loss   = models.FloatField(default=0.0)
    accuracy     = models.FloatField(default=0.0)
    is_active    = models.BooleanField(default=True)        # latest version
    notes        = models.TextField(blank=True)

    class Meta:
        ordering = ["-trained_at"]

    def __str__(self):
        return f"{self.version} ({self.trained_at.strftime('%Y-%m-%d %H:%M')})"

    def save(self, *args, **kwargs):
        # Deactivate all previous versions when a new one is saved as active
        if self.is_active:
            ModelVersion.objects.exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)


class AutoTrainingLog(models.Model):
    """Log of every auto-training event."""
    triggered_at    = models.DateTimeField(auto_now_add=True)
    trigger_reason  = models.CharField(max_length=200)   # e.g. "5 files accumulated"
    status          = models.CharField(
        max_length=20,
        choices=[("running", "Running"), ("complete", "Complete"), ("failed", "Failed")],
        default="running"
    )
    model_version   = models.ForeignKey(
        ModelVersion, null=True, blank=True, on_delete=models.SET_NULL
    )
    error_message   = models.TextField(blank=True)

    class Meta:
        ordering = ["-triggered_at"]

    def __str__(self):
        return f"AutoTrain {self.triggered_at.strftime('%Y-%m-%d %H:%M')} — {self.status}"
