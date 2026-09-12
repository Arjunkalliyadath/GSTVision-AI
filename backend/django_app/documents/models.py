from django.db import models
import uuid


class UploadJob(models.Model):
    """Tracks each file upload and extraction job."""
    
    STATUS_CHOICES = [
        ("queued", "Queued"),
        ("processing", "Processing"),
        ("complete", "Complete"),
        ("failed", "Failed"),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_filename = models.CharField(max_length=255)
    uploaded_file = models.FileField(upload_to="uploads/")
    excel_file_path = models.CharField(max_length=500, blank=True, null=True)
    excel_filename = models.CharField(max_length=255, blank=True, null=True)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="queued")
    progress_percent = models.IntegerField(default=0)
    current_step = models.CharField(max_length=100, blank=True)
    error_message = models.TextField(blank=True, null=True)
    
    records_extracted = models.IntegerField(default=0)
    confidence_score = models.FloatField(default=0.0)
    processing_time = models.FloatField(default=0.0)  # seconds
    
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.original_filename} ({self.status})"
    
    class Meta:
        ordering = ["-created_at"]


class ExtractedRecord(models.Model):
    """Stores individual invoice records extracted from GST 2A statements."""
    
    job = models.ForeignKey(UploadJob, on_delete=models.CASCADE, related_name="records")
    
    # GST 2A Fields
    gstin = models.CharField(max_length=20, blank=True)
    trade_name = models.CharField(max_length=255, blank=True)
    invoice_number = models.CharField(max_length=100, blank=True)
    invoice_date = models.CharField(max_length=20, blank=True)
    invoice_value = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    place_of_supply = models.CharField(max_length=50, blank=True)
    taxable_value = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    igst = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    cgst = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    sgst = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    cess = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True)
    return_period = models.CharField(max_length=20, blank=True)
    
    # Metadata
    confidence_score = models.FloatField(default=0.0)
    is_corrected = models.BooleanField(default=False)  # True if user corrected this record
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.gstin} - {self.invoice_number}"
    
    class Meta:
        ordering = ["created_at"]


class UserCorrection(models.Model):
    """
    Stores corrections made by users to extracted data.
    This is the key data for continuous learning!
    """
    record = models.OneToOneField(ExtractedRecord, on_delete=models.CASCADE, related_name="correction")
    
    # Corrected values (store what the user changed)
    corrected_data = models.JSONField()  # {field_name: corrected_value}
    original_data = models.JSONField()   # {field_name: original_value}
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Correction for record {self.record.id}"