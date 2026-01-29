from django.db import models
import uuid

# This table is the "Chain of Custody" for every file that touches the platform.

class ForensicReport(models.Model):
    # Unique ID for the report itself
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # File Metadata
    file_name = models.CharField(max_length=255)
    sha256_hash = models.CharField(max_length=64, unique=True, db_index=True)
    
    # Path inside the 1TB vault (relative to MEDIA_ROOT)
    vault_path = models.CharField(max_length=500)
    
    # Analysis Verdict
    ai_confidence_score = models.FloatField()
    is_altered = models.BooleanField(default=False)
    
    # Notarization (The Security Layer)
    rsa_signature = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report {self.sha256_hash[:8]} | {self.file_name}"

    class Meta:
        ordering = ['-timestamp']