from django import forms
from django.core.validators import FileExtensionValidator

class ForensicUploadForm(forms.Form):
    media_file = forms.FileField(
        label="Upload Evidence",
        help_text="Supported formats: JPG, PNG, MP4, AVI, MOV",
        validators=[
            FileExtensionValidator(
                allowed_extensions=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi']
            )
        ],
        widget=forms.ClearableFileInput(attrs={'class': 'file-input'})
    )