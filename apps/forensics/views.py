import hashlib
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from .models import ForensicReport
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64




def upload_and_analyze(request):
    if request.method == 'POST' and request.FILES.get('media_file'):
        uploaded_file = request.FILES['media_file']
        
        # 1. Generate Hash
        sha256 = hashlib.sha256()
        for chunk in uploaded_file.chunks():
            sha256.update(chunk)
        file_hash = sha256.hexdigest()

        # 2. Check for Existing
        existing_report = ForensicReport.objects.filter(sha256_hash=file_hash).first()
        
        if existing_report:
            # RENDER HTML instead of JSON
            return render(request, 'forensics/report.html', {
                'status': 'Exists',
                'report': existing_report
            })

        # 3. Process New File (RSA Signing)
        with open(settings.RSA_KEY_PATH, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(), password=None
            )

        signature = private_key.sign(
            file_hash.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        encoded_sig = base64.b64encode(signature).decode('utf-8')

        # 4. Save
        report = ForensicReport.objects.create(
            file_name=uploaded_file.name,
            sha256_hash=file_hash,
            vault_path=f"storage/vault/{uploaded_file.name}",
            ai_confidence_score=0.98,
            rsa_signature=encoded_sig
        )

        # RENDER HTML for success
        return render(request, 'forensics/report.html', {
            'status': 'Success',
            'report': report
        })

    return render(request, 'forensics/upload.html')