import hashlib
import base64
from django.shortcuts import render
from django.conf import settings
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from .models import ForensicReport
from .forms import ForensicUploadForm  # <--- Import the new form
import os
import shutil

def upload_and_analyze(request):
    if request.method == 'POST':
        form = ForensicUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES['media_file']
            
            # 1. Hashing
            sha256 = hashlib.sha256()
            for chunk in uploaded_file.chunks():
                sha256.update(chunk)
            file_hash = sha256.hexdigest()

            # 2. Duplicate Check
            existing_report = ForensicReport.objects.filter(sha256_hash=file_hash).first()
            if existing_report:
                return render(request, 'forensics/report.html', {
                    'status': 'Exists',
                    'report': existing_report
                })

            # --- CORRECTED: SAVE TO QUARANTINE ---
            # We save to 'quarantine' first. This is the "Suspect" folder.
            quarantine_path = os.path.join(settings.MEDIA_ROOT, 'quarantine', uploaded_file.name)
            
            os.makedirs(os.path.dirname(quarantine_path), exist_ok=True)

            with open(quarantine_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            
            # NOTE: In the future, the Malware/AI check happens HERE.
            # Only if it passes would we move it to 'vault'.
            # For now, we leave it in quarantine as "Pending Analysis".
            # -------------------------------------

            # 3. RSA Signing (Sign the hash of the file in quarantine)
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

            # 4. Save to DB 
            # Note: We record the path as 'quarantine' for now.
            report = ForensicReport.objects.create(
                file_name=uploaded_file.name,
                sha256_hash=file_hash,
                vault_path=f"quarantine/{uploaded_file.name}", 
                ai_confidence_score=0.98,
                rsa_signature=encoded_sig
            )

            return render(request, 'forensics/report.html', {
                'status': 'Success',
                'report': report
            })
    else:
        form = ForensicUploadForm()

    return render(request, 'forensics/upload.html', {'form': form})