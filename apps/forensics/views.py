from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import ForensicReport
# --- FIX: Only import the form that actually exists ---
from .forms import ForensicUploadForm 
from .core import VeritasForensicEngine
import os

def upload_and_analyze(request):
    if request.method == 'POST':
        # Use the correct form class here
        form = ForensicUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            # Check forms.py to see if the field is 'media_file' or 'file'
            # Based on your previous snippet, it looked like 'media_file'
            uploaded_file = request.FILES.get('media_file')
            
            # 1. Save to Quarantine (Temporary Holding Area)
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'quarantine'))
            filename = fs.save(uploaded_file.name, uploaded_file)
            quarantine_path = fs.path(filename)

            try:
                # 2. Initialize the Engine
                engine = VeritasForensicEngine(
                    input_file_path=quarantine_path,
                    private_key_path=settings.RSA_KEY_PATH 
                )

                # --- DUPLICATE CHECK OPTIMIZATION ---
                file_hash = engine.get_file_hash()
                existing_report = ForensicReport.objects.filter(sha256_hash=file_hash).first()
                
                if existing_report:
                    # Clean up the quarantine file since we don't need it
                    os.remove(quarantine_path)
                    return render(request, 'forensics/report.html', {
                        'status': 'Exists',
                        'report': existing_report
                    })

                # 3. The "Big Red Button" -> Execute Pipeline (AI + Signing + Vaulting)
                result = engine.execute_full_pipeline()

                # 4. Save the Result to the Database
                report = ForensicReport.objects.create(
                    file_name=uploaded_file.name,
                    sha256_hash=result['hash'],
                    vault_path=result['vault_location'],
                    ai_confidence_score=result['analysis']['ai_confidence'],
                    rsa_signature=result['signature']
                )

                return render(request, 'forensics/report.html', {
                    'status': 'Success',
                    'report': report,
                    # We don't need 'metadata' here as 'report' has the score
                })

            except Exception as e:
                print(f"❌ PIPELINE CRASHED: {str(e)}")
                return render(request, 'forensics/upload.html', {'form': form, 'error': str(e)})

    else:
        # Use the correct form class here as well
        form = ForensicUploadForm()

    return render(request, 'forensics/upload.html', {'form': form})