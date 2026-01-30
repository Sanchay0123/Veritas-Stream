from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import ForensicReport
from .forms import ForensicUploadForm
from .core import VeritasForensicEngine  # <--- The new Engine
import os

def upload_and_analyze(request):
    if request.method == 'POST':
        form = ForensicUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES['media_file']
            
            # 1. Save to Quarantine (Temporary Holding Area)
            # We use FileSystemStorage to safely save the file first so the Engine can read it
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'quarantine'))
            filename = fs.save(uploaded_file.name, uploaded_file)
            quarantine_path = fs.path(filename)

            try:
                # 2. Initialize the Engine
                # We use your existing RSA_KEY_PATH setting
                engine = VeritasForensicEngine(
                    input_file_path=quarantine_path,
                    private_key_path=settings.RSA_KEY_PATH 
                )

                # --- DUPLICATE CHECK OPTIMIZATION ---
                # We can verify the hash before running the heavy AI analysis
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
                    vault_path=result['vault_location'], # Engine moved it to the Vault
                    ai_confidence_score=result['analysis']['ai_confidence'],
                    rsa_signature=result['signature']
                )

                return render(request, 'forensics/report.html', {
                    'status': 'Success',
                    'report': report,
                    'metadata': result['analysis']
                })

            except Exception as e:
                # If the pipeline crashes, show the error
                return render(request, 'forensics/upload.html', {'form': form, 'error': str(e)})

    else:
        form = ForensicUploadForm()

    return render(request, 'forensics/upload.html', {'form': form})