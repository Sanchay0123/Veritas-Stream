from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import ForensicReport
from .forms import ForensicUploadForm 
from .core import VeritasForensicEngine
import os

engine = VeritasForensicEngine(private_key_path=settings.RSA_KEY_PATH)

def upload_and_analyze(request):
    if request.method == 'POST':
        form = ForensicUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES.get('media_file')
            
            # 1. Save to Quarantine
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'quarantine'))
            filename = fs.save(uploaded_file.name, uploaded_file)
            quarantine_path = fs.path(filename)

            try:
                # 2. Pre-check for duplicates
                file_hash = engine.get_file_hash(quarantine_path)
                existing_report = ForensicReport.objects.filter(sha256_hash=file_hash).first()
                
                if existing_report:
                    os.remove(quarantine_path) # Clean up
                    
                    # 🎨 Consumer UI Translation (For Existing Reports)
                    score = existing_report.ai_confidence_score
                    is_fake = score >= 0.50
                    
                    return render(request, 'forensics/report.html', {
                        'status': 'Exists',
                        'report': existing_report,
                        'verdict': 'AI GENERATED' if is_fake else 'AUTHENTIC',
                        'confidence': f"{score * 100:.1f}%",
                        'is_fake': is_fake
                    })

                # 3. Execute Pipeline
                result = engine.execute_full_pipeline(quarantine_path)

                # 4. Save Report
                report, created = ForensicReport.objects.get_or_create(
                    sha256_hash=result['hash'],
                    defaults={
                        'file_name': uploaded_file.name,
                        'vault_path': result['vault_location'],
                        'ai_confidence_score': result['analysis']['ai_confidence'],
                        'rsa_signature': result['signature']
                    }
                )

                # 5. 🎨 Consumer UI Translation (For New Reports)
                # We hide the multi-stage complexity and provide a clean verdict
                score = report.ai_confidence_score
                is_fake = score >= 0.50

                return render(request, 'forensics/report.html', {
                    'status': 'Success',
                    'report': report,
                    'verdict': 'AI GENERATED' if is_fake else 'AUTHENTIC',
                    'confidence': f"{score * 100:.1f}%",
                    'is_fake': is_fake
                })

            except Exception as e:
                print(f"❌ PIPELINE CRASHED: {str(e)}")
                return render(request, 'forensics/upload.html', {'form': form, 'error': str(e)})

    else:
        form = ForensicUploadForm()

    return render(request, 'forensics/upload.html', {'form': form})