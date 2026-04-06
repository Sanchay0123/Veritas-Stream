import os
import re
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .models import ForensicReport
from .forms import ForensicUploadForm 
from .core import VeritasForensicEngine

engine = VeritasForensicEngine(private_key_path=settings.RSA_KEY_PATH)

def get_guaranteed_url(report_vault_path, django_filename):
    """
    🧠 The Auto-Locator: Hunts down the physical image file no matter 
    what Django or the Engine did to its name or location.
    """
    raw_path = str(report_vault_path).replace('\\', '/')
    
    # Extract expected path
    if 'vault/' in raw_path:
        expected_rel_path = 'vault/' + raw_path.split('vault/')[-1]
    else:
        expected_rel_path = f"vault/{django_filename}"

    # Scenario A: File is exactly where the Engine says it is (Vault)
    if os.path.exists(os.path.join(settings.MEDIA_ROOT, expected_rel_path)):
        return f"/media/{expected_rel_path}"

    # Scenario B: File is still sitting in Quarantine (Engine didn't move it)
    if os.path.exists(os.path.join(settings.MEDIA_ROOT, 'quarantine', django_filename)):
        return f"/media/quarantine/{django_filename}"

    # Scenario C: Engine stripped Django's collision suffix (_Z8HkEN2) during the move
    clean_filename = re.sub(r'_[a-zA-Z0-9]{7}\.', '.', django_filename)
    clean_rel_path = expected_rel_path.replace(django_filename, clean_filename)
    if os.path.exists(os.path.join(settings.MEDIA_ROOT, clean_rel_path)):
        return f"/media/{clean_rel_path}"

    # Fallback (If all fails, let the browser try the original path)
    return f"/media/{expected_rel_path}"


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
                    # Clean up the duplicate from quarantine
                    if os.path.exists(quarantine_path):
                        os.remove(quarantine_path) 
                    
                    # 🎨 Consumer UI Translation (For Existing Reports)
                    score = existing_report.ai_confidence_score
                    is_fake = score >= 0.50
                    
                    # 🚀 Hit the Auto-Locator
                    final_image_url = get_guaranteed_url(existing_report.vault_path, existing_report.file_name)
                    
                    return render(request, 'forensics/report.html', {
                        'status': 'Exists',
                        'report': existing_report,
                        'verdict': 'AI GENERATED' if is_fake else 'AUTHENTIC',
                        'confidence': f"{score * 100:.1f}%",
                        'is_fake': is_fake,
                        'image_url': final_image_url
                    })

                # 3. Execute Pipeline
                result = engine.execute_full_pipeline(quarantine_path)

                # 4. Save Report
                report, created = ForensicReport.objects.get_or_create(
                    sha256_hash=result['hash'],
                    defaults={
                        'file_name': filename,
                        'vault_path': result['vault_location'],
                        'ai_confidence_score': result['analysis']['ai_confidence'],
                        'rsa_signature': result['signature']
                    }
                )

                # 5. 🎨 Consumer UI Translation (For New Reports)
                score = report.ai_confidence_score
                is_fake = score >= 0.50

                # 🚀 Hit the Auto-Locator
                final_image_url = get_guaranteed_url(report.vault_path, report.file_name)

                return render(request, 'forensics/report.html', {
                    'status': 'Success',
                    'report': report,
                    'verdict': 'AI GENERATED' if is_fake else 'AUTHENTIC',
                    'confidence': f"{score * 100:.1f}%",
                    'is_fake': is_fake,
                    'image_url': final_image_url
                })

            except Exception as e:
                print(f"❌ PIPELINE CRASHED: {str(e)}")
                return render(request, 'forensics/upload.html', {'form': form, 'error': str(e)})

    else:
        form = ForensicUploadForm()

    return render(request, 'forensics/upload.html', {'form': form})