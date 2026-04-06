import os
import re
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .models import ForensicReport
from .forms import ForensicUploadForm 
from .core import VeritasForensicEngine
from .utils import sterilize_file  # 🛡️ Defense Utility

engine = VeritasForensicEngine(private_key_path=settings.RSA_KEY_PATH)

def get_guaranteed_url(report_vault_path, django_filename):
    """
    🧠 The Auto-Locator: Hunts down the physical image file no matter 
    what Django or the Engine did to its name or location.
    """
    raw_path = str(report_vault_path).replace('\\', '/')
    
    if 'vault/' in raw_path:
        expected_rel_path = 'vault/' + raw_path.split('vault/')[-1]
    else:
        expected_rel_path = f"vault/{django_filename}"

    if os.path.exists(os.path.join(settings.MEDIA_ROOT, expected_rel_path)):
        return f"/media/{expected_rel_path}"

    if os.path.exists(os.path.join(settings.MEDIA_ROOT, 'quarantine', django_filename)):
        return f"/media/quarantine/{django_filename}"

    clean_filename = re.sub(r'_[a-zA-Z0-9]{7}\.', '.', django_filename)
    clean_rel_path = expected_rel_path.replace(django_filename, clean_filename)
    if os.path.exists(os.path.join(settings.MEDIA_ROOT, clean_rel_path)):
        return f"/media/{clean_rel_path}"

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
                # 🛡️ STEP 1.5: MALWARE STERILIZATION (The Iron Curtain)
                # Generate hash early for VirusTotal check
                temp_hash = engine.get_file_hash(quarantine_path)
                
                is_safe, security_msg = sterilize_file(quarantine_path, temp_hash)
                
                if not is_safe:
                    # Instant Nuke: Delete the malicious file immediately
                    if os.path.exists(quarantine_path):
                        os.remove(quarantine_path)
                    
                    print(f"🚨 SECURITY INTERCEPTION: {security_msg}")
                    return render(request, 'forensics/upload.html', {
                        'form': form, 
                        'error': security_msg  # Alert the user they've been blocked
                    })

                # 2. Pre-check for duplicates (Proceeds only if CLEAN)
                file_hash = temp_hash
                existing_report = ForensicReport.objects.filter(sha256_hash=file_hash).first()
                
                if existing_report:
                    if os.path.exists(quarantine_path):
                        os.remove(quarantine_path) 
                    
                    final_image_url = get_guaranteed_url(existing_report.vault_path, existing_report.file_name)

                    try:
                        is_valid = engine.verify_signature(
                            existing_report.sha256_hash, 
                            existing_report.rsa_signature,
                            existing_report.ai_confidence_score
                        )
                    except Exception as e:
                        print(f"⚠️ SIGNATURE ERROR: {str(e)}")
                        is_valid = False

                    if not is_valid:
                        return render(request, 'forensics/report.html', {
                            'status': 'BREACH_DETECTED',
                            'verdict': 'DATA COMPROMISED',
                            'confidence': 'ERR_SIG_INVALID',
                            'is_fake': True,
                            'image_url': final_image_url,
                            'error_message': 'CRYPTOGRAPHIC FAILURE: The forensic archive has been altered.'
                        })
                    
                    score = existing_report.ai_confidence_score
                    is_fake = score >= 0.50
                    
                    return render(request, 'forensics/report.html', {
                        'status': 'Exists',
                        'report': existing_report,
                        'verdict': 'AI GENERATED' if is_fake else 'AUTHENTIC',
                        'confidence': f"{score * 100:.1f}%",
                        'is_fake': is_fake,
                        'image_url': final_image_url
                    })

                # 3. Execute Pipeline (Only reached if safe and new)
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

                # 5. UI Translation
                score = report.ai_confidence_score
                is_fake = score >= 0.50
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