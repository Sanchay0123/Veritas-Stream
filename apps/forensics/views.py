import os
import re
import boto3
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage

from .models import ForensicReport
from .forms import ForensicUploadForm 
from .core import VeritasForensicEngine
from .utils import sterilize_file, extract_deep_metadata  # 🛡️ Global Security Utils

from io import BytesIO
from django.http import HttpResponse
from xhtml2pdf import pisa
from django.template.loader import get_template
from django.utils import timezone

# Initialize the Forensic Engine with your RSA Private Key
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
            
            # 1. Save to Quarantine (The Airlock)
            fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'quarantine'))
            filename = fs.save(uploaded_file.name, uploaded_file)
            quarantine_path = fs.path(filename)

            try:
                # 🛡️ STEP 1.5: MALWARE STERILIZATION (The Iron Curtain)
                temp_hash = engine.get_file_hash(quarantine_path)
                is_safe, security_msg = sterilize_file(quarantine_path, temp_hash)
                
                if not is_safe:
                    if os.path.exists(quarantine_path):
                        os.remove(quarantine_path)
                    
                    print(f"🚨 SECURITY INTERCEPTION: {security_msg}")
                    return render(request, 'forensics/upload.html', {
                        'form': form, 
                        'error': security_msg 
                    })

                # 2. Duplicate Detection
                file_hash = temp_hash
                existing_report = ForensicReport.objects.filter(sha256_hash=file_hash).first()
                
                if existing_report:
                    if os.path.exists(quarantine_path):
                        os.remove(quarantine_path) 
                    
                    final_image_url = get_guaranteed_url(existing_report.vault_path, existing_report.file_name)

                    # 🕵️‍♂️ Unpack 3 values (metadata, risk_tags, info_flags)
                    abs_vault_path = os.path.join(settings.MEDIA_ROOT, str(existing_report.vault_path))
                    meta_data, risk_tags, info_flags = extract_deep_metadata(abs_vault_path)

                    # 🗺️ NEW: Hunt for the historical ELA Heatmap
                    historical_ela_url = None
                    expected_ela_filename = f"ELA_{existing_report.file_name}"
                    expected_ela_path = os.path.join(os.path.dirname(abs_vault_path), expected_ela_filename)
                    
                    if os.path.exists(expected_ela_path):
                        raw_dir = str(existing_report.vault_path).replace('\\', '/')
                        if 'vault/' in raw_dir:
                            rel_dir = 'vault/' + raw_dir.split('vault/')[-1]
                            rel_dir = os.path.dirname(rel_dir)
                            historical_ela_url = f"/media/{rel_dir}/{expected_ela_filename}"

                    # Verify RSA Signature
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
                            'error_message': 'CRYPTOGRAPHIC FAILURE: This record has been altered since notarization.'
                        })
                    
                    score = existing_report.ai_confidence_score
                    is_fake = score >= 0.50
                    
                    return render(request, 'forensics/report.html', {
                        'status': 'Exists',
                        'report': existing_report,
                        'verdict': 'AI GENERATED' if is_fake else 'AUTHENTIC',
                        'confidence': f"{score * 100:.1f}%",
                        'is_fake': is_fake,
                        'image_url': final_image_url,
                        'ela_heatmap_url': historical_ela_url, # 👈 Added ELA to historical duplicates
                        'metadata': meta_data,
                        'risk_tags': risk_tags,
                        'info_flags': info_flags 
                    })

                # 3. Execute Pipeline (AI Gauntlet + Move to Vault)
                result = engine.execute_full_pipeline(quarantine_path)

                # 🛡️ 3.5: Deep Metadata Extraction (Post-Vault Move)
                abs_vault_path = os.path.join(settings.MEDIA_ROOT, result['vault_location'])
                meta_data, risk_tags, info_flags = extract_deep_metadata(abs_vault_path)

                # 4. Save New Report
                report, created = ForensicReport.objects.get_or_create(
                    sha256_hash=result['hash'],
                    defaults={
                        'file_name': filename,
                        'vault_path': result['vault_location'],
                        'ai_confidence_score': result['analysis']['ai_confidence'],
                        'rsa_signature': result['signature']
                    }
                )

                # Comment out if s3 is inactive
                # ☁️ STEP 4.5: THE REDUNDANT MIRROR (AWS S3 Backup)
                abs_vault_path = os.path.join(settings.MEDIA_ROOT, str(report.vault_path))
                backup_to_s3(abs_vault_path, report.file_name)

                # 5. Final UI Translation
                score = report.ai_confidence_score
                is_fake = score >= 0.50
                final_image_url = get_guaranteed_url(report.vault_path, report.file_name)

                # 🗺️ Format the ELA Heatmap URL for the frontend
                ela_heatmap_url = None
                if result.get('ela_heatmap'):
                    ela_heatmap_url = f"/media/{result['ela_heatmap']}"

                return render(request, 'forensics/report.html', {
                    'status': 'Success',
                    'report': report,
                    'verdict': 'AI GENERATED' if is_fake else 'AUTHENTIC',
                    'confidence': f"{score * 100:.1f}%",
                    'is_fake': is_fake,
                    'image_url': final_image_url,
                    'ela_heatmap_url': ela_heatmap_url,
                    'metadata': meta_data,
                    'risk_tags': risk_tags,
                    'info_flags': info_flags 
                })

            except Exception as e:
                print(f"❌ PIPELINE CRASHED: {str(e)}")
                return render(request, 'forensics/upload.html', {'form': form, 'error': str(e)})

    else:
        form = ForensicUploadForm()

    return render(request, 'forensics/upload.html', {'form': form})


def backup_to_s3(local_vault_path, s3_file_name):
    """
    ☁️ The Redundant Mirror: Copies the clean file to AWS S3 (Learner's Lab Edition)
    """
    try:
        s3 = boto3.client(
            's3', 
            region_name=settings.AWS_S3_REGION_NAME,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID, 
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            aws_session_token=settings.AWS_SESSION_TOKEN  
        )
                          
        s3.upload_file(local_vault_path, settings.AWS_STORAGE_BUCKET_NAME, s3_file_name)
        print(f"☁️ [Backup] Successfully mirrored {s3_file_name} to AWS S3.")
        
    except Exception as e:
        print(f"⚠️ [Backup Warning] S3 Mirror failed: {e}. File is safe locally.")

def export_pdf_report(request, report_id):
    """
    📄 The Notarizer: Generates a tamper-proof PDF forensic report.
    """
    try:
        report = ForensicReport.objects.get(id=report_id)
        
        # 🕵️‍♂️ Unpack 3 values to prevent ValueError
        abs_path = os.path.join(settings.MEDIA_ROOT, str(report.vault_path))
        metadata, risk_tags, info_flags = extract_deep_metadata(abs_path)
        
        context = {
            'report': report,
            'metadata': metadata,
            'risk_tags': risk_tags,
            'info_flags': info_flags,
            'verdict': 'AI GENERATED' if report.ai_confidence_score >= 0.5 else 'AUTHENTIC',
            'confidence': f"{report.ai_confidence_score * 100:.1f}%",
            'analysis_date': timezone.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        }

        # Render HTML to PDF
        template = get_template('forensics/pdf_template.html')
        html = template.render(context)
        result = BytesIO()
        
        pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), result)
        
        if not pdf.err:
            response = HttpResponse(result.getvalue(), content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="Veritas_Forensic_Report_{report.id}.pdf"'
            return response
        
        return HttpResponse("❌ PDF Engine Error: Could not generate document.", status=500)

    except ForensicReport.DoesNotExist:
        return HttpResponse("❌ Error: Report not found in the archive.", status=404)
    except Exception as e:
        return HttpResponse(f"❌ Critical Error: {str(e)}", status=500)

def landing_page(request):
    """
    🚀 The Front Door: Introductory landing page for Veritas Stream.
    """
    return render(request, 'landing.html')