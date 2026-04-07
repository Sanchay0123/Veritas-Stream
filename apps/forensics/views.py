import os
import re
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
                # We generate the hash early to check VirusTotal and Duplicates
                temp_hash = engine.get_file_hash(quarantine_path)
                
                is_safe, security_msg = sterilize_file(quarantine_path, temp_hash)
                
                if not is_safe:
                    # Instant Nuke: Remove malicious/malformed file
                    if os.path.exists(quarantine_path):
                        os.remove(quarantine_path)
                    
                    print(f"🚨 SECURITY INTERCEPTION: {security_msg}")
                    return render(request, 'forensics/upload.html', {
                        'form': form, 
                        'error': security_msg 
                    })

                # 2. Duplicate Detection (Proceeds only if file is CLEAN)
                file_hash = temp_hash
                existing_report = ForensicReport.objects.filter(sha256_hash=file_hash).first()
                
                if existing_report:
                    # Clean up the upload; we already have this data
                    if os.path.exists(quarantine_path):
                        os.remove(quarantine_path) 
                    
                    final_image_url = get_guaranteed_url(existing_report.vault_path, existing_report.file_name)

                    # 🕵️‍♂️ Extract Metadata for the existing archived file
                    abs_vault_path = os.path.join(settings.MEDIA_ROOT, str(existing_report.vault_path))
                    meta_data, risk_tags = extract_deep_metadata(abs_vault_path)

                    # Verify RSA Signature to ensure no one manually changed the DB score
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
                        'metadata': meta_data,
                        'risk_tags': risk_tags
                    })

                # 3. Execute Pipeline (AI Gauntlet + Move to Vault)
                result = engine.execute_full_pipeline(quarantine_path)

                # 🛡️ 3.5: Deep Metadata Extraction (Post-Vault Move)
                abs_vault_path = os.path.join(settings.MEDIA_ROOT, result['vault_location'])
                meta_data, risk_tags = extract_deep_metadata(abs_vault_path)

                # 4. Save New Report to Database
                report, created = ForensicReport.objects.get_or_create(
                    sha256_hash=result['hash'],
                    defaults={
                        'file_name': filename,
                        'vault_path': result['vault_location'],
                        'ai_confidence_score': result['analysis']['ai_confidence'],
                        'rsa_signature': result['signature']
                    }
                )

                # 5. Final UI Translation
                score = report.ai_confidence_score
                is_fake = score >= 0.50
                final_image_url = get_guaranteed_url(report.vault_path, report.file_name)

                return render(request, 'forensics/report.html', {
                    'status': 'Success',
                    'report': report,
                    'verdict': 'AI GENERATED' if is_fake else 'AUTHENTIC',
                    'confidence': f"{score * 100:.1f}%",
                    'is_fake': is_fake,
                    'image_url': final_image_url,
                    'metadata': meta_data,
                    'risk_tags': risk_tags
                })

            except Exception as e:
                print(f"❌ PIPELINE CRASHED: {str(e)}")
                return render(request, 'forensics/upload.html', {'form': form, 'error': str(e)})

    else:
        form = ForensicUploadForm()

    return render(request, 'forensics/upload.html', {'form': form})


def export_pdf_report(request, report_id):
    """
    📄 The Notarizer: Generates a tamper-proof PDF forensic report.
    """
    try:
        report = ForensicReport.objects.get(id=report_id)
        
        # Re-extract metadata for the PDF context
        abs_path = os.path.join(settings.MEDIA_ROOT, str(report.vault_path))
        metadata, risk_tags = extract_deep_metadata(abs_path)
        
        context = {
            'report': report,
            'metadata': metadata,
            'risk_tags': risk_tags,
            'verdict': 'AI GENERATED' if report.ai_confidence_score >= 0.5 else 'AUTHENTIC',
            'confidence': f"{report.ai_confidence_score * 100:.1f}%",
            'analysis_date': timezone.now().strftime("%Y-%m-%d %H:%M:%S %Z"),
        }

        # Render HTML to PDF
        template = get_template('forensics/pdf_template.html')
        html = template.render(context)
        result = BytesIO()
        
        # The pisaDocument function does the heavy lifting
        pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8")), result)
        
        if not pdf.err:
            response = HttpResponse(result.getvalue(), content_type='application/pdf')
            # 'attachment' forces the browser to download it
            response['Content-Disposition'] = f'attachment; filename="Veritas_Forensic_Report_{report.id}.pdf"'
            return response
        
        return HttpResponse("❌ PDF Engine Error: Could not generate document.", status=500)

    except ForensicReport.DoesNotExist:
        return HttpResponse("❌ Error: Report not found in the archive.", status=404)
    except Exception as e:
        return HttpResponse(f"❌ Critical Error: {str(e)}", status=500)