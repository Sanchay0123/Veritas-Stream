from django.shortcuts import render
import hashlib
import os
from .models import ForensicReport
from .forms import ForensicUploadForm 

# 🛰️ THE CRITICAL CHANGE: Pointing to the correct Cloud-Native Engine
from apps.analysis.core import engine 

def upload_and_analyze(request):
    if request.method == 'POST':
        form = ForensicUploadForm(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = request.FILES.get('media_file')
            
            try:
                # ⚡ 1. In-Memory Duplicate Check
                sha256_hash = hashlib.sha256()
                for chunk in uploaded_file.chunks():
                    sha256_hash.update(chunk)
                file_hash = sha256_hash.hexdigest()
                
                # Rewind the file stream so the engine can read it from the start
                uploaded_file.seek(0) 

                existing_report = ForensicReport.objects.filter(sha256_hash=file_hash).first()
                
                if existing_report:
                    score = existing_report.ai_confidence_score
                    is_fake = score >= 0.50
                    
                    return render(request, 'forensics/report.html', {
                        'status': 'Exists',
                        'report': existing_report,
                        'verdict': 'AI GENERATED' if is_fake else 'AUTHENTIC',
                        'confidence': f"{score * 100:.1f}%",
                        'is_fake': is_fake
                    })

                # ☁️ 2. Execute the AWS Double-Triage Pipeline
                # This triggers Stage 0 (Metadata), Stage 1 (ELA), and Stage 2-4 (AI Models)
                result = engine.process_cloud_triage(uploaded_file)

                # 🗄️ 3. Save the Database Report
                report = ForensicReport.objects.create(
                    sha256_hash=file_hash,
                    file_name=uploaded_file.name,
                    vault_path=result.get('vault_path'), 
                    ai_confidence_score=result.get('ai_confidence_score')
                )

                # 🎨 4. Final Verdict for UI
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
                # This will now print the REAL error if AWS or the AI fails
                print(f"❌ CLOUD PIPELINE CRASHED: {str(e)}")
                return render(request, 'forensics/upload.html', {
                    'form': form, 
                    'error': f"Forensic Engine Error: {str(e)}"
                })

    else:
        form = ForensicUploadForm()

    return render(request, 'forensics/upload.html', {'form': form})