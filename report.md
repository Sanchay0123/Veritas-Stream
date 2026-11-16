Ran list directory contents
Ran inspect directory tree · Exited with code 127
Ran generate directory tree
Ran check weights, models, and datasets
Ran check storage directory files
Ran check apps/storage directory files
Ran grep dataset in train.py
Ran list storage/dataset
Ran read .gitignore
Ran find forensic templates
Ran count lines in target files
Ran count lines in templates and analysis core
Ran list keys directory
Analyzed core.py:L1-236
Analyzed core.py:L1-193
Analyzed models.py:L1-29
Analyzed views.py:L1-262
Analyzed utils.py:L1-132
Analyzed forms.py:L1-14
Analyzed debug_veritas.py:L1-68
Analyzed inject.py:L1-81
Analyzed settings.py:L1-120
Analyzed base.py:L1-17
Analyzed urls.py:L1-25
Analyzed mount_vault.sh:L1-54
Analyzed README.md:L1-71
Analyzed pdf_template.html:L1-59
Analyzed report.html:L1-331
Ran list notebooks directory
Ran list apps/core and apps/users
Analyzed apps.py:L1-6
Analyzed 0001_initial.py:L1-31
Ran check upload template header

# Technical Architecture & Integration Report: Veritas-Stream

**Audit Status:** Complete  
**Operating Constraint:** Strict Read-Only Mode (Zero workspace file mutations, zero script executions, zero secret leaks).

---

## 1. Project Layout & Model/Dataset Storage

### 1.1 Architectural Directory Hierarchy
The project uses the **`apps/analysis` + `apps/forensics`** structure (namespaced under the `apps/` directory). In `config/settings.py`, `INSTALLED_APPS` explicitly registers `'apps.forensics'` and `'apps.analysis'`, and `apps/forensics/apps.py` specifies `name = 'apps.forensics'`, `label = 'forensics'`.

```
veritas_stream/
├── .env                          # Local environment secrets (read-only, uncommitted)
├── .gitignore                    # Excludes heavy datasets, weights, vaults, virtualenvs
├── README.md                     # Platform overview & 4-stage pipeline documentation
├── manage.py                     # Django CLI management entrypoint
├── debug_veritas.py              # MTCNN face extraction & diagnostic visualization script
├── inject.py                     # Dataset augmentation utility (GAN -> Diffusion fakes)
├── mint_brain.py                 # Weight minting / packaging script
├── mount_vault.sh                # WSL2 DrvFS drive mount & bind-mount script for 1TB vault
├── requirements.txt              # Python runtime dependencies
├── train.py                      # SmartDataset + ViT training pipeline
├── db.sqlite3                    # Local SQLite database (ForensicReport ledger)
├── apps/
│   ├── analysis/                 # AI & Deepfake Detection Engine
│   │   ├── core.py               # EnterpriseDeepFakeDetector / DeepFakeDetector
│   │   ├── networks.py           # Neural architectures (Meso4, etc.)
│   │   ├── ai_models/            # Directory reserved for serialized AI model definitions
│   │   └── weights/              # Serialized PyTorch weight matrices (.pkl checkpoints)
│   ├── forensics/                # Forensic Orchestration, Notarization & Web UI
│   │   ├── apps.py               # ForensicsConfig app specification
│   │   ├── core.py               # VeritasForensicEngine (Hash, ELA, RSA, Vault)
│   │   ├── forms.py              # ForensicUploadForm evidence validator
│   │   ├── keys/                 # RSA key directory
│   │   ├── migrations/           # Database migration files (0001_initial.py)
│   │   ├── models.py             # ForensicReport schema (Chain of Custody)
│   │   ├── templates/            # Forensic UI and PDF export templates
│   │   │   ├── landing.html
│   │   │   └── forensics/
│   │   │       ├── pdf_template.html
│   │   │       ├── report.html
│   │   │       └── upload.html
│   │   ├── utils.py              # Iron Curtain sterilizer & deep EXIF metadata scraper
│   │   └── views.py              # Upload controller, duplicate tripwire, PDF generator
│   ├── storage/                  # Active Django media storage (settings.MEDIA_ROOT)
│   │   ├── quarantine/           # Pre-analysis airlock for malware sterilization
│   │   └── vault/                # Secure notarized storage organized as YYYY/MM/DD
│   ├── core/                     # Reserved core package
│   └── users/                    # Reserved authentication package
├── config/
│   ├── __init__.py
│   ├── settings.py               # Active Django settings (reads from .env via django-environ)
│   ├── urls.py                   # URL routing & tunnel media serving
│   ├── wsgi.py                   # WSGI deployment configuration
│   └── settings/
│       └── base.py               # Base settings fallback
├── storage/                      # Top-level workspace storage & training cache
│   ├── dataset/                  # Primary dataset root (referenced in training scripts)
│   ├── modern_fakes/             # Diffusion fakes staging
│   ├── temp_gan/                 # Unzipped GAN training splits (train.csv, valid.csv, test.csv)
│   ├── quarantine/               # Workspace quarantine buffer
│   ├── vault/                    # External physical drive bind mount target
│   ├── media/                    # Media root
│   └── logs/                     # System logs
└── notebooks/                    # Research and exploratory Jupyter notebooks
```

---

### 1.2 Model Weights & Dataset Storage Audit

#### `apps/analysis/weights/`
Contains 9 pre-trained weight checkpoints:
| Filename | File Size (MB / KB) | Exact Bytes | Description |
| :--- | :--- | :--- | :--- |
| `veritas_vit.pkl` | **327.36 MB** | 343,261,746 B | Vision Transformer (`vit_b_16`) full forensic model weights |
| `veritas_noise.pkl` | **15.59 MB** | 16,343,850 B | Noise-residual frequency domain expert weights |
| `veritas_pro1.pkl` | **15.58 MB** | 16,334,226 B | Production ensemble candidate iteration 1 |
| `veritas_pro.pkl` | **15.58 MB** | 16,334,226 B | Active production CNN expert weights |
| `veritas_spectral1.pkl` | **15.59 MB** | 16,344,942 B | Spectral FFT artifact detector checkpoint 1 |
| `veritas_pro_backup.pkl` | **15.58 MB** | 16,334,226 B | Production backup checkpoint |
| `veritas_fresh.pkl` | **119.14 KB** | 121,998 B | Lightweight / partial head checkpoint |
| `veritas_forensic.pkl` | **15.59 MB** | 16,345,333 B | Calibrated forensic classification checkpoint |
| `veritas_spectral.pkl` | **15.59 MB** | 16,344,942 B | Spectral FFT artifact detector primary checkpoint |

#### `apps/analysis/ai_models/`
- **Status:** Empty directory (0 files). HuggingFace pipelines (`dima806/deepfake_vs_real_image_detection` and `Organika/sdxl-detector`) and MTCNN are pulled dynamically into the PyTorch/HuggingFace cache, while custom weights are housed in `weights/`.

#### `storage/dataset/`
- **Status:** Empty folder in local workspace tree (0 images). Heavy datasets are excluded via `.gitignore` (`storage/`, `apps/storage/`, `*.pkl`).
- **Expected Data Layout (from `train.py` & `inject.py`):**
  - **Path:** `storage/dataset/130k_faces/images/`
  - **Structure:** Standard `torchvision.datasets.ImageFolder` format with two class directories:
    - `storage/dataset/130k_faces/images/real/` (Authentic human portrait dataset)
    - `storage/dataset/130k_faces/images/fake/` (Synthetic portrait dataset: Diffusion fakes + 20,000 injected StyleGAN images prefixed with `gan_stylegan_` from `storage/temp_gan/real_vs_fake/...`, aiming for an ~80,000 sample mixed synthetic set).

---

## 2. Full Source Code of Unseen Core Python Files

### 2.1 `apps/forensics/core.py`
```python
import os
import cv2
import hashlib
import base64
import shutil
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from django.conf import settings
from django.utils import timezone

# 🔬 NEW: Added Pillow imports for Forensic ELA Math
from PIL import Image, ImageChops, ImageEnhance

# Import the AI Brain
from apps.analysis.core import DeepFakeDetector 

class VeritasForensicEngine:
    # 🛑 LAZY LOAD CACHE: Stores the AI in memory across all requests
    _cached_detector = None

    def __init__(self, private_key_path=None):
        self.private_key_path = private_key_path
        self.timestamp = timezone.now()

    def _get_detector(self):
        """Loads the AI ONLY when needed, then saves it for future use."""
        if VeritasForensicEngine._cached_detector is None:
            VeritasForensicEngine._cached_detector = DeepFakeDetector()
        return VeritasForensicEngine._cached_detector

    def get_file_hash(self, input_path):
        """Step 1: Generate SHA-256 Hash"""
        sha256 = hashlib.sha256()
        with open(input_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def analyze_media(self, input_path):
        """Step 2: AI Visual Analysis"""
        cap = cv2.VideoCapture(input_path)
        success, frame = cap.read()
        
        if not success:
            if input_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(input_path)
                if img is None:
                    raise ValueError("Could not read image file.")
                h, w, _ = img.shape
                res = f"{w}x{h}"
                fps = 0
                frame_count = 1
            else:
                raise ValueError("Incompatible or corrupt media file.")
        else:
            res = f"{int(cap.get(3))}x{int(cap.get(4))}"
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()

        # 🛑 Trigger Lazy Load: Only boots the AI if it hasn't been booted yet
        detector = self._get_detector()
        real_confidence = detector.analyze(input_path)

        metadata = {
            "resolution": res,
            "fps": fps,
            "frame_count": frame_count,
            "ai_confidence": real_confidence 
        }
        return metadata

    def sign_report(self, data_to_sign):
        """Step 3: RSA Notarization"""
        if not self.private_key_path or not os.path.exists(self.private_key_path):
            return "NO_KEY_FOUND"

        with open(self.private_key_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(), password=None
            )

        signature = private_key.sign(
            data_to_sign.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')

    def verify_signature(self, file_hash, signature_b64, ai_confidence):
        """
        🚨 THE TRIPWIRE: Reconstructs the 'envelope' and verifies it 
        against the RSA Public Key.
        """
        if not self.private_key_path or not os.path.exists(self.private_key_path):
            return False

        try:
            # 1. Reconstruct the EXACT string used during signing
            data_to_verify = f"Hash:{file_hash}|Result:{ai_confidence}"
            
            # 2. Load Public Key from your Private Key
            with open(self.private_key_path, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(), password=None
                )
            public_key = private_key.public_key()

            # 3. Decode and Verify
            signature_bytes = base64.b64decode(signature_b64)
            public_key.verify(
                signature_bytes,
                data_to_verify.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            print(f"🛑 CRYPTOGRAPHIC VERIFICATION FAILED: {e}")
            return False

    def archive_to_vault(self, input_path):
        """Step 4: Secure Storage on External Drive"""
        filename = os.path.basename(input_path)
        
        vault_root = os.path.join(settings.MEDIA_ROOT, 'vault')
        
        if os.path.exists(vault_root) and not os.path.isdir(vault_root):
            os.remove(vault_root)
            
        os.makedirs(vault_root, exist_ok=True)

        date_path = self.timestamp.strftime('%Y/%m/%d')
        vault_dir = os.path.join(vault_root, date_path)
        os.makedirs(vault_dir, exist_ok=True)
        
        vault_path = os.path.join(vault_dir, filename)

        try:
            shutil.copyfile(input_path, vault_path)
            if os.path.getsize(vault_path) == os.path.getsize(input_path):
                os.remove(input_path)
            else:
                raise Exception("Copy size mismatch")
        except Exception as e:
            print(f"⚠️ Vault Transfer Failed: {e}")
            raise e

        return vault_path

    def generate_ela_heatmap(self, image_path, output_dir, quality=90, multiplier=15):
        """
        🔬 The Visual Veto: Generates an ELA heatmap to visualize compression artifacts.
        """
        try:
            original = Image.open(image_path).convert('RGB')
            
            # STEP 1: The Resave
            temp_filename = os.path.join(output_dir, "temp_ela_buffer.jpg")
            original.save(temp_filename, 'JPEG', quality=quality)
            compressed = Image.open(temp_filename)
            
            # STEP 2: The Subtraction
            diff = ImageChops.difference(original, compressed)
            
            # STEP 3: The Amplification
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1 
                
            enhancer = ImageEnhance.Brightness(diff)
            heatmap = enhancer.enhance(multiplier)
            
            # STEP 4: Save to the Vault next to the original
            base_name = os.path.basename(image_path)
            heatmap_filename = f"ELA_{base_name}"
            heatmap_path = os.path.join(output_dir, heatmap_filename)
            
            heatmap.save(heatmap_path, 'JPEG', quality=100)
            
            # 🧹 Cleanup
            compressed.close()
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
            print(f"   [Forensic Math] 🗺️ ELA Heatmap generated: {heatmap_filename}")
            
            # Return the relative path so Django can serve it easily via /media/
            return heatmap_path
            
        except Exception as e:
            print(f"❌ [Forensic Math] ELA Heatmap Generation Failed: {e}")
            return None

    def execute_full_pipeline(self, input_path):
        """The Big Red Button"""
        file_hash = self.get_file_hash(input_path)
        analysis = self.analyze_media(input_path)
        
        report_summary = f"Hash:{file_hash}|Result:{analysis['ai_confidence']}"
        signature = self.sign_report(report_summary)
        
        # Move the clean file to the Vault FIRST
        vault_path = self.archive_to_vault(input_path)
        
        # 🗺️ Generate the ELA Heatmap directly in the Vault directory
        vault_dir = os.path.dirname(vault_path)
        
        # Only attempt ELA on images (skip videos for now to save processing time)
        ela_path = None
        if vault_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            ela_path = self.generate_ela_heatmap(vault_path, vault_dir)
        
        # Extract just the relative portion starting from 'vault/' so it works with your URLs
        if ela_path:
            ela_rel_path = 'vault/' + ela_path.split('vault/')[-1]
        else:
            ela_rel_path = None
        
        return {
            "hash": file_hash,
            "signature": signature,
            "vault_location": vault_path,
            "ela_heatmap": ela_rel_path,  # 👈 Now available for your views.py to pass to the UI
            "analysis": analysis
        }
```

---

### 2.2 `apps/forensics/models.py`
```python
from django.db import models
import uuid

# This table is the "Chain of Custody" for every file that touches the platform.

class ForensicReport(models.Model):
    # Unique ID for the report itself
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # File Metadata
    file_name = models.CharField(max_length=255)
    sha256_hash = models.CharField(max_length=64, unique=True, db_index=True)
    
    # Path inside the 1TB vault (relative to MEDIA_ROOT)
    vault_path = models.CharField(max_length=500)
    
    # Analysis Verdict
    ai_confidence_score = models.FloatField()
    is_altered = models.BooleanField(default=False)
    
    # Notarization (The Security Layer)
    rsa_signature = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report {self.sha256_hash[:8]} | {self.file_name}"

    class Meta:
        ordering = ['-timestamp']
```

---

### 2.3 `apps/forensics/views.py`
```python
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
```

---

### 2.4 `apps/forensics/utils.py`
```python
import magic
import subprocess
import requests
from PIL import Image # Moved to top for efficiency
from django.conf import settings
import exifread
import os

def sterilize_file(file_path, file_hash):
    """
    🛡️ The Iron Curtain: 4-Stage Malware & Integrity Sterilization
    """
    # 1. MAGIC BYTE CHECK (The Bouncer)
    mime = magic.Magic(mime=True)
    detected_type = mime.from_file(file_path)
    print(f"🛡️  [Bouncer] Detected MIME: {detected_type}")
    
    if not detected_type.startswith('image/'):
        print(f"❌ [Bouncer] Blocked: {detected_type} is not a valid image.")
        return False, f"SECURITY ALERT: Magic Bytes suggest {detected_type}, not an image."

    # 2. VIRUSTOTAL HASH CHECK (Global Check)
    print(f"🌐 [Global Check] Querying VirusTotal for: {file_hash[:12]}...")
    vt_url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
    headers = {"x-apikey": settings.VT_API_KEY}
    
    try:
        response = requests.get(vt_url, headers=headers, timeout=5)
        if response.status_code == 200:
            stats = response.json()['data']['attributes']['last_analysis_stats']
            malicious_count = stats.get('malicious', 0)
            print(f"✅ [Global Check] VT hits found: {malicious_count}")
            if malicious_count > 0:
                return False, f"THREAT DETECTED: VirusTotal flagged this hash ({malicious_count} hits)."
        elif response.status_code == 404:
            print(f"ℹ️  [Global Check] Hash not found in VT database (Clean/New).")
    except Exception as e:
        print(f"⚠️  [Global Check] VT API Timeout or Error: {e}")

    # 3. CLAMAV LOCAL SCAN (Guard Dog)
    print(f"🐕 [Guard Dog] Running local clamscan...")
    try:
        result = subprocess.run(
            ['clamscan', '--allmatch', '--no-summary', file_path], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 1: 
            print(f"❌ [Guard Dog] VIRUS DETECTED by ClamAV!")
            return False, "MALWARE DETECTED: Local antivirus flagged the file buffer."
        print(f"✅ [Guard Dog] Local scan complete.")
    except Exception as e:
        print(f"❌ [Guard Dog] ClamAV Execution Error: {e}")

    # 4. STRUCTURAL INTEGRITY CHECK (The Insurance Policy)
    # This prevents the 'No JPEG data found' crash in the AI pipeline
    print(f"🔬 [Integrity] Validating image structure...")
    try:
        with Image.open(file_path) as img:
            img.verify() 
        print("✅ [Integrity] File structure is valid.")
    except Exception as e:
        print(f"❌ [Integrity] Malformed Image detected: {e}")
        return False, "CORRUPT DATA: File header is valid, but image data is unreadable or malformed."

    # --- ALL GATES CLEARED ---
    print("🏁 [Sterilizer] File cleared all security gates. Proceeding to AI Gauntlet.")
    return True, "CLEAN"




def extract_deep_metadata(file_path):
    """
    🕵️‍♂️ Deep Metadata Scraper: Prioritizing AI Detection over Privacy Scrubbing.
    """
    metadata = {}
    risk_flags = []
    info_flags = []
    
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f)
            
            # --- 🛡️ REFINED: AI vs. Social Media Logic ---
            if not tags:
                ext = file_path.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg']:
                    # 🟡 Yellow Alert: Logic in reports.html looks for "SCRUBBED"
                    risk_flags.append("SCRUBBED METADATA: Standard JPEG headers missing. Typical of WhatsApp/Discord compression.")
                else:
                    # 🔴 Red Alert: No "SCRUBBED" keyword, so template defaults to Red "Anomaly"
                    risk_flags.append("SYNTHETIC SIGNATURE: Total absence of forensic headers in a non-JPEG container. High possibility of AI generation (Gemini/Midjourney).")
            
            if tags:
                # 1. THE HARDWARE FINGERPRINT
                metadata['Make'] = str(tags.get('Image Make', 'Unknown'))
                metadata['Model'] = str(tags.get('Image Model', 'Unknown'))
                metadata['Software'] = str(tags.get('Image Software', 'None Detected'))
                
                # 🚨 RED FLAG: Software Manipulation
                software_blacklist = ['photoshop', 'gimp', 'picsart', 'midjourney', 'stable diffusion', 'dall-e']
                software_lower = metadata['Software'].lower()
                
                if any(s in software_lower for s in software_blacklist):
                    risk_flags.append(f"MANIPULATION TRACE: File edited via {metadata['Software']}")
                
                # 🚨 Specific Check: Outdated/Suspect Software
                if 'picasa' in software_lower:
                    risk_flags.append("LEGACY SOFTWARE: Use of Picasa detected. Highly unusual for modern captures; often used in AI-upscaling or legacy re-encoding.")

                # 2. THE TEMPORAL TRAP (Timestamps)
                metadata['DateTaken'] = str(tags.get('EXIF DateTimeOriginal', 'N/A'))
                metadata['DateDigitized'] = str(tags.get('EXIF DateTimeDigitized', 'N/A'))
                
                if metadata['DateTaken'] != 'N/A' and metadata['DateDigitized'] != 'N/A':
                    if metadata['DateTaken'] != metadata['DateDigitized']:
                        risk_flags.append("TEMPORAL ANOMALY: Creation and Digitization dates do not match.")

                # 3. GPS SURVEILLANCE
                has_gps = any('GPS' in key for key in tags.keys())
                metadata['Geotagged'] = "Yes" if has_gps else "No"
                if has_gps:
                    info_flags.append("LOCATION DATA: This image contains embedded GPS coordinates.")
            else:
                metadata['Header Status'] = "No metadata found in the bitstream."

    except Exception as e:
        print(f"⚠️ Metadata Extraction Error: {e}")
        return {"error": "Failed to parse EXIF data."}, []

    return metadata, risk_flags, info_flags
```

---

### 2.5 `apps/forensics/forms.py`
```python
from django import forms
from django.core.validators import FileExtensionValidator

class ForensicUploadForm(forms.Form):
    media_file = forms.FileField(
        label="Upload Evidence",
        help_text="Supported formats: JPG, PNG, MP4, AVI, MOV",
        validators=[
            FileExtensionValidator(
                allowed_extensions=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi']
            )
        ],
        widget=forms.ClearableFileInput(attrs={'class': 'file-input'})
    )
```

---

### 2.6 `debug_veritas.py`
```python
import torch
from apps.analysis.core import DeepFakeDetector
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# CHANGE THIS to your image path
TEST_IMAGE = r"/mnt/c/Users/Lenovo/Desktop/random codes/Design Thinking and Innovation/Images/test2.jpg"

def debug_image(image_path):
    print(f"🔍 Debugging: {image_path}")
    detector = DeepFakeDetector()
    
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        print("❌ Could not open image.")
        return

    print("🕵️ Scanning for faces...")
    boxes, _ = detector.face_detector.detect(img)
    
    if boxes is None:
        print("❌ NO FACES DETECTED.")
        return

    print(f"✅ Found {len(boxes)} faces.")
    
    # Draw setup
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        # 1. Capture the Crop
        face_crop = detector.make_ffhq_crop(img, box)
        
        # 2. Check Resolution
        w, h = face_crop.size
        print(f"\n👤 Face {i+1} Resolution: {w}x{h} pixels")
        
        if w < 100 or h < 100:
            print("   ⚠️ WARNING: Face is very small! Analysis will be unreliable.")
        
        # 3. Save the Crop (So you can see it)
        crop_name = f"debug_face_{i+1}.jpg"
        face_crop.save(crop_name)
        print(f"   📸 Saved crop to: {crop_name}")

        # 4. Predict
        score = detector._predict_single(face_crop)
        print(f"   📊 Score: {score:.4f}")

        # Draw
        x1, y1, x2, y2 = box
        color = "red" if score > 0.5 else "green"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.text((x1, y1 - 40), f"{score:.2f}", fill=color, font=font)

    img.save("debug_output.jpg")
    print("\n✅ Done. Check 'debug_face_1.jpg' and 'debug_face_2.jpg'.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEST_IMAGE = sys.argv[1]
    debug_image(TEST_IMAGE)
```

---

### 2.7 `inject.py`
```python
import os
import shutil
import random
from tqdm import tqdm

# CONFIG
# Adjust this path if your unzip structure is different!
# Usually it's storage/temp_gan/real_vs_fake/real-vs-fake/train/fake
INITIAL_SOURCE_GAN_DIR = "storage/temp_gan/real_vs_fake/real-vs-fake/train/fake" 
DEST_FAKE_DIR = "storage/dataset/130k_faces/images/fake"
TARGET_COUNT = 20000

def inject():
    # Fix: Use a local variable first, then decide on the final path
    source_dir = INITIAL_SOURCE_GAN_DIR
    
    print(f"💉 Injecting {TARGET_COUNT} GAN images into your dataset...")
    
    # 1. Check if source exists
    if not os.path.exists(source_dir):
        print(f"⚠️ Could not find default path: {source_dir}")
        print("   Searching 'storage/temp_gan' for a valid 'fake' folder...")
        
        found = False
        # Walk through the temp folder to find where the images actually are
        for root, dirs, files in os.walk("storage/temp_gan"):
            # Look for a folder named 'fake' that has images in it
            if "fake" in os.path.basename(root).lower():
                # Check if it actually contains images (avoid empty folders)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if len(image_files) > 1000:
                    print(f"   ✅ Found candidate: {root} ({len(image_files)} images)")
                    source_dir = root
                    found = True
                    break
        
        if not found:
            print("❌ ERROR: Could not find any folder with GAN images.")
            return

    # 2. Get all GAN files from the valid source_dir
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   Found {len(all_files)} GAN images available.")
    
    if len(all_files) == 0:
        print("❌ ERROR: Source folder is empty.")
        return

    # 3. Select 20,000 (or all if fewer)
    if len(all_files) < TARGET_COUNT:
        print("⚠️ Not enough images! Taking all of them.")
        selected_files = all_files
    else:
        # Randomly select 20k to avoid alphabetical bias
        selected_files = random.sample(all_files, TARGET_COUNT)
        
    # 4. Move them
    print(f"   Moving {len(selected_files)} images to {DEST_FAKE_DIR}...")
    
    # Ensure destination exists
    os.makedirs(DEST_FAKE_DIR, exist_ok=True)
    
    count = 0
    for file in tqdm(selected_files):
        try:
            src = os.path.join(source_dir, file)
            # Rename to 'gan_' so we know origin, and avoid overwriting Diffusion files
            dst_name = f"gan_stylegan_{file}"
            dst = os.path.join(DEST_FAKE_DIR, dst_name)
            
            # Use copy instead of move if you want to keep the temp folder for now
            shutil.move(src, dst)
            count += 1
        except Exception as e:
            print(f"Error moving {file}: {e}")
            
    print(f"✅ Success! Added {count} GAN images.")
    print(f"   Total Fakes should now be ~80,000 (Mix of Diffusion + GAN).")

if __name__ == "__main__":
    inject()
```

---

### 2.8 `config/settings.py` & `config/settings/base.py`

#### `config/settings.py`
```python
import environ
import os
from pathlib import Path

# Initialize environ with a strict default for production safety
env = environ.Env(
    DEBUG=(bool, False)
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Read the .env file
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

# ==========================================
# 🛡️ SECURITY & CORE SETTINGS
# ==========================================
SECRET_KEY = env('SECRET_KEY')
# Pulls from .env, but defaults to False if missing (Critical for Prod!)
DEBUG = env('DEBUG') 
RSA_KEY_PATH = os.path.expanduser('~/.veritas_keys/private_key.pem')
VT_API_KEY = env('VT_API_KEY')  # VirusTotal API Key from .env

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = 'config.wsgi.application'

# 🚀 SUBDOMAIN CONFIGURATION
# We allow the tunnel address and the local loopback
ALLOWED_HOSTS = [
    'veritas-stream.sanchayjain.in', 
    'localhost', 
    '127.0.0.1',
    '[::1]'
]

# Required for Django's CSRF protection to work over the HTTPS tunnel
CSRF_TRUSTED_ORIGINS = ['https://veritas-stream.sanchayjain.in']


# ==========================================
# 💾 DATABASE
# ==========================================
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}



# ☁️ AWS LEARNER'S LAB CREDENTIALS
# WARNING: These expire every 4 hours! Update them before your presentation.
AWS_ACCESS_KEY_ID = env('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env('AWS_SECRET_ACCESS_KEY')
AWS_SESSION_TOKEN = env('AWS_SESSION_TOKEN')

AWS_STORAGE_BUCKET_NAME = 'veritass3bucket' # Your unique bucket name
AWS_S3_REGION_NAME = 'us-east-1'


# ==========================================
# 🧩 APPLICATION DEFINITION
# ==========================================
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # 🧠 Veritas Stream Custom Apps
    'apps.forensics',
    'apps.analysis',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [], 
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]


# ==========================================
# 📁 STATIC & MEDIA STORAGE
# ==========================================
# The base URL for static files
STATIC_URL = '/static/'
# 🛑 NEW: Where collectstatic will dump your assets for Cloudflare
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# The base URL that serves the media files to the browser
MEDIA_URL = '/media/'
# The absolute path routing directly to your WSL2 storage architecture
MEDIA_ROOT = os.path.join(BASE_DIR, 'apps', 'storage')

# Keep your vault root for your background engine pipeline
VAULT_ROOT = env('VAULT_STORAGE_PATH')
```

#### `config/settings/base.py`
```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Lead Dev Hack: Check if we are on WSL2 or Windows to set storage
# This will be your 1TB drive symlink location
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'storage')

INSTALLED_APPS = [
    # Django apps...
    'rest_framework',
    'apps.core',
    'apps.analysis',
    'apps.forensics',
]
```

---

### 2.9 `config/urls.py`
```python
from django.contrib import admin
from django.urls import path, re_path
from django.conf import settings
from django.views.static import serve
from apps.forensics.views import upload_and_analyze
from apps.forensics import views

urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('admin/', admin.site.urls),  
    path('upload/', upload_and_analyze, name='upload'),
    path('', upload_and_analyze, name='home'),
    path('report/<uuid:report_id>/export/', views.export_pdf_report, name='export_pdf'),
]

# 🔓 THE FIX: Force Django to serve static and media files through the tunnel
# This explicitly bypasses the DEBUG=False restriction
urlpatterns += [
    re_path(r'^media/(?P<path>.*)$', serve, {
        'document_root': settings.MEDIA_ROOT,
    }),
    re_path(r'^static/(?P<path>.*)$', serve, {
        'document_root': settings.STATIC_ROOT,
    }),
]
```

---

## 3. Data-Flow Contract Between `analysis` and `forensics`

### 3.1 End-to-End Execution Trace on Media Upload

```
[User Browser]
       │
       ▼ (multipart/form-data POST)
[forensics/views.py: upload_and_analyze()]
       │
       ├─► 1. Airlock Ingestion: Save incoming file buffer to `apps/storage/quarantine/`
       │
       ├─► 2. Hash Calculation: `engine.get_file_hash(quarantine_path)` -> SHA-256
       │
       ├─► 3. Iron Curtain Sterilization: `sterilize_file(quarantine_path, temp_hash)`
       │      ├── Magic Byte Check: `mime.from_file()` ensures starts with 'image/'
       │      ├── VirusTotal API Check: SHA-256 hash lookup via `VT_API_KEY`
       │      ├── Local Antivirus: `clamscan --allmatch --no-summary`
       │      └── Structural Integrity: `PIL.Image.open().verify()`
       │      (If any gate fails: purge file from quarantine, render upload.html with error)
       │
       ├─► 4. Duplicate Check & Cryptographic Tripwire:
       │      Queries `ForensicReport.objects.filter(sha256_hash=file_hash).first()`
       │      If Found:
       │         - Reconstructs canonical string: `"Hash:<hash>|Result:<score>"`
       │         - Verifies RSA-PSS digital signature via `engine.verify_signature()`
       │         - If signature verification fails: renders `report.html` with status `BREACH_DETECTED`
       │         - If signature valid: unpacks deep metadata, locates historical ELA, renders `report.html` (status `Exists`)
       │
       └─► 5. Execution Pipeline (New Evidence):
              Calls `engine.execute_full_pipeline(quarantine_path)`
                     │
                     ├─► a. `analyze_media(input_path)`:
                     │      - Resolves geometry/FPS using OpenCV `cv2.VideoCapture` / `cv2.imread`
                     │      - Retrieves singleton: `self._get_detector()` (`DeepFakeDetector`)
                     │      - Invokes `detector.analyze(input_path)` (The Gauntlet)
                     │
                     ├─► b. RSA Notarization:
                     │      - Constructs: `"Hash:<hash>|Result:<ai_confidence>"`
                     │      - Signs payload with RSA-4096 private key (`sign_report`)
                     │
                     ├─► c. Vault Isolation:
                     │      - `archive_to_vault()` transfers clean file from `quarantine/` to
                     │        `apps/storage/vault/YYYY/MM/DD/<filename>` (deleting quarantine source)
                     │
                     └─► d. Visual Veto (ELA Generation):
                            - `generate_ela_heatmap()` saves `ELA_<filename>` (15x brightness scaled)
                              directly adjacent to the evidence file in the vault.
```

---

### 3.2 Exact Return Type & Contract of `EnterpriseDeepFakeDetector.analyze()`

In `apps/analysis/core.py`, the detector is exposed via:
```python
DeepFakeDetector = EnterpriseDeepFakeDetector
```

#### Method Signature & Return Value
```python
def analyze(self, image_path: str) -> float:
```
- **Return Type:** Strictly a single Python `float` in the range `[0.0, 1.0]`.
  - `0.0` represents fully authentic / human-captured.
  - `1.0` represents fully synthetic / deepfake.
- **Contract Reality & Nuances:**
  1. `EnterpriseDeepFakeDetector.analyze()` **does NOT return** a dictionary, heatmaps, or stage breakdown to `VeritasForensicEngine`.
  2. In `forensics/core.py`, `analyze_media()` wraps the detector output:
     ```python
     real_confidence = detector.analyze(input_path)
     metadata = {
         "resolution": res,
         "fps": fps,
         "frame_count": frame_count,
         "ai_confidence": real_confidence,  # <-- Single float passed here
     }
     return metadata
```
  3. **Dual ELA Implementations:**
     - `analysis/core.py` has an internal `_generate_ela()` that calculates a mathematical variance float for its **Forensic Veto** rule (`raw_max >= 0.80 and ela_variance > 150`).
     - `forensics/core.py` independently runs `generate_ela_heatmap()` on the saved vault file to generate the visual heatmap image (`ELA_<filename>.jpg`) for the UI frontend.

#### Integration Recommendation for `VeritasHybridNet`
To act as a seamless drop-in replacement, `VeritasHybridNet` must implement an `analyze(image_path)` method returning a single `float` score. If you choose to expose richer multi-task outputs (e.g. per-stage GAN vs. Diffusion probabilities, spatial heatmaps, or spectral maps), `VeritasForensicEngine.analyze_media()` and `execute_full_pipeline()` can be updated without altering the underlying database or UI contract.

---

### 3.3 Frontend Template Contract

#### 1. Web Report Template (`apps/forensics/templates/forensics/report.html`)
The view passes the following context dictionary to `report.html`:

```python
context = {
    "status": "Success",  # 'Success' | 'Exists' | 'BREACH_DETECTED'
    "report": report,  # ForensicReport instance
    "verdict": "AI GENERATED",  # 'AI GENERATED' | 'AUTHENTIC' | 'DATA COMPROMISED'
    "confidence": "94.2%",  # Formatted string: f"{score * 100:.1f}%" or "ERR_SIG_INVALID"
    "is_fake": True,  # Boolean: score >= 0.50
    "image_url": "/media/vault/2026/11/16/photo.jpg",  # Resolved URL via get_guaranteed_url()
    "ela_heatmap_url": (
        "/media/vault/2026/11/16/ELA_photo.jpg"  # Path or None
    ),
    "metadata": {  # Dict from extract_deep_metadata()
        "Make": "Apple",
        "Model": "iPhone 15 Pro",
        "Software": "None Detected",
        "DateTaken": "2026:04:10 12:45:00",
        "DateDigitized": "2026:04:10 12:45:00",
        "Geotagged": "Yes",
        "Header Status": "...",
    },
    "risk_tags": [  # List of risk strings parsed by template
        # e.g., "SCRUBBED METADATA: Standard JPEG headers missing...",
        # e.g., "SYNTHETIC SIGNATURE: Total absence of forensic headers...",
        # e.g., "MANIPULATION TRACE: File edited via Photoshop..."
    ],
    "info_flags": [  # Informational alerts
        # e.g., "LOCATION DATA: This image contains embedded GPS coordinates."
    ],
    "error_message": (
        "..."  # Present only during BREACH_DETECTED (cryptographic signature mismatch)
    ),
}
```

- **Special UI Behaviors in `report.html`:**
  - `status == 'BREACH_DETECTED'`: Turns the UI into red alert mode, pulses red borders, grayscales the evidence image with sepia saturation, displays `DATA COMPROMISED`, and prints `error_message`.
  - `ela_heatmap_url`: When non-null, conditionally displays the Error Level Analysis container below the verdict cards.
  - `risk_tags`: Tags containing the substring `"SCRUBBED"` render as yellow privacy alerts (WhatsApp/Discord proxy scrubbing); all other tags render as red forensic anomalies.

#### 2. PDF Notarization Template (`apps/forensics/templates/forensics/pdf_template.html`)
The view `export_pdf_report(request, report_id)` passes the following context to `pdf_template.html`:

```python
context = {
    "report": report,  # ForensicReport instance
    "metadata": metadata,  # Dict of parsed EXIF tags
    "risk_tags": risk_tags,  # List of forensic anomaly / scrubbing tags
    "info_flags": info_flags,  # Informational flags (GPS, etc.)
    "verdict": "AI GENERATED",  # 'AI GENERATED' if score >= 0.5 else 'AUTHENTIC'
    "confidence": "94.2%",  # Formatted string: f"{score * 100:.1f}%"
    "analysis_date": (
        "2026-11-16 19:40:00 UTC"  # Formatted timezone string
    ),
}
```
- **PDF Elements Rendered:**
  - Dynamic A4 layout with `#{{ report.id }}` and SHA-256 hash truncated to 20 chars.
  - Color-coded verdict box (Red `#ff525f` for `AI GENERATED`, Green `#05e777` for `AUTHENTIC`).
  - Heuristic risk assessment list (`risk_tags`).
  - Formatted 2-column table of extracted metadata.
  - Cryptographic notary footer with truncated `report.rsa_signature` (RSA-4096) and analysis timestamp.

---

## 4. Vault & Shell Script Architecture

### 4.1 `mount_vault.sh`
```bash
#!/bin/bash
WINDOWS_DRIVE_LETTER="E:"   # How Windows sees it
LINUX_MOUNT_POINT="/mnt/e"  # Where we want it in Linux
VAULT_FOLDER_NAME="veritas_vault"
PROJECT_VAULT="./storage/vault"
```

#### Workflow & Operational Mechanics:
1. **Host-to-WSL2 DrvFS Mount:** Checks if Windows drive `E:` is accessible inside WSL2 at `/mnt/e`. If unmounted, executes:
   ```bash
   sudo mount -t drvfs "$WINDOWS_DRIVE_LETTER" "$LINUX_MOUNT_POINT"
   ```
2. **Directory Initialization:** Asserts that the target directory `/mnt/e/veritas_vault` exists on the physical storage device.
3. **Linux Bind Mount:** Connects the project's relative `./storage/vault` path directly to the external drive:
   ```bash
   sudo mount --bind "$EXTERNAL_PATH" "$PROJECT_VAULT"
   ```
4. **Architectural Purpose:** Isolates volatile, high-capacity forensic media and uncompressed archives (terabytes of evidence) onto high-capacity external drives while allowing the Django application in WSL2/Linux to interact with standard POSIX relative paths.

---

### 4.2 Cryptographic Signing, Notarization & Tripwire Architecture

As detailed in `apps/forensics/core.py`, `apps/forensics/views.py`, and `README.md`:

```
               [Raw File in Quarantine]
                          │
                          ▼
             SHA-256 Digest Calculation
                          │
                          ▼
           AI Dual-Brain Ensemble Score (0.0 - 1.0)
                          │
                          ▼
     Canonical Envelope: "Hash:<sha256>|Result:<score>"
                          │
                          ▼
             RSA-4096 PSS Digital Signature
              (Private Key: ~/.veritas_keys/private_key.pem)
                          │
                          ▼
    Persisted to DB: `ForensicReport` Table
    ├── sha256_hash (Indexed, Unique)
    ├── ai_confidence_score (Float)
    ├── rsa_signature (Base64 Encoded)
    └── vault_path (apps/storage/vault/YYYY/MM/DD/file.ext)
```

1. **Notarization Step:** Once the file clears the "Iron Curtain" and passes through the AI Gauntlet, `VeritasForensicEngine.sign_report()` formats a canonical plaintext payload:
   $$\text{Payload} = \texttt{"Hash:"} + \text{file\_hash} + \texttt{"|Result:"} + \text{ai\_confidence}$$
   This payload is signed using **RSA-4096** with **PSS padding** (MGF1 with SHA-256, maximum salt length) and SHA-256 digest, then stored in `ForensicReport.rsa_signature`.
2. **The Cryptographic Tripwire (Duplicate / Query Verification):**
   When an evidence file is queried or re-uploaded:
   - The engine retrieves the existing database record.
   - It reconstructs the identical canonical envelope string: `f"Hash:{existing_report.sha256_hash}|Result:{existing_report.ai_confidence_score}"`.
   - `engine.verify_signature()` verifies the envelope against the public key extracted from `RSA_KEY_PATH`.
   - **Tamper Detection:** If an adversary or unauthorized database operator manually modifies the `ai_confidence_score` (e.g. flipping a deepfake score from `0.98` to `0.02`), the RSA-PSS signature check fails mathematically. The system traps this error and triggers the `BREACH_DETECTED` status, flashing the cryptographic compromise alert in the UI.
3. **Cold Storage Archival:** After notarization, the file is physically moved via `shutil.copyfile` into `apps/storage/vault/YYYY/MM/DD/` with file size validation, and the quarantine copy is purged.
4. **Redundant Mirroring:** If configured, `backup_to_s3()` in `apps/forensics/views.py` mirrors the quarantined-and-cleared file to an AWS S3 bucket (`settings.AWS_STORAGE_BUCKET_NAME`).