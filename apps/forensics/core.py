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