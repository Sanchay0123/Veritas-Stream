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

# Import the AI Brain
from apps.analysis.core import DeepFakeDetector 

class VeritasForensicEngine:
    def __init__(self, input_file_path, private_key_path=None):
        self.input_path = input_file_path
        self.filename = os.path.basename(input_file_path)
        self.private_key_path = private_key_path
        self.timestamp = timezone.now()

    def get_file_hash(self):
        """Step 1: Generate SHA-256 Hash"""
        sha256 = hashlib.sha256()
        with open(self.input_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def analyze_media(self):
        """Step 2: AI Visual Analysis"""
        cap = cv2.VideoCapture(self.input_path)
        success, frame = cap.read()
        
        if not success:
            if self.input_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(self.input_path)
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

        # Trigger the AI Brain
        detector = DeepFakeDetector()
        real_confidence = detector.analyze(self.input_path)

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




    def archive_to_vault(self):
        """Step 4: Secure Storage on External Drive"""
        
        # --- FIX: Robust Directory Creation ---
        vault_root = os.path.join(settings.MEDIA_ROOT, 'vault')
        
        # If it exists as a FILE, delete it so we can make a DIRECTORY
        if os.path.exists(vault_root) and not os.path.isdir(vault_root):
            os.remove(vault_root)
            
        os.makedirs(vault_root, exist_ok=True)
        # --------------------------------------

        date_path = self.timestamp.strftime('%Y/%m/%d')
        vault_dir = os.path.join(vault_root, date_path)
        os.makedirs(vault_dir, exist_ok=True)
        
        vault_path = os.path.join(vault_dir, self.filename)

        # Use copyfile (Pure Data Copy) to avoid permission errors
        try:
            shutil.copyfile(self.input_path, vault_path)
            # Verify copy size before deleting
            if os.path.getsize(vault_path) == os.path.getsize(self.input_path):
                os.remove(self.input_path)
            else:
                raise Exception("Copy size mismatch")
        except Exception as e:
            print(f"⚠️ Vault Transfer Failed: {e}")
            raise e

        return vault_path


        

    def execute_full_pipeline(self):
        """The Big Red Button"""
        file_hash = self.get_file_hash()
        analysis = self.analyze_media()
        
        report_summary = f"Hash:{file_hash}|Result:{analysis['ai_confidence']}"
        signature = self.sign_report(report_summary)
        
        vault_path = self.archive_to_vault()
        
        return {
            "hash": file_hash,
            "signature": signature,
            "vault_location": vault_path,
            "analysis": analysis
        }
