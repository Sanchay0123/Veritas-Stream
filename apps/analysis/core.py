import os
import io
import torch
import tempfile
import boto3
import hashlib
from PIL import Image, ImageChops, ImageEnhance, ImageStat
from facenet_pytorch import MTCNN
from transformers import pipeline
from django.conf import settings

class VeritasForensicEngine:
    def __init__(self):
        print("🧠 Initializing Veritas Stream Cloud Engine (Dual-Brain + ELA)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ Hardware: {self.device}")
        
        # 1. Face Hunter
        self.face_detector = MTCNN(keep_all=True, device=self.device, post_process=False, margin=0, thresholds=[0.5, 0.6, 0.6])
        
        # 2. AI Expert Ensemble
        print("⏳ Loading Expert 1 (GANs)...")
        self.gan_detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection", device=0 if torch.cuda.is_available() else -1)
        
        print("⏳ Loading Expert 2 (Diffusions)...")
        self.diffusion_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=0 if torch.cuda.is_available() else -1)
        print("✅ Ultimate Brain Loaded and Cloud Ready")

    # --- FORENSIC UTILITIES ---

    def _scan_metadata(self, img):
        ai_signatures = ['gemini', 'midjourney', 'dall-e', 'stable diffusion', 'ai generated', 'google']
        for key, value in img.info.items():
            if any(sig in str(value).lower() for sig in ai_signatures):
                return 1.0 
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                if any(sig in str(value).lower() for sig in ai_signatures):
                    return 1.0 
        return 0.0

    def _generate_ela(self, img):
        temp_io = io.BytesIO()
        img.save(temp_io, 'JPEG', quality=90)
        temp_io.seek(0)
        compressed_img = Image.open(temp_io)
        ela_map = ImageChops.difference(img, compressed_img)
        extrema = ela_map.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        scale = 255.0 / (max_diff if max_diff > 0 else 1)
        ela_map = ImageEnhance.Brightness(ela_map).enhance(scale)
        stat = ImageStat.Stat(ela_map)
        variance = sum(stat.var) / len(stat.var)
        return variance

    def make_ffhq_crop(self, img, box):
        width, height = img.size
        x1, y1, x2, y2 = [int(b) for b in box]
        w, h = x2 - x1, y2 - y1
        padding = max(w, h) * 0.40
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_side = max(w, h) + (padding * 2)
        new_x1 = max(0, int(cx - new_side / 2))
        new_y1 = max(0, int(cy - new_side / 2))
        new_x2 = min(width, int(cx + new_side / 2))
        new_y2 = min(height, int(cy + new_side / 2))
        return img.crop((new_x1, new_y1, new_x2, new_y2))

    def _extract_score(self, hf_output):
        for pred in hf_output:
            label = pred['label'].lower()
            if label in ['fake', 'artificial', 'ai']:
                return pred['score']
        top_label = hf_output[0]['label'].lower()
        if top_label in ['real', 'human', 'original']:
            return 1.0 - hf_output[0]['score']
        return 0.0

    def _run_ensemble(self, img, stage_name):
        score_1 = self._extract_score(self.gan_detector(img))
        score_2 = self._extract_score(self.diffusion_detector(img))
        final_score = max(score_1, score_2)
        print(f"   [{stage_name}] Ensemble Score: {final_score * 100:.2f}%")
        return final_score

    # --- THE ANALYZE ENGINE ---

    def analyze(self, image_path):
        try:
            full_image = Image.open(image_path).convert('RGB')
            width, height = full_image.size
            scores = []
            
            if self._scan_metadata(full_image) == 1.0:
                return 1.0

            self._generate_ela(full_image)
            scores.append(self._run_ensemble(full_image, "Full Context"))

            corner_crop = full_image.crop((int(width * 0.85), int(height * 0.85), width, height))
            scores.append(self._run_ensemble(corner_crop, "Watermark Hunter"))

            boxes, _ = self.face_detector.detect(full_image)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    face_crop = self.make_ffhq_crop(full_image, box)
                    scores.append(self._run_ensemble(face_crop, f"Face Hunter {i+1}"))

            suspicious_stages = [s for s in scores if s >= 0.25]
            raw_max = max(scores)
            final_score = min(0.99, raw_max * 1.5) if len(suspicious_stages) >= 2 and raw_max < 0.50 else raw_max

            return final_score
        except Exception as e:
            print(f"❌ Core Analysis Error: {e}")
            return 0.0

    # --- CLOUD TRIAGE PIPELINE ---

    def process_cloud_triage(self, uploaded_file):
        """
        The Main Cloud Logic. Handles AI check and S3 Upload.
        """
        try:
            # AWS Client Initialization
            s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                aws_session_token=settings.AWS_SESSION_TOKEN,
                region_name=settings.AWS_S3_REGION_NAME
            )

            # Create Temp File
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
                temp_path = tmp.name

            # 1. AI Analysis
            final_score = self.analyze(temp_path)
            is_fake = final_score >= 0.50

            # 2. Define S3 Keys
            safe_name = uploaded_file.name.replace(" ", "_")
            quarantine_key = f"quarantine/{safe_name}"
            vault_key = f"vault/{safe_name}"

            # 3. S3 Quarantine Upload
            print(f"☁️ Uploading {safe_name} to S3...")
            s3_client.upload_file(temp_path, settings.AWS_STORAGE_BUCKET_NAME, quarantine_key)

            # 4. Vault Mirror (If Real)
            vault_path = None
            if not is_fake:
                print(f"✅ Authentic. Mirroring to Vault...")
                copy_source = {'Bucket': settings.AWS_STORAGE_BUCKET_NAME, 'Key': quarantine_key}
                s3_client.copy(copy_source, settings.AWS_STORAGE_BUCKET_NAME, vault_key)
                vault_path = vault_key

            # Clean Up
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return {
                'ai_confidence_score': final_score,
                'is_fake': is_fake,
                'vault_path': vault_path
            }

        except Exception as aws_err:
            print(f"❌ AWS/Processing Error: {aws_err}")
            raise aws_err

# Create the Engine Instance
engine = VeritasForensicEngine()