import os
import io
import torch
from PIL import Image, ImageChops, ImageEnhance, ImageStat
from facenet_pytorch import MTCNN
from transformers import pipeline

class EnterpriseDeepFakeDetector:
    def __init__(self):
        print("🧠 Initializing Veritas Stream (Dual-Brain Ensemble + ELA)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ Hardware: {self.device}")
        
        self.face_detector = MTCNN(keep_all=True, device=self.device, post_process=False, margin=0, thresholds=[0.5, 0.6, 0.6])
        
        print("⏳ Loading Expert 1 (GANs)...")
        self.gan_detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection", device=0 if torch.cuda.is_available() else -1)
        
        print("⏳ Loading Expert 2 (Diffusions)...")
        self.diffusion_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=0 if torch.cuda.is_available() else -1)
        print("✅ Ultimate Brain Loaded")

    def _scan_metadata(self, img):
        print("   [Stage 0: Metadata] Scanning file headers...")
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
        """Calculates the Error Level Analysis variance."""
        print("   [Stage 1: Forensic Math] Running Error Level Analysis...")
        
        # 1. Save temporary compressed version
        temp_io = io.BytesIO()
        img.save(temp_io, 'JPEG', quality=90)
        temp_io.seek(0)
        compressed_img = Image.open(temp_io)
        
        # 2. Calculate pixel difference
        ela_map = ImageChops.difference(img, compressed_img)
        
        # 3. Enhance brightness to see the hidden noise
        extrema = ela_map.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        ela_map = ImageEnhance.Brightness(ela_map).enhance(scale)
        
        # 4. Statistical Variance (AI images are usually very low variance/too clean)
        stat = ImageStat.Stat(ela_map)
        variance = sum(stat.var) / len(stat.var)
        
        print(f"      -> ELA Noise Variance: {variance:.2f}")
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
        print(f"   [{stage_name}] Ensemble Score: {final_score * 100:.2f}% (GAN: {score_1*100:.0f}%, Diff: {score_2*100:.0f}%)")
        return final_score

    def analyze(self, image_path):
        try:
            full_image = Image.open(image_path).convert('RGB')
            width, height = full_image.size
            scores = []
            
            print(f"\n🧠 [AI BRAIN DUMP - THE GAUNTLET]")

            # STAGE 0: METADATA
            if self._scan_metadata(full_image) == 1.0:
                print("   🚨 Metadata Triggered! Auto-Failing Image.")
                return 1.0

            # STAGE 1: FORENSIC MATH (ELA)
            # Keeping ELA active for raw/uncompressed dataset testing!
            ela_variance = self._generate_ela(full_image)

            # STAGE 2: FULL CONTEXT
            scores.append(self._run_ensemble(full_image, "Stage 2: Full Context"))

            # STAGE 3: WATERMARK HUNTER
            corner_crop = full_image.crop((int(width * 0.85), int(height * 0.85), width, height))
            scores.append(self._run_ensemble(corner_crop, "Stage 3: Watermark Hunter"))

            # STAGE 4: FACE HUNTER
            boxes, _ = self.face_detector.detect(full_image)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    face_crop = self.make_ffhq_crop(full_image, box)
                    scores.append(self._run_ensemble(face_crop, f"Stage 4: Face Crop {i+1}"))
            else:
                print("   [Face Hunter] ⚠️ No faces detected.")

            # 🧠 THE BIG BRAIN FIX: Heuristic Threat Amplification
            # Count how many stages are showing "mild" suspicion (>= 25%)
            suspicious_stages = [s for s in scores if s >= 0.25]
            raw_max = max(scores)
            
            if len(suspicious_stages) >= 2 and raw_max < 0.50:
                print(f"   🔍 [Heuristic] {len(suspicious_stages)} stages report systemic suspicion. Amplifying threat score!")
                final_score = min(0.99, raw_max * 1.5) # Apply a 1.5x Multiplier
            else:
                final_score = raw_max

            print(f"🚩 FINAL PIPELINE SCORE: {final_score * 100:.2f}% Fake\n")
            return final_score

        except Exception as e:
            print(f"❌ Analysis Failed: {e}")
            return 0.0

DeepFakeDetector = EnterpriseDeepFakeDetector