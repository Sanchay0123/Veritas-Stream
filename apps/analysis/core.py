import os
import io
import torch
import gc
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
        self.gan_detector = pipeline("image-classification",model="dima806/deepfake_vs_real_image_detection",device=0 if torch.cuda.is_available() else -1)        
        print("⏳ Loading Expert 2 (Diffusions)...")
        self.diffusion_detector = pipeline("image-classification", model="Organika/sdxl-detector", device=0 if torch.cuda.is_available() else -1)
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
        """Calculates the Error Level Analysis variance with calibrated thresholds."""
        print("   [Stage 1: Forensic Math] Running Error Level Analysis...")
        
        temp_io = io.BytesIO()
        img.save(temp_io, 'JPEG', quality=90)
        temp_io.seek(0)
        compressed_img = Image.open(temp_io)
        
        ela_map = ImageChops.difference(img, compressed_img)
        
        extrema = ela_map.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        if max_diff == 0: max_diff = 1
        scale = 255.0 / max_diff
        ela_map = ImageEnhance.Brightness(ela_map).enhance(scale)
        
        stat = ImageStat.Stat(ela_map)
        variance = sum(stat.var) / len(stat.var)
        
        # --- 🧪 RECALIBRATED INTERPRETATION ---
        # ISO noise in real photos often hits 60. We only flag extremes now.
        status = "NOMINAL"
        if variance > 150: 
            status = "ANOMALOUS HIGH (Possible Splicing)"
        elif variance < 5: 
            status = "ANOMALOUS LOW (Pure Synthetic/Too Clean)"
        
        print(f"      -> ELA Status: {status} (Variance: {variance:.2f})")
        return variance

    def _run_ensemble(self, img, stage_name, ela_variance=0): # 👈 Added ela_variance
        s1 = self._extract_score(self.gan_detector(img))
        s2 = self._extract_score(self.diffusion_detector(img))
        
        raw_max = max(s1, s2)
        raw_min = min(s1, s2)
        
        if raw_max >= 0.50 and raw_min < 0.50:
            # 🚨 THE FORENSIC VETO 🚨
            # If the neural networks are hung, check the hard math!
            # If Diffusion is screaming (>80%) AND ELA variance is extremely high (>150), it's a deepfake.
            if raw_max >= 0.80 and ela_variance > 150:
                final_score = raw_max
                print(f"   [{stage_name}] 🚨 FORENSIC VETO: ELA Math confirms anomaly. Overruling Acquittal. (GAN: {s1*100:.0f}%, Diff: {s2*100:.0f}%)")
            else:
                # ⚖️ Normal Weighted Acquittal (For actual real HDR photos)
                organic_score = (raw_min * 0.6) + (raw_max * 0.4)
                final_score = min(0.49, organic_score)
                print(f"   [{stage_name}] ⚖️ Hung Jury. Weighted Acquittal to {final_score * 100:.2f}% (GAN: {s1*100:.0f}%, Diff: {s2*100:.0f}%)")
        else:
            final_score = raw_max
            
        print(f"   [{stage_name}] Ensemble Score: {final_score * 100:.2f}%")
        return final_score

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
            ela_variance = self._generate_ela(full_image)

            # STAGE 2: FULL CONTEXT
            # 👈 Injecting ELA Variance for the Forensic Veto
            scores.append(self._run_ensemble(full_image, "Stage 2: Full Context", ela_variance))

            # STAGE 3: WATERMARK HUNTER
            corner_crop = full_image.crop((int(width * 0.85), int(height * 0.85), width, height))
            # 👈 Injecting ELA Variance for the Forensic Veto
            scores.append(self._run_ensemble(corner_crop, "Stage 3: Watermark Hunter", ela_variance))

            # STAGE 4: FACE HUNTER
            boxes, _ = self.face_detector.detect(full_image)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    face_crop = self.make_ffhq_crop(full_image, box)
                    # 👈 Injecting ELA Variance for the Forensic Veto
                    scores.append(self._run_ensemble(face_crop, f"Stage 4: Face Crop {i+1}", ela_variance))
            else:
                print("   [Face Hunter] ⚠️ No faces detected.")

            # 🧠 THE BIG BRAIN FIX: The Contextual Anchor
            # scores[0] is always "Stage 2: Full Context"
            # scores[1] is "Watermark"
            # scores[2:] are the "Face Crops"
            
            full_context_score = scores[0]
            sub_scores = scores[1:]
            
            if sub_scores:
                highest_sub_score = max(sub_scores)
                
                # ⚖️ Anchor the highest local anomaly to the global context.
                if highest_sub_score > full_context_score:
                    # Pull the high crop score down using the clean full context
                    final_score = (full_context_score * 0.5) + (highest_sub_score * 0.5)
                else:
                    final_score = full_context_score
            else:
                final_score = full_context_score

            # 🧹 THE VRAM VACUUM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            print(f"🚩 FINAL PIPELINE SCORE: {final_score * 100:.2f}% Fake\n")
            return final_score

        except Exception as e:
            print(f"❌ Analysis Failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            return 0.0


DeepFakeDetector = EnterpriseDeepFakeDetector