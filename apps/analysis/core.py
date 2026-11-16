import torch
from PIL import Image
from facenet_pytorch import MTCNN
from apps.analysis.networks import VeritasHybridNet

class EnterpriseDeepFakeDetector:
    def __init__(self):
        print("🧠 Initializing Veritas Stream (VeritasHybridNet Local Engine)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️ Hardware: {self.device}")
        
        self.face_detector = MTCNN(keep_all=True, device=self.device, post_process=False, margin=0, thresholds=[0.5, 0.6, 0.6])
        
        print("⏳ Loading VeritasHybridNet Weights...")
        self.model = VeritasHybridNet(num_classes=2, pretrained=False).to(self.device)
        weights_path = "apps/analysis/weights/veritas_hybrid.pth"
        
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(weights_path))
        else:
            self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            
        self.model.eval()
        print("✅ Local VeritasHybridNet Brain Loaded Successfully!")

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

            # STAGE 2: FULL CONTEXT
            scores.append(self.model.analyze(full_image, device=self.device))

            # STAGE 3: WATERMARK HUNTER
            corner_crop = full_image.crop((int(width * 0.85), int(height * 0.85), width, height))
            scores.append(self.model.analyze(corner_crop, device=self.device))

            # STAGE 4: FACE HUNTER
            boxes, _ = self.face_detector.detect(full_image)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    face_crop = self.make_ffhq_crop(full_image, box)
                    scores.append(self.model.analyze(face_crop, device=self.device))
            else:
                print("   [Face Hunter] ⚠️ No faces detected.")

            # Contextual Anchor Logic
            full_context_score = scores[0]
            sub_scores = scores[1:]
            
            if sub_scores:
                highest_sub_score = max(sub_scores)
                if highest_sub_score > full_context_score:
                    final_score = (full_context_score * 0.5) + (highest_sub_score * 0.5)
                else:
                    final_score = full_context_score
            else:
                final_score = full_context_score

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"🚩 FINAL PIPELINE SCORE: {final_score * 100:.2f}% Fake\n")
            return final_score

        except Exception as e:
            print(f"❌ Analysis Failed: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0.0

DeepFakeDetector = EnterpriseDeepFakeDetector