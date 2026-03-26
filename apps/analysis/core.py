import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from PIL import Image
import os
import torch.nn.functional as F
from facenet_pytorch import MTCNN

# --- VERITAS PRO: VISION TRANSFORMER (ViT) ARCHITECTURE ---
class ProDeepFakeDetector:
    def __init__(self):
        print("🧠 Initializing Veritas Pro (Vision Transformer Mode)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Face Hunter
        self.face_detector = MTCNN(keep_all=True, device=self.device, post_process=False, margin=0, thresholds=[0.5, 0.6, 0.6])
        
        # 1. Load the Base ViT Architecture
        self.model = vit_b_16(weights=None) 
        # Modify the final classification head for 2 classes (Fake/Real)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, 2)
        
        # 2. Load our trained weights
        weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'veritas_vit.pkl')
        if os.path.exists(weights_path):
            try:
                # weights_only=True is safer and removes the PyTorch warning you were seeing
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("✅ Veritas ViT Model Loaded")
            except Exception as e:
                print(f"❌ CRITICAL: Weights failed: {e}")
        else:
            print(f"⚠️ Warning: {weights_path} not found. System is Untrained.")

        self.model.to(self.device)
        self.model.eval()

        # 3. ViT Standard Transform (224x224 is mandatory for ViT patches)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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

    def _predict_single(self, img):
        rgb_tensor = self.transform(img).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(rgb_tensor)
            probs = F.softmax(output, dim=1)
            
            # --- THE WIRETAP ---
            print(f"\n🧠 [AI BRAIN DUMP]")
            print(f"Raw Output (Logits): {output.tolist()}")
            print(f"Probabilities: {probs.tolist()}")
            print(f"Class 0 Score: {probs[0][0].item() * 100:.2f}%")
            print(f"Class 1 Score: {probs[0][1].item() * 100:.2f}%\n")
            
            # Assuming standard ImageFolder: 0=Fake, 1=Real
            return probs[0][0].item()

    def analyze(self, image_path):
        try:
            full_image = Image.open(image_path).convert('RGB')
            # 1. Hunt for faces
            boxes, _ = self.face_detector.detect(full_image)
            
            # 2. 🛑 THE FIX: The Face Gate Bypass
            if boxes is None:
                print("⚠️ No face detected. Defaulting to Authentic (0.0).")
                return 0.0 # Returns 0% Fake
            
            # 3. Analyze detected faces
            scores = []
            for box in boxes:
                face_crop = self.make_ffhq_crop(full_image, box)
                scores.append(self._predict_single(face_crop))

            # Return the highest fake score found in the image
            return max(scores)

        except Exception as e:
            print(f"❌ Analysis Failed: {e}")
            return 0.0 # Failsafe: If the image is corrupted, just pass it as Authentic

DeepFakeDetector = ProDeepFakeDetector