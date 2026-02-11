import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from facenet_pytorch import MTCNN
import numpy as np

# --- VERITAS PRO ARCHITECTURE (FFHQ Matcher) ---
class ProDeepFakeDetector:
    def __init__(self):
        print("🧠 Initializing Veritas Pro Engine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Initialize Face Hunter
        # margin=0 because we will handle expansion manually
        self.face_detector = MTCNN(keep_all=False, select_largest=True, device=self.device, post_process=False, margin=0)

        # 2. Define Brain
        self.model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
        
        # 3. Load Weights
        weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'veritas_pro.pkl')
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("✅ Veritas Pro Loaded (Mode: FFHQ-Crop)")
            except Exception as e:
                print(f"❌ CRITICAL: Weights failed: {e}")
        else:
            print(f"⚠️ Warning: {weights_path} not found.")

        self.model.to(self.device)
        self.model.eval()

        # 4. Standard Transform (No degradation needed now)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # Match FFHQ size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def make_ffhq_crop(self, img, box):
        """
        Expands the tight MTCNN box to include hair/ears/neck
        mimicking the FFHQ training data style.
        """
        width, height = img.size
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Calculate size of the face
        w = x2 - x1
        h = y2 - y1
        
        # FFHQ Expansion Factor (40% padding is the sweet spot)
        padding = max(w, h) * 0.40
        
        # Find Center
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # New Square Box
        new_side = max(w, h) + (padding * 2)
        
        new_x1 = max(0, int(cx - new_side / 2))
        new_y1 = max(0, int(cy - new_side / 2))
        new_x2 = min(width, int(cx + new_side / 2))
        new_y2 = min(height, int(cy + new_side / 2))
        
        # Debug Print
        print(f"📐 Original: {w}x{h} -> Expanded: {new_x2-new_x1}x{new_y2-new_y1}")
        
        return img.crop((new_x1, new_y1, new_x2, new_y2))

    def analyze(self, image_path):
        try:
            full_image = Image.open(image_path).convert('RGB')
            w, h = full_image.size
            
            final_input = None
            
            # BYPASS: Small images (Dataset check)
            if w < 300 and h < 300:
                print(f"🔹 Dataset Image ({w}x{h}). Skipping Crop.")
                final_input = self.transform(full_image).unsqueeze(0).to(self.device)
                
            # DETECT: Real photos get the FFHQ Expansion
            else:
                boxes, _ = self.face_detector.detect(full_image)
                if boxes is not None:
                    # Use the "Generous" Cropper
                    face_image = self.make_ffhq_crop(full_image, boxes[0])
                    print(f"🕵️ Face Found & FFHQ-Matched (Zoomed Out)")
                    final_input = self.transform(face_image).unsqueeze(0).to(self.device)
                else:
                    print("⚠️ No face found! Using full image.")
                    final_input = self.transform(full_image).unsqueeze(0).to(self.device)
            
            # INFERENCE
            with torch.no_grad():
                output = self.model(final_input)
                probabilities = F.softmax(output, dim=1)
                
                # IMPORTANT: Check your train.py output! 
                # If classes=['fake', 'real'] -> fake=0, real=1.
                # If classes=['real', 'fake'] -> fake=1, real=0.
                # Assuming Alphabetical (fake=0, real=1):
                fake_prob = probabilities[0][0].item()
                real_prob = probabilities[0][1].item()
                
                print(f"📊 Score: Fake={fake_prob:.4f} | Real={real_prob:.4f}")
                return fake_prob
                
        except Exception as e:
            print(f"❌ Analysis Failed: {e}")
            return 0.5 

DeepFakeDetector = ProDeepFakeDetector