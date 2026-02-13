import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from facenet_pytorch import MTCNN
import numpy as np

# --- VERITAS PRO: MULTI-FACE SPECTRAL ARCHITECTURE ---
class FrequencyEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Load Standard B0
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
        
        # 2. SURGERY: Modify Input Layer to accept 6 Channels (RGB + 3 Freq)
        old_conv = self.backbone._conv_stem
        new_conv = nn.Conv2d(6, old_conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Copy RGB weights to the first 3 channels & init Frequency weights
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight
        
        self.backbone._conv_stem = new_conv

    def forward(self, x):
        return self.backbone(x)

class ProDeepFakeDetector:
    def __init__(self):
        print("🧠 Initializing Veritas Pro (Multi-Face Spectral)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Face Hunter: keep_all=True finds EVERY face, not just the largest one
        self.face_detector = MTCNN(keep_all=True, device=self.device, post_process=False, margin=0)

        # 2. Spectral Brain
        self.model = FrequencyEfficientNet()
        
        # 3. Load Weights
        weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'veritas_spectral.pkl')
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("✅ Veritas Spectral Loaded")
            except Exception as e:
                print(f"❌ CRITICAL: Weights failed: {e}")
        else:
            print(f"⚠️ Warning: {weights_path} not found.")

        self.model.to(self.device)
        self.model.eval()

        # 4. Transform (Standard 256x256)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_spectrum(self, img_tensor):
        """Matched to Training Logic"""
        fft = torch.fft.fft2(img_tensor)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        spectrum = torch.log(magnitude + 1e-8)
        # Instance Norm
        spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)
        return spectrum

    def make_ffhq_crop(self, img, box):
        """Expands the crop to include hair/neck (FFHQ Style)"""
        width, height = img.size
        x1, y1, x2, y2 = [int(b) for b in box]
        w, h = x2 - x1, y2 - y1
        
        # 40% Padding
        padding = max(w, h) * 0.40
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_side = max(w, h) + (padding * 2)
        
        new_x1 = max(0, int(cx - new_side / 2))
        new_y1 = max(0, int(cy - new_side / 2))
        new_x2 = min(width, int(cx + new_side / 2))
        new_y2 = min(height, int(cy + new_side / 2))
        
        return img.crop((new_x1, new_y1, new_x2, new_y2))

    def _predict_single(self, img):
        """Helper to run inference on a single face crop"""
        rgb_tensor = self.transform(img).to(self.device)
        spectrum_tensor = self.get_spectrum(rgb_tensor).to(self.device)
        
        # Stack: 3 RGB + 3 Spectrum = 6 Channels
        combined_input = torch.cat([rgb_tensor, spectrum_tensor], dim=0).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(combined_input)
            probs = F.softmax(output, dim=1)
            # Assuming fake=0, real=1 (Standard ImageFolder order)
            return probs[0][0].item()

    def analyze(self, image_path):
        try:
            full_image = Image.open(image_path).convert('RGB')
            w, h = full_image.size
            
            # 1. Detect ALL faces
            boxes, _ = self.face_detector.detect(full_image)
            
            if boxes is None:
                print("⚠️ No face found! Analyzing full image.")
                # Fallback: Analyze the whole image as one
                return self._predict_single(full_image)

            print(f"🕵️ Faces Found: {len(boxes)}")
            
            scores = []
            
            # 2. Loop through every face found
            for i, box in enumerate(boxes):
                face_crop = self.make_ffhq_crop(full_image, box)
                score = self._predict_single(face_crop)
                scores.append(score)
                print(f"   👤 Face {i+1}: Fake Score = {score:.4f}")

            # 3. RETURN THE WORST SCORE (Most Fake)
            # If any face is fake, the image is fake.
            max_fake_score = max(scores)
            print(f"📊 Final Verdict: {max_fake_score:.4f}")
            return max_fake_score

        except Exception as e:
            print(f"❌ Analysis Failed: {e}")
            return 0.5 

DeepFakeDetector = ProDeepFakeDetector