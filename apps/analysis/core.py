import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

# --- VERITAS PRO ARCHITECTURE (EfficientNet-B0) ---
class ProDeepFakeDetector:
    def __init__(self):
        print("🧠 Initializing Veritas Pro Engine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Define the Empty Architecture (The Skeleton)
        # We use 'from_name' because we will load our own weights, not internet ones.
        self.model = EfficientNet.from_name('efficientnet-b0', num_classes=2)
        
        # 2. Load Your 98% Accuracy Weights (The Brain)
        weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'veritas_pro.pkl')
        
        if os.path.exists(weights_path):
            print(f"📥 Loading Custom Trained Weights: {weights_path}")
            try:
                # Load the state dictionary
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                print("✅ Veritas Pro Successfully Loaded (Accuracy: ~98%)")
            except Exception as e:
                print(f"❌ CRITICAL: Custom weights failed to load: {e}")
                print("⚠️ Falling back to untrained initialization.")
        else:
            print(f"⚠️ Warning: {weights_path} not found.")
            print("⚠️ System is running in UNTRAINED mode.")

        self.model.to(self.device)
        self.model.eval()

        # 3. Define Image Preprocessing
        # MUST match the transformation used in train.py
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def analyze(self, image_path):
        """Run the AI inference on a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                # Softmax to get probabilities [Real_Prob, Fake_Prob]
                probabilities = F.softmax(output, dim=1)
                # Get Fake Probability (Index 1)
                confidence = probabilities[0][0].item()
                
            return confidence
        except Exception as e:
            print(f"❌ AI Analysis Failed: {e}")
            return 0.5 

# Alias for compatibility with views.py
DeepFakeDetector = ProDeepFakeDetector