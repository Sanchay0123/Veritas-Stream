import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Import the brain we just built
from .networks import Meso4 

class DeepFakeDetector:
    def __init__(self):
        print("🧠 Initializing MesoNet AI...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Load the Structure
        self.model = Meso4().to(self.device)
        
        # 2. Load Weights (The Knowledge)
        weights_path = os.path.join(os.path.dirname(__file__), 'weights', 'meso4_faceforensics.pkl')
        
        if os.path.exists(weights_path):
            print(f"📥 Loading weights from: {weights_path}")
            try:
                # Load the state dictionary
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval() # Set to evaluation mode (frozen)
                print("✅ AI Brain Successfully Loaded!")
            except Exception as e:
                print(f"⚠️ Error loading weights: {e}")
                print("⚠️ Running with RANDOM weights (Untrained Mode)")
        else:
            print(f"⚠️ No weights found at {weights_path}")
            print("⚠️ Running with RANDOM weights (Untrained Mode)")
            # Initialize with random weights but set to eval mode
            self.model.eval()
        
        # 3. Define Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        print(f"✅ Model loaded on {self.device}")





    def analyze(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Evidence not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.jpg', '.png', '.jpeg']:
            return self._analyze_image(file_path)
        elif ext in ['.mp4', '.avi', '.mov']:
            return self._analyze_video(file_path)
        else:
            return 0.0

    def _analyze_image(self, path):
        try:
            # Open Image
            img = Image.open(path).convert('RGB')
            
            # Preprocess (Resize -> Tensor -> Normalize)
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.model(img_tensor)
                score = output.item() # Returns float between 0.0 (Real) and 1.0 (Fake)
            
            print(f"👁️ Image Scan Result: {score:.4f}")
            return score
        except Exception as e:
            print(f"❌ Analysis Failed: {e}")
            return 0.0

    def _analyze_video(self, path):
        # Strategy: Grab 5 random frames, analyze them, and average the score
        cap = cv2.VideoCapture(path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Pick 5 timestamps to sample
        sample_indices = np.linspace(0, frame_count - 10, 5).astype(int)
        
        scores = []
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if success:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Preprocess
                img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    output = self.model(img_tensor)
                    scores.append(output.item())
        
        cap.release()
        
        if not scores:
            return 0.0
            
        final_score = sum(scores) / len(scores)
        print(f"🎥 Video Scan Result: {final_score:.4f}")
        return final_score