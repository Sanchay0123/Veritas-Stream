import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, random_split, Dataset
import os
from tqdm import tqdm
from PIL import Image, ImageFilter
import random
import io
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "storage/dataset/130k_faces/images" 
BATCH_SIZE = 32  
LEARNING_RATE = 1e-5 
WEIGHT_DECAY = 0.01
EPOCHS = 4
SAVE_PATH = "apps/analysis/weights/veritas_vit.pkl"

class RandomJPEGCompression(object):
    def __init__(self, quality_range=(50, 90), p=0.5):
        self.quality_range = quality_range
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=random.randint(*self.quality_range))
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        return img

def simulate_diffusion(img):
    """Applies modern AI flaws ONLY to fake images"""
    # 1. Micro-blur to destroy GAN checkerboards and simulate Diffusion smoothness
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    # 2. Add synthetic latent noise
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(0, random.uniform(2, 5), img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

# 🛑 THE FIX: SMART DATASET WRAPPER
class SmartDataset(Dataset):
    def __init__(self, base_dataset, base_transform):
        self.base_dataset = base_dataset
        self.base_transform = base_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        
        # 🛑 ONLY apply the simulator if the image is FAKE (label == 0)
        if label == 0 and random.random() < 0.4:
            img = simulate_diffusion(img)
            
        # Apply standard transforms (Resize, Normalize, etc) to ALL images
        img_tensor = self.base_transform(img)
        return img_tensor, label

def train_model():
    print("🧠 Training Veritas (Un-Poisoned Smart Dataset)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Hardware: {device}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # Base transforms applied to EVERYTHING
    base_transform = transforms.Compose([
        RandomJPEGCompression(quality_range=(50, 90), p=0.3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📂 Loading Dataset...")
    raw_dataset = datasets.ImageFolder(DATA_DIR) # Load without transforms first
    print(f"✅ Class Map Confirmed: {raw_dataset.class_to_idx}")
    
    # Wrap it in our smart logic
    full_dataset = SmartDataset(raw_dataset, base_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
        
    for block in model.encoder.layers[-1:]:
        for param in block.parameters():
            param.requires_grad = True
            
    for param in model.encoder.ln.parameters():
        param.requires_grad = True
    
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    model = model.to(device)

    unfrozen_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(unfrozen_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Keeping the class weights to force it to hunt fakes
    class_weights = torch.tensor([2.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"   Validation Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print("💾 Saved Lethal ViT Model.")

if __name__ == "__main__":
    train_model()