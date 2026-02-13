import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import os
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from PIL import Image
import random
import io
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "storage/dataset/130k_faces/images" 
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 4
SAVE_PATH = "apps/analysis/weights/veritas_spectral.pkl"

# --- 1. ROBUST FREQUENCY CONVERTER ---
def get_spectrum(img_tensor):
    """
    FIXED: Uses Instance Normalization (Mean=0, Std=1).
    This reveals structural patterns without breaking the neural net with wild values.
    """
    fft = torch.fft.fft2(img_tensor)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shift)
    
    # Log scale
    spectrum = torch.log(magnitude + 1e-8)
    
    # INSTANCE NORMALIZATION (The Fix)
    # Forces the spectrum to be a clean, readable map for the AI
    spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-8)
    
    return spectrum

# --- 2. DATASET ---
class SpectralDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.classes = self.data.classes
        print(f"✅ Classes: {self.classes}") 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        spectrum = get_spectrum(img_tensor)
        combined_input = torch.cat([img_tensor, spectrum], dim=0)
        
        return combined_input, label

# --- 3. WHATSAPP SIMULATOR ---
class RandomJPEGCompression(object):
    def __init__(self, quality_range=(40, 90), p=0.5):
        self.quality_range = quality_range
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=random.randint(*self.quality_range))
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        return img

# --- 4. STABILIZED MODEL ---
class FrequencyEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
        
        old_conv = self.backbone._conv_stem
        new_conv = nn.Conv2d(6, old_conv.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        
        with torch.no_grad():
            # Keep RGB weights (Channels 0-2)
            new_conv.weight[:, :3] = old_conv.weight
            
            # ZERO-INIT FREQUENCY WEIGHTS (Channels 3-5)
            # This prevents the model from getting confused at the start.
            # It will learn to use these channels gradually.
            nn.init.constant_(new_conv.weight[:, 3:], 0.0)
            
        self.backbone._conv_stem = new_conv

    def forward(self, x):
        return self.backbone(x)

def train_model():
    print("🧠 Training Veritas SPECTRAL (Stabilized Mode)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Hardware: {device}")

    train_transform = transforms.Compose([
        RandomJPEGCompression(quality_range=(40, 80), p=0.5), 
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("📂 Loading Dataset...")
    full_dataset = SpectralDataset(DATA_DIR, transform=train_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = FrequencyEfficientNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

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

        # Validation
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
            print("💾 Model Saved.")

if __name__ == "__main__":
    train_model()