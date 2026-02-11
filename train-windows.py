# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install efficientnet_pytorch tqdm numpy
# VeritasTrain/
    # train.py
    # real_vs_fake/
    #     real-vs-fake/
    #         train/
    #         valid/


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet 

# --- WINDOWS CONFIGURATION ---
# Use raw string (r"...") or forward slashes to avoid path errors
DATA_DIR = r"real_vs_fake/real-vs-fake" 
BATCH_SIZE = 48   # Safe size for 8GB VRAM (64 might OOM with heavy gradients)
LEARNING_RATE = 0.0001
EPOCHS = 4
SAVE_PATH = "veritas_pro_windows.pkl"

# --- DEFINE MODEL CLASS LOCALLY (To avoid importing complications) ---
class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    
    def forward(self, x):
        return self.model(x)

def train_model():
    print("🚀 Veritas Windows Native Training: i7 + RTX 4060...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Hardware: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   VRAM: {props.total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ WARNING: CPU Mode Detected! Did you install CUDA PyTorch?")

    # 1. HEAVY AUGMENTATION (Enabled for your i7)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Data
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'valid')
    
    # Check if paths exist to prevent frustration
    if not os.path.exists(train_dir):
        print(f"❌ ERROR: Could not find {train_dir}")
        print("   Make sure the 'real_vs_fake' folder is next to this script!")
        return

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    
    # 3. WINDOWS-SAFE LOADERS
    # num_workers=4 is safe. If it crashes with "Broken Pipe", set to 0.
    train_loader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,         
        pin_memory=True,
        persistent_workers=True 
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"✅ Data Loaded. Images: {len(train_data)}")

    # 4. Initialize Brain
    model = DeepFakeDetector().to(device)

    # 5. Class Balancing
    weights = torch.tensor([1.0, 1.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n🥊 Epoch {epoch+1}/{EPOCHS}")
        model.train()
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc="Training")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images) # Using the wrapper class forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        # Validation
        print("   Running Validation...")
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f"   Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"💾 Saving Best Model to {SAVE_PATH}")
            # Saving just the state dict (standard practice)
            torch.save(model.model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    # multiprocessing support on Windows requires this check
    try:
        train_model()
    except KeyboardInterrupt:
        print("\n🛑 Training Stopped by User.")