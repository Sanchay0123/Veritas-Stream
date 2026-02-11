import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

# --- CONFIGURATION ---
# Points directly to the 'images' folder containing 'real' and 'fake'
DATA_DIR = "storage/dataset/130k_faces/images" 
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 3
SAVE_PATH = "apps/analysis/weights/veritas_pro.pkl"

def train_model():
    print("🧠 Training Veritas on 130k Modern Faces (Flux/SDXL)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Hardware: {device}")

    # 1. High-Res Transform (256x256)
    # No blurring, no degrading. We trust this high-quality data.
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Load Data
    print(f"📂 Loading data from: {DATA_DIR}")
    
    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    except FileNotFoundError:
        print("❌ ERROR: Could not find folder. Did you run setup_dataset.py?")
        return

    # 3. Check Classes
    # Expected: ['fake', 'real'] -> fake=0, real=1
    classes = full_dataset.classes
    print(f"✅ Classes Found: {classes}")
    if classes[0] != 'fake':
        print("⚠️ WARNING: Class order is unusual! Check your folder names.")
        # If 'fake' is not 0, the logic in core.py will need flipping.
        # But alphabetically, 'f' < 'r', so it should be fine.

    # 4. Auto-Split (80% Train, 20% Val)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transform
    val_data.dataset.transform = val_transform 
    
    print(f"   Training Images: {len(train_data)}")
    print(f"   Validation Images: {len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 5. Model
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model = model.to(device)

    # 6. Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. Training Loop
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
            print("💾 Saving Veritas Pro...")
            torch.save(model.state_dict(), SAVE_PATH)

if __name__ == "__main__":
    train_model()