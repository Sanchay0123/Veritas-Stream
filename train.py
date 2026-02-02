import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from apps.analysis.core import DeepFakeDetector # Loads your EfficientNet

# --- CONFIGURATION ---
# The path you found in your 'ls' command
DATA_DIR = "storage/dataset/real_vs_fake/real-vs-fake"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 1 # Start with 1 to test. Industry standard is 10-20.
SAVE_PATH = "apps/analysis/weights/veritas_pro.pkl"

def train_model():
    print("🚀 Veritas Training System Initiated...")
    
    # 1. Check Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Hardware: {device} (CUDA is {'ON' if device.type=='cuda' else 'OFF'})")

    # 2. Prepare Data Transformations (Must match EfficientNet requirements)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load the Dataset
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'valid')
    
    if not os.path.exists(train_dir):
        print(f"❌ ERROR: Could not find training data at {train_dir}")
        return

    print("📁 Loading 140k Images (This may take a moment)...")
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"✅ Data Loaded. Training Images: {len(train_data)} | Validation Images: {len(val_data)}")

    # 4. Initialize the Brain
    detector = DeepFakeDetector() # This calls your class from core.py
    model = detector.model
    model = model.to(device)

    # 5. Define Rules of the Game
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. The Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n🥊 Epoch {epoch+1}/{EPOCHS}")
        print("-" * 20)
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress Bar
        loop = tqdm(train_loader, desc="Training")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        epoch_acc = 100 * correct / total
        print(f"   Training Accuracy: {epoch_acc:.2f}%")

        # --- VALIDATION PHASE ---
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

        # 7. Save the Best Brain
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"💾 New Record! Saving model to {SAVE_PATH}")
            torch.save(model.state_dict(), SAVE_PATH)

    print("\n🏁 Training Complete.")
    print(f"🏆 Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()
