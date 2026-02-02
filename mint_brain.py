import torch
import torch.nn as nn
import os

# --- STANDARD MESONET ARCHITECTURE ---
class Meso4(nn.Module):
    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        
        self.maxpool = nn.MaxPool2d(2, 2)
        self.maxpool4 = nn.MaxPool2d(4, 4)
        self.dropout = nn.Dropout2d(0.5)
        
        self.fc1 = nn.Linear(1024, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        pass # We only need init for saving weights

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🧠 Minting a Fresh AI Brain...")
    
    # 1. Initialize Model
    model = Meso4()
    
    # 2. Define Save Path
    save_dir = "apps/analysis/weights"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "veritas_fresh.pkl")
    
    # 3. Save the Clean Weights
    torch.save(model.state_dict(), save_path)
    
    print(f"✅ Success! Fresh weights saved to: {save_path}")
    print("🚀 You can now point your Engine to this file.")
