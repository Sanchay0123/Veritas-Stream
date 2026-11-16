import os
import io
import random
import argparse
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from apps.analysis.networks import VeritasHybridNet

# --- CONFIGURATION DEFAULTS ---
DEFAULT_DATA_DIR = "storage/dataset/130k_faces/images"
DEFAULT_SAVE_PATH = "apps/analysis/weights/veritas_hybrid.pth"
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 5
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4


# ==============================================================================
# 1. SYMMETRIC FORENSIC AUGMENTATIONS (NO LABEL LEAKAGE)
# ==============================================================================
class RandomJPEGCompression(object):
    """
    Applies JPEG compression artifact simulation with randomized quality factor.
    CRITICAL: Applied equally to both real and fake images to avoid shortcut learning.
    """
    def __init__(self, quality_range=(50, 95), p=0.4):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            buffer = io.BytesIO()
            q = random.randint(*self.quality_range)
            img.save(buffer, format='JPEG', quality=q)
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        return img


class RandomGaussianBlur(object):
    """
    Applies subtle Gaussian blur to simulate anti-aliasing / lens blur.
    CRITICAL: Applied equally to both real and fake classes to prevent label leakage.
    """
    def __init__(self, radius_range=(0.2, 1.0), p=0.3):
        self.radius_range = radius_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            r = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=r))
        return img


# ==============================================================================
# 2. DATASET WRAPPER
# ==============================================================================
class ForensicImageDataset(Dataset):
    """
    Lightweight Dataset wrapper loading from pre-partitioned sample tuples (path, label).
    Ensures exact 1:1 pixel crops without resampling blur.
    """
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert('RGB')
            tensor = self.transform(img)
        return tensor, label


# ==============================================================================
# 3. TRAINING PIPELINE
# ==============================================================================
def train_model(
    data_dir=DEFAULT_DATA_DIR,
    save_path=DEFAULT_SAVE_PATH,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    max_samples=10000,
    full_dataset=False,
    num_workers=4
):
    print("=" * 70)
    print("🧠 VeritasHybridNet Training Pipeline")
    print("   Dual-Stream Semantic (ConvNeXt-Tiny) + Forensic (NPR + FFT) Engine")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Compute Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"   GPU: {gpu_name} ({total_vram:.2f} GB Total VRAM)")
        torch.cuda.reset_peak_memory_stats(device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --------------------------------------------------------------------------
    # Data Augmentation & Forensic Pixel Transforms
    # --------------------------------------------------------------------------
    # Symmetric augmentations applied to ALL images:
    # RandomCrop(224) extracts 1:1 patches preserving local pixel grid relationships
    train_transform = transforms.Compose([
        RandomJPEGCompression(quality_range=(50, 95), p=0.4),
        RandomGaussianBlur(radius_range=(0.2, 1.0), p=0.3),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --------------------------------------------------------------------------
    # Dataset Ingestion & Balanced Partitioning
    # --------------------------------------------------------------------------
    print(f"\n📂 Loading Dataset from: {data_dir}...")
    raw_dataset = datasets.ImageFolder(data_dir)
    class_to_idx = raw_dataset.class_to_idx
    print(f"✅ Class Mapping Confirmed: {class_to_idx}")
    print(f"   Total Images Available on Disk: {len(raw_dataset):,}")

    fake_idx = class_to_idx.get("fake", 0)
    real_idx = class_to_idx.get("real", 1)

    fake_samples = [s for s in raw_dataset.samples if s[1] == fake_idx]
    real_samples = [s for s in raw_dataset.samples if s[1] == real_idx]
    print(f"   Available breakdown: {len(fake_samples):,} fake, {len(real_samples):,} real")

    # Balanced sampling
    random.seed(42)
    random.shuffle(fake_samples)
    random.shuffle(real_samples)

    if not full_dataset and max_samples is not None and max_samples < len(raw_dataset):
        samples_per_class = max_samples // 2
        fake_pool = fake_samples[:samples_per_class]
        real_pool = real_samples[:samples_per_class]
        print(f"🎯 Using balanced sub-pool of {len(fake_pool) + len(real_pool):,} samples ({len(fake_pool):,} per class)")
    else:
        min_count = min(len(fake_samples), len(real_samples))
        fake_pool = fake_samples[:min_count]
        real_pool = real_samples[:min_count]
        print(f"🎯 Using balanced full-pool of {len(fake_pool) + len(real_pool):,} samples")

    # 80/20 train/val split per class
    fake_split = int(0.8 * len(fake_pool))
    real_split = int(0.8 * len(real_pool))

    train_samples = fake_pool[:fake_split] + real_pool[:real_split]
    val_samples = fake_pool[fake_split:] + real_pool[real_split:]
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"📊 Partitions: {len(train_samples):,} Training Samples | {len(val_samples):,} Validation Samples")

    train_dataset = ForensicImageDataset(train_samples, train_transform)
    val_dataset = ForensicImageDataset(val_samples, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------------------------
    # Initialize VeritasHybridNet Architecture
    # --------------------------------------------------------------------------
    print("\n🏗️ Instantiating VeritasHybridNet...")
    model = VeritasHybridNet(num_classes=2, pretrained=True, fake_class_idx=fake_idx)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_count:,} (Stages 2/3 + Forensic Stem + Fusion)")
    print(f"   Frozen Parameters: {total_params - trainable_count:,} (Early ConvNeXt Stages 0/1)")

    # --------------------------------------------------------------------------
    # Optimizer, Loss Criterion & Scheduler
    # --------------------------------------------------------------------------
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Class weighting: higher penalty for missing a fake (false negative)
    # fake_idx=0 -> weight 1.2, real_idx=1 -> weight 1.0
    weights_tensor = torch.zeros(2, device=device)
    weights_tensor[fake_idx] = 1.2
    weights_tensor[real_idx] = 1.0
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)

    # FP16 Automatic Mixed Precision Scaler for RTX 3050 VRAM safety
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    best_acc = 0.0
    print("\n🚀 Beginning Training Loop...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # PyTorch FP16 Mixed Precision
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "train_acc": f"{100.0 * train_correct / train_total:.2f}%",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        epoch_loss = running_loss / train_total
        train_acc = 100.0 * train_correct / train_total

        # ----------------------------------------------------------------------
        # Validation Evaluation
        # ----------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / val_total
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # VRAM monitoring
        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            vram_str = f"Peak VRAM: {peak_vram_mb:.1f} MB ({peak_vram_mb/1024:.2f} GB)"
        else:
            vram_str = "CPU Mode"

        print(f"\n📊 Epoch {epoch}/{epochs} Results:")
        print(f"   Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}% (Target: >90%)")
        print(f"   {vram_str}")

        # Checkpoint Saving
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Saved new best model checkpoint to {save_path} (Val Acc: {best_acc:.2f}%)\n")
        else:
            print(f"   Best Val Acc remains: {best_acc:.2f}%\n")

    print("=" * 70)
    print(f"🏁 Training Complete!")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")
    print(f"   Model Weights Saved: {save_path}")
    if device.type == "cuda":
        final_peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(f"   Final Peak VRAM Usage: {final_peak_vram:.1f} MB (Constraint: < 3072 MB)")
        assert final_peak_vram < 3072, "VRAM exceeded 3GB constraint!"
    print("=" * 70)

    return model, best_acc


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VeritasHybridNet for Deepfake Detection")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to image dataset")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save model weights")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs (default: 5)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay (default: 1e-4)")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max balanced samples for training (default: 10000)")
    parser.add_argument("--full", action="store_true", help="Train on full dataset without sample cap")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        save_path=args.save_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_samples=args.max_samples,
        full_dataset=args.full,
        num_workers=args.workers
    )
