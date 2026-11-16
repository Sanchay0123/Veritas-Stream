import os
import io
import random
import argparse
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
DEFAULT_EPOCHS = 4
DEFAULT_LR = 8e-5
DEFAULT_WEIGHT_DECAY = 1e-4


# ==============================================================================
# 1. SPATIAL & FORENSIC AUGMENTATIONS
# ==============================================================================
class PreserveGridOrResize(object):
    """
    Dual-mode spatial adaptation:
    - 50% probability: Native-resolution RandomCrop(224) to preserve raw VAE
      decoder grids, micro-textures, and high-frequency NPR residuals.
    - 50% probability: Resize((224, 224)) to capture global semantic context,
      full-scene composition, and diffusion lighting/reflection artifacts.
    """
    def __init__(self, size=224, p_crop=0.5):
        self.size = size
        self.p_crop = p_crop
        self.crop = transforms.RandomCrop(size)
        self.resize = transforms.Resize((size, size))

    def __call__(self, img):
        w, h = img.size
        # Fallback to resize if image is smaller than target crop
        if w < self.size or h < self.size:
            return self.resize(img)
            
        if random.random() < self.p_crop:
            return self.crop(img)
        else:
            return self.resize(img)


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
# 2. BALANCED FORENSIC DATASET WRAPPER
# ==============================================================================
class ForensicImageDataset(Dataset):
    """
    Fast Dataset wrapper loading from pre-partitioned sample tuples (path, label).
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
# 3. FINE-TUNING PIPELINE
# ==============================================================================
def train_model(
    data_dir=DEFAULT_DATA_DIR,
    save_path=DEFAULT_SAVE_PATH,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    num_workers=4
):
    print("=" * 72)
    print("🚀 VeritasHybridNet: Emergency Modern Diffusion Fine-Tuning")
    print("   Adapting Dual-Stream Architecture for StyleGAN + Latent Diffusion")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Compute Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"   GPU: {gpu_name} ({total_vram:.2f} GB Total VRAM)")
        torch.cuda.reset_peak_memory_stats(device)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # --------------------------------------------------------------------------
    # Transforms: PreserveGridOrResize + Symmetric Augmentations
    # --------------------------------------------------------------------------
    train_transform = transforms.Compose([
        RandomJPEGCompression(quality_range=(50, 95), p=0.4),
        RandomGaussianBlur(radius_range=(0.2, 1.0), p=0.3),
        PreserveGridOrResize(size=224, p_crop=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --------------------------------------------------------------------------
    # Ingest Balanced 16,000-Image Dataset
    # --------------------------------------------------------------------------
    print(f"\n📂 Loading Dataset from: {data_dir}...")
    fake_dir = os.path.join(data_dir, "fake")
    real_dir = os.path.join(data_dir, "real")

    fake_files = sorted(os.listdir(fake_dir))
    real_files = sorted(os.listdir(real_dir))

    # Stratify fake and real sub-domains
    sg_fakes = [os.path.join(fake_dir, f) for f in fake_files if f.startswith("sg_")]
    diff_fakes = [os.path.join(fake_dir, f) for f in fake_files if f.startswith("diff_")]
    face_reals = [os.path.join(real_dir, f) for f in real_files if f.startswith("face_")]
    pool_reals = [os.path.join(real_dir, f) for f in real_files if f.startswith("real_")]

    print(f"   Breakdown: {len(sg_fakes):,} StyleGAN Fakes | {len(diff_fakes):,} Modern Diffusion Fakes")
    print(f"              {len(face_reals):,} Real Faces    | {len(pool_reals):,} Real General Scenes")
    print(f"   Total Balanced Samples: {len(sg_fakes) + len(diff_fakes) + len(face_reals) + len(pool_reals):,}")

    # Deterministic 80/20 train/validation split per sub-domain
    random.seed(42)
    random.shuffle(sg_fakes)
    random.shuffle(diff_fakes)
    random.shuffle(face_reals)
    random.shuffle(pool_reals)

    train_samples = (
        [(p, 0) for p in sg_fakes[:3200]] +
        [(p, 0) for p in diff_fakes[:3200]] +
        [(p, 1) for p in face_reals[:3200]] +
        [(p, 1) for p in pool_reals[:3200]]
    )

    val_samples = (
        [(p, 0) for p in sg_fakes[3200:]] +
        [(p, 0) for p in diff_fakes[3200:]] +
        [(p, 1) for p in face_reals[3200:]] +
        [(p, 1) for p in pool_reals[3200:]]
    )

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"📊 Partitions: {len(train_samples):,} Training Samples (6,400 Fake, 6,400 Real)")
    print(f"              {len(val_samples):,} Validation Samples (1,600 Fake, 1,600 Real)")

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
    # Initialize VeritasHybridNet & Load Pre-trained Weights
    # --------------------------------------------------------------------------
    print("\n🏗️ Instantiating VeritasHybridNet...")
    model = VeritasHybridNet(num_classes=2, pretrained=True, fake_class_idx=0)
    model = model.to(device)

    if os.path.exists(save_path):
        print(f"📦 Loading pre-existing checkpoint from {save_path} for fine-tuning...")
        state_dict = torch.load(save_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Pre-trained checkpoint loaded successfully!")
    else:
        print("⚠️ No existing checkpoint found; training from base pre-trained ConvNeXt.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"   Trainable Parameters: {trainable_count:,} (Stages 2/3 + Forensic Stem + Fusion)")

    # --------------------------------------------------------------------------
    # Optimizer, Loss Criterion & Scheduler
    # --------------------------------------------------------------------------
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Forensic class weighting: 1.2x weight for fake class
    class_weights = torch.tensor([1.2, 1.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # FP16 Automatic Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # --------------------------------------------------------------------------
    # Fine-Tuning Loop
    # --------------------------------------------------------------------------
    best_acc = 0.0
    print("\n🚀 Commencing Fine-Tuning...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Fine-Tune]", unit="batch")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

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
        # Validation Evaluation with Domain Breakdown
        # ----------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Sub-domain tracking: sg_fake, diff_fake, face_real, pool_real
        domain_stats = {"sg": [0, 0], "diff": [0, 0], "face": [0, 0], "pool": [0, 0]}

        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # Attribute batch to domains
                batch_start = i * batch_size
                for b_idx in range(len(labels)):
                    sample_path, _ = val_samples[batch_start + b_idx]
                    base_name = os.path.basename(sample_path)
                    is_correct = int(preds[b_idx] == labels[b_idx])
                    if base_name.startswith("sg_"):
                        domain_stats["sg"][0] += is_correct
                        domain_stats["sg"][1] += 1
                    elif base_name.startswith("diff_"):
                        domain_stats["diff"][0] += is_correct
                        domain_stats["diff"][1] += 1
                    elif base_name.startswith("face_"):
                        domain_stats["face"][0] += is_correct
                        domain_stats["face"][1] += 1
                    elif base_name.startswith("real_"):
                        domain_stats["pool"][0] += is_correct
                        domain_stats["pool"][1] += 1

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / val_total
        scheduler.step()

        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            vram_str = f"Peak VRAM: {peak_vram_mb:.1f} MB ({peak_vram_mb/1024:.2f} GB)"
        else:
            vram_str = "CPU Mode"

        sg_acc = 100.0 * domain_stats["sg"][0] / max(1, domain_stats["sg"][1])
        diff_acc = 100.0 * domain_stats["diff"][0] / max(1, domain_stats["diff"][1])
        face_acc = 100.0 * domain_stats["face"][0] / max(1, domain_stats["face"][1])
        pool_acc = 100.0 * domain_stats["pool"][0] / max(1, domain_stats["pool"][1])

        print(f"\n📊 Epoch {epoch}/{epochs} Results:")
        print(f"   Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   🔍 Domain Accuracy: GAN: {sg_acc:.1f}% | Diffusion: {diff_acc:.1f}% | Face Real: {face_acc:.1f}% | Scene Real: {pool_acc:.1f}%")
        print(f"   {vram_str}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Saved new best fine-tuned checkpoint to {save_path} (Val Acc: {best_acc:.2f}%)\n")
        else:
            print(f"   Best Val Acc remains: {best_acc:.2f}%\n")

    print("=" * 72)
    print(f"🏁 Fine-Tuning Complete!")
    print(f"   Best Combined Validation Accuracy: {best_acc:.2f}%")
    print(f"   Final Checkpoint: {save_path}")
    print("=" * 72)

    return model, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tune VeritasHybridNet for Diffusion Adaptation")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Path to image dataset")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Path to save model weights")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (default: 32)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate (default: 8e-5)")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay (default: 1e-4)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        save_path=args.save_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.workers
    )
