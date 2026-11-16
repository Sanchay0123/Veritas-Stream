import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms
from PIL import Image


class Meso4(nn.Module):
    """
    MesoNet: a Compact Facial Video Forgery Detection Network.
    Paper: https://arxiv.org/abs/1809.00888
    """
    def __init__(self, num_classes=1):
        super(Meso4, self).__init__()
        
        # Layer 1: Conv2d -> BatchNorm -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(2, 2)

        # Layer 2
        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(8)

        # Layer 3
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(16)

        # Layer 4
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 16 * 16, 16) 
        self.fc2 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.sigmoid(x)


class NPRForensicStem(nn.Module):
    """
    Forensic feature extractor operating on high-frequency noise & spectral domain artifacts.
    
    Takes a 3-channel image and produces a 6-channel representation:
    - Channels 0..2: Neighboring Pixel Relationships (NPR) 2x2 upsampling residual:
      I - Nearest(AvgPool(I))
    - Channels 3..5: 2D FFT log-magnitude spectra via torch.fft.rfft2
    
    Passes the 6-channel map through a lightweight convolutional stem to produce a 256-d vector.
    """
    def __init__(self, in_channels=6, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
        
        # Lightweight convolutional stem (stride 2 downsampling stages)
        self.stem = nn.Sequential(
            # Stage 1: 224x224 -> 112x112
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            
            # Stage 2: 112x112 -> 56x56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            
            # Stage 3: 56x56 -> 28x28
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            # Stage 4: 28x28 -> 14x14
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            
            # Global aggregation
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def extract_forensic_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the 6-channel forensic map from input tensor x (B, 3, H, W).
        """
        # 1. Neighboring Pixel Relationships (NPR) residual: I - Nearest(AvgPool(I))
        avg_pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
        nearest_up = F.interpolate(avg_pooled, scale_factor=2, mode='nearest')
        npr_residual = x - nearest_up  # (B, 3, H, W)
        
        # 2. 2D FFT log-magnitude spectra via torch.fft.rfft2
        # Ensure float32 for FFT stability across AMP/autocast
        x_float = x.float()
        fft_complex = torch.fft.rfft2(x_float, norm='ortho')  # (B, 3, H, W//2 + 1)
        # Compute magnitude: sqrt(real^2 + imag^2)
        fft_mag = torch.sqrt(fft_complex.real ** 2 + fft_complex.imag ** 2 + 1e-8)  # (B, 3, H, W//2 + 1)
        fft_log = torch.log(fft_mag + 1e-6)
        
        # Resize/interpolate FFT log-magnitude to match input spatial dimensions (H, W)
        fft_spatial = F.interpolate(
            fft_log,
            size=(x.shape[-2], x.shape[-1]),
            mode='bilinear',
            align_corners=False
        )
        
        # Cast FFT spatial back to match npr_residual dtype (fp16 under autocast)
        fft_spatial = fft_spatial.to(npr_residual.dtype)
        
        # Concatenate along channel dimension -> (B, 6, H, W)
        return torch.cat([npr_residual, fft_spatial], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forensic_input = self.extract_forensic_channels(x)
        return self.stem(forensic_input)


class VeritasHybridNet(nn.Module):
    """
    VeritasHybridNet: Generalized Dual-Stream Deepfake Detection Architecture.
    
    Combines:
    1. Semantic Branch: ConvNeXt-Tiny with frozen early stages (0 and 1) and
       trainable domain adaptation stages (2 and 3) -> 768-d vector.
    2. Forensic Branch: NPRForensicStem capturing interpolation residuals and
       2D FFT log-magnitude spectrum anomalies -> 256-d vector.
    3. Fusion Head: 1024-d concatenation passed through LayerNorm + GELU MLP
       to a 2-class linear classifier (real vs fake).
    """
    def __init__(self, num_classes=2, pretrained=True, fake_class_idx=0):
        super().__init__()
        self.num_classes = num_classes
        self.fake_class_idx = fake_class_idx  # 0 for ImageFolder ['fake', 'real'] mapping
        
        # --- 1. SEMANTIC BRANCH (ConvNeXt-Tiny) ---
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.convnext = convnext_tiny(weights=weights)
        
        # Freeze early stages: stage 0 (index 0, 1) and stage 1 (index 2, 3)
        # convnext.features:
        # 0: Stem Conv4x4, 1: Stage 0 (3 blocks), 2: Downsample, 3: Stage 1 (3 blocks)
        # 4: Downsample, 5: Stage 2 (9 blocks), 6: Downsample, 7: Stage 3 (3 blocks)
        for i in range(4):
            for param in self.convnext.features[i].parameters():
                param.requires_grad = False
                
        # Keep later stages trainable for domain adaptation (stages 2 and 3: indices 4, 5, 6, 7)
        for i in range(4, len(self.convnext.features)):
            for param in self.convnext.features[i].parameters():
                param.requires_grad = True
                
        # Norm layer before classifier in ConvNeXt: LayerNorm2d((768,))
        self.semantic_norm = self.convnext.classifier[0]
        
        # --- 2. FORENSIC BRANCH (NPRForensicStem) ---
        self.forensic_stem = NPRForensicStem(in_channels=6, out_dim=256)
        
        # --- 3. FUSION HEAD ---
        # 768 (semantic) + 256 (forensic) = 1024-d
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Semantic features (768-d)
        s_feat = self.convnext.features(x)
        s_feat = self.convnext.avgpool(s_feat)
        s_feat = self.semantic_norm(s_feat)
        s_feat = torch.flatten(s_feat, 1)  # (B, 768)
        
        # 2. Forensic features (256-d)
        f_feat = self.forensic_stem(x)     # (B, 256)
        
        # 3. Concatenate and classify (1024-d -> 2)
        fused = torch.cat([s_feat, f_feat], dim=1)  # (B, 1024)
        logits = self.fusion_head(fused)            # (B, num_classes)
        return logits

    @torch.no_grad()
    def analyze(self, image_input, device=None) -> float:
        """
        Forensic inference helper method for EnterpriseDeepFakeDetector.
        Evaluates both global scene context and native-resolution VAE pixel grid crops
        to reliably detect both GANs and modern Latent Diffusion (Gemini, SD, Midjourney).
        
        Accepts:
            image_input: File path (str), PIL.Image.Image, or torch.Tensor.
            device: Optional torch.device.
        Returns:
            float score in [0.0, 1.0] representing the probability that the image is FAKE.
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
        self.eval()
        
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert('RGB')
        elif isinstance(image_input, torch.Tensor):
            pil_img = None
            tensor = image_input
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}. Expected str, PIL.Image, or Tensor.")

        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if pil_img is not None:
            w, h = pil_img.size
            # View A: Global context resized to 224x224
            t_resized = norm(transforms.ToTensor()(pil_img.resize((224, 224)))).unsqueeze(0).to(device)

            # View B: If image is high-res, evaluate 1:1 native crop to inspect VAE grid
            if w >= 256 and h >= 256:
                crop_img = transforms.CenterCrop(224)(pil_img)
                t_crop = norm(transforms.ToTensor()(crop_img)).unsqueeze(0).to(device)
                batch = torch.cat([t_resized, t_crop], dim=0)
            else:
                batch = t_resized

            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = self.forward(batch)
            else:
                logits = self.forward(batch)
                
            probs = F.softmax(logits, dim=1)
            fake_probs = probs[:, self.fake_class_idx]
            # Max fake probability across full context and local native patch
            return float(torch.max(fake_probs).item())
        else:
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device)
            
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = self.forward(tensor)
            else:
                logits = self.forward(tensor)
                
            probs = F.softmax(logits, dim=1)
            fake_prob = float(probs[0, self.fake_class_idx].item())
            return fake_prob
