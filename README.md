# 🛡️ Veritas Stream: Neural Forensic Engine
> **Advanced Deepfake Detection & Cryptographic Notarization Suite**  
> *Bridging Neural Representation Learning, High-Frequency Signal Forensics, and Immutable Chain of Custody.*

---

## 🧬 System Overview

**Veritas Stream** is an enterprise-grade digital forensics platform engineered to detect modern generative manipulation (GANs, Latent Diffusion, and Auto-Regressive synthesis) while maintaining a strict, tamper-evident legal chain of custody. 

Unlike standard neural classifiers that degrade when exposed to out-of-domain generators, Veritas Stream couples a custom **Dual-Stream Neural Architecture (`VeritasHybridNet`)** with **multi-stage sterilization**, **Error Level Analysis (ELA)**, **adversarial metadata heuristics**, and **RSA-4096 cryptographic notarization with live tripwires**.

---

## 🏗️ End-to-End System Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │             Incoming Media Upload            │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
    ================== STAGE 1: THE IRON CURTAIN ==================
    │  • Magic Byte Invariant Check (Anti-Spoofing MIME Filter)    │
    │  • VirusTotal v3 Global Intelligence SHA-256 Threat Lookup   │
    │  • ClamAV Real-time Local Heuristic Antivirus Quarantine     │
    │  • PIL Bitstream Structural Integrity Insurance Policy       │
    =====================================┬=========================
                                         │ Clean Buffer
                                         ▼
    =============== STAGE 2: VERITAS HYBRID NET GAUNTLET ===========
    │  • Stage 0: EXIF / XMP Generative Signature Trap             │
    │  • Stage 1: Error Level Analysis (ELA) Compression Variance  │
    │  • Stage 2: Global Context Feature Extraction                │
    │  • Stage 3: Watermark & Border Anomaly Hunter                │
    │  • Stage 4: MTCNN Face Hunter (FFHQ 40% Margin Padding)      │
    │  • Dynamic Contextual Anchor Voting                          │
    =====================================┬=========================
                                         │ Threat Score (0.0 - 1.0)
                                         ▼
    ============= STAGE 3: FORENSIC MATH & ELA HEATMAP =============
    │  • 15x Amplified ELA Difference Heatmap Saved to Vault       │
    │  • Deep EXIF Scraper (GPS, Camera Make/Model, Timestamps)    │
    │  • Proxy Privacy Scrubbing Flag (WhatsApp/Discord Filter)    │
    =====================================┬=========================
                                         │
                                         ▼
    ========== STAGE 4: CRYPTOGRAPHIC NOTARIZATION & VAULT =========
    │  • Canonical Payload: "Hash:<sha256>|Result:<confidence>"    │
    │  • RSA-4096 Digital Signature with PSS Padding (SHA-256)     │
    │  • Cold Storage Transfer: `vault/YYYY/MM/DD/<evidence>`      │
    │  • Tamper-Evident Tripwire (Live DB Signature Verification)  │
    │  • Court-Admissible Notarized PDF Generation via xhtml2pdf   │
    │  • Redundant AWS S3 Mirroring Archive                        │
    ================================================================
```

---

## 🧠 VeritasHybridNet Architecture

At the heart of the detection pipeline is **`VeritasHybridNet`**, a dual-stream neural network designed to capture both semantic layout flaws and micro-scale signal inconsistencies:

```
                      Input Image (3 x 224 x 224)
                        │                     │
      ┌─────────────────┴────────┐   ┌────────┴─────────────────┐
      │     SEMANTIC BRANCH      │   │     FORENSIC BRANCH      │
      │   (ConvNeXt-Tiny)        │   │   (NPRForensicStem)      │
      │                          │   │                          │
      │ • Stages 0 & 1: Frozen   │   │ • 2x2 Upsampling NPR:    │
      │ • Stages 2 & 3: Fine-    │   │   I - Nearest(AvgPool(I))│
      │   tuned Domain Adapt.    │   │ • 2D FFT Log-Magnitude:  │
      │ • Global Pooling & Norm  │   │   log(|rfft2(I)| + 1e-6) │
      │                          │   │ • Lightweight CNN Stem   │
      └─────────────┬────────────┘   └─────────────┬────────────┘
                    │                              │
                (768-d)                        (256-d)
                    │                              │
                    └──────────────┬───────────────┘
                                   │ Concatenation
                                   ▼
                       Fusion Vector (1024-d)
                                   │
                                   ▼
                       LayerNorm + GELU MLP Head
                                   │
                                   ▼
                     Linear Classifier (Real vs Fake)
```

1. **Semantic Branch (`ConvNeXt-Tiny`):**
   - Extracts 768-dimensional high-level semantic representations.
   - Early stages (0 and 1) remain frozen to preserve general vision primitives; stages 2 and 3 are fine-tuned for domain adaptation to synthetic lighting, unnatural reflections, and structural distortions.
2. **Forensic Branch (`NPRForensicStem`):**
   - Generates a **6-channel input** combining:
     - **Neighboring Pixel Relationships (NPR):** $R_{\text{NPR}} = I - \text{Nearest}(\text{AvgPool}_{2\times2}(I))$ capturing interpolation and resampling residuals.
     - **2D Fast Fourier Transform (FFT):** Spatial log-magnitude spectra via `torch.fft.rfft2` isolating periodic lattice artifacts and high-frequency energy decay anomalies.
   - Passes through a 4-stage convolutional stem ($6 \to 64 \to 128 \to 256 \to 256$) to output a **256-dimensional** forensic feature vector.
3. **Fusion Head:**
   - Concatenates representations into a **1024-dimensional** vector passed through `nn.LayerNorm(1024)` $\to$ `nn.Linear(1024, 512)` $\to$ `nn.GELU()` $\to$ `nn.Dropout(0.2)` $\to$ `nn.Linear(512, 2)`.
4. **Spatial Grid Preservation (`PreserveGridOrResize`):**
   - Applies 50% native-resolution `RandomCrop(224)` during training to preserve unresampled Latent VAE decoder grids alongside 50% full-scene `Resize((224, 224))`.
   - Dual-view inference in `model.analyze()` inspects both global scene composition and native pixel patches.

---

## 📊 Detection Benchmarks & Performance

`VeritasHybridNet` was trained on a balanced 16,000-image dataset (StyleGAN, Midjourney, Stable Diffusion, FFHQ human portraits, and real-world scenes) using FP16 mixed precision on an **NVIDIA GeForce RTX 3050 (6GB VRAM)**:

| Generative Engine / Media Type | Sample Evaluation | Deepfake Probability | Detection Verdict |
| :--- | :--- | :---: | :---: |
| **Google Gemini / Imagen** | `Gemini_Generated_Image_3jm47z.png` | **97.46%** | 🚨 **AI GENERATED** |
| **Google Gemini (Variant 2)** | `Gemini_Generated_Image_Hl2OIc.png` | **97.46%** | 🚨 **AI GENERATED** |
| **Stable Diffusion (SDXL / SD 1.5)** | `diff_sd_00500.png` | **98.58%** | 🚨 **AI GENERATED** |
| **Midjourney (v5 / v6)** | `diff_mj_00000.png` | **96.09%** | 🚨 **AI GENERATED** |
| **StyleGAN / ProGAN (Legacy)** | `sg_001DDU0NI4.jpg` | **98.44%** | 🚨 **AI GENERATED** |
| **Authentic Human Face** | `face_00000.jpg` | **3.23%** | 🛡️ **AUTHENTIC** |
| **Authentic General Scene** | `real_00000.JPEG` | **8.20%** | 🛡️ **AUTHENTIC** |

- **Combined Validation Accuracy:** **97.62%**
- **GAN Accuracy:** **99.8%** | **Diffusion Accuracy:** **97.6%**
- **Real Specificity:** **99.2%** (Faces) | **93.9%** (General Scenes)
- **Peak Hardware Footprint:** **1.51 GB VRAM** (Operating well within consumer 4GB/6GB GPU limits).

### 🧪 Out-of-Distribution Stress Test (46 Unseen Real-World Samples)

To measure true generalization beyond the validation split, `EnterpriseDeepFakeDetector` was evaluated (without modifying any model weights) against **46 out-of-distribution samples** across unseen generators (DALL-E 3, EXIF-stripped Google Gemini), diverse real-world photography (`Flickr30k`, `Unsplash/Picsum`), multi-resolution scaling, extreme JPEG compression, and post-generation edits (`0.50` decision threshold):

| Stress-Test Category | Samples Tested | Correct / Total | Accuracy | Key Findings |
| :--- | :--- | :---: | :---: | :--- |
| **1. Completely Unseen Generators** | DALL-E 3 (8) + Gemini w/ & w/o EXIF (2) | **10 / 10** | **100.00%** | Strong zero-shot transfer (`80.64%`–`98.63%` Fake); Gemini scored `97.53%` even with EXIF stripped. |
| **2. Real Photographs** | `Flickr30k` diverse environments (10) | **9 / 10** | **90.00%** | 1 borderline false positive (`50.70%`); real scenes show tighter margins (`41%`–`49.8%`). |
| **3. Different Sources & Resolutions** | `Unsplash/Picsum` Real + Resized DALL-E 3 (`256×256` to `2560×1440`) | **6 / 8** | **75.00%** | Resized DALL-E 3 detected **4/4** (`70.51%`–`96.53%`); real Unsplash had **2/4** false positives (`1280×720`, `1920×1080`). |
| **4. Heavy JPEG Compression** | Real & DALL-E 3 at JPEG `Q=25` and `Q=10` | **8 / 12** | **66.67%** | Extreme `Q=10`/`Q=25` compression attenuates high-frequency NPR/FFT traces (3 fake misses, 1 real FP). |
| **5. Edited After Generation** | Crop (70%), Contrast/Color, Blur, Sharpen, Border Pad, WebP→JPG | **6 / 6** | **100.00%** | High resilience across all 6 post-processing edits (`87.18%`–`98.24%` Fake). |
| **Overall Stress-Test Accuracy** | **Combined Out-of-Distribution Suite** | **39 / 46** | **84.78%** | High AI recall across unseen generators/edits; extreme JPEG compression and OOD real scenes remain primary challenges. |

---

## 🔏 Cryptographic Chain of Custody & The Tripwire

Veritas Stream provides mathematically verifiable evidence handling:
1. **RSA-4096 Digital Signatures:**
   Every report generates a canonical envelope:
   $$\text{Payload} = \texttt{"Hash:"} + \text{SHA256} + \texttt{"|Result:"} + \text{Score}$$
   This string is signed with RSA-4096 using PSS padding (MGF1 with SHA-256) and base64-encoded into the `ForensicReport` database record.
2. **The Cryptographic Tripwire:**
   When an archived record or duplicate is queried, the engine reconstructs the canonical envelope and validates it against the RSA Public Key. If any database record has been altered (e.g. flipping a verdict from Fake to Real), verification fails immediately, putting the UI into **`BREACH_DETECTED`** alert mode.
3. **Forensic Storage Vault:**
   Raw evidence is sanitized, stripped from quarantine, and moved to physical cold storage (`storage/vault/YYYY/MM/DD/`) with bind-mount isolation (`mount_vault.sh`) and optional AWS S3 mirroring.
4. **Notarized PDF Exports:**
   Generates tamper-proof legal forensic reports with embedded cryptographic seals via `xhtml2pdf`.

---

## 🛠️ Tech Stack

| Domain | Technology Stack |
| :--- | :--- |
| **AI / Deep Learning** | PyTorch 2.x, Torchvision (`ConvNeXt-Tiny`), Facenet-PyTorch (`MTCNN`) |
| **Forensic Signal Math** | Pillow (`ImageChops`, `ImageEnhance`), NumPy, PyTorch FFT (`torch.fft.rfft2`) |
| **Security & Ingestion** | `python-magic` (libmagic), ClamAV (`clamscan`), VirusTotal v3 REST API |
| **Cryptography & Ledger** | Python `cryptography` (RSA-4096, PSS, SHA-256), Django ORM (SQLite / PostgreSQL) |
| **Backend & Web UI** | Django 6.0, TailwindCSS, HTML5 Canvas, Space Grotesk Typography |
| **Reporting & Cloud** | `xhtml2pdf` (PISA Engine), AWS Boto3 (S3 Cold Storage Mirroring) |

---

## 📁 Repository Directory Structure

```text
veritas_stream/
├── apps/
│   ├── analysis/                 # Neural Detection Core
│   │   ├── core.py               # EnterpriseDeepFakeDetector & Gauntlet Runner
│   │   ├── networks.py           # VeritasHybridNet, NPRForensicStem & Meso4
│   │   └── weights/              # Serialized Weights (veritas_hybrid.pth)
│   ├── forensics/                # Forensics Engine, Web UI & Notarization
│   │   ├── core.py               # VeritasForensicEngine (Hash, ELA, Vault, RSA)
│   │   ├── forms.py              # Upload Evidence Validation Form
│   │   ├── models.py             # ForensicReport Database Schema
│   │   ├── utils.py              # Iron Curtain Sterilizer & Deep EXIF Scraper
│   │   ├── views.py              # Pipeline Controller & PDF Generator
│   │   └── templates/            # Tailwind Forensic UI & Notarized PDF Template
│   └── storage/                  # Active Django Media Storage (Quarantine & Vault)
├── config/                       # Django Project Configuration & URL Routing
├── storage/                      # Storage Architecture (Dataset, Vault, Temp Pools)
├── debug_veritas.py              # Single-Image Diagnostic & Bounding-Box Visualizer
├── mount_vault.sh                # WSL2 DrvFS External 1TB Vault Bind-Mount Utility
├── train.py                      # VeritasHybridNet Training & Fine-Tuning Pipeline
└── requirements.txt              # Production Dependencies
```

---

## 🚀 Quickstart & Installation

### 1. Prerequisites (Ubuntu / WSL2)
```bash
sudo apt update && sudo apt install -y \
    clamav \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libmagic1
sudo freshclam  # Update ClamAV virus signatures
```

### 2. Environment Setup
```bash
git clone git@github.com:Sanchay0123/Veritas-Stream.git
cd Veritas-Stream

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate RSA-4096 Forensic Keys
```bash
mkdir -p ~/.veritas_keys
openssl genpkey -algorithm RSA -out ~/.veritas_keys/private_key.pem -pkeyopt rsa_keygen_bits:4096
openssl rsa -pubout -in ~/.veritas_keys/private_key.pem -out ~/.veritas_keys/public_key.pem
```

### 4. Configuration (`.env`)
Create a `.env` file in the root directory:
```ini
SECRET_KEY=your_django_secret_key_here
DEBUG=True
VT_API_KEY=your_virustotal_api_key_here
VAULT_STORAGE_PATH=./storage/vault

# Optional: AWS S3 Mirror
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_SESSION_TOKEN=your_token
```

### 5. Mount Vault & Run Development Server
```bash
# Optional: Mount external storage vault
# chmod +x mount_vault.sh && ./mount_vault.sh

python manage.py migrate
python manage.py runserver
```
Navigate to `http://127.0.0.1:8000/` to access the Veritas Stream interface.

### 6. Training & Fine-Tuning VeritasHybridNet
To train or fine-tune on custom datasets:
```bash
python train.py \
    --data-dir storage/dataset/130k_faces/images \
    --save-path apps/analysis/weights/veritas_hybrid.pth \
    --epochs 4 \
    --batch-size 32 \
    --lr 8e-5
```

---

## 📄 Forensic & Legal Disclosure
Veritas Stream is an academic and forensic intelligence platform. While the system demonstrates high empirical accuracy against SOTA diffusion models, generative engines continuously evolve. Veritas Stream is designed to function as a verified, tamper-evident forensic instrument within a broader multi-modal chain of custody.
