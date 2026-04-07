🛡️ Veritas Stream: Neural Forensic Engine
Advanced Deepfake Detection & Cryptographic Notarization Suite

🧬 The Core Mission
Veritas Stream is a digital forensics platform designed to bridge the gap between AI Analysis and Chain of Custody. Unlike standard detectors, Veritas implements a multi-stage "Iron Curtain" sterilization process to ensure that forensic artifacts are safe, untampered, and mathematically verified.

🏗️ System Architecture: The 4-Stage Pipeline
1. 🧊 The Iron Curtain (Sterilization)
Before any neural analysis occurs, the file passes through a security airlock:

Magic Byte Validation: Prevents MIME-type spoofing and extension-renaming attacks.

Global Threat Intelligence: Real-time SHA-256 querying via VirusTotal API.

Local Heuristics: Signature-based malware detection using ClamAV.

Structural Integrity: Header verification via PIL to prevent "Logic Bomb" image crashes.

2. 🧠 The Dual-Brain Ensemble (Neural Analysis)
Veritas utilizes a high-confidence ensemble of Vision Transformers (ViT):

Expert A (GAN-Hunter): Specialized in detecting upscaling and architectural patterns from Generative Adversarial Networks.

Expert B (Diffusion-Hunter): Optimized for Stable Diffusion, Midjourney, and DALL-E 3 latent-space artifacts.

ELA Engine: Error Level Analysis to detect non-uniform compression grids.

3. 🔍 Adversarial Metadata Heuristics
A custom expert system that interprets Data Absence as a forensic signal:

Red Flag (Synthetic): Total absence of EXIF/XMP headers in PNG/WebP containers.

Yellow Warning (Scrubbed): Detection of Social Media Proxy scrubbing (WhatsApp/Discord).

Legacy Detection: Identifying retired software signatures (e.g., Picasa) used in AI re-encoding.

4. 🔏 Cryptographic Notarization
Every report is notarized using RSA-4096 Digital Signatures. The final analysis score and file hash are signed with a private key, ensuring that the database record cannot be altered without breaking the cryptographic seal.

🛠️ Tech Stack
Backend: Django (Python 3.12)

AI/ML: PyTorch, HuggingFace Transformers (ViT)

Security: ClamScan, VirusTotal API, Cryptography (RSA)

Forensics: ExifRead, Pillow (ELA), Magic-Python

Reporting: xhtml2pdf (Notarized PDF Export)

🚀 Installation (WSL2 / Ubuntu)
Prerequisites
Bash
sudo apt update && sudo apt install -y clamav libcairo2-dev pkg-config python3-dev
Setup
Clone the repository:

Bash
git clone https://github.com/theimmortalcreator/veritas_stream.git
Configure Environment:
Create a .env file in the root directory:

Code snippet
VT_API_KEY=your_virustotal_key_here
RSA_PRIVATE_KEY_PATH=/path/to/your/private.pem
Run the Engine:

Bash
python manage.py runserver
📄 Forensic Disclosure
Veritas Stream is an academic research project. While it achieves high accuracy against current SOTA diffusion models, it should be used as a supplementary tool in a broader forensic investigation.