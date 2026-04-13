import magic
import subprocess
import requests
from PIL import Image # Moved to top for efficiency
from django.conf import settings
import exifread
import os

def sterilize_file(file_path, file_hash):
    """
    🛡️ The Iron Curtain: 4-Stage Malware & Integrity Sterilization
    """
    # 1. MAGIC BYTE CHECK (The Bouncer)
    mime = magic.Magic(mime=True)
    detected_type = mime.from_file(file_path)
    print(f"🛡️  [Bouncer] Detected MIME: {detected_type}")
    
    if not detected_type.startswith('image/'):
        print(f"❌ [Bouncer] Blocked: {detected_type} is not a valid image.")
        return False, f"SECURITY ALERT: Magic Bytes suggest {detected_type}, not an image."

    # 2. VIRUSTOTAL HASH CHECK (Global Check)
    print(f"🌐 [Global Check] Querying VirusTotal for: {file_hash[:12]}...")
    vt_url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
    headers = {"x-apikey": settings.VT_API_KEY}
    
    try:
        response = requests.get(vt_url, headers=headers, timeout=5)
        if response.status_code == 200:
            stats = response.json()['data']['attributes']['last_analysis_stats']
            malicious_count = stats.get('malicious', 0)
            print(f"✅ [Global Check] VT hits found: {malicious_count}")
            if malicious_count > 0:
                return False, f"THREAT DETECTED: VirusTotal flagged this hash ({malicious_count} hits)."
        elif response.status_code == 404:
            print(f"ℹ️  [Global Check] Hash not found in VT database (Clean/New).")
    except Exception as e:
        print(f"⚠️  [Global Check] VT API Timeout or Error: {e}")

    # 3. CLAMAV LOCAL SCAN (Guard Dog)
    print(f"🐕 [Guard Dog] Running local clamscan...")
    try:
        result = subprocess.run(
            ['clamscan', '--allmatch', '--no-summary', file_path], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 1: 
            print(f"❌ [Guard Dog] VIRUS DETECTED by ClamAV!")
            return False, "MALWARE DETECTED: Local antivirus flagged the file buffer."
        print(f"✅ [Guard Dog] Local scan complete.")
    except Exception as e:
        print(f"❌ [Guard Dog] ClamAV Execution Error: {e}")

    # 4. STRUCTURAL INTEGRITY CHECK (The Insurance Policy)
    # This prevents the 'No JPEG data found' crash in the AI pipeline
    print(f"🔬 [Integrity] Validating image structure...")
    try:
        with Image.open(file_path) as img:
            img.verify() 
        print("✅ [Integrity] File structure is valid.")
    except Exception as e:
        print(f"❌ [Integrity] Malformed Image detected: {e}")
        return False, "CORRUPT DATA: File header is valid, but image data is unreadable or malformed."

    # --- ALL GATES CLEARED ---
    print("🏁 [Sterilizer] File cleared all security gates. Proceeding to AI Gauntlet.")
    return True, "CLEAN"




def extract_deep_metadata(file_path):
    """
    🕵️‍♂️ Deep Metadata Scraper: Prioritizing AI Detection over Privacy Scrubbing.
    """
    metadata = {}
    risk_flags = []
    info_flags = []
    
    try:
        with open(file_path, 'rb') as f:
            tags = exifread.process_file(f)
            
            # --- 🛡️ REFINED: AI vs. Social Media Logic ---
            if not tags:
                ext = file_path.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg']:
                    # 🟡 Yellow Alert: Logic in reports.html looks for "SCRUBBED"
                    risk_flags.append("SCRUBBED METADATA: Standard JPEG headers missing. Typical of WhatsApp/Discord compression.")
                else:
                    # 🔴 Red Alert: No "SCRUBBED" keyword, so template defaults to Red "Anomaly"
                    risk_flags.append("SYNTHETIC SIGNATURE: Total absence of forensic headers in a non-JPEG container. High possibility of AI generation (Gemini/Midjourney).")
            
            if tags:
                # 1. THE HARDWARE FINGERPRINT
                metadata['Make'] = str(tags.get('Image Make', 'Unknown'))
                metadata['Model'] = str(tags.get('Image Model', 'Unknown'))
                metadata['Software'] = str(tags.get('Image Software', 'None Detected'))
                
                # 🚨 RED FLAG: Software Manipulation
                software_blacklist = ['photoshop', 'gimp', 'picsart', 'midjourney', 'stable diffusion', 'dall-e']
                software_lower = metadata['Software'].lower()
                
                if any(s in software_lower for s in software_blacklist):
                    risk_flags.append(f"MANIPULATION TRACE: File edited via {metadata['Software']}")
                
                # 🚨 Specific Check: Outdated/Suspect Software
                if 'picasa' in software_lower:
                    risk_flags.append("LEGACY SOFTWARE: Use of Picasa detected. Highly unusual for modern captures; often used in AI-upscaling or legacy re-encoding.")

                # 2. THE TEMPORAL TRAP (Timestamps)
                metadata['DateTaken'] = str(tags.get('EXIF DateTimeOriginal', 'N/A'))
                metadata['DateDigitized'] = str(tags.get('EXIF DateTimeDigitized', 'N/A'))
                
                if metadata['DateTaken'] != 'N/A' and metadata['DateDigitized'] != 'N/A':
                    if metadata['DateTaken'] != metadata['DateDigitized']:
                        risk_flags.append("TEMPORAL ANOMALY: Creation and Digitization dates do not match.")

                # 3. GPS SURVEILLANCE
                has_gps = any('GPS' in key for key in tags.keys())
                metadata['Geotagged'] = "Yes" if has_gps else "No"
                if has_gps:
                    info_flags.append("LOCATION DATA: This image contains embedded GPS coordinates.")
            else:
                metadata['Header Status'] = "No metadata found in the bitstream."

    except Exception as e:
        print(f"⚠️ Metadata Extraction Error: {e}")
        return {"error": "Failed to parse EXIF data."}, []

    return metadata, risk_flags, info_flags