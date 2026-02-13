import torch
from apps.analysis.core import DeepFakeDetector
from PIL import Image, ImageDraw, ImageFont
import sys
import os

# CHANGE THIS to your image path
TEST_IMAGE = r"/mnt/c/Users/Lenovo/Desktop/random codes/Design Thinking and Innovation/Images/test2.jpg"

def debug_image(image_path):
    print(f"🔍 Debugging: {image_path}")
    detector = DeepFakeDetector()
    
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        print("❌ Could not open image.")
        return

    print("🕵️ Scanning for faces...")
    boxes, _ = detector.face_detector.detect(img)
    
    if boxes is None:
        print("❌ NO FACES DETECTED.")
        return

    print(f"✅ Found {len(boxes)} faces.")
    
    # Draw setup
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        # 1. Capture the Crop
        face_crop = detector.make_ffhq_crop(img, box)
        
        # 2. Check Resolution
        w, h = face_crop.size
        print(f"\n👤 Face {i+1} Resolution: {w}x{h} pixels")
        
        if w < 100 or h < 100:
            print("   ⚠️ WARNING: Face is very small! Analysis will be unreliable.")
        
        # 3. Save the Crop (So you can see it)
        crop_name = f"debug_face_{i+1}.jpg"
        face_crop.save(crop_name)
        print(f"   📸 Saved crop to: {crop_name}")

        # 4. Predict
        score = detector._predict_single(face_crop)
        print(f"   📊 Score: {score:.4f}")

        # Draw
        x1, y1, x2, y2 = box
        color = "red" if score > 0.5 else "green"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
        draw.text((x1, y1 - 40), f"{score:.2f}", fill=color, font=font)

    img.save("debug_output.jpg")
    print("\n✅ Done. Check 'debug_face_1.jpg' and 'debug_face_2.jpg'.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEST_IMAGE = sys.argv[1]
    debug_image(TEST_IMAGE)