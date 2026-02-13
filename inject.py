import os
import shutil
import random
from tqdm import tqdm

# CONFIG
# Adjust this path if your unzip structure is different!
# Usually it's storage/temp_gan/real_vs_fake/real-vs-fake/train/fake
INITIAL_SOURCE_GAN_DIR = "storage/temp_gan/real_vs_fake/real-vs-fake/train/fake" 
DEST_FAKE_DIR = "storage/dataset/130k_faces/images/fake"
TARGET_COUNT = 20000

def inject():
    # Fix: Use a local variable first, then decide on the final path
    source_dir = INITIAL_SOURCE_GAN_DIR
    
    print(f"💉 Injecting {TARGET_COUNT} GAN images into your dataset...")
    
    # 1. Check if source exists
    if not os.path.exists(source_dir):
        print(f"⚠️ Could not find default path: {source_dir}")
        print("   Searching 'storage/temp_gan' for a valid 'fake' folder...")
        
        found = False
        # Walk through the temp folder to find where the images actually are
        for root, dirs, files in os.walk("storage/temp_gan"):
            # Look for a folder named 'fake' that has images in it
            if "fake" in os.path.basename(root).lower():
                # Check if it actually contains images (avoid empty folders)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if len(image_files) > 1000:
                    print(f"   ✅ Found candidate: {root} ({len(image_files)} images)")
                    source_dir = root
                    found = True
                    break
        
        if not found:
            print("❌ ERROR: Could not find any folder with GAN images.")
            return

    # 2. Get all GAN files from the valid source_dir
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   Found {len(all_files)} GAN images available.")
    
    if len(all_files) == 0:
        print("❌ ERROR: Source folder is empty.")
        return

    # 3. Select 20,000 (or all if fewer)
    if len(all_files) < TARGET_COUNT:
        print("⚠️ Not enough images! Taking all of them.")
        selected_files = all_files
    else:
        # Randomly select 20k to avoid alphabetical bias
        selected_files = random.sample(all_files, TARGET_COUNT)
        
    # 4. Move them
    print(f"   Moving {len(selected_files)} images to {DEST_FAKE_DIR}...")
    
    # Ensure destination exists
    os.makedirs(DEST_FAKE_DIR, exist_ok=True)
    
    count = 0
    for file in tqdm(selected_files):
        try:
            src = os.path.join(source_dir, file)
            # Rename to 'gan_' so we know origin, and avoid overwriting Diffusion files
            dst_name = f"gan_stylegan_{file}"
            dst = os.path.join(DEST_FAKE_DIR, dst_name)
            
            # Use copy instead of move if you want to keep the temp folder for now
            shutil.move(src, dst)
            count += 1
        except Exception as e:
            print(f"Error moving {file}: {e}")
            
    print(f"✅ Success! Added {count} GAN images.")
    print(f"   Total Fakes should now be ~80,000 (Mix of Diffusion + GAN).")

if __name__ == "__main__":
    inject()