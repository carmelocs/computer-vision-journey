# resize_celeba.py
import os
from PIL import Image
from tqdm import tqdm

# ----------------------------
# Config
# ----------------------------
input_dir = "./data/celeba/img_align_celeba"   # Original high-res CelebA
output_dir = "./data/celeba_64x64"             # New folder for resized images
target_size = (64, 64)

# ----------------------------
# Create Output Folder
# ----------------------------
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Get All Image Files
# ----------------------------
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(image_files)} images. Resizing to {target_size}...")

# ----------------------------
# Resize Images
# ----------------------------
for filename in tqdm(image_files, desc="Resizing"):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with Image.open(input_path) as img:
        # Resize using LANCZOS (high-quality downsampling)
        img_resized = img.resize(target_size, Image.LANCZOS)
        img_resized.save(output_path, quality=95)  # Preserve quality

print(f"✅ Resized {len(image_files)} images to {target_size}. Saved to: {output_dir}")