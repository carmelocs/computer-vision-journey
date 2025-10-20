# fid_helper.py
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.generator import Generator
from tqdm import tqdm
import subprocess
import shutil

def generate_images_for_fid(generator, z_dim, num_images, output_dir, device="cpu", batch_size=64):
    """
    Generate `num_images` using the generator and save them as PNGs in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Clear existing images (optional)
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    generator.eval()
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            current_batch = min(batch_size, num_images - i)
            noise = torch.randn(current_batch, z_dim).to(device)
            fake = generator(noise).cpu()
            fake = (fake + 1) / 2  # Denormalize to [0, 1]

            for j, img_tensor in enumerate(fake):
                img_pil = transforms.ToPILImage()(img_tensor)
                img_pil.save(os.path.join(output_dir, f"{i+j:06d}.png"))

    print(f"✅ Generated {num_images} images to {output_dir}")


def compute_fid(real_path, fake_path, device="cuda"):
    """
    Compute FID using pytorch-fid CLI.
    Assumes real_path and fake_path contain .png or .jpg images.
    """
    device_flag = "--device cuda" if "cuda" in device else "--device cpu"

    cmd = f"pytorch-fid '{real_path}' '{fake_path}' {device_flag} --num-workers 4"
    print(f"Running FID command: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        # Parse FID from output (e.g., "FID: 42.31")
        for line in result.stdout.splitlines():
            if "FID:" in line:
                fid_score = float(line.split("FID:")[-1].strip())
                return fid_score
        print("⚠️ Could not parse FID from output.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ FID computation failed:\n{e.stderr}")
        return None


def compute_fid_with_generator(
    generator,
    z_dim,
    real_images_path,
    num_fake_images=1000,
    device="cuda",
    temp_fake_dir="./outputs/fid_temp"
):
    """
    High-level function: generate fake images + compute FID.
    """
    print("🔄 Generating fake images for FID...")
    generate_images_for_fid(generator, z_dim, num_fake_images, temp_fake_dir, device=device)

    print("📊 Computing FID...")
    fid = compute_fid(real_images_path, temp_fake_dir, device=device)

    # Optional: clean up temp files
    # shutil.rmtree(temp_fake_dir)

    return fid

if __name__ == "__main__":
    # Example usage
    z_dim = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a pre-trained generator
    generator = Generator(z_dim=z_dim).to(device)
    generator.load_state_dict(torch.load("checkpoints/celeba/generator_epoch_20.pth", weights_only=True))

    real_images_path = "./data/celeba_64x64/celeba_train"  # Path to real images for FID comparison

    fid_score = compute_fid_with_generator(
        generator,
        z_dim,
        real_images_path,
        num_fake_images=1000,
        device=device
    )

    print(f"Calculated FID: {fid_score}")