# fid_eval.py
import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from models.generator import Generator

# ----------------------------
# Config
# ----------------------------
config = {
    "z_dim": 100,
    "img_size": 64,
    "channels": 3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_images": 5000,
    "batch_size": 64,
    "generator_path": "checkpoints/celeba/generator_epoch_020.pth"
}

# ----------------------------
# Load Generator
# ----------------------------
generator = Generator(z_dim=config["z_dim"], img_channels=config["channels"]).to(config["device"])
generator.load_state_dict(torch.load(config["generator_path"], map_location=config["device"], weights_only=True))
generator.eval()

# ----------------------------
# Create Output Folder
# ----------------------------
os.makedirs("outputs/fid_generated", exist_ok=True)

# ----------------------------
# Generate Images
# ----------------------------
with torch.no_grad():
    for i in tqdm(range(0, config["num_images"], config["batch_size"]), desc="Generating"):
        batch_size = min(config["batch_size"], config["num_images"] - i)
        noise = torch.randn(batch_size, config["z_dim"]).to(config["device"])
        fake_images = generator(noise).cpu()

        # Denormalize: [-1,1] → [0,1]
        fake_images = (fake_images + 1) / 2

        # Save each image
        for j, img in enumerate(fake_images):
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(f"outputs/fid_generated/{i+j+1:05d}.png")

print(f"✅ Generated {config['num_images']} images to outputs/fid_generated/")