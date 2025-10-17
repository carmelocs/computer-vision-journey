# dcgan_train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
config = {
    "epochs": 1,
    "batch_size": 64,
    # "lr": 0.0002,
    "d_lr": 0.0004,  # Two-time scale update rule (TTUR): Train G slower than D
    "g_lr": 0.0002,
    "z_dim": 100,
    "img_size": 64,
    "channels": 3,
    "sample_interval": 1,  # Save samples every epoch
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_save_dir": "checkpoints/celeba",
    "images_save_dir": "outputs/samples/celeba",
    "resume": "checkpoints/celeba/generator_epoch_20.pth",
    "dataset": "data/celeba_64x64",
}

# ----------------------------
# Initialize W&B
# ----------------------------
wandb.init(project="dcgan-celeba", config=config, mode="disabled")
config = wandb.config

# ----------------------------
# Data: CelebA
# ----------------------------
transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.CenterCrop(config.img_size),
    transforms.RandomHorizontalFlip(p=0.5),  # Better Data Augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # [-1, 1]
])

dataset = datasets.CelebA(
    root="./data",
    split="all",
    download=True,  # First run will download (~1.4GB). Or manually download: https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
    transform=transform
)

# dataset = datasets.ImageFolder(root=config.dataset, transform=transform)

dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

# ----------------------------
# Models
# ----------------------------
generator = Generator(z_dim=config.z_dim, img_channels=config.channels).to(config.device)
discriminator = Discriminator(img_channels=config.channels).to(config.device)

# ----------------------------
# Use current Model as warm start
# ----------------------------
if config.resume is not None:
    generator.load_state_dict(torch.load(config.resume, weights_only=True))  # Only load tensors (weights), no code or custom objects

# ----------------------------
# Optimizers & Loss
# ----------------------------
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=config.g_lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(0.5, 0.999))

# ----------------------------
# Fixed noise for consistent samples
# ----------------------------
fixed_noise = torch.randn(64, config.z_dim).to(config.device)

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(config.epochs):
    g_losses = []
    d_losses = []

    for i, (real_images, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")):
        real_images = real_images.to(config.device)
        batch_size = real_images.size(0)

        # Labels
        # real_labels = torch.ones(batch_size, 1).to(config.device)
        # fake_labels = torch.zeros(batch_size, 1).to(config.device)

        # Label Smoothing (Avoid Overconfidence)
        real_labels = torch.full((batch_size, 1), 0.9).to(config.device)
        fake_labels = torch.full((batch_size, 1), 0.1).to(config.device)

        # ------------------------
        # Train Discriminator
        # ------------------------
        d_optimizer.zero_grad()

        # Real loss
        pred_real = discriminator(real_images)
        loss_real = criterion(pred_real, real_labels)

        # Fake loss
        noise = torch.randn(batch_size, config.z_dim).to(config.device)
        fake_images = generator(noise)
        pred_fake = discriminator(fake_images.detach())
        loss_fake = criterion(pred_fake, fake_labels)

        d_loss = loss_real + loss_fake
        d_loss.backward()
        d_optimizer.step()

        # ------------------------
        # Train Generator
        # ------------------------
        g_optimizer.zero_grad()
        pred_fake_g = discriminator(fake_images)
        g_loss = criterion(pred_fake_g, real_labels)
        g_loss.backward()
        g_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    # ------------------------
    # Log & Save
    # ------------------------
    avg_g_loss = sum(g_losses) / len(g_losses)
    avg_d_loss = sum(d_losses) / len(d_losses)

    print(f"Epoch {epoch+1} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")

    # Generate samples
    with torch.no_grad():
        fake_samples = generator(fixed_noise).cpu()
        fake_samples = (fake_samples + 1) / 2  # Denormalize to [0,1]

    # Save grid
    grid_img = torchvision.utils.make_grid(fake_samples, nrow=8, padding=2)
    plt.figure(figsize=(8,8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig(f"{config.images_save_dir}/epoch_{epoch+1:03d}.png", bbox_inches="tight")
    plt.close()

    # Log to W&B
    wandb.log({
        "g_loss": avg_g_loss,
        "d_loss": avg_d_loss,
        "generated_images": [wandb.Image(grid_img)]
    })

# Save generator
torch.save(generator.state_dict(), f"{config.model_save_dir}/generator_epoch_{epoch+1}.pth")

print("✅ Training complete!")
wandb.finish()