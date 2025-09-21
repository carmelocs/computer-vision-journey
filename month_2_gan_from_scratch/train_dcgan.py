# train_dcgan.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary

import yaml
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb  # Optional but recommended

from dcgan_model import Generator, Discriminator

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create save dirs
os.makedirs(config["save_dir"], exist_ok=True)
os.makedirs(config["model_save_dir"], exist_ok=True)

# Transformations
transform = transforms.Compose([
    transforms.Resize(config["image_size"]),
    transforms.CenterCrop(config["image_size"]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Range: [-1, 1]
])

# Load dataset
if config["dataset"] == "cifar10":
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
elif config["dataset"] == "celeba":
    dataset = datasets.ImageFolder(root="data/celeba", transform=transform)
else:
    raise ValueError("Dataset not supported")

loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

# Initialize models
gen = Generator(
    latent_dim=config["latent_dim"],
    g_features=config["g_features"],
    channels=config["channels"]
).to(device)

disc = Discriminator(
    d_features=config["d_features"],
    channels=config["channels"]
).to(device)

# Initialize weights (DCGAN paper: mean=0, std=0.02)
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

gen.apply(weights_init)
disc.apply(weights_init)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=config["lr_g"], betas=(config["beta1"], 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=config["lr_d"], betas=(config["beta1"], 0.999))

# Loss
criterion = nn.BCELoss()

# Fixed noise for consistent sampling
fixed_noise = torch.randn(64, config["latent_dim"], 1, 1).to(device)

# Initialize W&B (optional but highly recommended)
wandb.init(project="dcgan-training", config=config, mode="online")  # Set mode="disabled" to skip

# Training loop
gen.train()
disc.train()

for epoch in range(config["num_epochs"]):
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

    for i, (real, _) in enumerate(pbar):
        batch_size = real.shape[0]
        real = real.to(device)
        ones = torch.ones(batch_size).to(device)  # Real labels
        zeros = torch.zeros(batch_size).to(device)  # Fake labels

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, config["latent_dim"], 1, 1).to(device)
        fake = gen(noise)

        disc_real = disc(real).reshape(-1)
        loss_real = criterion(disc_real, ones)

        disc_fake = disc(fake.detach()).reshape(-1)
        loss_fake = criterion(disc_fake, zeros)

        loss_disc = (loss_real + loss_fake) / 2
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) => max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, ones)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Log to W&B
        if i % config["print_every"] == 0:
            wandb.log({
                "loss_gen": loss_gen.item(),
                "loss_disc": loss_disc.item(),
                "epoch": epoch,
            })

        pbar.set_postfix(
            loss_gen=loss_gen.item(),
            loss_disc=loss_disc.item()
        )

    # Sample images
    if (epoch + 1) % config["sample_every"] == 0:
        gen.eval()
        with torch.no_grad():
            fake_samples = gen(fixed_noise).cpu()
            fake_grid = torch.clamp((fake_samples + 1) / 2, 0, 1)  # Denormalize to [0,1]
            plt.figure(figsize=(8, 8))
            plt.imshow(torchvision.utils.make_grid(fake_grid, nrow=8).permute(1, 2, 0))
            plt.axis("off")
            plt.title(f"Generated Images - Epoch {epoch+1}")
            sample_path = f"{config['save_dir']}/epoch_{epoch+1}.png"
            plt.savefig(sample_path, bbox_inches='tight', dpi=150)
            plt.close()

            # Log to W&B
            wandb.log({"generated_images": wandb.Image(sample_path)})

        gen.train()

    # Save generator
    if (epoch + 1) % 10 == 0:
        torch.save(gen.state_dict(), f"{config['model_save_dir']}/generator_epoch_{epoch+1}.pth")

# Finish W&B run
wandb.finish()
print("✅ Training completed!")