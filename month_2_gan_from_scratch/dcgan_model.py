# dcgan_model.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, g_features=64, channels=3):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: (N, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, g_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_features * 8, g_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_features * 4, g_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_features * 2, g_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_features),
            nn.ReLU(True),

            nn.ConvTranspose2d(g_features, channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output: (N, 3, 64, 64), pixel range [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, d_features=64, channels=3):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: (N, 3, 64, 64)
            nn.Conv2d(channels, d_features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_features, d_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_features * 2, d_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_features * 4, d_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(d_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Output: real/fake probability
        )

    def forward(self, x):
        return self.net(x).view(-1)