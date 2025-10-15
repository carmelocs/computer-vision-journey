# models/generator.py
import torch.nn as nn

class Generator(nn.Module):
    """
    Simple DCGAN-style Generator.
    Input: latent vector (z) of size (batch_size, z_dim)
    Output: RGB image of size (batch_size, img_channels, 64, 64)
    """
    def __init__(self, z_dim=100, img_channels=3, hidden_dim=128):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * 8 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (hidden_dim * 8, 4, 4)),

            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.ConvTranspose2d(hidden_dim, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)