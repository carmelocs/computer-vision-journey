# models/discriminator.py
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Simple DCGAN-style Discriminator.
    Input: RGB image of size (batch_size, img_channels, 64, 64)
    Output: probability of real image (batch_size, 1)
    """
    def __init__(self, img_channels=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)