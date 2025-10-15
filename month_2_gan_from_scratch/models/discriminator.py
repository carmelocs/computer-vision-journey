# models# models/discriminator.py

import torch.nn as nn

class Discriminator(nn.Module):

    """
    Simple DCGAN-style Discriminator.
    Input: RGB image of size (batch_size, img_channels, 32, 32)
    Output: probability of real image (batch_size, 1)
    """

    def __init__(self, img_channels=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: (batch_size, 3, 32, 32)
            nn.Conv2d(img_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)