# test_setup.py
import torch
import cv2
from torch import nn

print("✅ Torch version:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ CuDNN enabled:", torch.backends.cudnn.enabled)
print("🎮 GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("🔢 CUDA capability:", torch.cuda.get_device_capability() if torch.cuda.is_available() else "N/A")
print("✅ OpenCV version:", cv2.__version__)

# Quick model test (on GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Linear(10, 5).to(device)
x = torch.randn(3, 10).to(device)
print("🧪 Model ran on:", x.device)