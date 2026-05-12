"""gpu_benchmark.py — Quick GPU throughput check using ResNet50 + a sample image batch.

Usage:
    python scripts/gpu_benchmark.py

Prints forward-pass timing and an estimated epoch duration based on 29 batches of 32.
"""

import glob
import time

import torch
from PIL import Image
from torchvision import models, transforms

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
model = models.resnet50(weights="IMAGENET1K_V2").to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load a batch of 32 images
files = glob.glob("data/images/*.JPG")[:32]
if not files:
    raise FileNotFoundError("No .JPG files found in data/images/ — check your path.")

imgs = torch.stack([tf(Image.open(f).convert("RGB")) for f in files]).to(device)
print(f"Loaded {len(files)} images into batch")

# Warm up GPU
with torch.no_grad():
    _ = model(imgs)

# Time 10 forward passes
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(imgs)
torch.cuda.synchronize()
elapsed = time.time() - t0

print(f"10 forward passes of batch=32: {elapsed:.2f}s")
print(f"Per batch: {elapsed / 10 * 1000:.0f}ms")
print(f"Estimated epoch (29 batches): {elapsed / 10 * 29:.1f}s")
