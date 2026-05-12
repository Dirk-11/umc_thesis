"""gpu_benchmark.py — Diagnose training bottleneck: data loading vs GPU compute.

Runs a real DataLoader with actual augmentations + a full forward/backward pass,
and samples GPU utilization in the background to show whether the GPU is sitting
idle waiting for data.

Usage:
    python scripts/gpu_benchmark.py
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, resolve_path

# ---------------------------------------------------------------------------
# GPU utilization sampler (background thread)
# ---------------------------------------------------------------------------

class GPUSampler:
    """Samples GPU utilization % every 0.5s in a background thread."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                self.samples.append(int(out))
            except Exception:
                pass
            time.sleep(self.interval)

    def report(self) -> str:
        if not self.samples:
            return "no samples"
        avg = sum(self.samples) / len(self.samples)
        mn, mx = min(self.samples), max(self.samples)
        return f"avg {avg:.0f}%  min {mn}%  max {mx}%  ({len(self.samples)} samples)"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    device = torch.device("cuda")
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {cfg['training']['batch_size']}")
    print(f"Workers:    {cfg['training']['num_workers']}")
    print()

    # Build a real DataLoader using the actual dataset + augmentations.
    import importlib
    ds = importlib.import_module("03_dataset")
    images_csv = resolve_path(cfg, "processed_images_csv")
    samples = ds.load_image_samples(images_csv)
    indices = list(range(len(samples)))
    cache = cfg["training"].get("cache_images", False)
    if cache:
        print("cache_images: true — images will be preloaded into RAM")
    loader = ds.make_dataloader(samples, indices, cfg, train=True)
    n_batches = min(30, len(loader))
    print(f"Dataset:    {len(samples)} images  →  {len(loader)} batches")
    print(f"Timing {n_batches} batches (forward + backward)...\n")

    # Model + loss
    model = models.resnet50(weights="IMAGENET1K_V2").to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Warm-up batch (not timed)
    imgs, labels, _ = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)
    with torch.no_grad():
        _ = model(imgs)

    # --- Timed run ---
    sampler = GPUSampler(interval=0.5)
    sampler.start()

    load_times, compute_times = [], []
    loader_iter = iter(loader)
    batches_done = 0

    while batches_done < n_batches:
        # Time data loading
        t0 = time.time()
        try:
            imgs, labels, _ = next(loader_iter)
        except StopIteration:
            break
        load_times.append(time.time() - t0)

        # Time forward + backward
        imgs, labels = imgs.to(device), labels.to(device)
        t1 = time.time()
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        compute_times.append(time.time() - t1)

        batches_done += 1

    sampler.stop()

    # --- Report ---
    avg_load    = sum(load_times)    / len(load_times)
    avg_compute = sum(compute_times) / len(compute_times)
    avg_total   = avg_load + avg_compute
    load_pct    = avg_load / avg_total * 100

    print(f"Results over {batches_done} batches:")
    print(f"  Data loading:      {avg_load*1000:6.0f}ms/batch  ({load_pct:.0f}% of total)")
    print(f"  GPU compute:       {avg_compute*1000:6.0f}ms/batch  ({100-load_pct:.0f}% of total)")
    print(f"  Total per batch:   {avg_total*1000:6.0f}ms/batch")
    print(f"  Est. epoch time:   {avg_total * len(loader):.0f}s")
    print()
    print(f"GPU utilization:   {sampler.report()}")
    print()

    if load_pct > 50:
        print("VERDICT: bottleneck is DATA LOADING (CPU/disk) — more vCPUs would help.")
    elif avg_compute * 1000 < 100 and load_pct < 30:
        print("VERDICT: GPU is fast and well-fed — overall training should be quick.")
    else:
        print("VERDICT: GPU compute is the main cost — a faster GPU would help.")


if __name__ == "__main__":
    main()
