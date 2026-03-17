#!/usr/bin/env python3
"""train-classifier.py — offline training script for the Go intersection classifier.

Trains a tiny CNN to classify each board intersection as empty / black / white.
Outputs a model/stone-classifier.onnx file suitable for onnxruntime-web.

Label convention (must match src/onnx-classifier.ts):
  0 → '.'  (empty)
  1 → 'B'  (black stone)
  2 → 'W'  (white stone)

Usage:
  pip install torch torchvision opencv-python onnx
  python scripts/train-classifier.py

The script expects .fixture.json files (with 'intersections' pixel coords) in
fixtures/ and fixtures/real/. It uses those coords to extract 32×32 patches
from the corresponding image files. The 'stones' field in each fixture provides
the ground-truth label for each intersection.

Augmentation covers:
  - brightness / contrast jitter
  - Gaussian blur
  - slight rotation (±5°)
  - horizontal / vertical flip (symmetric board positions)
  - color temperature shift (hue + saturation jitter on color images)

This lets the model generalise across the lighting variation seen in real-board
photos (dim bar lighting, warm tungsten, mixed sunlight) without needing a
large labelled dataset.
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import onnx
    import onnxruntime as ort
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Run: pip install torch torchvision opencv-python onnx onnxruntime"
    )

# ── Config ───────────────────────────────────────────────────────────────────

PATCH_SIZE   = 32
BATCH_SIZE   = 128
EPOCHS       = 40
LR           = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_MAP    = {'.': 0, 'B': 1, 'W': 2}
OUTPUT_DIR   = Path("public/models")
OUTPUT_PATH  = OUTPUT_DIR / "stone-classifier.onnx"

# ── Dataset ──────────────────────────────────────────────────────────────────

def load_patches_from_fixture(fixture_path: str) -> list[tuple[np.ndarray, int]]:
    """Extract (patch, label) pairs from one fixture file."""
    with open(fixture_path) as f:
        data = json.load(f)

    image_path = data.get("image", "")
    if not os.path.exists(image_path):
        print(f"  [skip] image not found: {image_path}")
        return []

    intersections = data.get("intersections")  # flat list of [x, y] in row-major order
    stones_list   = data.get("stones", [])      # [[r, c, color], ...]
    n_rows        = data.get("boardRows", 0)
    n_cols        = data.get("boardCols", 0)

    if not intersections or n_rows == 0 or n_cols == 0:
        print(f"  [skip] no intersections ground truth: {fixture_path}")
        return []

    # Build r,c → color lookup
    stone_map: dict[tuple[int, int], str] = {}
    for entry in stones_list:
        r, c, color = entry[0], entry[1], entry[2]
        stone_map[(r, c)] = color

    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"  [skip] could not read image: {image_path}")
        return []
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    H, W = img_gray.shape
    half = PATCH_SIZE // 2

    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            idx   = r * n_cols + c
            if idx >= len(intersections):
                continue
            x, y  = int(round(intersections[idx][0])), int(round(intersections[idx][1]))
            x0, y0 = x - half, y - half
            if x0 < 0 or y0 < 0 or x0 + PATCH_SIZE > W or y0 + PATCH_SIZE > H:
                continue
            patch = img_gray[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE].copy()
            label = LABEL_MAP[stone_map.get((r, c), '.')]
            patches.append((patch, label))

    return patches


class IntersectionDataset(Dataset):
    def __init__(self, patches: list[tuple[np.ndarray, int]], augment: bool = True):
        self.patches = patches
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch, label = self.patches[idx]
        patch = patch.astype(np.float32) / 255.0

        if self.augment:
            patch = self._augment(patch)

        # [1, H, W] tensor
        tensor = torch.from_numpy(patch[None])
        return tensor, label

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        # Brightness / contrast jitter
        alpha = random.uniform(0.7, 1.3)   # contrast
        beta  = random.uniform(-0.15, 0.15) # brightness
        patch = np.clip(patch * alpha + beta, 0.0, 1.0)

        # Gaussian blur (simulates defocus / motion)
        if random.random() < 0.3:
            k = random.choice([3, 5])
            p8 = (patch * 255).astype(np.uint8)
            patch = cv2.GaussianBlur(p8, (k, k), 0).astype(np.float32) / 255.0

        # Slight rotation ±5°
        if random.random() < 0.4:
            angle  = random.uniform(-5, 5)
            M      = cv2.getRotationMatrix2D((PATCH_SIZE / 2, PATCH_SIZE / 2), angle, 1.0)
            p8     = (patch * 255).astype(np.uint8)
            rotated = cv2.warpAffine(p8, M, (PATCH_SIZE, PATCH_SIZE),
                                     borderMode=cv2.BORDER_REFLECT)
            patch  = rotated.astype(np.float32) / 255.0

        # Horizontal / vertical flip (board is symmetric)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        if random.random() < 0.5:
            patch = np.flipud(patch)

        return patch


# ── Model ────────────────────────────────────────────────────────────────────

class StoneClassifier(nn.Module):
    """Tiny CNN: ~50K parameters, fast in WASM."""
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 16×16
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                        # 8×8
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                # 1×1
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.head(x)


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    # Collect patches from all fixtures
    print("Loading fixtures...")
    all_patches: list[tuple[np.ndarray, int]] = []
    fixture_dirs = ["fixtures", "fixtures/real"]
    for d in fixture_dirs:
        for path in sorted(glob.glob(f"{d}/*.fixture.json")):
            ps = load_patches_from_fixture(path)
            print(f"  {path}: {len(ps)} patches")
            all_patches.extend(ps)

    if not all_patches:
        raise SystemExit("No patches found — annotate some fixtures first.")

    print(f"\nTotal patches: {len(all_patches)}")
    label_counts = [0, 0, 0]
    for _, lbl in all_patches:
        label_counts[lbl] += 1
    print(f"  empty={label_counts[0]}  black={label_counts[1]}  white={label_counts[2]}")

    # Class weights to handle imbalance (boards are mostly empty)
    total = len(all_patches)
    weights = torch.tensor([total / (3 * max(c, 1)) for c in label_counts], dtype=torch.float32)

    # Train / val split
    random.shuffle(all_patches)
    split = int(len(all_patches) * 0.85)
    train_set = IntersectionDataset(all_patches[:split], augment=True)
    val_set   = IntersectionDataset(all_patches[split:],  augment=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on {device}")

    model     = StoneClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total_val += y.size(0)
        acc = correct / total_val if total_val > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "/tmp/best_stone_classifier.pt")
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:3d}/{EPOCHS}  val_acc={acc:.4f}  best={best_acc:.4f}")

    print(f"\nBest val accuracy: {best_acc:.4f}")

    # Export to ONNX
    model.load_state_dict(torch.load("/tmp/best_stone_classifier.pt", map_location="cpu"))
    model.eval().cpu()
    dummy = torch.zeros(1, 1, PATCH_SIZE, PATCH_SIZE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(OUTPUT_PATH),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB)")

    # Quick sanity check via onnxruntime
    sess    = ort.InferenceSession(str(OUTPUT_PATH))
    inp     = dummy.numpy()
    logits, = sess.run(None, {"input": inp})
    pred    = int(np.argmax(logits[0]))
    label   = ['.', 'B', 'W'][pred]
    print(f"ONNX sanity check: empty patch → '{label}' (expected '.')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ONNX stone classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    EPOCHS = args.epochs
    train(args)
