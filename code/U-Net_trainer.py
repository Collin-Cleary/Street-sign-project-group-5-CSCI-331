#!/usr/bin/env python3
# U-Net_trainer_precompute_via.py
# Run: python U-Net_trainer_precompute_via.py

import os
import json
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# -------------------- CONFIG --------------------
ROOT = "../data/archive(4)/sign_dataset"
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR   = os.path.join(ROOT, "val")

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-3

SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# -------------------- VIA PARSER --------------------
def load_via_annotations(via_json_path):
    """Load VIA JSON → return {image_filename: [shape_attributes]}"""
    with open(via_json_path, "r") as f:
        data = json.load(f)

    mapping = {}

    for _, entry in data.items():
        filename = entry.get("filename")
        regions_dict = entry.get("regions", {})

        if not filename:
            continue

        shapes = []
        for _, region in regions_dict.items():
            shapes.append(region.get("shape_attributes", {}))

        mapping[filename] = shapes

    return mapping


# -------------------- CREATE MASK --------------------
def rasterize_regions(regions, w, h):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for shape in regions:
        name = shape.get("name")

        if name == "polygon":
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if len(xs) >= 3:
                draw.polygon(list(zip(xs, ys)), fill=255)

        elif name in ("circle", "ellipse"):
            cx = shape.get("cx")
            cy = shape.get("cy")
            rx = shape.get("rx") or shape.get("r")
            ry = shape.get("ry") or shape.get("r")
            if None not in (cx, cy, rx, ry):
                draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=255)

        elif name in ("rect", "rectangle"):
            x = shape.get("x")
            y = shape.get("y")
            w2 = shape.get("width")
            h2 = shape.get("height")
            if None not in (x, y, w2, h2):
                draw.rectangle([x, y, x+w2, y+h2], fill=255)

    return mask


# -------------------- MASK PRECOMPUTE --------------------
def precompute_masks(folder):
    """
    Reads:
        folder/via_region_data.json
        folder/*.jpg
    Writes:
        folder/masks/*.png
    """
    via_path = os.path.join(folder, "via_region_data.json")
    if not os.path.exists(via_path):
        raise RuntimeError(f"No VIA file found in {folder}")

    via = load_via_annotations(via_path)

    mask_dir = os.path.join(folder, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    print(f"\nPrecomputing masks in {folder} ...")
    skipped = 0
    written = 0

    for img_name in tqdm(image_files):
        if img_name not in via:
            skipped += 1
            continue

        img_path = os.path.join(folder, img_name)
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", ".png"))

        # Skip if mask already exists
        if os.path.exists(mask_path):
            continue

        # Load image to obtain size
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Rasterize VIA shapes
        mask_pil = rasterize_regions(via[img_name], w, h)

        # Save mask PNG
        mask_pil.save(mask_path)
        written += 1

    print(f"✓ Masks written: {written}")
    print(f"✓ Images skipped (no VIA annotation): {skipped}")


# -------------------- DATASET --------------------
class RussianSignDatasetPrecomputed(Dataset):
    """
    Loads precomputed PNG masks.
    Requires:
        folder/image.jpg
        folder/masks/image.png
    """

    def __init__(self, folder, transform, target_transform):
        self.folder = folder
        self.images = []

        mask_dir = os.path.join(folder, "masks")

        # Build a set of available mask filenames (e.g. '1.png') for fast lookup.
        mask_files = set(os.listdir(mask_dir)) if os.path.isdir(mask_dir) else set()

        for f in os.listdir(folder):
            # only consider image files
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # expected mask filename is the image basename with .png extension
            base, _ = os.path.splitext(f)
            expected_mask = base + ".png"
            if expected_mask in mask_files:
                self.images.append(f)

        self.transform = transform
        self.target_transform = target_transform

        print(f"Loaded dataset from {folder}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img_path = os.path.join(self.folder, name)
        mask_path = os.path.join(self.folder, "masks", name.replace(".jpg", ".png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.target_transform(mask)
        mask = (mask > 0).float()

        return image, mask, name


# -------------------- UNET --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        mid = self.mid(self.pool4(d4))

        u1 = self.up1(mid)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.conv2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.conv3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.conv4(u4)

        return self.out(u4)


# -------------------- TRAINING --------------------
def main():

    # 1. Precompute masks for train & val
    precompute_masks(TRAIN_DIR)
    precompute_masks(VAL_DIR)

    # 2. Load dataset
    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    tf_mask = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])

    train_ds = RussianSignDatasetPrecomputed(TRAIN_DIR, tf, tf_mask)
    val_ds   = RussianSignDatasetPrecomputed(VAL_DIR, tf, tf_mask)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 3. Build U-Net
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Train
    for epoch in range(NUM_EPOCHS):
        model.train()
        total = 0
        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, masks)
            loss.backward()
            optimizer.step()
            total += loss.item() * imgs.size(0)

        avg = total / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Train Loss = {avg:.4f}")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
