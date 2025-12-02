#!/usr/bin/env python3

import os
import json
import random
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


# CONFIG

ROOT = "../data/archive(4)/sign_dataset"
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR   = os.path.join(ROOT, "val")

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-3
SEED = 1337

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


#  PART 1 — VIA PARSER 

def load_via_annotations(via_json_path):
    """
    Read VIA JSON and return:
        { filename : [ {shape_attributes}, ... ] }
    This is the version that WORKS for your dataset.
    """
    with open(via_json_path, "r") as f:
        data = json.load(f)

    mapping = {}

    for _, entry in data.items():
        fname = entry.get("filename")
        if not fname:
            continue

        regions = entry.get("regions", {})

        
        shapes = []
        if isinstance(regions, dict):
            
            for _, reg in regions.items():
                if isinstance(reg, dict):
                    shape = reg.get("shape_attributes", {})
                    if isinstance(shape, dict):
                        shapes.append(shape)

        elif isinstance(regions, list):
            
            for reg in regions:
                if isinstance(reg, dict):
                    shape = reg.get("shape_attributes", {})
                    if isinstance(shape, dict):
                        shapes.append(shape)

        mapping[fname] = shapes

    return mapping



#  MASK RENDERING

def rasterize_regions(regions, w, h):
    """
    Draws a binary mask from a list of shape_attributes dicts.
    Supports polygon, rect, ellipse, circle.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for shape in regions:
        name = shape.get("name")

        if name == "polygon":
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if len(xs) >= 3:
                draw.polygon(list(zip(xs, ys)), fill=255)

        elif name in ("rect", "rectangle"):
            x = shape.get("x")
            y = shape.get("y")
            w2 = shape.get("width")
            h2 = shape.get("height")
            if None not in (x, y, w2, h2):
                draw.rectangle([x, y, x+w2, y+h2], fill=255)

        elif name in ("circle", "ellipse"):
            cx = shape.get("cx")
            cy = shape.get("cy")
            rx = shape.get("rx") or shape.get("r")
            ry = shape.get("ry") or shape.get("r")
            if None not in (cx, cy, rx, ry):
                draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=255)

    return mask



#  PRECOMPUTE MASK PNGs

def precompute_masks(folder):
    """
    Uses the fully working VIA parser → outputs PNG masks.
    If a VIA mask doesn't exist for an image, it is skipped.
    """
    via_path = os.path.join(folder, "via_region_data.json")
    if not os.path.exists(via_path):
        print(f"No VIA JSON found in {folder}. Skipping this folder.")
        return

    via = load_via_annotations(via_path)

    mask_dir = os.path.join(folder, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"\nPrecomputing masks in {folder} ...")
    written = 0
    skipped = 0

    for img_name in tqdm(image_files):
        if img_name not in via:
            skipped += 1
            continue

        regions = via[img_name]
        if len(regions) == 0:
            skipped += 1
            continue

        img_path = os.path.join(folder, img_name)
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        mask = rasterize_regions(regions, w, h)

        base, _ = os.path.splitext(img_name)
        outpath = os.path.join(mask_dir, base + ".png")
        mask.save(outpath)
        written += 1

    print(f"✓ Masks written: {written}")
    print(f"✓ Images skipped (no VIA annotation): {skipped}")



#  DATASET — loads PNG masks

class RussianSignDatasetPrecomputed(Dataset):
    def __init__(self, folder, transform, target_transform):
        self.folder = folder
        self.transform = transform
        self.target_transform = target_transform

        self.mask_dir = os.path.join(folder, "masks")
        mask_files = set(os.listdir(self.mask_dir))

        self.images = []
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                base = os.path.splitext(f)[0]
                expected_mask = base + ".png"
                if expected_mask in mask_files:
                    self.images.append(f)

        print(f"Loaded dataset from {folder}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]

        img_path = os.path.join(self.folder, fname)
        base = os.path.splitext(fname)[0]
        mask_path = os.path.join(self.mask_dir, base + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.transform(img)
        mask = self.target_transform(mask)
        mask = (mask > 0).float()

        return img, mask, fname



#  UNET ARCHITECTURE

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



#  IOU METRIC

def iou(pred, true, eps=1e-7):
    pred = pred > 0.5
    true = true > 0.5
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return inter / (union + eps) if union > 0 else 1.0



#  MAIN TRAIN LOOP

def main():

    # ---- PRECOMPUTE MASKS ----
    precompute_masks(TRAIN_DIR)
    precompute_masks(VAL_DIR)

    # ---- DATASETS ----
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
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    #  MODEL 
    model = UNet().to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    #  TRAINING LOOP 
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - train"):
            imgs = imgs.to(device)
            masks = masks.to(device)

            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            opt.step()

            total_loss += loss.item() * imgs.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        #  VALIDATION 
        model.eval()
        val_loss = 0
        ious = []

        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - val", leave=False):
                imgs = imgs.to(device)
                masks = masks.to(device)

                out = model(imgs)
                loss = criterion(out, masks)
                val_loss += loss.item() * imgs.size(0)

                probs = torch.sigmoid(out).cpu().numpy()
                true = masks.cpu().numpy()

                for p, t in zip(probs, true):
                    ious.append(iou(p.squeeze(), t.squeeze()))

        val_loss /= len(val_loader.dataset)
        mean_iou = float(np.mean(ious))

        print(f"Epoch {epoch}/{NUM_EPOCHS}  Train={train_loss:.4f}  Val={val_loss:.4f}  IoU={mean_iou:.4f}")

    #  FINAL VISUALIZATION

    model.eval()
    HARDCODED_FILENAME = "231.jpg"

    try:
        idx = val_ds.images.index(HARDCODED_FILENAME)
    except:
        raise RuntimeError(f"{HARDCODED_FILENAME} not found in validation dataset.")
    img_t, mask_t, fname = val_ds[idx]

    img_input = img_t.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_input)
        pred = torch.sigmoid(out).squeeze().cpu().numpy()
    pred_bin = (pred > 0.5).astype(np.float32)

    img_np = np.transpose(img_t.numpy(), (1, 2, 0))
    mask_np = mask_t.squeeze().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title(f"Image: {fname}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_bin, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
