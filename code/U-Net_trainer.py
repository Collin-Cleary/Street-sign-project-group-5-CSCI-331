#!/usr/bin/env python3
# U-Net_trainer.py
# Final trainer: VIA-only masks -> precompute PNGs -> train U-Net -> metrics -> 5-fold eval (val only)
# Save as U-Net_trainer.py and run: python U-Net_trainer.py

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

from sklearn.model_selection import KFold

#  CONFIG 
ROOT = "../data/archive/sign_dataset"
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR = os.path.join(ROOT, "val")

IMAGE_SIZE = (256, 256)  
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-3
SEED = 1337
KFOLD = 5   # 5-fold evaluation on validation set

BEST_MODEL_PATH = "best_model_by_iou.pth"

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
    Handles both 'regions' as dict or list.
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

#  PART 2 — process VIA shapes into binary mask 
def rasterize_regions(regions, w, h):
    """
    Convert a list of shape_attributes (VIA) into a PIL L-mode mask (0/255).
    Supports polygon, rect/rectangle, circle/ellipse.
    """
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    for shape in regions:
        name = shape.get("name")

        if name == "polygon":
            xs = shape.get("all_points_x", [])
            ys = shape.get("all_points_y", [])
            if len(xs) >= 3:
                pts = list(zip(xs, ys))
                draw.polygon(pts, fill=255)

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

        else:
            # unknown/unsupported shape - skip
            continue

    return mask

#  PART 3 — Precompute masks into PNG files 
def precompute_masks(folder):
    via_path = os.path.join(folder, "via_region_data.json")
    if not os.path.exists(via_path):
        print(f"No VIA JSON in {folder} -> skipping precompute.")
        return

    via = load_via_annotations(via_path)
    mask_dir = os.path.join(folder, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    image_files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"\nPrecomputing masks in {folder} ...")
    written = 0
    skipped = 0

    for img_name in tqdm(image_files):
        # only process images that appear in VIA mapping (all of them btw)
        if img_name not in via:
            skipped += 1
            continue

        regions = via[img_name]
        if not regions:
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
    print(f"✓ Images skipped (no VIA annotation or empty): {skipped}")

#  PART 4 — Dataset that loads precomputed PNG masks 
class RussianSignDatasetPrecomputed(Dataset):
    """
    Loads images and corresponding precomputed masks at folder/masks/<basename>.png
    Returns: imageTensor, maskTensor, filename
    """
    def __init__(self, folder, transform=None, target_transform=None):
        self.folder = folder
        self.mask_dir = os.path.join(folder, "masks")
        self.transform = transform
        self.target_transform = target_transform

        # Ensure mask_dir exists
        if not os.path.isdir(self.mask_dir):
            raise RuntimeError(f"Mask directory missing: {self.mask_dir} (run precompute_masks first)")

        mask_files = set(os.listdir(self.mask_dir))
        self.images = []
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base = os.path.splitext(f)[0]
            expected_mask = base + ".png"
            if expected_mask in mask_files:
                self.images.append(f)

        if len(self.images) == 0:
            raise RuntimeError(f"No images with precomputed masks found in {folder}")

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

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        mask = (mask > 0.5).float()
        return img, mask, fname

#  PART 5 — U-Net  
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.down1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.mid = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        mid = self.mid(p4)

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

#  Metrics helpers 
def iou_score(pred_arr, true_arr, eps=1e-7):
    pred = pred_arr > 0.5
    true = true_arr > 0.5
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return inter / (union + eps)

def precision_recall_f1(pred_arr, true_arr, eps=1e-7):
    pred = pred_arr > 0.5
    true = true_arr > 0.5
    TP = np.logical_and(pred, true).sum()
    FP = np.logical_and(pred, np.logical_not(true)).sum()
    FN = np.logical_and(np.logical_not(pred), true).sum()
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

#  Transforms 
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

#  Main training + eval pipeline 
def run_training_and_evaluation():
    # Precompute masks
    precompute_masks(TRAIN_DIR)
    precompute_masks(VAL_DIR)

    # Build datasets
    train_ds = RussianSignDatasetPrecomputed(TRAIN_DIR, transform, target_transform)
    val_ds   = RussianSignDatasetPrecomputed(VAL_DIR, transform, target_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_iou = -1.0

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - train")
        for imgs, masks, _ in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        ious = []
        precisions = []
        recalls = []
        f1s = []

        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} - val", leave=False):
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                loss = criterion(logits, masks)
                val_running_loss += loss.item() * imgs.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                trues = masks.cpu().numpy()

                for p, t in zip(probs, trues):
                    p_arr = p.squeeze()
                    t_arr = t.squeeze()
                    ious.append(iou_score(p_arr, t_arr))
                    prec, rec, f1 = precision_recall_f1(p_arr, t_arr)
                    precisions.append(prec)
                    recalls.append(rec)
                    f1s.append(f1)

        val_loss = val_running_loss / len(val_loader.dataset)
        mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
        mean_prec = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
        mean_rec = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
        mean_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0

        print(f"Epoch {epoch}/{NUM_EPOCHS}  Train={train_loss:.4f}  Val={val_loss:.4f}  IoU={mean_iou:.4f}  Precision={mean_prec:.4f}  Recall={mean_rec:.4f}  F1={mean_f1:.4f}")

        # Save best model by IoU
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved best model (IoU={best_iou:.4f}) to {BEST_MODEL_PATH}")

    # Final visualization (use original-resolution mask & image for alignment)
    model.eval()

    # By default select a random file from val
    HARDCODED_FILENAME = "231.jpg"

    try:
        idx = val_ds.images.index(HARDCODED_FILENAME)
    except:
        raise RuntimeError(f"{HARDCODED_FILENAME} not found in validation dataset.")
    img_t, mask_t, fname = val_ds[idx]

    # Visualize using original resolution image & mask so alignment is perfect
    orig_img_path = os.path.join(val_ds.folder, fname)
    orig_mask_path = os.path.join(val_ds.folder, "masks", os.path.splitext(fname)[0] + ".png")
    orig_img = Image.open(orig_img_path).convert("RGB")
    orig_mask = Image.open(orig_mask_path).convert("L")
    orig_w, orig_h = orig_img.size

    # Prepare model input at training resolution
    inp = transform(orig_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
        prob = torch.sigmoid(out).squeeze().cpu().numpy()
    pred_bin = (prob > 0.5).astype(np.uint8) * 255
    pred_full = Image.fromarray(pred_bin).resize((orig_w, orig_h), Image.NEAREST)

    # Convert to arrays for plotting
    img_np = np.array(orig_img)
    mask_np = np.array(orig_mask)
    pred_np = np.array(pred_full)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title(f"Image: {fname}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("True Mask (orig res)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap="gray")
    plt.title("Predicted Mask (orig res)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    #  5-Fold evaluation on validation dataset only 
    print("\nRunning 5-fold evaluation on validation dataset")
    # Build arrays of predictions and truths for each image 
    all_probs = []
    all_trues = []
    filenames = []

    with torch.no_grad():
        for imgs, masks, fnames in tqdm(val_loader, desc="Predicting full val set"):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()  # (B,1,H,W)
            trues = masks.cpu().numpy()                       # (B,1,H,W)
            for p, t, fn in zip(probs, trues, fnames):
                all_probs.append(p.squeeze())
                all_trues.append(t.squeeze())
                filenames.append(fn)

    all_probs = np.array(all_probs)
    all_trues = np.array(all_trues)
    n = len(all_probs)
    assert n == len(val_ds), f"val set mismatch: {n} vs {len(val_ds)}"

    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    fold = 0
    fold_ious = []
    fold_precs = []
    fold_recs = []
    fold_f1s = []

    for train_idx, test_idx in kf.split(np.arange(n)):
        fold += 1
        ious_fold = []
        precs_fold = []
        recs_fold = []
        f1s_fold = []

        for ii in test_idx:
            p = all_probs[ii]
            t = all_trues[ii]
            ious_fold.append(iou_score(p, t))
            prec, rec, f1 = precision_recall_f1(p, t)
            precs_fold.append(prec)
            recs_fold.append(rec)
            f1s_fold.append(f1)

        mean_i = float(np.mean(ious_fold))
        mean_p = float(np.mean(precs_fold))
        mean_r = float(np.mean(recs_fold))
        mean_f = float(np.mean(f1s_fold))

        fold_ious.append(mean_i)
        fold_precs.append(mean_p)
        fold_recs.append(mean_r)
        fold_f1s.append(mean_f)

        print(f"Fold {fold}/{KFOLD}: IoU={mean_i:.4f}, Precision={mean_p:.4f}, Recall={mean_r:.4f}, F1={mean_f:.4f}")

    print("\nK-Fold (validation set) results:")
    print(f"Mean IoU = {np.mean(fold_ious):.4f}  Std = {np.std(fold_ious):.4f}")
    print(f"Mean Precision = {np.mean(fold_precs):.4f}  Std = {np.std(fold_precs):.4f}")
    print(f"Mean Recall = {np.mean(fold_recs):.4f}  Std = {np.std(fold_recs):.4f}")
    print(f"Mean F1 = {np.mean(fold_f1s):.4f}  Std = {np.std(fold_f1s):.4f}")

if __name__ == "__main__":
    run_training_and_evaluation()  
