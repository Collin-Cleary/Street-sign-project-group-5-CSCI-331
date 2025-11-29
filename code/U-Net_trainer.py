#!/usr/bin/env python3
# code/U-Net_trainer.py
# Run: python U-Net_trainer.py

import os
import json
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ---------------- Config ----------------
# Paths: script lives in code/, data/ is sibling
TRAIN_DIR = "../data/archive(4)/sign_dataset/train"
VAL_DIR   = "../data/archive(4)/sign_dataset/val"

BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-3
IMAGE_SIZE = (256, 256)   # (H, W) used for model input
# The native annotation mask shape inside the JSON (height, width)
ANNOT_MASK_SHAPE = None  # optional hint; code will infer mask shapes from JSON when present
SEED = 1337
# ----------------------------------------

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Dataset ----------------
class RussianSignUNetDataset(Dataset):
    """
    Expects directory with:
        <name>.jpg
        <name>.jpg_coco.json

    JSON format:
        { "masks": [ mask1, mask2, ... ] }
    where each mask is a 2D boolean list of shape ANNOT_MASK_SHAPE.
    We merge instance masks with OR (Option A), upscale from ANNOT_MASK_SHAPE -> image size (nearest),
    then apply target_transform (resize to IMAGE_SIZE) using NEAREST for masks.
    """
    def __init__(self, root_dir, transform=None, target_transform=None, annot_shape=None):
        self.root = root_dir
        self.transform = transform
        self.target_transform = target_transform
        # annot_shape is optional — prefer to infer from JSON mask arrays
        self.annot_shape = tuple(annot_shape) if annot_shape is not None else None

        if not os.path.isdir(root_dir):
            raise ValueError(f"root_dir does not exist: {root_dir}")

        all_files = sorted(os.listdir(root_dir))
        self.images = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(self.images) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.images)

    def _load_semantic_mask_from_json(self, json_path, target_img_w, target_img_h):
        """Load masks from JSON and render into image space.

        Dataset structure (discovered):
        - masks: 56 × 56 × N list where N = number of bboxes
          - masks[h][w][bbox_idx] = boolean for whether pixel (h,w) in canonical space
            belongs to bbox_idx
          - Canonical space is 56×56 (downsampled from image)
        - bbox: [x1, y1, x2, y2] for each bbox in image coordinates
        
        Many images in the dataset have out-of-bounds bboxes, indicating they were
        originally at 2x resolution before being downsampled. We scale bboxes down
        by 0.5 if any are out of bounds.
        
        For each bbox:
        1. Extract the mask for that bbox (56×56 channel)
        2. Resize the mask to bbox size
        3. Place at bbox location in full image
        
        Returns a PIL Image in mode 'L' (grayscale, 0..255).
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        masks = data.get("masks", [])
        bboxes = data.get('bbox') or []
        
        if not masks or not bboxes:
            return Image.fromarray(np.zeros((target_img_h, target_img_w), dtype=np.uint8))

        # Convert masks to numpy array
        # masks[h][w] is a list of len(bboxes) booleans
        # We want shape (56, 56, num_bboxes)
        try:
            masks_arr = np.array(masks, dtype=bool)  # (56, 56, N)
        except Exception:
            # Fallback if structure is unexpected
            return Image.fromarray(np.zeros((target_img_h, target_img_w), dtype=np.uint8))
        
        if masks_arr.ndim != 3:
            # Unexpected structure, return empty mask
            return Image.fromarray(np.zeros((target_img_h, target_img_w), dtype=np.uint8))
        
        canonical_h, canonical_w, num_bboxes_in_masks = masks_arr.shape
        num_bboxes = len(bboxes)
        
        if num_bboxes_in_masks != num_bboxes:
            # Mismatch, return empty mask
            return Image.fromarray(np.zeros((target_img_h, target_img_w), dtype=np.uint8))
        
        # Check if bboxes are out of bounds (indicating 2x resolution scaling needed)
        # This happens with ~65% of images in the dataset
        has_out_of_bounds = any(
            int(bbox[0]) < 0 or int(bbox[2]) > target_img_w or
            int(bbox[1]) < 0 or int(bbox[3]) > target_img_h
            for bbox in bboxes
        )
        
        # If out of bounds, scale bbox coordinates down by 0.5
        # (original images were likely 2x resolution before downsampling)
        if has_out_of_bounds:
            bboxes = [[x * 0.5 for x in bbox] for bbox in bboxes]
        
        # Render each bbox
        full_mask = np.zeros((target_img_h, target_img_w), dtype=np.uint8)
        
        for bbox_idx, bbox in enumerate(bboxes):
            try:
                x1, y1, x2, y2 = map(float, bbox)
            except (ValueError, TypeError):
                continue
            
            # Clip to image bounds
            x1 = max(0, min(x1, target_img_w - 1))
            x2 = max(x1 + 1, min(x2, target_img_w))
            y1 = max(0, min(y1, target_img_h - 1))
            y2 = max(y1 + 1, min(y2, target_img_h))
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            bw = x2 - x1
            bh = y2 - y1
            
            # Extract the mask for this bbox from canonical space
            bbox_mask_canonical = masks_arr[:, :, bbox_idx]  # (canonical_h, canonical_w)
            
            # Resize to bbox dimensions
            pil_mask = Image.fromarray((bbox_mask_canonical.astype(np.uint8) * 255))
            pil_resized = pil_mask.resize((bw, bh), Image.NEAREST)
            arr_resized = (np.array(pil_resized) > 0).astype(np.uint8)
            
            # Paste into full image with OR (to combine overlapping bboxes)
            full_mask[y1:y2, x1:x2] = np.logical_or(full_mask[y1:y2, x1:x2], arr_resized)
        
        return Image.fromarray(full_mask * 255)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root, img_name)
        json_path = img_path + "_coco.json"

        image = Image.open(img_path).convert("RGB")
        w, h = image.size  # (width, height)

        if os.path.exists(json_path):
            mask_pil = self._load_semantic_mask_from_json(json_path, w, h)
        else:
            mask_pil = Image.fromarray(np.zeros((h, w), dtype=np.uint8))

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask_pil)
        else:
            mask = transforms.ToTensor()(mask_pil)

        # Ensure binary 0/1 float tensor
        mask = (mask > 0).float()

        return image, mask


# --------------- UNet model (unchanged) ---------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.middle = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        d4 = self.down4(p3)
        p4 = self.pool4(d4)

        m = self.middle(p4)

        u1 = self.up1(m)
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


# ---------------- Transforms, Datasets ----------------
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# For masks: use NEAREST interpolation to preserve binary labels
target_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
    transforms.ToTensor(),
])

train_dataset = RussianSignUNetDataset(root_dir=TRAIN_DIR,
                                      transform=transform,
                                      target_transform=target_transform,
                                      annot_shape=ANNOT_MASK_SHAPE)
val_dataset = RussianSignUNetDataset(root_dir=VAL_DIR,
                                    transform=transform,
                                    target_transform=target_transform,
                                    annot_shape=ANNOT_MASK_SHAPE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train samples: {len(train_dataset)}   Val samples: {len(val_dataset)}")

# ---------------- Model, loss, optimizer ----------------
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- Utility: IoU (robust) ----------------
def iou_score(pred_mask, true_mask, eps=1e-7):
    pred = (pred_mask > 0.5).astype(bool)
    true = (true_mask > 0.5).astype(bool)
    inter = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    if union == 0:
        # both empty -> perfect match
        return 1.0
    return inter / (union + eps)

# ---------------- Training loop ----------------
def train_and_visualize():
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - train")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)                # logits
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix({'loss': loss.item()})

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        iou_list = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - val", leave=False):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(np.float32)
                trues = masks.cpu().numpy()

                for p, t in zip(preds, trues):
                    p_arr = p.squeeze()
                    t_arr = t.squeeze()
                    iou_list.append(iou_score(p_arr, t_arr))

        val_loss = val_running_loss / len(val_loader.dataset)
        mean_iou = float(np.mean(iou_list)) if len(iou_list) > 0 else 0.0

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val IoU: {mean_iou:.4f}")

    # ---------------- Show a random non-empty validation example ----------------
    model.eval()
    # Try to pick a random non-empty validation example for visualization
    nonempty_indices = []
    for i in range(len(val_dataset)):
        # use dataset directly to get mask without model
        img_t, mask_t = val_dataset[i]
        if mask_t.sum().item() > 0:
            nonempty_indices.append(i)

    if len(nonempty_indices) > 0:
        chosen_idx = random.choice(nonempty_indices)
    else:
        chosen_idx = 0

    with torch.no_grad():
        sample_img, sample_mask = val_dataset[chosen_idx]
        sample_img_tensor = sample_img.unsqueeze(0).to(device)
        out = model(sample_img_tensor)
        pred = torch.sigmoid(out).squeeze().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)

    img_np = np.transpose(sample_img.numpy(), (1, 2, 0))
    true_mask_np = sample_mask.squeeze().numpy()
    pred_mask_np = pred_binary

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title(f"Image (idx={chosen_idx})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask_np, cmap='gray')
    plt.title("True Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_mask_np, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_and_visualize()
