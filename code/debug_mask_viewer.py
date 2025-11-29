import os
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR = "../data/archive(4)/sign_dataset/train"
ANNOT_MASK_SHAPE = (56, 8)      # (H, W) — tiny mask format from JSON
NUM_SAMPLES = 6                 # how many random images to visualize
SHOW_INSTANCE_MASKS = False     # set to True to show each mask separately
# ----------------------------------------

def load_masks_from_json(json_path, annot_shape):
    """
    Loads all instance masks from JSON, returning a list of 2D boolean numpy arrays.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    masks = data.get("masks", [])
    instance_masks = []

    for m in masks:
        arr = np.array(m, dtype=np.uint8)

        # Fix orientation / shape if needed
        if arr.shape != annot_shape:
            if arr.shape == (annot_shape[1], annot_shape[0]):
                arr = arr.T
            else:
                # fallback resize
                pil = Image.fromarray(arr * 255)
                pil = pil.resize((annot_shape[1], annot_shape[0]), Image.NEAREST)
                arr = (np.array(pil) > 0).astype(np.uint8)

        instance_masks.append(arr.astype(bool))

    return instance_masks


def upscale_to_image(mask_small, target_w, target_h):
    """
    Upscale tiny mask (56x8) into full image size using NEAREST neighbor.
    """
    pil_small = Image.fromarray(mask_small.astype(np.uint8) * 255)
    pil_big = pil_small.resize((target_w, target_h), Image.NEAREST)
    return np.array(pil_big) > 0


def visualize_sample(img_path, annot_shape):
    """
    Loads an image and its mask JSON, visualizes mask stages.
    """
    json_path = img_path + "_coco.json"
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    if not os.path.exists(json_path):
        print(f"No mask JSON found for {img_path}, skipping.")
        return

    instance_masks = load_masks_from_json(json_path, annot_shape)

    if len(instance_masks) == 0:
        print(f"{os.path.basename(img_path)} - NO MASKS")
        combined_small = np.zeros(annot_shape, dtype=bool)
    else:
        combined_small = np.any(np.stack(instance_masks, axis=0), axis=0)

    combined_big = upscale_to_image(combined_small, w, h)

    # ---- PLOT ----
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(combined_small, cmap="gray")
    plt.title(f"Combined mask (tiny {annot_shape})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(combined_big, cmap="gray")
    plt.title("Upscaled Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Optional: show instance masks individually
    if SHOW_INSTANCE_MASKS and len(instance_masks) > 0:
        rows = 1
        cols = len(instance_masks)
        plt.figure(figsize=(3*cols, 3))

        for i, m in enumerate(instance_masks):
            plt.subplot(rows, cols, i+1)
            plt.imshow(m, cmap="gray")
            plt.title(f"Instance {i}")
            plt.axis("off")

        plt.show()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Collect all images in the folder
    all_files = sorted(os.listdir(DATA_DIR))
    image_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(image_files) == 0:
        raise ValueError("No images found in dataset folder.")

    # Pick random samples
    chosen = random.sample(image_files, min(NUM_SAMPLES, len(image_files)))

    print(f"Visualizing {len(chosen)} samples from {DATA_DIR} ...")

    for f in chosen:
        img_path = os.path.join(DATA_DIR, f)
        print(f"\n--- {f} ---")
        visualize_sample(img_path, ANNOT_MASK_SHAPE)
