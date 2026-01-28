"""
Training Script for Vehicle Detection SVM.

Project Requirement: 
- 3. Data Classification
- 4. Results Evaluation

Features:
- Loads KITTI data (YOLO format labels).
- Performs Data Augmentation (Horizontal Flipping).
- Enforces strict 50/50 Class Balancing.
- Trains an SVM with RBF Kernel for high-precision separation.
- Cross-platform path compatibility.
"""
import argparse
import os
import random
import time
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from .config import VehicleDetectionConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM using KITTI YOLO format data")
    parser.add_argument("--kitti_root", type=str, 
                        default="kitti",
                        help="Path to the root of KITTI dataset containing images/ and labels/")
    parser.add_argument("--samples", type=int, default=20000, 
                        help="Max number of positive/negative samples to extract")
    return parser.parse_args()

def iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) to avoid overlapping negative samples."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def load_kitti_data(root_dir: str, config: VehicleDetectionConfig, max_samples: int):
    """
    Loads images and extracts car/non-car patches.
    Strategy: Filter small images (<32px) + Augmentation + Hard Balancing.
    """
    root = Path(root_dir)
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"Error: Could not find 'images/train' or 'labels/train' in {root.resolve()}")
        return [], []

    image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"Found {len(image_paths)} images. Extracting patches...")

    pos_patches = []
    neg_patches = []
    win_w, win_h = config.hog_win_size

    for i, img_path in enumerate(image_paths):
        if len(pos_patches) >= max_samples and len(neg_patches) >= max_samples:
            break
            
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        boxes = [] 
        
        # --- PROCESS POSITIVE SAMPLES (CARS) ---
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if int(parts[0]) == 0: # Class 0 = Car
                        ncx, ncy, nw, nh = map(float, parts[1:])
                        # Convert YOLO format (normalized) to pixels
                        w_px = int(nw * w_img)
                        h_px = int(nh * h_img)
                        x_px = int((ncx * w_img) - (w_px / 2))
                        y_px = int((ncy * h_img) - (h_px / 2))
                        boxes.append([x_px, y_px, w_px, h_px])
                        
                        if len(pos_patches) < max_samples:
                            # Filter small images < 32px (HOG is unreliable for tiny objects)
                            if w_px < 32 or h_px < 32: continue 
                            x1, y1 = max(0, x_px), max(0, y_px)
                            x2, y2 = min(w_img, x_px + w_px), min(h_img, y_px + h_px)
                            
                            if x2 > x1 and y2 > y1:
                                patch = img[y1:y2, x1:x2]
                                patch = cv2.resize(patch, (win_w, win_h))
                                pos_patches.append(patch)
                                # Data Augmentation: Horizontal Flip
                                pos_patches.append(cv2.flip(patch, 1))

        # --- PROCESS NEGATIVE SAMPLES (BACKGROUND) ---
        patches_needed = 6 
        patches_found = 0
        if len(neg_patches) < max_samples:
            for _ in range(30): # Try 30 times to find non-overlapping background
                rx = random.randint(0, w_img - win_w)
                ry = random.randint(0, h_img - win_h)
                r_box = [rx, ry, win_w, win_h]
                
                # Check overlap with any car in the image
                overlap = False
                for b in boxes:
                    if iou(r_box, b) > 0.05: 
                        overlap = True
                        break
                if not overlap:
                    patch = img[ry:ry+win_h, rx:rx+win_w]
                    neg_patches.append(patch)
                    patches_found += 1
                    if patches_found >= patches_needed: break 

        if i % 500 == 0:
            print(f"Processed {i} images... (Pos: {len(pos_patches)}, Neg: {len(neg_patches)})")

    # --- STRICT CLASS BALANCING ---
    # SVM is sensitive to class imbalance. We trim the larger set to match the smaller one.
    min_len = min(len(pos_patches), len(neg_patches))
    print(f"\n[CRITICAL] Balancing Data... Raw Counts -> Pos: {len(pos_patches)}, Neg: {len(neg_patches)}")
    print(f"Trimming both to {min_len} to ensure 50/50 balance.")
    pos_patches = pos_patches[:min_len]
    neg_patches = neg_patches[:min_len]
    
    return pos_patches, neg_patches

def extract_hog_features(images, config):
    """Computes HOG descriptors for a list of images."""
    hog = config.create_hog()
    features = []
    print("Computing HOG descriptors...")
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feat = hog.compute(gray).flatten()
        features.append(feat)
        if i % 5000 == 0 and i > 0:
            print(f"  Feature extraction: {i}/{len(images)}")
    return np.array(features, dtype=np.float32)

def main():
    args = parse_args()
    config = VehicleDetectionConfig()
    
    # 1. Load Data
    print("--- Step 1: Loading & Cropping Data ---")
    pos_imgs, neg_imgs = load_kitti_data(args.kitti_root, config, args.samples)
    
    if not pos_imgs or not neg_imgs:
        print("Error: Not enough data found. Check paths.")
        return

    # Create Labels: 1 for Car, -1 for Background
    y_pos = np.ones(len(pos_imgs), dtype=np.int32)
    y_neg = -np.ones(len(neg_imgs), dtype=np.int32)
    
    X_images = pos_imgs + neg_imgs
    y_labels = np.concatenate((y_pos, y_neg))
    
    # 2. Extract Features
    print("\n--- Step 2: Extracting HOG Features ---")
    X_features = extract_hog_features(X_images, config)
    
    # Split into Train/Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.2, random_state=42, shuffle=True
    )
    
    # 3. Train
    print("\n--- Step 3: Training SVM (RBF KERNEL) ---")
    print("Configuration: C=10.0 (High Penalty), Kernel=RBF")
    print("WARNING: This may take 5-10 minutes. Please wait...")
    
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF) # Radial Basis Function for non-linear separation
    svm.setC(10.0)                # High C = stricter boundaries (less misclassification allowed)
    svm.setGamma(0.1)             # Gamma controls the influence of a single training example
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 2000, 1e-6))
    
    t_start = time.time()
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    print(f"Training finished in {time.time() - t_start:.2f}s.")
    
    # 4. Evaluate
    print("\n--- Step 4: Internal Evaluation (Test Split) ---")
    _, y_pred = svm.predict(X_test)
    y_pred = y_pred.ravel().astype(np.int32)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Non-Vehicle', 'Vehicle']))
    
    # 5. Save Model
    output_dir = "models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, "vehicle_svm.xml")
    svm.save(save_path)
    print(f"\n[SUCCESS] Model saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()