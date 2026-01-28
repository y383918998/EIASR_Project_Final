"""
Evaluation script for Vehicle Detection SVM using KITTI Validation Set.
Satisfies Project Requirement: 4. Results Evaluation
"""
import argparse
import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from .config import VehicleDetectionConfig
# Reuse the IoU function and feature extraction from the training script
from .train_svm import iou, extract_hog_features

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SVM on KITTI Validation Set")
    parser.add_argument("--kitti_root", type=str, 
                        default=r"D:\python_code\EIASR\prototype\kitti",
                        help="Path to KITTI root (must contain images/val and labels/val)")
    parser.add_argument("--model", type=str, default="vehicle_svm.xml",
                        help="Path to the trained SVM XML file")
    # Using 5000 samples is enough for a good confusion matrix plot
    parser.add_argument("--samples", type=int, default=5000,
                        help="Max validation samples to extract")
    return parser.parse_args()

def load_val_data(root_dir: str, config: VehicleDetectionConfig, max_samples: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load data specifically from the 'val' subfolder.
    Includes filtering (<32px) to match training logic for fair evaluation.
    """
    root = Path(root_dir)
    img_dir = root / "images" / "val"
    lbl_dir = root / "labels" / "val"
    
    if not img_dir.exists():
        print(f"Error: {img_dir} does not exist!")
        return [], []

    print(f"Loading Validation Data from: {img_dir}")
    
    image_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
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
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if int(parts[0]) == 0: # Car
                        ncx, ncy, nw, nh = map(float, parts[1:])
                        w_px = int(nw * w_img)
                        h_px = int(nh * h_img)
                        x_px = int((ncx * w_img) - (w_px / 2))
                        y_px = int((ncy * h_img) - (h_px / 2))
                        
                        boxes.append([x_px, y_px, w_px, h_px])
                        
                        # --- Extract Positive Sample (Car) ---
                        if len(pos_patches) < max_samples:
                            # CONSISTENCY: Filter small cars just like in training
                            if w_px < 32 or h_px < 32: continue 

                            x1, y1 = max(0, x_px), max(0, y_px)
                            x2, y2 = min(w_img, x_px + w_px), min(h_img, y_px + h_px)
                            
                            if x2 > x1 and y2 > y1:
                                patch = img[y1:y2, x1:x2]
                                patch = cv2.resize(patch, (win_w, win_h))
                                pos_patches.append(patch)
        
        # --- Extract Negative Sample (Background) ---
        if len(neg_patches) < max_samples:
            for _ in range(5): 
                rx = random.randint(0, w_img - win_w)
                ry = random.randint(0, h_img - win_h)
                r_box = [rx, ry, win_w, win_h]
                
                overlap = False
                for b in boxes:
                    if iou(r_box, b) > 0.05:
                        overlap = True
                        break
                if not overlap:
                    patch = img[ry:ry+win_h, rx:rx+win_w]
                    neg_patches.append(patch)
                    break 

        if i % 100 == 0:
            print(f"Val Processed {i} images... (Pos: {len(pos_patches)}, Neg: {len(neg_patches)})")
            
    return pos_patches, neg_patches

def main():
    args = parse_args()
    config = VehicleDetectionConfig()
    
    # 1. Load Pre-trained Model
    print(f"Loading model from {args.model}...")
    if not Path(args.model).exists():
        print(f"Error: Model file '{args.model}' not found.")
        return

    try:
        svm = cv2.ml.SVM_load(args.model)
    except Exception as e:
        print(f"Error loading SVM: {e}")
        return

    # 2. Load Validation Data
    pos_imgs, neg_imgs = load_val_data(args.kitti_root, config, args.samples)
    
    if not pos_imgs or not neg_imgs:
        print("Error: No validation data found.")
        return
        
    print(f"\nFinal Validation Set: {len(pos_imgs)} Cars, {len(neg_imgs)} Non-Cars")
    
    # Labels
    y_pos = np.ones(len(pos_imgs), dtype=np.int32)
    y_neg = -np.ones(len(neg_imgs), dtype=np.int32)
    X_images = pos_imgs + neg_imgs
    y_true = np.concatenate((y_pos, y_neg))
    
    # 3. Extract Features
    print("Extracting features...")
    X_features = extract_hog_features(X_images, config)
    
    # 4. Predict
    print("Predicting...")
    _, y_pred = svm.predict(X_features)
    y_pred = y_pred.ravel().astype(np.int32)
    
    # 5. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"VALIDATION SET RESULTS")
    print("="*40)
    print(f"Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    
    # 6. Save Plot
    try:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Vehicle', 'Vehicle'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Validation Set Confusion Matrix")
        plt.savefig("confusion_matrix_val.png")
        print(f"\n[Success] Plot saved to 'confusion_matrix_val.png'")
    except Exception as e:
        print(f"Plotting error: {e}")

if __name__ == "__main__":
    main()