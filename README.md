# EIASR Lane & Vehicle Detection System

## Project Overview

This project implements a robust **Advanced Driver Assistance System (ADAS)** prototype capable of detecting **lane lines** and **vehicles** in video streams and static images.

The system utilizes classical **Computer Vision** and **Machine Learning** techniques:

- **Lane Detection**
  - Canny Edge Detection
  - Hough Transform
  - 2nd-order Polynomial Fitting

- **Vehicle Detection**
  - Histogram of Oriented Gradients (HOG)
  - Support Vector Machine (SVM) with RBF kernel

- **Optimization**
  - Safe Mode ROI filtering
  - Absolute Margin Scoring (based on SVM decision boundaries)  
    → Minimizes false positives on road textures while maintaining recall

---

## Directory Structure

Ensure your project directory is organized as follows before running any scripts:

```text
Project_final/
│
├── prototype/              # Source Code Package
│   ├── __init__.py         # Package initializer
│   ├── config.py           # Configuration (Thresholds, ROI, Parameters)
│   ├── pipeline.py         # Core Logic (Lane & Vehicle Detectors)
│   ├── train.py            # SVM Training Script (RBF Kernel, C=10.0)
│   ├── evaluate.py         # Model Evaluation Script
│   └── demo.py             # Main Entry Point for Images/Videos
│
├── kitti/                  # Dataset Directory
│   ├── images/             # Train / validation images
│   └── labels/             # YOLO-format labels
│
├── models/                 # Model Output Directory
│   └── vehicle_svm.xml     # Trained SVM Model
│
├── output/                 # Results (Videos / Images / Plots)
│
└── 2.mp4                   # Test Video File

## Installation

### 1. Environment Setup

It is recommended to use a virtual environment to avoid path conflicts.

Create virtual environment:

    python -m venv .venv

Activate (Windows PowerShell):

    .\.venv\Scripts\activate

### 2. Install Dependencies

    pip install numpy opencv-python opencv-contrib-python matplotlib scikit-learn

------------------------------------------------------------------------

## Usage Guide

### 1. Train the Vehicle Detector

You must train the SVM model before running any detection. The training
script handles data augmentation and class balancing.

Training configuration: - Kernel: RBF (Radial Basis Function) - Penalty
parameter (C): 10.0 (Strict mode to reduce road texture noise)

    python -m prototype.train --kitti_root "kitti"

Wait for: - Training finished - Model saved

------------------------------------------------------------------------

### 2. Evaluate Model Performance

    python -m prototype.evaluate --kitti_root "kitti" --model "models/vehicle_svm.xml"

The confusion matrix will be saved in the output/ directory.

------------------------------------------------------------------------

### 3. Run Demo (Single Image Debugging)

    python -m prototype.demo --input "kitti/images/val/000006.png" --svm "models/vehicle_svm.xml" --display

Output: output/debug_000006.png

------------------------------------------------------------------------

### 4. Run Demo (Video Processing)

    python -m prototype.demo --input "2.mp4" --svm "models/vehicle_svm.xml" --display

Output: output/processed_2.mp4

------------------------------------------------------------------------

## Configuration & Tuning

All parameters are centralized in prototype/config.py.

### Vehicle Configuration

-   score_threshold (default: 1.0)
    -   Lower it (e.g., 0.5) to increase recall
    -   Raise it (e.g., 1.5) to reduce false positives
-   hog_win_size: 64 x 64

### Safe Mode ROI

-   X-axis: 2% -- 98% of image width
-   Y-axis: 40% -- 90% of image height

------------------------------------------------------------------------

## Limitations

-   Truncated Vehicles
-   Low Contrast Conditions

------------------------------------------------------------------------

## Author

\[Your Name / Student ID\] EIASR Project --- Final Submission
