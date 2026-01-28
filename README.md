# EIASR Lane & Vehicle Detection System

## Project Overview
This project implements a robust Advanced Driver Assistance System (ADAS) prototype capable of detecting Lane Lines and Vehicles in video streams and static images.

The system utilizes classical Computer Vision and Machine Learning techniques:
* Lane Detection: Canny Edge Detection, Hough Transform, and 2nd-order Polynomial Fitting.
* Vehicle Detection: Histogram of Oriented Gradients (HOG) features combined with a Support Vector Machine (SVM) classifier using an RBF kernel.
* Optimization: Implements "Safe Mode" ROI filtering and Absolute Margin Scoring (based on SVM decision boundaries) to minimize false positives on road textures while maintaining recall for legitimate vehicles.

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
│   ├── images/             # train/val images
│   └── labels/             # YOLO format labels
│
├── models/                 # Model Output Directory
│   └── vehicle_svm.xml     # Trained SVM Model (Generated after training)
│
├── output/                 # Results Directory (Videos/Images/Plots)
│
└── 2.mp4                   # Test Video File
```
---

## Installation

1. Environment Setup
It is recommended to use a virtual environment to avoid path conflicts.

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\activate

2. Install Dependencies
pip install numpy opencv-python opencv-contrib-python matplotlib scikit-learn

---

## Usage Guide

1. Train the Vehicle Detector
You must train the SVM model before running any detection. The script handles data augmentation and class balancing.
* Kernel: RBF (Radial Basis Function)
* Penalty (C): 10.0 (Strict mode to reduce road texture noise)

python -m prototype.train --kitti_root "kitti"

(Wait for the "Training finished" and "Model saved" success messages.)

2. Evaluate Model Performance
Verify accuracy and generate a confusion matrix on the validation set.

python -m prototype.evaluate --kitti_root "kitti" --model "models/vehicle_svm.xml"

(Check the output/ folder for the confusion matrix image.)

3. Run Demo (Single Image Debugging)
Analyzes a frame with "Quad-View" output (Edges, Hough Lines, ROI Mask, Final Result).

python -m prototype.demo --input "kitti/images/val/000006.png" --svm "models/vehicle_svm.xml" --display

(Result saved to: output/debug_000006.png)

4. Run Demo (Video Processing)
Process the test video with real-time lane tracking and vehicle distance estimation.

python -m prototype.demo --input "2.mp4" --svm "models/vehicle_svm.xml" --display

(Result saved to: output/processed_2.mp4)

---

## Configuration & Tuning

The system parameters are centralized in "prototype/config.py".

Vehicle Configuration:
* score_threshold: Default is 1.0. 
    This uses Absolute Margin Scoring. It means the SVM must be confident (distance from hyperplane > 1.0) to classify an object as a vehicle.
    - Lower it (e.g., 0.5) if valid vehicles are being missed.
    - Raise it (e.g., 1.5) if you see false positives on the road.
* hog_win_size: Default 64x64.

"Safe Mode" ROI (in pipeline.py):
To prevent the model from mistaking road texture for vehicles, the code enforces a strict search area:
* X-axis: Scans 2% to 98% of the image width (allows detecting cars at edges).
* Y-axis: Scans 40% to 90% of the image height (skips sky and immediate foreground).

---

## Limitations

* Truncated Vehicles: While the ROI has been widened, vehicles that are heavily cut off by the image frame may still be ignored due to HOG descriptor limitations.
* Low Contrast Conditions: Dark vehicles on dark asphalt (e.g., black cars in shadows) typically generate weak gradients and may have lower detection recall.

---

## Author
[Your Name / Student ID]
EIASR Project - Final Submission
