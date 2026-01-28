# EIASR Project (2025/2026)
Driving assistance prototype for the EIASR course.

## Repository layout (expected folders)
```
EIASR_Project_Final/
├── kitti/                # KITTI dataset (images/ + labels/)
├── models/               # Trained SVM models (vehicle_svm.xml)
├── output/               # Demo outputs (debug images, annotated videos)
├── prototype/            # Python pipeline code
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training the vehicle SVM
Train using KITTI `images/train` + `labels/train` (YOLO format).
```bash
python -m prototype.train --kitti_root kitti --samples 20000
```
The model is saved to `models/vehicle_svm.xml`.

## Evaluating on KITTI validation set
Evaluate using KITTI `images/val` + `labels/val`:
```bash
python -m prototype.evaluate --kitti_root kitti --model models/vehicle_svm.xml --samples 5000
```
This writes `confusion_matrix_val.png` in the repository root.

## Running the demo
### Single image (debug montage)
```bash
python -m prototype.demo --input "kitti/images/val/001576.png" --svm models/vehicle_svm.xml --display
```
The output montage is saved to `output/debug_001576.png`.

### Video
```bash
python -m prototype.demo --input "kitti/videos/sample.mp4" --svm models/vehicle_svm.xml --display
```
The annotated video is saved to `output/processed_sample.mp4`.

## Configuration tips
- `prototype/config.py` contains all thresholds and parameters.
- For vehicle detection, adjust `VehicleDetectionConfig.score_threshold` to control false positives
  (higher values are stricter).
- Lane detection uses ROI + Canny + Hough; adjust `LaneDetectionConfig` to tune line density.

## Module overview
- `prototype/config.py`: Configuration objects for lane/vehicle detection.
- `prototype/pipeline.py`: Full pipeline (lane detection + vehicle detection + rendering).
- `prototype/train.py`: SVM training for vehicle detection.
- `prototype/evaluate.py`: Validation set evaluation and confusion matrix plotting.
- `prototype/demo.py`: CLI for processing images/videos and generating debug montages.
