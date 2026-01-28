# EIASR_Project
2025/2026 Project for EIASR, topic is "driving assistance"


## Python prototype
A minimal Python prototype was added to align with the preliminary report requirements:
- Lane detection with Gaussian smoothing, Canny edge extraction, ROI masking, and Hough line grouping followed by quadratic fitting and lane-type heuristics.
- Vehicle detection hook using HOG features with an external linear SVM (optional) and monocular distance estimation using a pinhole model.

### Installation
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Running on a video
```bash
python -m prototype.video_demo --video /path/to/kitti_clip.mp4 --focal 700 \
    --svm /path/to/linear_svm.yml --output annotated.mp4 --display
```
- `--svm` is optional; when omitted, only lane overlays are produced.
- `--focal` sets the focal length in pixels for distance estimation. Without it, distance fields remain empty.
- Press `q` to stop when `--display` is enabled.

### Module overview
- `prototype/config.py`: Configuration and calibration helpers.
- `prototype/pipeline.py`: Lane detection, vehicle detection, and distance estimation pipeline.
- `prototype/video_demo.py`: CLI wrapper to process and optionally save annotated videos.# EIASR_Project
