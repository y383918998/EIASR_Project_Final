"""
Core Logic Module: Lane Detection & Vehicle Detection Pipeline.

This module implements the primary processing algorithms:
1. LaneDetector: Computer Vision pipeline (Canny -> Hough -> Polyfit).
2. VehicleDetector: Machine Learning pipeline (HOG -> SVM -> NMS).
3. Optimizations: Hardware acceleration (OpenCL) and ROI filtering.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np
from .config import LaneDetectionConfig, PipelineConfig, VehicleDetectionConfig

# === Hardware Acceleration Initialization ===
try:
    cv2.ocl.setUseOpenCL(True)
    is_opencl_active = cv2.ocl.useOpenCL()
    device_name = cv2.ocl.Device.getDefault().name() if is_opencl_active else "CPU"
    print(f"[System] Hardware Acceleration: {'ENABLED' if is_opencl_active else 'DISABLED'} ({device_name})")
except Exception as e:
    print(f"[System] Warning: OpenCL initialization failed. Fallback to CPU. Error: {e}")

@dataclass
class Lane:
    """Represents a detected lane line model."""
    points: np.ndarray      # Raw points from Hough Transform
    polynomial: np.ndarray  # Coefficients of the 2nd-order polynomial (ax^2 + bx + c)
    side: str               # 'left' or 'right'

    def eval_at(self, y: float) -> float:
        """Evaluate the polynomial at a given Y coordinate to find X."""
        return np.polyval(self.polynomial, y)

@dataclass
class VehicleDetection:
    """Represents a single detected vehicle with scoring and distance metadata."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    score: float                     # SVM Decision Margin (Confidence)
    distance_m: Optional[float]      # Estimated distance in meters

class LaneDetector:
    """Detects lane boundaries using classical Computer Vision techniques."""
    def __init__(self, config: LaneDetectionConfig) -> None:
        self.config = config
        self.debug_images: Dict[str, np.ndarray] = {}

    def detect_lanes(self, frame: np.ndarray) -> Tuple[List[Lane], Dict[str, np.ndarray]]:
        self.debug_images = {}
        # 1. Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.config.gaussian_kernel, self.config.gaussian_sigma)
        
        # 2. ROI Masking
        mask = np.zeros_like(blurred)
        vertices = self.config.roi_vertices if self.config.roi_vertices else self.config.default_roi(frame.shape)
        cv2.fillPoly(mask, [vertices], 255)
        self.debug_images['roi_mask'] = mask
        masked = cv2.bitwise_and(blurred, mask)

        # 3. Edge Detection
        edges = cv2.Canny(masked, self.config.canny_thresholds[0], self.config.canny_thresholds[1])
        self.debug_images['edges'] = edges

        # 4. Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.config.hough_threshold, 
                                self.config.hough_min_line_length, self.config.hough_max_line_gap)
        
        hough_viz = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        left_pts, right_pts = [], []
        height, width = frame.shape[:2]

        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = line
                cv2.line(hough_viz, (x1, y1), (x2, y2), (0, 0, 255), 1)
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)
                # Slope Filtering
                if slope < -0.5 and x1 < width//2:
                    left_pts.extend([[x1, y1], [x2, y2]])
                elif slope > 0.5 and x1 > width//2:
                    right_pts.extend([[x1, y1], [x2, y2]])
        
        self.debug_images['hough_lines'] = hough_viz
        
        # 5. Polynomial Fitting
        lanes = []
        for pts, side in [(left_pts, 'left'), (right_pts, 'right')]:
            if pts:
                p = np.polyfit(np.array(pts)[:, 1], np.array(pts)[:, 0], 2)
                lanes.append(Lane(np.array(pts), p, side))
        return lanes, self.debug_images

    def render(self, frame: np.ndarray, lanes: List[Lane]) -> np.ndarray:
        out = frame.copy()
        h = out.shape[0]
        for lane in lanes:
            ys = np.linspace(int(h*0.6), h-1, h//2)
            xs = lane.eval_at(ys)
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            color = (255, 0, 0) if lane.side == 'left' else (0, 0, 255)
            cv2.polylines(out, [pts], False, color, 5)
        return out

class VehicleDetector:
    """Detects vehicles using a Sliding Window approach with HOG features and SVM classification."""
    def __init__(self, config: VehicleDetectionConfig):
        self.config = config
        self.hog = config.create_hog()
        self.svm = cv2.ml.SVM_load(config.svm_model_path) if config.svm_model_path else None

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        if not self.svm: return []
        
        # === RESTORED ACCURACY: No Downsampling ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        win = self.config.hog_win_size
        
        # === KEY FIX: Add 0.5 scale to see distant cars ===
        # 0.5 means "Resize image to 200%", making small 30px cars into 60px cars
        scales = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0] 
        raw_dets = []

        for scale in scales:
            if scale != 1.0:
                s_img = cv2.resize(gray, (int(gray.shape[1]/scale), int(gray.shape[0]/scale)))
            else:
                s_img = gray
            
            if s_img.shape[0] < win[1] or s_img.shape[1] < win[0]: continue
            
            img_w = s_img.shape[1]
            img_h = s_img.shape[0]
            
            # === ROI Limits ===
            x_start_limit = int(img_w * 0.02) 
            x_end_limit = int(img_w * 0.98)   
            y_start = int(img_h * 0.40)
            y_end = int(img_h * 0.90)

            # Step size 16 for better accuracy
            step = 16  
            
            for y in range(y_start, y_end - win[1], step):
                for x in range(0, img_w - win[0], step):
                    
                    if x < x_start_limit or (x + win[0]) > x_end_limit:
                        continue

                    window = s_img[y:y+win[1], x:x+win[0]]
                    feat = self.hog.compute(window).flatten().reshape(1, -1)
                    
                    # === Logic Check ===
                    # 1. Get Prediction
                    _, pred_res = self.svm.predict(feat)
                    label = int(pred_res[0][0])
                    
                    # 2. Get Score
                    raw_score = float(self.svm.predict(feat, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1][0][0])
                    score = abs(raw_score)
                    
                    # 3. Combined Filter (Label MUST be 1, Score MUST be > Threshold)
                    if label == 1 and score > self.config.score_threshold:
                        # Coordinate Mapping
                        ox = int(x * scale)
                        oy = int(y * scale)
                        ow = int(win[0] * scale)
                        oh = int(win[1] * scale)
                        
                        dist = None
                        if self.config.focal_length_px:
                            dist = (self.config.focal_length_px * 1.8 / ow)
                            
                        raw_dets.append(VehicleDetection((ox, oy, ow, oh), score, dist))
        
        # === NMS ===
        raw_dets.sort(key=lambda x: x.score, reverse=True)
        keep = []
        while raw_dets:
            curr = raw_dets.pop(0)
            keep.append(curr)
            raw_dets = [d for d in raw_dets if self._iou(curr.bbox, d.bbox) < 0.3]
        return keep

    def _iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[0]+b1[2], b2[0]+b2[2]), min(b1[1]+b1[3], b2[1]+b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        return inter / (b1[2]*b1[3] + b2[2]*b2[3] - inter + 1e-6)

    def render(self, frame: np.ndarray, dets: List[VehicleDetection]) -> np.ndarray:
        out = frame.copy()
        for d in dets:
            x, y, w, h = d.bbox
            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if d.distance_m: 
                cv2.putText(out, f"{d.distance_m:.1f}m", (x, y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out

class LaneVehiclePipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.lanes = LaneDetector(self.config.lane)
        self.vehicles = VehicleDetector(self.config.vehicles)

    def process_frame(self, frame: np.ndarray, return_debug: bool = False):
        lanes, l_debug = self.lanes.detect_lanes(frame)
        cars = self.vehicles.detect(frame)
        out = self.lanes.render(frame, lanes)
        out = self.vehicles.render(out, cars)
        
        if len(lanes) == 2:
            h, w = frame.shape[:2]
            center = (lanes[0].eval_at(h) + lanes[1].eval_at(h)) / 2
            deviation = w/2 - center
            cv2.putText(out, f"Offset: {deviation:.1f} px", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if return_debug:
            top = np.hstack((cv2.cvtColor(l_debug['edges'], cv2.COLOR_GRAY2BGR), l_debug['hough_lines']))
            bot = np.hstack((cv2.cvtColor(l_debug['roi_mask'], cv2.COLOR_GRAY2BGR), out))
            montage = np.vstack((top, bot))
            return cv2.resize(montage, (frame.shape[1], frame.shape[0])), l_debug
            
        return out, l_debug