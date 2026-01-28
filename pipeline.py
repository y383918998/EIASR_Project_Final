"""Core Logic: Lane Detection + Vehicle Detection Pipeline."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np
from .config import LaneDetectionConfig, PipelineConfig, VehicleDetectionConfig

@dataclass
class Lane:
    points: np.ndarray
    polynomial: np.ndarray
    side: str
    def eval_at(self, y: float) -> float: return np.polyval(self.polynomial, y)

@dataclass
class VehicleDetection:
    bbox: Tuple[int, int, int, int]
    score: float
    distance_m: Optional[float]

class LaneDetector:
    def __init__(self, config: LaneDetectionConfig) -> None:
        self.config = config
        self.debug_images: Dict[str, np.ndarray] = {}

    def detect_lanes(self, frame: np.ndarray) -> Tuple[List[Lane], Dict[str, np.ndarray]]:
        self.debug_images = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.config.gaussian_kernel, self.config.gaussian_sigma)
        
        # ROI Mask
        mask = np.zeros_like(blurred)
        vertices = self.config.roi_vertices if self.config.roi_vertices else self.config.default_roi(frame.shape)
        cv2.fillPoly(mask, [vertices], 255)
        self.debug_images['roi_mask'] = mask
        masked = cv2.bitwise_and(blurred, mask)

        # Edges
        edges = cv2.Canny(masked, self.config.canny_thresholds[0], self.config.canny_thresholds[1])
        self.debug_images['edges'] = edges

        # Hough Transform
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
                # Filter lines based on slope to separate left/right lanes
                if slope < -0.5 and x1 < width//2: left_pts.extend([[x1, y1], [x2, y2]])
                elif slope > 0.5 and x1 > width//2: right_pts.extend([[x1, y1], [x2, y2]])
        
        self.debug_images['hough_lines'] = hough_viz
        lanes = []
        for pts, side in [(left_pts, 'left'), (right_pts, 'right')]:
            if pts:
                # Fit 2nd order polynomial
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
            cv2.polylines(out, [pts], False, (255, 0, 0) if lane.side=='left' else (0, 0, 255), 5)
        return out

class VehicleDetector:
    def __init__(self, config: VehicleDetectionConfig):
        self.config = config
        self.hog = config.create_hog()
        self.svm = cv2.ml.SVM_load(config.svm_model_path) if config.svm_model_path else None

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        if not self.svm: return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        win = self.config.hog_win_size
        
        # === Scale Selection ===
        # 0.75 for distant vehicles, up to 3.0 for close/large vehicles
        scales = [0.75, 1.0, 1.5, 2.0, 3.0] 
        raw_dets = []

        for scale in scales:
            # Resize image according to scale
            if scale != 1.0:
                s_img = cv2.resize(gray, (int(gray.shape[1]/scale), int(gray.shape[0]/scale)))
            else:
                s_img = gray
            
            # Skip if scaled image is smaller than detection window
            if s_img.shape[0] < win[1] or s_img.shape[1] < win[0]: continue
            
            # === CRITICAL: Define ROI Limits to Prevent False Positives ===
            img_w = s_img.shape[1]
            img_h = s_img.shape[0]
            
            # 1. X-axis Limits: Skip left 10% (curb) and right 10% (trees)
            x_start_limit = int(img_w * 0.02) 
            x_end_limit = int(img_w * 0.98)   
            
            # 2. Y-axis Limits: Scan only the middle band (Road Horizon to Near Road)
            # Start at 40% height (skip sky/background)
            y_start = int(img_h * 0.4)
            # Stop at 85% height (skip bottom 15% - prevents road texture false positives)
            y_end = int(img_h * 0.90)

            step = 16
            
            # Apply Y-axis limits in the loop range
            for y in range(y_start, y_end - win[1], step):
                for x in range(0, img_w - win[0], step):
                    
                    # Apply X-axis filter: Skip if window is too far left or right
                    if x < x_start_limit or (x + win[0]) > x_end_limit:
                        continue

                    window = s_img[y:y+win[1], x:x+win[0]]
                    feat = self.hog.compute(window).flatten().reshape(1, -1)
                    
                    # Predict label and decision margin (raw distance to hyperplane).
                    _, pred = self.svm.predict(feat)
                    pred_label = int(pred[0][0])
                    raw_score = float(
                        self.svm.predict(feat, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1][0][0]
                    )
                    margin = abs(raw_score)

                    # Only keep positive predictions with sufficient margin.
                    if pred_label == 1 and margin >= self.config.score_threshold:
                        ox, oy = int(x*scale), int(y*scale)
                        ow, oh = int(win[0]*scale), int(win[1]*scale)
                        
                        # Distance estimation
                        dist = None
                        if self.config.focal_length_px:
                            dist = (self.config.focal_length_px * 1.8 / ow)
                            
                        raw_dets.append(VehicleDetection((ox, oy, ow, oh), margin, dist))
        
        # === NMS (Non-Maximum Suppression) ===
        raw_dets.sort(key=lambda x: x.score, reverse=True)
        keep = []
        while raw_dets:
            curr = raw_dets.pop(0)
            keep.append(curr)
            # IoU threshold 0.3
            raw_dets = [d for d in raw_dets if self._iou(curr.bbox, d.bbox) < 0.3]
        return keep

    def _iou(self, b1, b2):
        """Calculate Intersection over Union (IoU) between two boxes."""
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[0]+b1[2], b2[0]+b2[2]), min(b1[1]+b1[3], b2[1]+b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        return inter / (b1[2]*b1[3] + b2[2]*b2[3] - inter + 1e-6)

    def render(self, frame: np.ndarray, dets: List[VehicleDetection]) -> np.ndarray:
        """Draw final detection results on the frame."""
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
        
        # Lateral Deviation
        if len(lanes) == 2:
            h, w = frame.shape[:2]
            center = (lanes[0].eval_at(h) + lanes[1].eval_at(h)) / 2
            cv2.putText(out, f"Offset: {w/2 - center:.1f} px", (50, 50), 0, 1, (255, 255, 255), 2)

        if return_debug:
            # 2x2 Montage for Professor
            top = np.hstack((cv2.cvtColor(l_debug['edges'], cv2.COLOR_GRAY2BGR), l_debug['hough_lines']))
            bot = np.hstack((cv2.cvtColor(l_debug['roi_mask'], cv2.COLOR_GRAY2BGR), out))
            montage = np.vstack((top, bot))
            return cv2.resize(montage, (frame.shape[1], frame.shape[0])), l_debug
            
        return out, l_debug
