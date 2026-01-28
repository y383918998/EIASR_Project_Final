"""Prototype implementation for lane detection and monocular vehicle distance estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import CameraCalibration, LaneDetectionConfig, PipelineConfig, VehicleDetectionConfig


@dataclass
class Lane:
    points: np.ndarray
    polynomial: np.ndarray
    fill_ratio: float

    @property
    def lane_type(self) -> str:
        return "solid" if self.fill_ratio > 0.6 else "dashed"


@dataclass
class VehicleDetection:
    bbox: Tuple[int, int, int, int]
    score: float
    distance_m: Optional[float]


class LaneDetector:
    def __init__(self, config: LaneDetectionConfig) -> None:
        self.config = config

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, self.config.gaussian_kernel, self.config.gaussian_sigma)
        return blurred

    def _region_of_interest(self, image: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(image)
        vertices = (
            self.config.roi_vertices
            if self.config.roi_vertices is not None
            else self.config.default_roi(image.shape)
        )
        cv2.fillPoly(mask, [vertices], 255)
        return cv2.bitwise_and(image, mask)

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(
            image,
            self.config.canny_thresholds[0],
            self.config.canny_thresholds[1],
        )
        return self._region_of_interest(edges)

    def _fit_polynomial(self, points: np.ndarray) -> np.ndarray:
        ys = points[:, 1]
        xs = points[:, 0]
        return np.polyfit(xs, ys, 2)

    def detect_lanes(self, frame: np.ndarray) -> List[Lane]:
        blurred = self.preprocess(frame)
        edges = self.detect_edges(blurred)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_max_line_gap,
        )
        if lines is None:
            return []

        lane_points: List[np.ndarray] = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = line
            lane_points.append(np.array([[x1, y1], [x2, y2]]))
        all_points = np.concatenate(lane_points, axis=0)

        poly = self._fit_polynomial(all_points)
        curve_x = all_points[:, 0]
        predicted_y = np.polyval(poly, curve_x)
        closeness = np.abs(predicted_y - all_points[:, 1])
        fill_ratio = float(np.mean(closeness < 5))
        return [Lane(points=all_points, polynomial=poly, fill_ratio=fill_ratio)]

    def render(self, frame: np.ndarray, lanes: List[Lane]) -> np.ndarray:
        output = frame.copy()
        height, _ = output.shape[:2]
        for lane in lanes:
            xs = np.linspace(0, output.shape[1], num=100)
            ys = np.polyval(lane.polynomial, xs)
            pts = np.stack([xs, ys], axis=1).astype(np.int32)
            pts = pts[(pts[:, 1] >= 0) & (pts[:, 1] < height)]
            cv2.polylines(output, [pts], isClosed=False, color=(0, 255, 0), thickness=3)
            label = f"{lane.lane_type} lane"
            if pts.size:
                cv2.putText(
                    output,
                    label,
                    (int(pts[0, 0]), int(max(pts[0, 1] - 10, 0))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
        return output


class VehicleDetector:
    def __init__(self, config: VehicleDetectionConfig):
        self.config = config
        self.hog = config.create_hog()
        self.svm = self._load_svm(config.svm_model_path)

    def _load_svm(self, path: Optional[str]) -> Optional[cv2.ml_SVM]:
        if path is None:
            return None
        svm = cv2.ml.SVM_load(path)
        return svm

    def _sliding_windows(self, image: np.ndarray, step: int = 16) -> List[Tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        win_w, win_h = self.config.hog_win_size
        boxes = []
        for y in range(0, h - win_h, step):
            for x in range(0, w - win_w, step):
                boxes.append((x, y, win_w, win_h))
        return boxes

    def _compute_distance(self, box: Tuple[int, int, int, int]) -> Optional[float]:
        _, _, w, _ = box
        focal = self.config.focal_length_px
        if focal is None or w == 0:
            return None
        return (self.config.car_real_width_m * focal) / w

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._sliding_windows(gray)
        detections: List[VehicleDetection] = []
        if self.svm is None:
            return detections

        for (x, y, w, h) in boxes:
            roi = gray[y : y + h, x : x + w]
            roi_resized = cv2.resize(roi, self.config.hog_win_size)
            descriptor = self.hog.compute(roi_resized).reshape(1, -1)
            _, score = self.svm.predict(descriptor, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            score_value = float(-score[0][0])
            if score_value > self.config.score_threshold:
                detections.append(
                    VehicleDetection(
                        bbox=(x, y, w, h),
                        score=score_value,
                        distance_m=self._compute_distance((x, y, w, h)),
                    )
                )
        return detections

    def render(self, frame: np.ndarray, detections: List[VehicleDetection]) -> np.ndarray:
        output = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label = f"car {det.score:.2f}"
            if det.distance_m is not None:
                label += f" | {det.distance_m:.1f}m"
            cv2.putText(output, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return output


class LaneVehiclePipeline:
    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.lanes = LaneDetector(self.config.lane)
        self.vehicles = VehicleDetector(self.config.vehicles)
        self.calibration: Optional[CameraCalibration] = None

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        calibration = self.calibration or self.config.ensure_calibration(frame.shape)
        self.calibration = calibration
        return cv2.undistort(frame, calibration.matrix, calibration.distortion)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Lane], List[VehicleDetection]]:
        undistorted = self._undistort(frame)
        lanes = self.lanes.detect_lanes(undistorted)
        vehicles = self.vehicles.detect(undistorted)

        overlay = self.lanes.render(undistorted, lanes)
        overlay = self.vehicles.render(overlay, vehicles)
        return overlay, lanes, vehicles


__all__ = [
    "LaneVehiclePipeline",
    "PipelineConfig",
    "LaneDetectionConfig",
    "VehicleDetectionConfig",
    "CameraCalibration",
    "Lane",
    "VehicleDetection",
]
