"""Configuration objects for the Python prototype lane and vehicle detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class CameraCalibration:
    """Simple container for intrinsic/extrinsic camera parameters.

    Args:
        matrix: 3x3 camera matrix.
        distortion: Distortion coefficients (k1, k2, p1, p2, k3).
        perspective_src: Source points (4,2) for perspective transform.
        perspective_dst: Destination points (4,2) for perspective transform.
    """

    matrix: np.ndarray
    distortion: np.ndarray
    perspective_src: Optional[np.ndarray] = None
    perspective_dst: Optional[np.ndarray] = None

    @classmethod
    def identity(cls, image_shape: Tuple[int, int]) -> "CameraCalibration":
        height, width = image_shape[:2]
        matrix = np.array(
            [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        distortion = np.zeros((5,), dtype=np.float32)
        return cls(matrix=matrix, distortion=distortion)


@dataclass
class LaneDetectionConfig:
    gaussian_kernel: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.0
    canny_thresholds: Tuple[int, int] = (50, 150)
    hough_threshold: int = 30
    hough_min_line_length: int = 20
    hough_max_line_gap: int = 20
    roi_vertices: Optional[np.ndarray] = None

    def default_roi(self, image_shape: Tuple[int, int]) -> np.ndarray:
        height, width = image_shape[:2]
        top_y = int(height * 0.6)
        bottom_y = height
        return np.array(
            [
                [int(width * 0.1), bottom_y],
                [int(width * 0.45), top_y],
                [int(width * 0.55), top_y],
                [int(width * 0.9), bottom_y],
            ],
            dtype=np.int32,
        )


@dataclass
class VehicleDetectionConfig:
    hog_win_size: Tuple[int, int] = (64, 64)
    hog_block_size: Tuple[int, int] = (16, 16)
    hog_block_stride: Tuple[int, int] = (8, 8)
    hog_cell_size: Tuple[int, int] = (8, 8)
    hog_bins: int = 9
    svm_model_path: Optional[str] = None
    score_threshold: float = 0.0
    car_real_width_m: float = 1.8
    focal_length_px: Optional[float] = None

    def create_hog(self) -> "cv2.HOGDescriptor":
        import cv2

        return cv2.HOGDescriptor(
            _winSize=self.hog_win_size,
            _blockSize=self.hog_block_size,
            _blockStride=self.hog_block_stride,
            _cellSize=self.hog_cell_size,
            _nbins=self.hog_bins,
        )


@dataclass
class PipelineConfig:
    lane: LaneDetectionConfig = field(default_factory=LaneDetectionConfig)
    vehicles: VehicleDetectionConfig = field(default_factory=VehicleDetectionConfig)
    calibration: Optional[CameraCalibration] = None

    def ensure_calibration(self, image_shape: Tuple[int, int]) -> CameraCalibration:
        if self.calibration is None:
            self.calibration = CameraCalibration.identity(image_shape)
        return self.calibration
