"""
Configuration module for the EIASR Lane & Vehicle Detection System.

This module centralizes all tunable parameters, ensuring consistency across
training, evaluation, and the detection pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import cv2

@dataclass
class CameraCalibration:
    """
    Stores intrinsic camera parameters for distance estimation and distortion correction.
    """
    matrix: np.ndarray
    distortion: np.ndarray
    perspective_src: Optional[np.ndarray] = None
    perspective_dst: Optional[np.ndarray] = None

    @classmethod
    def identity(cls, image_shape: Tuple[int, int]) -> "CameraCalibration":
        """Returns an identity matrix (no calibration) for testing purposes."""
        height, width = image_shape[:2]
        matrix = np.array(
            [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
            dtype=np.float32,
        )
        return cls(matrix=matrix, distortion=np.zeros((5,), dtype=np.float32))

@dataclass
class LaneDetectionConfig:
    """
    Parameters for the Computer Vision-based Lane Detection Pipeline.
    """
    # Gaussian Blur: Smoothes image to reduce noise before Canny
    gaussian_kernel: Tuple[int, int] = (5, 5)
    gaussian_sigma: float = 1.0
    
    # Canny Edge Detection: Hysteresis thresholds
    # High threshold (150) starts strong edges; Low (50) continues them.
    canny_thresholds: Tuple[int, int] = (50, 150)
    
    # Hough Transform: Parameters for line segment detection
    hough_threshold: int = 30         # Min votes to accept a line
    hough_min_line_length: int = 20   # Min pixels to constitute a line
    hough_max_line_gap: int = 20      # Max pixels to bridge a gap
    
    # Region of Interest (ROI) Mask
    roi_vertices: Optional[np.ndarray] = None

    def default_roi(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generates a trapezoidal mask to focus on the road surface.
        Logic: Lane lines usually converge towards the center-top of the lower image half.
        """
        height, width = image_shape[:2]
        top_y = int(height * 0.6)  # Horizon line approx at 60% height
        bottom_y = height
        return np.array(
            [[int(width * 0.1), bottom_y],   # Bottom-Left
             [int(width * 0.45), top_y],     # Top-Left
             [int(width * 0.55), top_y],     # Top-Right
             [int(width * 0.9), bottom_y]],  # Bottom-Right
            dtype=np.int32,
        )

@dataclass
class VehicleDetectionConfig:
    """
    Parameters for the HOG + SVM Vehicle Detection Pipeline.
    """
    # HOG Descriptor Parameters (Standard for Pedestrian/Vehicle detection)
    hog_win_size: Tuple[int, int] = (64, 64)      # Detection window size
    hog_block_size: Tuple[int, int] = (16, 16)    # Normalization block
    hog_block_stride: Tuple[int, int] = (8, 8)    # Block overlap
    hog_cell_size: Tuple[int, int] = (8, 8)       # Gradient cell
    hog_bins: int = 9                             # Orientation bins (0-180 degrees)
    
    svm_model_path: Optional[str] = "models/vehicle_svm.xml"
    
    # === CRITICAL PARAMETER: SVM Decision Threshold ===
    # This value represents the Absolute Margin (distance from the hyperplane).
    # Since we use an RBF kernel with C=10.0, the model is strict.
    # 0.3 - 1.0 provides a good balance between Precision (few false positives)
    # and Recall (detecting most cars).
    score_threshold: float = 0.3
    
    # Distance Estimation Constants
    car_real_width_m: float = 1.8  # Avg vehicle width in meters
    focal_length_px: Optional[float] = None

    def create_hog(self) -> "cv2.HOGDescriptor":
        """Factory method to create an OpenCV HOG descriptor instance."""
        return cv2.HOGDescriptor(
            _winSize=self.hog_win_size, _blockSize=self.hog_block_size,
            _blockStride=self.hog_block_stride, _cellSize=self.hog_cell_size,
            _nbins=self.hog_bins
        )

@dataclass
class PipelineConfig:
    """Master configuration object aggregating sub-configs."""
    lane: LaneDetectionConfig = field(default_factory=LaneDetectionConfig)
    vehicles: VehicleDetectionConfig = field(default_factory=VehicleDetectionConfig)
    calibration: Optional[CameraCalibration] = None