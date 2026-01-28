"""
Unified Demo Script: Supports Images (Debug Montage) and Videos (Real-time Detection).

Features:
- Handles both static images and video files.
- Generates a "Quad-View" debug montage (Edges, Hough, Mask, Result).
- Implements Frame Skipping for optimized video processing speed.
"""
import argparse
import cv2
import os
import numpy as np
from pathlib import Path
from .config import PipelineConfig
from .pipeline import LaneVehiclePipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to image or video")
    parser.add_argument("--svm", type=str, default="models/vehicle_svm.xml")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Pipeline
    config = PipelineConfig()
    config.vehicles.svm_model_path = args.svm
    pipeline = LaneVehiclePipeline(config)

    input_path = Path(args.input)
    
    # === Mode 1: Single Image Processing ===
    if input_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
        frame = cv2.imread(str(input_path))
        if frame is None: return
        print(f"Processing image: {input_path.name}")
        
        # Request debug information to generate the Quad-View montage
        montage, _ = pipeline.process_frame(frame, return_debug=True)
        
        save_path = os.path.join(args.output_dir, f"debug_{input_path.name}")
        cv2.imwrite(save_path, montage)
        print(f"Saved montage to: {save_path}")
        
        if args.display:
            cv2.imshow("Debug View", montage)
            cv2.waitKey(0)

    # === Mode 2: Video Processing (Optimized) ===
    else:
        cap = cv2.VideoCapture(str(input_path))
        save_path = os.path.join(args.output_dir, f"processed_quad_{input_path.name}")
        writer = None
        print(f"Processing video: {input_path.name}")
        
        # Frame Skipping Logic variables
        frame_count = 0
        detect_interval = 3  # Run expensive AI detection every 3 frames
        
        # Cache for detection results to smooth output
        last_lanes = []
        last_cars = []
        last_lane_debug = None 

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # === Detection Phase (With Frame Skipping) ===
            if frame_count % detect_interval == 0:
                # Capture debug info (l_debug) for the Quad-View
                last_lanes, last_lane_debug = pipeline.lanes.detect_lanes(frame)
                last_cars = pipeline.vehicles.detect(frame)
            
            # === Rendering Phase ===
            # 1. Generate Main Result View (Bottom-Right)
            out_main = pipeline.lanes.render(frame, last_lanes)
            out_main = pipeline.vehicles.render(out_main, last_cars)
            
            # 2. Generate Quad-View Montage
            if last_lane_debug is not None:
                # Convert grayscale debug images to BGR for stacking
                top_left = cv2.cvtColor(last_lane_debug['edges'], cv2.COLOR_GRAY2BGR)
                top_right = last_lane_debug['hough_lines']
                bot_left = cv2.cvtColor(last_lane_debug['roi_mask'], cv2.COLOR_GRAY2BGR)
                bot_right = out_main
                
                # Stack images: (TL, TR) over (BL, BR)
                top_row = np.hstack((top_left, top_right))
                bot_row = np.hstack((bot_left, bot_right))
                final_output = np.vstack((top_row, bot_row))
                
                # Resize to maintain original resolution (prevents massive file size)
                final_output = cv2.resize(final_output, (frame.shape[1], frame.shape[0]))
            else:
                final_output = out_main

            # === Saving Video ===
            if writer is None:
                h, w = final_output.shape[:2]
                # Lower framerate slightly (24fps) for better visualization speed
                writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (w, h))
            
            writer.write(final_output)
            
            # === Display ===
            if args.display:
                # Resize for screen display (1280px width)
                display_h = 720
                display_w = int(display_h * (final_output.shape[1] / final_output.shape[0]))
                display_frame = cv2.resize(final_output, (display_w, display_h))
                
                cv2.imshow("Quad-View Video", display_frame)
                if cv2.waitKey(1) == ord('q'): break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()