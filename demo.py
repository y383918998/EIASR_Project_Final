"""Unified Demo Script: Supports Images (Montage Output) and Videos."""
import argparse
import cv2
import os
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
    
    config = PipelineConfig()
    config.vehicles.svm_model_path = args.svm
    pipeline = LaneVehiclePipeline(config)

    input_path = Path(args.input)
    if input_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
        # === 处理单张图片 (生成四分图) ===
        frame = cv2.imread(str(input_path))
        if frame is None: return
        print(f"Processing image: {input_path.name}")
        
        # 强制请求 debug 信息以生成四分图
        montage, _ = pipeline.process_frame(frame, return_debug=True)
        
        save_path = os.path.join(args.output_dir, f"debug_{input_path.name}")
        cv2.imwrite(save_path, montage)
        print(f"Saved montage to: {save_path}")
        
        if args.display:
            cv2.imshow("Debug View", montage)
            cv2.waitKey(0)

    else:
        # === 处理视频 ===
        cap = cv2.VideoCapture(str(input_path))
        save_path = os.path.join(args.output_dir, f"processed_{input_path.name}")
        writer = None
        print(f"Processing video: {input_path.name}")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            output, _ = pipeline.process_frame(frame)
            
            if writer is None:
                h, w = output.shape[:2]
                writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
            
            writer.write(output)
            
            if args.display:
                cv2.imshow("Video", output)
                if cv2.waitKey(1) == ord('q'): break
        
        cap.release()
        if writer: writer.release()
        print(f"Saved video to: {save_path}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()