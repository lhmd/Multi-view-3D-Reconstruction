import os
import cv2
from transformers import pipeline
from PIL import Image
import numpy as np

def depth_estimation(video_name, output_dir):
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    cap = cv2.VideoCapture(video_name)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    desired_frames = 60
    skip_frames = total_frames // desired_frames
    extracted_frames = 0
    print(f"Total frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有帧了，就退出循环

        if frame_number % skip_frames == 0:
            # 将BGR格式的视频帧转换为RGB格式，因为模型需要RGB格式的图片
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # 使用模型估计深度
            depth = pipe(pil_image)["depth"]
            depth = np.array(depth)
            depth = (depth / np.max(depth) * 255).astype(np.uint8)
            
            depth_path = os.path.join(depth_dir, f"{frame_number}.png")
            cv2.imwrite(depth_path, depth)
            rgb_path = os.path.join(rgb_dir, f"{frame_number}.png")
            cv2.imwrite(rgb_path, frame)

            extracted_frames += 1
            print(f"Processed frame {frame_number}")
        frame_number += 1

    cap.release()

if __name__ == '__main__':
    depth_estimation("data/video.mp4", "/nas/home/wangweijie/Multi-view-3D-Reconstruction/output")