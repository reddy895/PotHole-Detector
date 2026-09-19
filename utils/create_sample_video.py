"""Generate a simulated road video (road.mp4) from road.jpg for testing."""
from pathlib import Path
import cv2
import numpy as np

def generate_sample_video(image_path: str = "road.jpg", output_path: str = "road.mp4", duration_sec: int = 5, fps: int = 30):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: {image_path} not found.")
        return False

    h, w = img.shape[:2]
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Create simulated dashcam driving motion (gradual zoom / forward motion)
    for i in range(total_frames):
        # Progress from 0 to 1
        progress = i / float(total_frames)
        # Gentle zoom factor from 1.0 to 1.15
        zoom = 1.0 + 0.15 * (progress ** 1.3)
        new_w = int(w / zoom)
        new_h = int(h / zoom)

        # Center crop
        x1 = (w - new_w) // 2
        y1 = int((h - new_h) * 0.7)  # bias towards road surface
        cropped = img[y1:y1 + new_h, x1:x1 + new_w]
        frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Add subtle vertical camera vibration / vehicle shake
        shake = int(np.sin(i * 0.8) * 2)
        if shake != 0:
            M = np.float32([[1, 0, 0], [0, 1, shake]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        out.write(frame)

    out.release()
    print(f"Successfully generated sample video: {output_path} ({total_frames} frames, {duration_sec}s @ {fps}fps)")
    return True

if __name__ == "__main__":
    generate_sample_video()
