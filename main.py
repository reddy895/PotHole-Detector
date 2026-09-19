#!/home/reddy895/Downloads/PotHole/.venv/bin/python
"""Terminal-Based AI Pothole Detection System.

Main application entry point supporting Webcam, Image, and Video detection modes.
"""
import argparse
import sys
from pathlib import Path
import time
import cv2
import numpy as np

# Ensure current directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config
from detector import PotholeDetector, DetectionResult
from utils.video_utils import (
    print_banner,
    print_detection_status,
    save_annotated_image,
    create_video_writer,
)


def parse_arguments() -> argparse.Namespace:
    """Parse terminal command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Terminal-Based AI Pothole Detection System (YOLO + OpenCV)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["webcam", "image", "video"],
        help="Input source mode: 'webcam', 'image', or 'video'",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input image or video file (required for 'image' and 'video' modes)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Path to custom YOLO .pt model weights (default: {config.MODEL_PATH})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Detection confidence threshold (default: {config.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--cam-idx",
        type=int,
        default=config.DEFAULT_CAMERA_INDEX,
        help="Camera device index for webcam mode (default: 0)",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Run without displaying OpenCV window (useful for headless / automated runs)",
    )
    return parser.parse_args()


# =========================================================================
# Mode Handlers
# =========================================================================

def run_image_mode(detector: PotholeDetector, image_path_str: str, no_view: bool = False) -> None:
    """Detect potholes in a single image, print stats, and save output."""
    image_path = Path(image_path_str)
    if not image_path.is_file():
        print(f"\n[ERROR] Input image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Print system status banner
    print_banner(
        source_name=f"Image ({image_path.name})",
        model_path=str(detector.model_path),
        device_name=detector.device_name,
    )

    # Load image
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Could not decode image: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Perform detection
    result = detector.detect(frame)
    annotated_frame = detector.draw_annotations(frame, result, show_hud=True)

    # Print terminal telemetry
    print_detection_status(
        count=result.count,
        highest_confidence=result.max_confidence,
        fps=result.fps,
        throttle_interval=0.0,
    )

    # Save processed image to outputs/
    saved_path = save_annotated_image(annotated_frame, image_path.name)
    print(f"\n[SAVED] Processed image saved to: {saved_path}")

    # Display window
    if not no_view:
        window_title = f"{config.WINDOW_TITLE} - {image_path.name}"
        cv2.imshow(window_title, annotated_frame)
        print("\nDisplaying image preview. Press any key in the image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video_mode(detector: PotholeDetector, video_path_str: str, no_view: bool = False) -> None:
    """Process road video frame-by-frame, display detection, and save annotated video."""
    video_path = Path(video_path_str)
    if not video_path.is_file():
        print(f"\n[ERROR] Input video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer to outputs/
    out_filename = f"annotated_{video_path.name}"
    writer, out_path = create_video_writer(out_filename, video_fps, (width, height))

    # Print system status banner
    print_banner(
        source_name=f"Video ({video_path.name})",
        model_path=str(detector.model_path),
        device_name=detector.device_name,
    )
    print(f"Video resolution: {width}x{height} | Total frames: {total_frames} | Target FPS: {video_fps:.1f}")
    print(f"Output destination: {out_path}\n")

    frame_idx = 0
    last_print_time = 0.0
    total_potholes_found = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1

            # Detect potholes
            result = detector.detect(frame)
            annotated_frame = detector.draw_annotations(frame, result, show_hud=True)

            # Write frame to output video
            writer.write(annotated_frame)
            total_potholes_found += result.count

            # Print terminal telemetry
            last_print_time = print_detection_status(
                count=result.count,
                highest_confidence=result.max_confidence,
                fps=result.fps,
                throttle_interval=0.4,
                last_print_time=last_print_time,
            )

            # Display window
            if not no_view:
                cv2.imshow(config.WINDOW_TITLE, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[INFO] Stopped by user (Q pressed).")
                    break

    finally:
        cap.release()
        writer.release()
        if not no_view:
            cv2.destroyAllWindows()

    print(f"\n[COMPLETE] Video processing finished.")
    print(f"Processed frames: {frame_idx}/{total_frames}")
    print(f"Total pothole instances detected: {total_potholes_found}")
    print(f"Annotated video saved to: {out_path}")


def run_webcam_mode(detector: PotholeDetector, cam_idx: int = 0, no_view: bool = False) -> None:
    """Capture live webcam frames, run real-time detection, and display OpenCV window."""
    print_banner(
        source_name=f"Webcam (Device {cam_idx})",
        model_path=str(detector.model_path),
        device_name=detector.device_name,
    )
    print(f"Initializing camera device {cam_idx}...")

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        # Fallback test with CAP_ANY
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not access webcam at index {cam_idx}.", file=sys.stderr)
        print("Check that a camera is connected and permissions are granted.", file=sys.stderr)
        sys.exit(1)

    print("Camera active. Press 'Q' inside the preview window to exit.\n")

    last_print_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from webcam. Retrying...", file=sys.stderr)
                time.sleep(0.05)
                continue

            # Run detection
            result = detector.detect(frame)
            annotated_frame = detector.draw_annotations(frame, result, show_hud=True)

            # Print terminal telemetry
            last_print_time = print_detection_status(
                count=result.count,
                highest_confidence=result.max_confidence,
                fps=result.fps,
                throttle_interval=0.4,
                last_print_time=last_print_time,
            )

            # Display preview window
            if not no_view:
                cv2.imshow(config.WINDOW_TITLE, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[INFO] Stopped by user (Q pressed).")
                    break

    finally:
        cap.release()
        if not no_view:
            cv2.destroyAllWindows()

    print("[INFO] Camera released and detection session closed.")


# =========================================================================
# Main Execution Entrypoint
# =========================================================================

def _pick_file_dialog() -> str | None:
    """Open a native file picker dialog and return selected path (or None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        import os
        root = tk.Tk()
        root.withdraw()          # hide the blank root window
        root.attributes("-topmost", True)
        file_path = filedialog.askopenfilename(
            title="Select a Video File",
            initialdir=os.path.expanduser("~"),   # start at home directory
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"  [WARN] File dialog unavailable ({e}). Enter path manually.")
        return None


def interactive_menu() -> argparse.Namespace:
    """Display interactive terminal menu and return populated Namespace."""
    print()
    print("=" * 48)
    print("      AI POTHOLE DETECTION SYSTEM")
    print("=" * 48)
    print()
    print("  Select detection mode:")
    print()
    print("  [1]  Live Video   — Real-time webcam detection")
    print("  [2]  Video File   — Browse & detect from a video")
    print()
    print("=" * 48)

    while True:
        choice = input("  Enter choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Invalid choice. Enter 1 or 2.")

    ns = argparse.Namespace(model=None, conf=config.CONFIDENCE_THRESHOLD,
                            cam_idx=config.DEFAULT_CAMERA_INDEX, no_view=False,
                            source=None, input=None)

    if choice == "1":
        ns.source = "webcam"
        cam_input = input(f"\n  Camera device index [default: 0]: ").strip()
        ns.cam_idx = int(cam_input) if cam_input.isdigit() else 0
    else:
        ns.source = "video"
        print("\n  Opening file browser... (select your video file)")
        picked = _pick_file_dialog()
        if picked:
            print(f"  Selected: {picked}")
            ns.input = picked
        else:
            # Fallback: manual entry
            while True:
                path_str = input("  Enter path to video file: ").strip().strip('"').strip("'")
                if Path(path_str).is_file():
                    ns.input = path_str
                    break
                print(f"  [ERROR] File not found: {path_str}. Try again.")

    print()
    return ns


def main():
    # If args were passed use the original argparse flow; otherwise show menu
    if len(sys.argv) > 1:
        args = parse_arguments()
    else:
        args = interactive_menu()

    # Input validation
    if args.source in ("image", "video") and not args.input:
        print(
            f"\n[ERROR] The --input argument is required when using '--source {args.source}'.\n"
            f"Example: python main.py --source {args.source} --input road.{'jpg' if args.source == 'image' else 'mp4'}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize YOLO Pothole Detector
    try:
        detector = PotholeDetector(
            model_path=args.model,
            confidence_threshold=args.conf,
        )
    except Exception as e:
        print(f"\n[INITIALIZATION ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Dispatch to appropriate mode
    if args.source == "image":
        run_image_mode(detector=detector, image_path_str=args.input, no_view=args.no_view)
    elif args.source == "video":
        run_video_mode(detector=detector, video_path_str=args.input, no_view=args.no_view)
    elif args.source == "webcam":
        run_webcam_mode(detector=detector, cam_idx=args.cam_idx, no_view=args.no_view)


if __name__ == "__main__":
    main()
