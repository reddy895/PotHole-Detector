#!/usr/bin/env python3
"""Terminal-Based AI Pothole Detection System.

Main application entry point supporting Webcam, Image, and Video detection modes.
"""
import argparse
import sys
from typing import Optional
from pathlib import Path
import time

import cv2

# Ensure current directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import config
from detector import (
    PotholeDetector,
    ThreadedInferencePipeline,
    PotholeTracker,
)
from utils.video_utils import (
    print_banner,
    print_progress_bar,
    print_webcam_status,
    print_final_summary,
    save_annotated_image,
    create_video_writer,
    save_detection_log,
)
from utils.whatsapp_notifier import WhatsAppNotifier




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
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Append a JSONL detection event log to outputs/detection_log.jsonl",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=config.DEFAULT_SKIP_FRAMES,
        metavar="N",
        help=f"Run inference every Nth frame (default: {config.DEFAULT_SKIP_FRAMES})",
    )
    parser.add_argument(
        "--whatsapp",
        action="store_true",
        default=config.WHATSAPP_ENABLED,
        help="Enable automated WhatsApp hazard alerts to authorities",
    )
    parser.add_argument(
        "--authority-phone",
        type=str,
        default=config.WHATSAPP_AUTHORITY_PHONE,
        metavar="PHONE",
        help=f"Authority WhatsApp phone number (default: {config.WHATSAPP_AUTHORITY_PHONE})",
    )
    return parser.parse_args()



# =========================================================================
# Mode Handlers
# =========================================================================

def run_image_mode(
    detector: PotholeDetector,
    image_path_str: str,
    no_view: bool = False,
    save_log: bool = False,
    notifier: Optional[WhatsAppNotifier] = None,
) -> None:

    """Detect potholes in a single image, print stats, and save output."""
    image_path = Path(image_path_str)
    if not image_path.is_file():
        print(f"\n[ERROR] Input image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print_banner(
        source_name=f"Image ({image_path.name})",
        model_path=str(detector.model_path),
        device_name=detector.device_name,
    )

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[ERROR] Could not decode image: {image_path}", file=sys.stderr)
        sys.exit(1)

    result = detector.detect(frame)
    annotated_frame = detector.draw_annotations(frame, result, show_hud=True)
    saved_path = save_annotated_image(annotated_frame, image_path.name)

    if save_log:
        save_detection_log(result, source_name=image_path.name, frame_idx=1)

    # Dispatch WhatsApp alert to authority if configured
    wa_msg = None
    if notifier and notifier.is_configured and result.count > 0:
        for det in result.detections:
            sent = notifier.send_pothole_alert(
                detection=det,
                frame=annotated_frame,
                source_name=image_path.name,
                async_dispatch=False,
            )
            if sent:
                wa_msg = f"WHATSAPP ALERT -> {notifier.authority_phone} | Pothole [{det.severity}]"
                print(f"\n📲 [WHATSAPP DISPATCH] Alert sent to {notifier.authority_phone} [{det.severity}]")
            break

    if wa_msg:
        annotated_frame = detector.draw_annotations(frame, result, show_hud=True, whatsapp_msg=wa_msg)

    log_file = (config.OUTPUTS_DIR / "detection_log.jsonl") if save_log else None

    print_final_summary(
        source_name=image_path.name,
        processed_frames=1,
        total_frames=1,
        unique_potholes=result.count,
        total_instances=result.count,
        highest_confidence=result.max_confidence,
        avg_fps=result.fps,
        output_path=saved_path,
        log_path=log_file,
    )

    if not no_view:
        window_title = f"{config.WINDOW_TITLE} - {image_path.name}"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_title, annotated_frame)
        print("\nDisplaying image preview (Full Screen). Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video_mode(
    detector: PotholeDetector,
    video_path_str: str,
    no_view: bool = False,
    save_log: bool = False,
    skip_frames: int = 0,
    notifier: Optional[WhatsAppNotifier] = None,
) -> None:

    """Process road video frame-by-frame, display detection, and save annotated video.

    Args:
        detector: Initialised PotholeDetector.
        video_path_str: Path to the input video file.
        no_view: Suppress the OpenCV preview window.
        save_log: Append detection events to outputs/detection_log.jsonl.
        skip_frames: Run inference every Nth frame; reuse last result otherwise.
            0 means every frame (no skipping).
    """
    video_path = Path(video_path_str)
    if not video_path.is_file():
        print(f"\n[ERROR] Input video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_fps = 15.0
    target_frame_time = 1.0 / target_fps
    output_fps = target_fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_filename = f"annotated_{video_path.name}"
    writer, out_path = create_video_writer(out_filename, output_fps, (width, height))

    print_banner(
        source_name=f"Video ({video_path.name})",
        model_path=str(detector.model_path),
        device_name=detector.device_name,
    )
    skip_info = f" | Frame-skip: {skip_frames}" if skip_frames > 0 else ""
    print(f"Resolution: {width}x{height} | Frames: {total_frames} | Max 15 FPS | Recursive Loop (Full Screen){skip_info}")
    print(f"Output: {out_path}\n")

    frame_idx = 0
    total_potholes_found = 0
    fps_history = []
    tracker = PotholeTracker(min_hits=min(3, max(1, total_frames // 15)))
    last_result = None
    last_whatsapp_msg = None
    last_whatsapp_time = 0.0

    if not no_view:
        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        with ThreadedInferencePipeline(detector) as pipe:
            while True:
                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Recursive continuous loop until stopped by user
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                frame_idx += 1

                # Frame-skip: feed only every Nth frame to the inference thread;
                # the pipeline automatically returns the last result for skipped frames.
                if skip_frames == 0 or frame_idx % (skip_frames + 1) == 1:
                    pipe.put(frame)

                result = pipe.get(timeout=0.05)
                if result is None:
                    # No result yet (very first frame) — run synchronously as fallback
                    result = detector.detect(frame)

                if result is not last_result:
                    tracker.update(result.detections, width, height)
                    last_result = result

                total_potholes_found += result.count

                work_duration = time.perf_counter() - loop_start
                remaining_time = max(0.0, target_frame_time - work_duration)
                effective_frame_time = work_duration + remaining_time
                current_fps = min(15.0, 1.0 / max(0.0001, effective_frame_time))
                fps_history.append(current_fps)

                # Dispatch automated WhatsApp alert for detected potholes
                if notifier and notifier.is_configured and result.count > 0:
                    for det in result.detections:
                        sent = notifier.send_pothole_alert(
                            detection=det,
                            frame=annotated_frame if 'annotated_frame' in locals() else frame,
                            source_name=video_path.name,
                            async_dispatch=True,
                        )
                        if sent:
                            track_str = f"#{det.track_id}" if det.track_id is not None else "N/A"
                            last_whatsapp_msg = f"WHATSAPP ALERT -> {notifier.authority_phone} | Pothole {track_str} [{det.severity}]"
                            last_whatsapp_time = time.time()
                            print(f"\n📲 [WHATSAPP DISPATCH] Alert sent to {notifier.authority_phone} | Pothole {track_str} [{det.severity}] ({det.confidence*100:.1f}%)")
                        break

                active_wa_msg = last_whatsapp_msg if (time.time() - last_whatsapp_time) < 4.0 else None

                annotated_frame = detector.draw_annotations(
                    frame,
                    result,
                    show_hud=True,
                    total_count=tracker.unique_count,
                    override_fps=current_fps,
                    whatsapp_msg=active_wa_msg,
                )
                writer.write(annotated_frame)

                if save_log and result.count > 0:
                    save_detection_log(result, source_name=video_path.name, frame_idx=frame_idx)

                print_progress_bar(
                    frame_idx=frame_idx % max(1, total_frames),
                    total_frames=total_frames,
                    fps=current_fps,
                    current_potholes=tracker.unique_count,
                    highest_confidence=tracker.highest_confidence,
                )

                if not no_view:
                    cv2.imshow(config.WINDOW_TITLE, annotated_frame)
                    wait_ms = max(1, int(remaining_time * 1000)) if remaining_time > 0 else 1
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("\n[INFO] Stopped by user (Q pressed).")
                        break
                else:
                    if remaining_time > 0:
                        time.sleep(remaining_time)

    finally:
        cap.release()
        writer.release()
        if not no_view:
            cv2.destroyAllWindows()

    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
    log_file = (config.OUTPUTS_DIR / "detection_log.jsonl") if save_log else None
    print_final_summary(
        source_name=video_path.name,
        processed_frames=frame_idx,
        total_frames=total_frames,
        unique_potholes=tracker.unique_count,
        total_instances=total_potholes_found,
        highest_confidence=tracker.highest_confidence,
        avg_fps=avg_fps,
        output_path=out_path,
        log_path=log_file,
    )


def run_webcam_mode(
    detector: PotholeDetector,
    cam_idx: int = 0,
    no_view: bool = False,
    save_log: bool = False,
    skip_frames: int = 0,
    notifier: Optional[WhatsAppNotifier] = None,
) -> None:

    """Capture live webcam frames, run real-time detection, and display OpenCV window.

    Args:
        detector: Initialised PotholeDetector.
        cam_idx: Camera device index.
        no_view: Suppress OpenCV preview window (headless mode).
        save_log: Append detection events to outputs/detection_log.jsonl.
        skip_frames: Run inference every Nth frame; reuse last result otherwise.
    """
    print_banner(
        source_name=f"Webcam (Device {cam_idx})",
        model_path=str(detector.model_path),
        device_name=detector.device_name,
    )
    print(f"Initializing camera device {cam_idx}...")

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"\n[ERROR] Could not access webcam at index {cam_idx}.", file=sys.stderr)
        print("Check that a camera is connected and permissions are granted.", file=sys.stderr)
        sys.exit(1)

    print("Camera active. Press 'Q' inside the preview window to exit.\n")

    target_fps = float(config.TARGET_VIDEO_FPS)
    target_frame_time = 1.0 / target_fps if target_fps > 0 else 0.080

    frame_idx = 0
    total_potholes_found = 0
    fps_history = []
    tracker = PotholeTracker(min_hits=3)
    last_result = None
    last_whatsapp_msg = None
    last_whatsapp_time = 0.0

    if not no_view:
        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        with ThreadedInferencePipeline(detector) as pipe:
            while True:
                loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[WARN] Failed to read frame from webcam. Retrying...", file=sys.stderr)
                    time.sleep(0.05)
                    continue

                frame_idx += 1
                h, w = frame.shape[:2]

                # Feed every Nth frame to inference thread; reuse last result otherwise
                if skip_frames == 0 or frame_idx % (skip_frames + 1) == 1:
                    pipe.put(frame)

                result = pipe.get(timeout=0.04)
                if result is None:
                    result = detector.detect(frame)  # fallback for very first frame

                if result is not last_result:
                    tracker.update(result.detections, w, h)
                    last_result = result

                total_potholes_found += result.count

                work_duration = time.perf_counter() - loop_start
                remaining_time = max(0.0, target_frame_time - work_duration)
                effective_frame_time = work_duration + remaining_time
                current_fps = min(15.0, 1.0 / max(0.0001, effective_frame_time))
                fps_history.append(current_fps)

                # Dispatch automated WhatsApp alert for detected potholes
                if notifier and notifier.is_configured and result.count > 0:
                    for det in result.detections:
                        sent = notifier.send_pothole_alert(
                            detection=det,
                            frame=annotated_frame if 'annotated_frame' in locals() else frame,
                            source_name=f"webcam:{cam_idx}",
                            async_dispatch=True,
                        )
                        if sent:
                            track_str = f"#{det.track_id}" if det.track_id is not None else "N/A"
                            last_whatsapp_msg = f"WHATSAPP ALERT -> {notifier.authority_phone} | Pothole {track_str} [{det.severity}]"
                            last_whatsapp_time = time.time()
                            print(f"\n📲 [WHATSAPP DISPATCH] Alert sent to {notifier.authority_phone} | Pothole {track_str} [{det.severity}] ({det.confidence*100:.1f}%)")
                        break

                active_wa_msg = last_whatsapp_msg if (time.time() - last_whatsapp_time) < 4.0 else None

                annotated_frame = detector.draw_annotations(
                    frame,
                    result,
                    show_hud=True,
                    total_count=tracker.unique_count,
                    override_fps=current_fps,
                    whatsapp_msg=active_wa_msg,
                )

                if save_log and result.count > 0:
                    save_detection_log(result, source_name=f"webcam:{cam_idx}", frame_idx=frame_idx)

                print_webcam_status(
                    frame_idx=frame_idx,
                    fps=current_fps,
                    current_count=result.count,
                    total_unique=tracker.unique_count,
                    highest_confidence=tracker.highest_confidence,
                )

                if not no_view:
                    cv2.imshow(config.WINDOW_TITLE, annotated_frame)
                    wait_ms = max(1, int(remaining_time * 1000)) if remaining_time > 0 else 1
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("\n[INFO] Stopped by user (Q pressed).")
                        break
                else:
                    if remaining_time > 0:
                        time.sleep(remaining_time)

    finally:
        cap.release()
        if not no_view:
            cv2.destroyAllWindows()

    avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0
    log_file = (config.OUTPUTS_DIR / "detection_log.jsonl") if save_log else None
    print_final_summary(
        source_name=f"Webcam ({cam_idx})",
        processed_frames=frame_idx,
        total_frames=frame_idx,
        unique_potholes=tracker.unique_count,
        total_instances=total_potholes_found,
        highest_confidence=tracker.highest_confidence,
        avg_fps=avg_fps,
        log_path=log_file,
    )



# =========================================================================
# Main Execution Entrypoint
# =========================================================================

def _pick_file_dialog(file_type: str = "video") -> str | None:
    """Open a native file picker dialog and return selected path (or None)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        import os
        root = tk.Tk()
        root.withdraw()          # hide the blank root window
        root.attributes("-topmost", True)
        if file_type == "image":
            file_path = filedialog.askopenfilename(
                title="Select an Image File",
                initialdir=os.path.expanduser("~"),
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                    ("All files", "*.*"),
                ],
            )
        else:
            file_path = filedialog.askopenfilename(
                title="Select a Video File",
                initialdir=os.path.expanduser("~"),
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm *.m4v"),
                    ("All files", "*.*"),
                ],
            )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"  [WARN] Native file dialog unavailable ({e}). Enter path manually.")
        return None


def interactive_menu() -> argparse.Namespace:
    """Display interactive terminal menu and return populated Namespace."""
    print()
    print("=" * 56)
    print("      AI POTHOLE DETECTION SYSTEM (TERMINAL & LIVE CV)")
    print("=" * 56)
    print()
    print("  Select detection mode:")
    print()
    print("  [1]  Live Webcam — Real-time camera detection in OpenCV")
    print("  [2]  Video File  — Browse & detect with live OpenCV tracking")
    print()
    print("=" * 56)

    while True:
        choice = input("  Enter choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Invalid choice. Enter 1 or 2.")

    ns = argparse.Namespace(
        model=None,
        conf=config.CONFIDENCE_THRESHOLD,
        cam_idx=config.DEFAULT_CAMERA_INDEX,
        no_view=False,
        source=None,
        input=None,
        save_log=False,
        skip_frames=config.DEFAULT_SKIP_FRAMES,
        whatsapp=True,
        authority_phone="+919591152862",
    )

    if choice == "1":
        ns.source = "webcam"
        cam_input = input("\n  Camera device index [default: 0]: ").strip()
        ns.cam_idx = int(cam_input) if cam_input.isdigit() else 0
    elif choice == "2":
        ns.source = "video"
        print("\n  Opening file browser... (select your video file)")
        picked = _pick_file_dialog(file_type="video")
        if picked:
            print(f"  Selected: {picked}")
            ns.input = picked
        else:
            # Fallback: manual entry
            while True:
                path_str = input("  Enter path to video file (or drag & drop): ").strip().strip('"').strip("'")
                if Path(path_str).is_file():
                    ns.input = path_str
                    break
                print(f"  [ERROR] File not found: {path_str}. Try again.")

    print(f"\n  [WHATSAPP ALERT BOT] Enabled -> Auto-dispatching alerts to: {ns.authority_phone}\n")
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

    # Initialize WhatsApp Notifier if requested
    notifier = None
    if getattr(args, "whatsapp", False) or config.WHATSAPP_ENABLED:
        authority = getattr(args, "authority_phone", None) or config.WHATSAPP_AUTHORITY_PHONE
        notifier = WhatsAppNotifier(
            authority_phone=authority,
            enabled=True,
            min_severity=config.WHATSAPP_MIN_SEVERITY,
            cooldown_seconds=config.WHATSAPP_COOLDOWN_SECONDS,
            port=config.WHATSAPP_PORT,
        )
        print(f"[WHATSAPP] Bot background service target: {notifier.authority_phone}")
        started = notifier.start_service()
        if started:
            notifier.wait_for_authentication(timeout_seconds=2)

    if args.source == "image":
        run_image_mode(
            detector=detector,
            image_path_str=args.input,
            no_view=args.no_view,
            save_log=args.save_log,
            notifier=notifier,
        )
    elif args.source == "video":
        run_video_mode(
            detector=detector,
            video_path_str=args.input,
            no_view=args.no_view,
            save_log=args.save_log,
            skip_frames=args.skip_frames,
            notifier=notifier,
        )
    elif args.source == "webcam":
        run_webcam_mode(
            detector=detector,
            cam_idx=args.cam_idx,
            no_view=args.no_view,
            save_log=args.save_log,
            skip_frames=args.skip_frames,
            notifier=notifier,
        )



if __name__ == "__main__":
    main()
