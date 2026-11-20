"""
Test script for Ultralytics YOLOv8 / YOLO11-Pose (Task #1: Python CV libraries analysis).

Features tested:
1. 2D anatomical skeleton detection (17 COCO keypoints + confidence scores).
2. Biomechanical joint angle calculation (e.g., knee and elbow 2D angles).
3. Performance benchmark (FPS & inference latency in ms).

Usage:
    # Run with webcam:
    python research/001-python-cv-libraries-analysis/test_yolo_pose.py --cam

    # Run with a video from data/ folder (with or without .mp4 / leading slash):
    python research/001-python-cv-libraries-analysis/test_yolo_pose.py --video /bench-press/bench-press-01

    # Choose YOLO model variant (default: yolo11n-pose.pt, can also use yolov8n-pose.pt, yolo11s-pose.pt, etc.):
    python research/001-python-cv-libraries-analysis/test_yolo_pose.py --video /bench-press/bench-press-01 --model yolo11s-pose.pt
"""

import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = SCRIPT_DIR / "models"

# COCO 17 Keypoint indices used by Ultralytics YOLO-Pose:
# 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear,
# 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow, 9: L-Wrist, 10: R-Wrist,
# 11: L-Hip, 12: R-Hip, 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
KP_LEFT_SHOULDER = 5
KP_LEFT_ELBOW = 7
KP_LEFT_WRIST = 9
KP_LEFT_HIP = 11
KP_LEFT_KNEE = 13
KP_LEFT_ANKLE = 15


def resolve_video_path(video_arg: str) -> Path:
    """
    Resolves a video path relative to the project's data/ directory.
    Supports inputs like:
      --video /bench-press/bench-press-01
      --video bench-press/bench-press-01.mp4
      --video data/bench-press/bench-press-01.mp4
    """
    clean_arg = video_arg.lstrip("/\\")
    candidates = [
        DATA_DIR / clean_arg,
        DATA_DIR / f"{clean_arg}.mp4",
        DATA_DIR / f"{clean_arg}.avi",
        DATA_DIR / f"{clean_arg}.mov",
        PROJECT_ROOT / clean_arg,
        PROJECT_ROOT / f"{clean_arg}.mp4",
        Path(video_arg),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find video file for '{video_arg}'. Checked in: {DATA_DIR / clean_arg}(.mp4)"
    )


def calculate_angle_2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """
    Calculates the 2D angle (in degrees, 0-180) at vertex B formed by points A-B-C.
    Example: A = Hip, B = Knee, C = Ankle.
    """
    a_vec = np.array(a)
    b_vec = np.array(b)
    c_vec = np.array(c)

    ba = a_vec - b_vec
    bc = c_vec - b_vec

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultralytics YOLO-Pose benchmark & biomechanical angle test.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--cam",
        nargs="?",
        const=0,
        type=int,
        help="Use webcam as video source (optional camera index, default: 0).",
    )
    source_group.add_argument(
        "--video",
        type=str,
        help="Path to video inside data/ directory, e.g. '/bench-press/bench-press-01' or 'bench-press/bench-press-01.mp4'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="YOLO Pose model name (e.g., yolo11n-pose.pt, yolo11s-pose.pt, yolov8n-pose.pt, yolov8s-pose.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for detection (default: 0.5).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.5,
        help="Video playback speed multiplier (default: 1.5x).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional maximum number of processed frames (0 = entire video).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without opening an OpenCV window (useful for pure FPS benchmarking).",
    )
    args = parser.parse_args()

    # Determine video source (default to webcam 0 if neither --cam nor --video was provided)
    if args.video:
        try:
            video_path = resolve_video_path(args.video)
        except FileNotFoundError as err:
            print(f"[ERROR] {err}")
            return
        source: int | str = str(video_path)
        source_label = f"Video ({video_path.relative_to(PROJECT_ROOT)})"
    else:
        cam_index = args.cam if args.cam is not None else 0
        source = cam_index
        source_label = f"Webcam (index={cam_index})"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    model_path = MODELS_DIR / model_name

    print(f"[INFO] Loading Ultralytics YOLO-Pose model: {model_path.name}...")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_label}")
        return

    inference_times_ms: list[float] = []
    detected_frames = 0
    total_frames = 0

    print(f"[INFO] Source : {source_label} (playback speed: {args.speed}x)")
    print(f"[INFO] Model  : {model_name} (ultralytics={ultralytics.__version__})")
    print("[INFO] Press 'q' in the video window to stop and see benchmark summary.")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_frame_time_s = (1.0 / (video_fps * args.speed)) if args.video and args.speed > 0 else 0.0
    raw_frame_idx = 0

    while cap.isOpened():
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        raw_frame_idx += 1

        # For video files at >1.0x speed, skip intermediate frames proportionally so playback is genuinely faster
        if args.video and args.speed > 1.0:
            if int(raw_frame_idx / args.speed) == int((raw_frame_idx - 1) / args.speed):
                continue

        total_frames += 1
        if args.max_frames > 0 and total_frames > args.max_frames:
            total_frames -= 1
            break

        # Measure pure inference time
        t_start = time.perf_counter()
        results = model(frame, conf=args.conf, verbose=False)
        t_end = time.perf_counter()

        inference_ms = (t_end - t_start) * 1000.0
        inference_times_ms.append(inference_ms)
        current_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

        result = results[0]
        annotated_frame = frame

        # Check if at least one person's keypoints were detected
        if result.keypoints is not None and len(result.keypoints.xy) > 0 and result.keypoints.xy[0].shape[0] >= 17:
            detected_frames += 1

            # Take primary person (index 0)
            kpts_xy = result.keypoints.xy[0].cpu().numpy()  # shape: (17, 2)
            kpts_conf = (
                result.keypoints.conf[0].cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(17, dtype=np.float32)
            )

            # Extract Left Leg (Hip -> Knee -> Ankle)
            hip_2d = tuple(kpts_xy[KP_LEFT_HIP])
            knee_2d = tuple(kpts_xy[KP_LEFT_KNEE])
            ankle_2d = tuple(kpts_xy[KP_LEFT_ANKLE])

            # Extract Left Arm (Shoulder -> Elbow -> Wrist)
            shoulder_2d = tuple(kpts_xy[KP_LEFT_SHOULDER])
            elbow_2d = tuple(kpts_xy[KP_LEFT_ELBOW])
            wrist_2d = tuple(kpts_xy[KP_LEFT_WRIST])

            knee_angle_2d = calculate_angle_2d(hip_2d, knee_2d, ankle_2d)
            elbow_angle_2d = calculate_angle_2d(shoulder_2d, elbow_2d, wrist_2d)

            if not args.no_display:
                # Draw YOLO skeleton and bounding box
                annotated_frame = result.plot(boxes=True)

                # Overlay Knee 2D angle if keypoints are visible
                if (
                    kpts_conf[KP_LEFT_HIP] > 0.3
                    and kpts_conf[KP_LEFT_KNEE] > 0.3
                    and kpts_conf[KP_LEFT_ANKLE] > 0.3
                ):
                    cv2.putText(
                        annotated_frame,
                        f"Knee 2D: {int(knee_angle_2d)} deg",
                        (int(knee_2d[0]) + 10, int(knee_2d[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                # Overlay Elbow 2D angle if keypoints are visible
                if (
                    kpts_conf[KP_LEFT_SHOULDER] > 0.3
                    and kpts_conf[KP_LEFT_ELBOW] > 0.3
                    and kpts_conf[KP_LEFT_WRIST] > 0.3
                ):
                    cv2.putText(
                        annotated_frame,
                        f"Elbow 2D: {int(elbow_angle_2d)} deg",
                        (int(elbow_2d[0]) + 10, int(elbow_2d[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 0),
                        2,
                    )

        if not args.no_display:
            cv2.putText(
                annotated_frame,
                f"YOLO-Pose ({model_name}) [{args.speed}x] | Inference: {inference_ms:.1f} ms ({current_fps:.1f} FPS)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Ultralytics YOLO-Pose - Biomechanical Analysis Test", annotated_frame)

            elapsed_s = time.perf_counter() - loop_start
            wait_ms = max(1, int((target_frame_time_s - elapsed_s) * 1000)) if target_frame_time_s > 0 else 1
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if inference_times_ms:
        # Exclude the first warmup frame from average calculation if more than 5 frames
        valid_times = inference_times_ms[1:] if len(inference_times_ms) > 5 else inference_times_ms
        avg_ms = float(np.mean(valid_times))
        p95_ms = float(np.percentile(valid_times, 95))
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        detection_rate = (detected_frames / total_frames) * 100.0 if total_frames > 0 else 0.0

        print("\n" + "=" * 55)
        print("ULTRALYTICS YOLO-POSE BENCHMARK SUMMARY")
        print("=" * 55)
        print(f"Source             : {source_label}")
        print(f"Ultralytics version: {ultralytics.__version__}")
        print(f"Model variant      : {model_name}")
        print(f"Total frames       : {total_frames}")
        print(f"Pose detected in   : {detected_frames}/{total_frames} frames ({detection_rate:.1f}%)")
        print(f"Avg inference time : {avg_ms:.2f} ms (after 1st frame warmup)")
        print(f"95th percentile    : {p95_ms:.2f} ms")
        print(f"Average FPS        : {avg_fps:.2f} FPS")
        print(f"Keypoints provided : 17 COCO (2D screen coordinates + confidence)")
        print("=" * 55)


if __name__ == "__main__":
    main()
