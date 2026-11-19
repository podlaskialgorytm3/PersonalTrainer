"""
Test script for MediaPipe Pose Landmarker (Task #1: Python CV libraries analysis).
Compatible with modern MediaPipe Tasks API (mediapipe >= 0.10 / 1.x).

Features tested:
1. 2D & 3D anatomical skeleton detection (33 BlazePose landmarks).
2. Biomechanical joint angle calculation (e.g., knee and elbow angles).
3. Performance benchmark (FPS & inference latency in ms).

Usage:
    # Run with webcam:
    python research/001-python-cv-libraries-analysis/test_mediapipe.py --cam

    # Run with a video from data/ folder (with or without .mp4 / leading slash):
    python research/001-python-cv-libraries-analysis/test_mediapipe.py --video /bench-press/bench-press-01
"""

import argparse
from pathlib import Path
import time
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def ensure_model_downloaded(variant: str, models_dir: Path) -> Path:
    """Downloads the MediaPipe .task model bundle if not already present locally."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"pose_landmarker_{variant}.task"
    if not model_path.exists():
        url = MODEL_URLS[variant]
        print(f"[INFO] Downloading MediaPipe model '{variant}' from:\n       {url}")
        urllib.request.urlretrieve(url, model_path)
        print(f"[INFO] Saved model to: {model_path}")
    return model_path


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


def calculate_angle(a: tuple[float, ...], b: tuple[float, ...], c: tuple[float, ...]) -> float:
    """
    Calculates the angle (in degrees, 0-180) at vertex B formed by points A-B-C (supports 2D and 3D).
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


def draw_pose_skeleton(frame: np.ndarray, landmarks: list) -> None:
    """Draws the 33 pose landmarks and connections on the BGR frame using MediaPipe Tasks drawing_utils."""
    vision.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=vision.drawing_styles.get_default_pose_landmarks_style(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MediaPipe Pose Landmarker benchmark & biomechanical angle test.")
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
        default="full",
        choices=["lite", "full", "heavy"],
        help="MediaPipe Pose model variant: 'lite', 'full' (default), or 'heavy'.",
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

    model_path = ensure_model_downloaded(args.model, SCRIPT_DIR / "models")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_label}")
        return

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    inference_times_ms: list[float] = []
    detected_frames = 0
    total_frames = 0
    start_wall_time = time.perf_counter()
    last_timestamp_ms = -1

    print(f"[INFO] Source : {source_label}")
    print(f"[INFO] Model  : MediaPipe Pose Landmarker (variant='{args.model}', version={mp.__version__})")
    print("[INFO] Press 'q' in the video window to stop and see benchmark summary.")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            h, w, _ = frame.shape

            # Monotonically increasing timestamp in ms required by RunningMode.VIDEO
            timestamp_ms = int((time.perf_counter() - start_wall_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Measure pure inference time
            t_start = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            t_end = time.perf_counter()

            inference_ms = (t_end - t_start) * 1000.0
            inference_times_ms.append(inference_ms)
            current_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

            if result.pose_landmarks:
                detected_frames += 1
                landmarks = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

                # Left Leg keypoints (Hip -> Knee -> Ankle)
                l_hip = landmarks[vision.PoseLandmark.LEFT_HIP]
                l_knee = landmarks[vision.PoseLandmark.LEFT_KNEE]
                l_ankle = landmarks[vision.PoseLandmark.LEFT_ANKLE]

                # Left Arm keypoints (Shoulder -> Elbow -> Wrist)
                l_shoulder = landmarks[vision.PoseLandmark.LEFT_SHOULDER]
                l_elbow = landmarks[vision.PoseLandmark.LEFT_ELBOW]
                l_wrist = landmarks[vision.PoseLandmark.LEFT_WRIST]

                # 2D pixel coordinates
                hip_2d = (l_hip.x * w, l_hip.y * h)
                knee_2d = (l_knee.x * w, l_knee.y * h)
                ankle_2d = (l_ankle.x * w, l_ankle.y * h)

                shoulder_2d = (l_shoulder.x * w, l_shoulder.y * h)
                elbow_2d = (l_elbow.x * w, l_elbow.y * h)
                wrist_2d = (l_wrist.x * w, l_wrist.y * h)

                knee_angle_2d = calculate_angle(hip_2d, knee_2d, ankle_2d)
                elbow_angle_2d = calculate_angle(shoulder_2d, elbow_2d, wrist_2d)

                # 3D world angle in meters
                knee_angle_3d = 0.0
                elbow_angle_3d = 0.0
                if world_landmarks:
                    w_hip = world_landmarks[vision.PoseLandmark.LEFT_HIP]
                    w_knee = world_landmarks[vision.PoseLandmark.LEFT_KNEE]
                    w_ankle = world_landmarks[vision.PoseLandmark.LEFT_ANKLE]
                    knee_angle_3d = calculate_angle(
                        (w_hip.x, w_hip.y, w_hip.z),
                        (w_knee.x, w_knee.y, w_knee.z),
                        (w_ankle.x, w_ankle.y, w_ankle.z),
                    )

                    w_shoulder = world_landmarks[vision.PoseLandmark.LEFT_SHOULDER]
                    w_elbow = world_landmarks[vision.PoseLandmark.LEFT_ELBOW]
                    w_wrist = world_landmarks[vision.PoseLandmark.LEFT_WRIST]
                    elbow_angle_3d = calculate_angle(
                        (w_shoulder.x, w_shoulder.y, w_shoulder.z),
                        (w_elbow.x, w_elbow.y, w_elbow.z),
                        (w_wrist.x, w_wrist.y, w_wrist.z),
                    )

                if not args.no_display:
                    draw_pose_skeleton(frame, landmarks)

                    cv2.putText(
                        frame,
                        f"Knee 2D: {int(knee_angle_2d)} deg (3D: {int(knee_angle_3d)} deg)",
                        (int(knee_2d[0]) + 10, int(knee_2d[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Elbow 2D: {int(elbow_angle_2d)} deg (3D: {int(elbow_angle_3d)} deg)",
                        (int(elbow_2d[0]) + 10, int(elbow_2d[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 0),
                        2,
                    )

            if not args.no_display:
                cv2.putText(
                    frame,
                    f"MediaPipe ({args.model}) | Inference: {inference_ms:.1f} ms ({current_fps:.1f} FPS)",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("MediaPipe Pose Landmarker - Biomechanical Analysis Test", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()

    if inference_times_ms:
        avg_ms = float(np.mean(inference_times_ms))
        p95_ms = float(np.percentile(inference_times_ms, 95))
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        detection_rate = (detected_frames / total_frames) * 100.0 if total_frames > 0 else 0.0

        print("\n" + "=" * 55)
        print("MEDIAPIPE POSE BENCHMARK SUMMARY")
        print("=" * 55)
        print(f"Source             : {source_label}")
        print(f"MediaPipe version  : {mp.__version__} (Tasks Vision API)")
        print(f"Model variant      : {args.model} (lite / full / heavy)")
        print(f"Total frames       : {total_frames}")
        print(f"Pose detected in   : {detected_frames}/{total_frames} frames ({detection_rate:.1f}%)")
        print(f"Avg inference time : {avg_ms:.2f} ms")
        print(f"95th percentile    : {p95_ms:.2f} ms")
        print(f"Average FPS        : {avg_fps:.2f} FPS")
        print(f"Keypoints provided : 33 (2D screen + 3D world coordinates + visibility)")
        print("=" * 55)


if __name__ == "__main__":
    main()
