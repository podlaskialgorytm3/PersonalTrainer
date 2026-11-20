"""
Test script for Google MoveNet (Lightning & Thunder) (Task #1: Python CV libraries analysis).

Features tested:
1. 2D anatomical skeleton detection (17 COCO keypoints + confidence scores).
2. Biomechanical joint angle calculation (e.g., knee and elbow 2D angles).
3. Performance benchmark (FPS & inference latency in ms) using TFLite (ai-edge-litert / tensorflow).

Usage:
    # Run with webcam (default model: lightning):
    python research/001-python-cv-libraries-analysis/test_movenet.py --cam

    # Run with a video from data/ folder:
    python research/001-python-cv-libraries-analysis/test_movenet.py --video /bench-press/bench-press-01

    # Choose MoveNet variant ('lightning' = 192x192 fast, 'thunder' = 256x256 accurate):
    python research/001-python-cv-libraries-analysis/test_movenet.py --video /bench-press/bench-press-01 --model thunder
"""

import argparse
import io
from pathlib import Path
import tarfile
import time
import urllib.request
import cv2
import numpy as np

# Support lightweight ai-edge-litert, tflite-runtime, or full tensorflow
try:
    from ai_edge_litert.interpreter import Interpreter

    RUNTIME_NAME = "ai-edge-litert"
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter

        RUNTIME_NAME = "tflite-runtime"
    except ImportError:
        try:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
            RUNTIME_NAME = f"tensorflow ({tf.__version__})"
        except ImportError as exc:
            raise ImportError(
                "Missing TFLite runtime for MoveNet. Install lightweight runtime with:\n"
                "  pip install ai-edge-litert\n"
                "or full TensorFlow with:\n"
                "  pip install tensorflow"
            ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = SCRIPT_DIR / "models"

# Official TFLite float16 MoveNet SinglePose models from Kaggle Models (successor to TF Hub)
MOVENET_MODELS = {
    "lightning": {
        "urls": [
            "https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-lightning-tflite-float16/1/download",
            "https://raw.githubusercontent.com/arunponnusamy/movenet-tensorflow/main/movenet_lightning_fp16.tflite",
        ],
        "input_size": 192,
        "filename": "movenet_singlepose_lightning_f16.tflite",
    },
    "thunder": {
        "urls": [
            "https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-thunder/4/download",
            "https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-thunder-tflite-float16/1/download",
            "https://www.kaggle.com/api/v1/models/google/movenet/tfLite/singlepose-thunder/1/download",
        ],
        "input_size": 256,
        "filename": "movenet_singlepose_thunder_f16.tflite",
    },
}

# MoveNet 17 COCO keypoint indices:
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
KP_LEFT_SHOULDER = 5
KP_LEFT_ELBOW = 7
KP_LEFT_WRIST = 9
KP_LEFT_HIP = 11
KP_LEFT_KNEE = 13
KP_LEFT_ANKLE = 15

SKELETON_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def ensure_model_downloaded(variant: str, models_dir: Path) -> tuple[Path, int]:
    """Downloads the MoveNet TFLite model (from Kaggle Models API or GitHub mirror) if not present locally."""
    models_dir.mkdir(parents=True, exist_ok=True)
    cfg = MOVENET_MODELS[variant]
    model_path = models_dir / cfg["filename"]
    if model_path.exists() and model_path.stat().st_size > 1000:
        return model_path, cfg["input_size"]

    last_err = None
    for url in cfg["urls"]:
        try:
            print(f"[INFO] Downloading MoveNet '{variant}' TFLite model from:\n       {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()

            # Kaggle API wraps the .tflite file inside a .tar.gz archive
            if data[:2] == b"\x1f\x8b":
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(".tflite"):
                            extracted = tar.extractfile(member)
                            if extracted is not None:
                                model_path.write_bytes(extracted.read())
                                break
            else:
                model_path.write_bytes(data)

            if model_path.exists() and model_path.stat().st_size > 1000:
                print(f"[INFO] Saved model to: {model_path}")
                return model_path, cfg["input_size"]
        except Exception as err:  # noqa: BLE001
            print(f"[WARN] Failed from {url}: {err}")
            last_err = err

    raise RuntimeError(f"Could not download MoveNet model '{variant}': {last_err}")


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
    """Calculates the 2D angle (in degrees, 0-180) at vertex B formed by points A-B-C."""
    a_vec = np.array(a)
    b_vec = np.array(b)
    c_vec = np.array(c)

    ba = a_vec - b_vec
    bc = c_vec - b_vec

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def preprocess_with_padding(frame: np.ndarray, target_size: int) -> tuple[np.ndarray, float, int, int]:
    """
    Resizes image with aspect-ratio-preserving padding to (target_size, target_size)
    so keypoints are not distorted on wide/tall videos.
    Returns: (input_tensor, scale, pad_top, pad_left)
    """
    h, w, _ = frame.shape
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(frame, (new_w, new_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded = cv2.copyMakeBorder(rgb, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    input_tensor = np.expand_dims(padded, axis=0).astype(np.uint8)
    return input_tensor, scale, pad_top, pad_left


def draw_movenet_skeleton(
    frame: np.ndarray,
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    conf_threshold: float,
) -> None:
    """Draws MoveNet keypoints and skeleton connections onto the BGR frame."""
    for i, j in SKELETON_EDGES:
        if kpts_conf[i] >= conf_threshold and kpts_conf[j] >= conf_threshold:
            pt1 = (int(kpts_xy[i, 0]), int(kpts_xy[i, 1]))
            pt2 = (int(kpts_xy[j, 0]), int(kpts_xy[j, 1]))
            cv2.line(frame, pt1, pt2, (0, 255, 128), 2)

    for idx in range(17):
        if kpts_conf[idx] >= conf_threshold:
            pt = (int(kpts_xy[idx, 0]), int(kpts_xy[idx, 1]))
            cv2.circle(frame, pt, 5, (0, 128, 255), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Google MoveNet benchmark & biomechanical angle test.")
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
        default="lightning",
        choices=["lightning", "thunder"],
        help="MoveNet model variant: 'lightning' (192x192, ultra-fast) or 'thunder' (256x256, higher accuracy).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="Minimum keypoint confidence threshold (default: 0.3).",
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

    model_path, input_size = ensure_model_downloaded(args.model, MODELS_DIR)

    interpreter = Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_label}")
        return

    inference_times_ms: list[float] = []
    detected_frames = 0
    total_frames = 0

    print(f"[INFO] Source  : {source_label} (playback speed: {args.speed}x)")
    print(f"[INFO] Model   : MoveNet SinglePose {args.model.capitalize()} ({input_size}x{input_size})")
    print(f"[INFO] Runtime : {RUNTIME_NAME}")
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

        input_tensor, scale, pad_top, pad_left = preprocess_with_padding(frame, input_size)
        if input_dtype != np.uint8:
            input_tensor = input_tensor.astype(input_dtype)

        # Measure pure inference time
        t_start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])  # shape: (1, 1, 17, 3) -> [y, x, score]
        t_end = time.perf_counter()

        inference_ms = (t_end - t_start) * 1000.0
        inference_times_ms.append(inference_ms)
        current_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

        kpts = keypoints_with_scores[0, 0]  # shape: (17, 3) where columns are [y_norm, x_norm, conf]
        kpts_conf = kpts[:, 2]

        # Convert normalized padded coordinates back to original frame pixel coordinates
        kpts_x = (kpts[:, 1] * input_size - pad_left) / scale
        kpts_y = (kpts[:, 0] * input_size - pad_top) / scale
        kpts_xy = np.stack([kpts_x, kpts_y], axis=-1)

        # Consider pose detected if at least 5 major keypoints exceed confidence threshold
        if int(np.sum(kpts_conf >= args.conf)) >= 5:
            detected_frames += 1

            hip_2d = tuple(kpts_xy[KP_LEFT_HIP])
            knee_2d = tuple(kpts_xy[KP_LEFT_KNEE])
            ankle_2d = tuple(kpts_xy[KP_LEFT_ANKLE])

            shoulder_2d = tuple(kpts_xy[KP_LEFT_SHOULDER])
            elbow_2d = tuple(kpts_xy[KP_LEFT_ELBOW])
            wrist_2d = tuple(kpts_xy[KP_LEFT_WRIST])

            knee_angle_2d = calculate_angle_2d(hip_2d, knee_2d, ankle_2d)
            elbow_angle_2d = calculate_angle_2d(shoulder_2d, elbow_2d, wrist_2d)

            if not args.no_display:
                draw_movenet_skeleton(frame, kpts_xy, kpts_conf, args.conf)

                if (
                    kpts_conf[KP_LEFT_HIP] >= args.conf
                    and kpts_conf[KP_LEFT_KNEE] >= args.conf
                    and kpts_conf[KP_LEFT_ANKLE] >= args.conf
                ):
                    cv2.putText(
                        frame,
                        f"Knee 2D: {int(knee_angle_2d)} deg",
                        (int(knee_2d[0]) + 10, int(knee_2d[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                if (
                    kpts_conf[KP_LEFT_SHOULDER] >= args.conf
                    and kpts_conf[KP_LEFT_ELBOW] >= args.conf
                    and kpts_conf[KP_LEFT_WRIST] >= args.conf
                ):
                    cv2.putText(
                        frame,
                        f"Elbow 2D: {int(elbow_angle_2d)} deg",
                        (int(elbow_2d[0]) + 10, int(elbow_2d[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 200, 0),
                        2,
                    )

        if not args.no_display:
            cv2.putText(
                frame,
                f"MoveNet ({args.model}) [{args.speed}x] | Inference: {inference_ms:.1f} ms ({current_fps:.1f} FPS)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Google MoveNet - Biomechanical Analysis Test", frame)

            elapsed_s = time.perf_counter() - loop_start
            wait_ms = max(1, int((target_frame_time_s - elapsed_s) * 1000)) if target_frame_time_s > 0 else 1
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    if inference_times_ms:
        valid_times = inference_times_ms[1:] if len(inference_times_ms) > 5 else inference_times_ms
        avg_ms = float(np.mean(valid_times))
        p95_ms = float(np.percentile(valid_times, 95))
        avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        detection_rate = (detected_frames / total_frames) * 100.0 if total_frames > 0 else 0.0

        print("\n" + "=" * 55)
        print("GOOGLE MOVENET BENCHMARK SUMMARY")
        print("=" * 55)
        print(f"Source             : {source_label}")
        print(f"TFLite Runtime     : {RUNTIME_NAME}")
        print(f"Model variant      : MoveNet SinglePose {args.model.capitalize()} ({input_size}x{input_size})")
        print(f"Total frames       : {total_frames}")
        print(f"Pose detected in   : {detected_frames}/{total_frames} frames ({detection_rate:.1f}%)")
        print(f"Avg inference time : {avg_ms:.2f} ms (after 1st frame warmup)")
        print(f"95th percentile    : {p95_ms:.2f} ms")
        print(f"Average FPS        : {avg_fps:.2f} FPS")
        print(f"Keypoints provided : 17 COCO (2D screen coordinates + confidence)")
        print("=" * 55)


if __name__ == "__main__":
    main()
