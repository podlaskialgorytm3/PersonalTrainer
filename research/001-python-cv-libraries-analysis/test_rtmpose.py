"""
Test script for OpenMMLab MMPose / RTMPose via rtmlib (Task #1: Python CV libraries analysis).

Why rtmlib + ONNXRuntime?
Full MMPose (`mmpose` + `mmcv` + `mmengine`) requires complex C++/CUDA compilation and strict PyTorch version pinning.
`rtmlib` is the official lightweight inference wrapper by the RTMPose authors that runs official MMPose RTMPose
models (RTMDet + RTMPose-t/s/m/l/x) directly on ONNXRuntime (CPU or GPU) with zero heavy dependencies.

Features tested:
1. 2D anatomical skeleton detection (17 COCO keypoints + SimCC confidence scores).
2. Biomechanical joint angle calculation (e.g., knee and elbow 2D angles).
3. Performance benchmark (FPS & inference latency in ms).

Usage:
    # Run with webcam (default mode: balanced):
    python research/001-python-cv-libraries-analysis/test_rtmpose.py --cam

    # Run with a video from data/ folder:
    python research/001-python-cv-libraries-analysis/test_rtmpose.py --video /bench-press/bench-press-01

    # Choose RTMPose mode ('lightweight' = RTMPose-s, 'balanced' = RTMPose-m, 'performance' = RTMPose-x):
    python research/001-python-cv-libraries-analysis/test_rtmpose.py --video /bench-press/bench-press-01 --mode lightweight
"""

import argparse
from pathlib import Path
import time
import cv2
import numpy as np

try:
    from rtmlib import Body, draw_skeleton
except ImportError as exc:
    raise ImportError(
        "Missing 'rtmlib' and 'onnxruntime' for RTMPose. Install them with:\n"
        "  pip install rtmlib onnxruntime"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# COCO 17 Keypoint indices used by MMPose / RTMPose Body:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="MMPose / RTMPose (via rtmlib) benchmark & biomechanical angle test.")
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
        "--mode",
        type=str,
        default="balanced",
        choices=["lightweight", "balanced", "performance"],
        help="RTMPose preset mode: 'lightweight' (RTMPose-s), 'balanced' (RTMPose-m, default), or 'performance' (RTMPose-x).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="onnxruntime",
        choices=["onnxruntime", "opencv"],
        help="Inference backend for RTMPose (default: onnxruntime).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Target device: 'cpu' (default) or 'cuda' (requires onnxruntime-gpu).",
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

    print(f"[INFO] Initializing MMPose RTMPose Body pipeline (mode='{args.mode}', backend='{args.backend}', device='{args.device}')...")
    body_estimator = Body(
        to_openpose=False,
        mode=args.mode,
        backend=args.backend,
        device=args.device,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_label}")
        return

    inference_times_ms: list[float] = []
    detected_frames = 0
    total_frames = 0

    print(f"[INFO] Source : {source_label} (playback speed: {args.speed}x)")
    print(f"[INFO] Model  : RTMPose ({args.mode}) via rtmlib/{args.backend}")
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

        # Measure pure inference time (RTMDet person detection + RTMPose keypoint estimation)
        t_start = time.perf_counter()
        keypoints, scores = body_estimator(frame)
        t_end = time.perf_counter()

        inference_ms = (t_end - t_start) * 1000.0
        inference_times_ms.append(inference_ms)
        current_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

        # keypoints shape: (N_persons, 17, 2), scores shape: (N_persons, 17)
        if keypoints is not None and len(keypoints) > 0 and keypoints[0].shape[0] >= 17:
            kpts_xy = keypoints[0]
            kpts_conf = scores[0]

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
                    frame = draw_skeleton(
                        frame,
                        keypoints,
                        scores,
                        openpose_skeleton=False,
                        kpt_thr=args.conf,
                    )

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
                f"RTMPose ({args.mode}) [{args.speed}x] | Inference: {inference_ms:.1f} ms ({current_fps:.1f} FPS)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.imshow("MMPose / RTMPose - Biomechanical Analysis Test", frame)

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
        print("MMPOSE / RTMPOSE BENCHMARK SUMMARY")
        print("=" * 55)
        print(f"Source             : {source_label}")
        print(f"Backend / Device   : {args.backend} ({args.device})")
        print(f"Model mode         : RTMPose '{args.mode}' (lightweight / balanced / performance)")
        print(f"Total frames       : {total_frames}")
        print(f"Pose detected in   : {detected_frames}/{total_frames} frames ({detection_rate:.1f}%)")
        print(f"Avg inference time : {avg_ms:.2f} ms (after 1st frame warmup)")
        print(f"95th percentile    : {p95_ms:.2f} ms")
        print(f"Average FPS        : {avg_fps:.2f} FPS")
        print(f"Keypoints provided : 17 COCO (2D screen coordinates + SimCC confidence)")
        print("=" * 55)


if __name__ == "__main__":
    main()
