"""
Shared tools for comparing Pose Estimation libraries on recordings of the selected exercises
(Issue #4: "Compare libraries proposed in python cv libraries ADR based on recordings provided").

The module:
1. Defines the reference recordings from docs/exercises/*.md (link + the segment showing the exercise).
2. Runs 8 model variants (the same as in docs/adr/001-.../benchmark_results.md) through a common
   interface. Each model returns 17 COCO keypoints (pixels) + a confidence score for every frame.
3. Computes joint angles, signal-quality metrics and simple rep counting
   based on the thresholds described in docs/exercises/*.md.

All models run on CPU (same as in the previous benchmark - the target is a mobile device).

Usage (download the recordings + extract into the results/exercise_benchmark/ cache):
    python research/001-python-cv-libraries-analysis/pose_compare.py --download
    python research/001-python-cv-libraries-analysis/pose_compare.py
    python research/001-python-cv-libraries-analysis/pose_compare.py --models mediapipe_full yolo11n --videos push_up_side
"""

from __future__ import annotations

import argparse
import io
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results" / "exercise_benchmark"

TARGET_FPS = 30.0          # 60 FPS recordings are processed every 2nd frame -> comparable metrics
WARMUP_FRAMES = 5          # first frames (model initialisation) are excluded from latency stats
MIN_KP_CONF = 0.3          # minimum keypoint confidence for the point to be used in an angle


# ---------------------------------------------------------------------------
# Keypoints (common COCO-17 format)
# ---------------------------------------------------------------------------

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Indices of the 33 BlazePose (MediaPipe) landmarks that correspond to the 17 COCO keypoints
MEDIAPIPE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_ear"), ("nose", "right_ear"),
]

# Angle at vertex B for the triple A-B-C
ANGLE_DEFS = {
    "elbow": ("shoulder", "elbow", "wrist"),
    "shoulder": ("hip", "shoulder", "elbow"),
    "hip": ("shoulder", "hip", "knee"),
    "knee": ("hip", "knee", "ankle"),
}


# ---------------------------------------------------------------------------
# Reference recordings (links from docs/exercises/*.md)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoSpec:
    key: str
    exercise: str                              # push_up | pull_up | ohp
    url: str
    segments: tuple[tuple[float, float], ...]  # segments (start_s, end_s) showing the exercise from the target camera angle
    view: str                                  # camera angle description
    side: str                                  # "auto" (side closer to the camera) | "both" (mean of both sides)
    true_reps: int                             # counted by hand (frame by frame) on these segments
    note: str = ""

    @property
    def path(self) -> Path:
        return DATA_DIR / self.exercise / f"{self.key}.mp4"

    @property
    def label(self) -> str:
        return f"{EXERCISE_LABELS[self.exercise]} – {self.view}"


EXERCISE_LABELS = {"push_up": "Pompki", "pull_up": "Podciąganie", "ohp": "OHP"}

# Segments and rep counts were set by hand from frames sampled every 0.25 s.
# One rep = a full cycle from docs/exercises/*.md (start -> full range of motion -> back to start).
VIDEOS: list[VideoSpec] = [
    VideoSpec(
        key="push_up_side",
        exercise="push_up",
        url="https://www.youtube.com/watch?v=hAhCweMfrF8",
        segments=((126.4, 135.7),),
        view="bok 90°",
        side="auto",
        true_reps=2,
        note="Video 2 z push_up.md (2:06–2:16, wolne powtórzenia, idealny profil)",
    ),
    VideoSpec(
        key="push_up_angles",
        exercise="push_up",
        url="https://www.youtube.com/watch?v=tWjBnQX3if0",
        segments=((185.5, 204.3),),
        view="ukos ~45°",
        side="auto",
        true_reps=8,
        note="Video 1 z push_up.md (3:05–3:24, ujęcie skośne bok/przód)",
    ),
    VideoSpec(
        key="pull_up_rear_diagonal",
        exercise="pull_up",
        url="https://www.youtube.com/watch?v=zuf5qRaFr-8",
        segments=((5.25, 21.5),),
        view="tył / lekki ukos",
        side="both",
        true_reps=6,
        note="Video 1 z pull_up.md (0:05–0:21, kamera blisko, nogi częściowo poza kadrem)",
    ),
    VideoSpec(
        key="pull_up_front",
        exercise="pull_up",
        url="https://www.youtube.com/watch?v=2SkV-3ckG0s",
        segments=((1.0, 15.6),),
        view="przód, z daleka",
        side="both",
        true_reps=11,
        note="Video 2 z pull_up.md (0:01–0:15, szybkie powtórzenia, osoba ~40% wysokości kadru)",
    ),
    VideoSpec(
        key="ohp_thrall",
        exercise="ohp",
        url="https://www.youtube.com/watch?v=wJlKUo2-P4o",
        segments=((120.2, 125.6), (129.8, 134.7)),
        view="ukos od przodu ~30°",
        side="both",
        true_reps=1,
        note="Video 1 z over_head_press.md (2:00–2:05 i 2:10–2:15, dwa ujęcia przedzielone zbliżeniem, klatka (rack) w tle)",
    ),
    VideoSpec(
        key="ohp_nasm",
        exercise="ohp",
        url="https://www.youtube.com/watch?v=cGnhixvC8uA",
        segments=((0.0, 11.4),),
        view="ukos od przodu 45°",
        side="both",
        true_reps=3,
        note="Video 2 z over_head_press.md (cały film, studyjne nagranie na sztucznym tle)",
    ),
    VideoSpec(
        key="ohp_scott_herman",
        exercise="ohp",
        url="https://www.youtube.com/watch?v=2yjwXTZQDDI",
        segments=((117.0, 143.9),),
        view="przód, bez nóg",
        side="both",
        true_reps=4,
        note="Video 3 z over_head_press.md (1:57–2:24, nogi poza kadrem, kamera lekko się rusza)",
    ),
]

VIDEOS_BY_KEY = {v.key: v for v in VIDEOS}


def download_videos(videos: list[VideoSpec] | None = None, max_height: int = 720) -> None:
    """Downloads the recordings from YouTube to data/<exercise>/<key>.mp4 (yt-dlp, H.264 codec preferred)."""
    import yt_dlp

    for video in videos or VIDEOS:
        if video.path.exists():
            print(f"[OK]   {video.path.relative_to(PROJECT_ROOT)} już istnieje")
            continue
        video.path.parent.mkdir(parents=True, exist_ok=True)
        opts = {
            "format": (
                f"bv*[height<={max_height}][vcodec^=avc1][ext=mp4]/"
                f"b[height<={max_height}][vcodec^=avc1][ext=mp4]/"
                f"bv*[height<={max_height}][ext=mp4]/b[ext=mp4]"
            ),
            "outtmpl": str(video.path),
            "noplaylist": True,
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
        }
        print(f"[GET]  {video.url} -> {video.path.relative_to(PROJECT_ROOT)}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video.url])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    library: str


MODELS: list[ModelSpec] = [
    ModelSpec("mediapipe_lite", "MediaPipe Lite", "MediaPipe"),
    ModelSpec("mediapipe_full", "MediaPipe Full", "MediaPipe"),
    ModelSpec("yolo11n", "YOLO11n-Pose", "YOLO-Pose"),
    ModelSpec("yolov8n", "YOLOv8n-Pose", "YOLO-Pose"),
    ModelSpec("movenet_lightning", "MoveNet Lightning", "MoveNet"),
    ModelSpec("movenet_thunder", "MoveNet Thunder", "MoveNet"),
    ModelSpec("rtmpose_lightweight", "RTMPose Lightweight", "RTMPose"),
    ModelSpec("rtmpose_balanced", "RTMPose Balanced", "RTMPose"),
]

MODELS_BY_KEY = {m.key: m for m in MODELS}
LIBRARIES = ["MediaPipe", "YOLO-Pose", "MoveNet", "RTMPose"]


@dataclass
class PoseResult:
    xy: np.ndarray                    # (17, 2) pixels
    conf: np.ndarray                  # (17,) confidence / visibility
    world: np.ndarray | None = None   # (17, 3) meters - MediaPipe only


def _download(urls: list[str], target: Path) -> Path:
    """Downloads a model file (handles .tar.gz archives from Kaggle). Returns the file path."""
    if target.exists() and target.stat().st_size > 1000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in urls:
        try:
            # Keep the default urllib User-Agent: with "Mozilla/5.0" (as in test_movenet.py) Kaggle/TF Hub returns 404
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if data[:2] == b"\x1f\x8b":
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                    member = next(m for m in tar.getmembers() if m.name.endswith((".tflite", ".task")))
                    data = tar.extractfile(member).read()
            if len(data) < 1000 or data[:15].lower().startswith((b"<!doctype", b"<html")):
                raise ValueError(f"serwer zwrócił {len(data)} B zamiast modelu")
            target.write_bytes(data)
            return target
        except Exception as err:  # noqa: BLE001
            last_err = err
            print(f"[WARN] Nie udało się pobrać {url}: {err}")
    raise RuntimeError(f"Nie udało się pobrać modelu {target.name}: {last_err}")


def _largest_person(boxes_xyxy: np.ndarray) -> int:
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    return int(np.argmax(areas))


class MediaPipeRunner:
    URLS = {
        "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    }

    def __init__(self, variant: str):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = _download([self.URLS[variant]], MODELS_DIR / f"pose_landmarker_{variant}.task")
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mp = mp
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple[PoseResult | None, float]:
        h, w = frame_bgr.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        t0 = time.perf_counter()
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        latency = (time.perf_counter() - t0) * 1000.0
        if not result.pose_landmarks:
            return None, latency
        lms = result.pose_landmarks[0]
        xy = np.array([[lms[i].x * w, lms[i].y * h] for i in MEDIAPIPE_TO_COCO], dtype=np.float32)
        conf = np.array([lms[i].visibility or 0.0 for i in MEDIAPIPE_TO_COCO], dtype=np.float32)
        world = None
        if result.pose_world_landmarks:
            wl = result.pose_world_landmarks[0]
            world = np.array([[wl[i].x, wl[i].y, wl[i].z] for i in MEDIAPIPE_TO_COCO], dtype=np.float32)
        return PoseResult(xy, conf, world), latency

    def close(self) -> None:
        self._landmarker.close()


class YoloRunner:
    def __init__(self, weights: str, conf: float = 0.5):
        from ultralytics import YOLO

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(MODELS_DIR / weights))
        self._conf = conf

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple[PoseResult | None, float]:
        t0 = time.perf_counter()
        result = self._model(frame_bgr, conf=self._conf, device="cpu", verbose=False)[0]
        latency = (time.perf_counter() - t0) * 1000.0
        if result.keypoints is None or len(result.keypoints.xy) == 0 or result.keypoints.xy.shape[1] < 17:
            return None, latency
        idx = _largest_person(result.boxes.xyxy.cpu().numpy())
        xy = result.keypoints.xy[idx].cpu().numpy().astype(np.float32)
        conf = result.keypoints.conf[idx].cpu().numpy().astype(np.float32)
        return PoseResult(xy, conf), latency

    def close(self) -> None:
        pass


class MoveNetRunner:
    # Official TF Hub links (they redirect to Kaggle). The Kaggle API links in test_movenet.py return 404, and the
    # "Mozilla/5.0" User-Agent header also blocks the TF Hub redirect - hence "ERROR" for Thunder in benchmark_results.md.
    MODELS = {
        "lightning": {
            "urls": [
                "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
                "https://raw.githubusercontent.com/arunponnusamy/movenet-tensorflow/main/movenet_lightning_fp16.tflite",
            ],
            "input_size": 192,
            "filename": "movenet_singlepose_lightning_f16.tflite",
        },
        "thunder": {
            "urls": [
                "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
            ],
            "input_size": 256,
            "filename": "movenet_singlepose_thunder_f16.tflite",
        },
    }

    def __init__(self, variant: str, min_conf: float = MIN_KP_CONF):
        from ai_edge_litert.interpreter import Interpreter

        cfg = self.MODELS[variant]
        model_path = _download(cfg["urls"], MODELS_DIR / cfg["filename"])
        self._size = cfg["input_size"]
        self._interpreter = Interpreter(model_path=str(model_path), num_threads=4)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self._min_conf = min_conf

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple[PoseResult | None, float]:
        h, w = frame_bgr.shape[:2]
        size = self._size
        scale = min(size / h, size / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (new_w, new_h)), cv2.COLOR_BGR2RGB)
        pad_top, pad_left = (size - new_h) // 2, (size - new_w) // 2
        padded = cv2.copyMakeBorder(
            rgb, pad_top, size - new_h - pad_top, pad_left, size - new_w - pad_left,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        tensor = np.expand_dims(padded, axis=0).astype(self._input["dtype"])

        t0 = time.perf_counter()
        self._interpreter.set_tensor(self._input["index"], tensor)
        self._interpreter.invoke()
        out = self._interpreter.get_tensor(self._output["index"])[0, 0]  # (17, 3): y, x, score
        latency = (time.perf_counter() - t0) * 1000.0

        conf = out[:, 2].astype(np.float32)
        # MoveNet always returns something - treat the person as detected when >= 5 keypoints have confidence >= threshold
        if int(np.sum(conf >= self._min_conf)) < 5:
            return None, latency
        x = (out[:, 1] * size - pad_left) / scale
        y = (out[:, 0] * size - pad_top) / scale
        return PoseResult(np.stack([x, y], axis=-1).astype(np.float32), conf), latency

    def close(self) -> None:
        pass


class RtmPoseRunner:
    def __init__(self, mode: str, min_conf: float = MIN_KP_CONF):
        from rtmlib import Body

        self._body = Body(to_openpose=False, mode=mode, backend="onnxruntime", device="cpu")
        self._min_conf = min_conf

    def infer(self, frame_bgr: np.ndarray, timestamp_ms: int) -> tuple[PoseResult | None, float]:
        t0 = time.perf_counter()
        keypoints, scores = self._body(frame_bgr)
        latency = (time.perf_counter() - t0) * 1000.0
        if keypoints is None or len(keypoints) == 0:
            return None, latency
        # Pick the largest person (box built from keypoints with confidence >= threshold)
        boxes = []
        for kp, sc in zip(keypoints, scores):
            good = kp[sc >= self._min_conf] if np.any(sc >= self._min_conf) else kp
            boxes.append([good[:, 0].min(), good[:, 1].min(), good[:, 0].max(), good[:, 1].max()])
        idx = _largest_person(np.array(boxes))
        conf = np.asarray(scores[idx], dtype=np.float32)
        if int(np.sum(conf >= self._min_conf)) < 5:
            return None, latency
        return PoseResult(np.asarray(keypoints[idx], dtype=np.float32), conf), latency

    def close(self) -> None:
        pass


def create_runner(model_key: str):
    factories = {
        "mediapipe_lite": lambda: MediaPipeRunner("lite"),
        "mediapipe_full": lambda: MediaPipeRunner("full"),
        "yolo11n": lambda: YoloRunner("yolo11n-pose.pt"),
        "yolov8n": lambda: YoloRunner("yolov8n-pose.pt"),
        "movenet_lightning": lambda: MoveNetRunner("lightning"),
        "movenet_thunder": lambda: MoveNetRunner("thunder"),
        "rtmpose_lightweight": lambda: RtmPoseRunner("lightweight"),
        "rtmpose_balanced": lambda: RtmPoseRunner("balanced"),
    }
    return factories[model_key]()


# ---------------------------------------------------------------------------
# Extraction (with cache)
# ---------------------------------------------------------------------------

def iter_segment_frames(video: VideoSpec, target_fps: float = TARGET_FPS):
    """
    Yields (segment_no, frame_index, seconds_since_first_segment_start, bgr_frame).
    Recordings above 30 FPS are sampled every n-th frame so the metrics stay comparable.
    """
    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Nie można otworzyć {video.path} (uruchom: pose_compare.py --download)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps / target_fps)))
    bounds = [(int(round(s * fps)), int(round(e * fps))) for s, e in video.segments]
    t0 = bounds[0][0]
    idx = 0
    try:
        for seg_no, (start, end) in enumerate(bounds):
            while idx < end:
                if not cap.grab():
                    return
                if idx >= start and (idx - start) % stride == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        yield seg_no, idx, (idx - t0) / fps, frame
                idx += 1
    finally:
        cap.release()


def result_path(video_key: str, model_key: str) -> Path:
    return RESULTS_DIR / f"{video_key}__{model_key}.csv.gz"


def extract(video: VideoSpec, model_key: str, overwrite: bool = False) -> pd.DataFrame:
    """Runs the model on the recording segments; the result (keypoints + per-frame inference time) is cached."""
    out = result_path(video.key, model_key)
    if out.exists() and not overwrite:
        return pd.read_csv(out)

    runner = create_runner(model_key)
    rows = []
    try:
        for seg_no, frame_idx, t_s, frame in iter_segment_frames(video):
            pose, latency = runner.infer(frame, int(round(t_s * 1000)))
            row = {"segment": seg_no, "frame": frame_idx, "t": t_s, "latency_ms": latency, "detected": pose is not None}
            if pose is not None:
                for i, name in enumerate(COCO_KEYPOINTS):
                    row[f"{name}_x"] = pose.xy[i, 0]
                    row[f"{name}_y"] = pose.xy[i, 1]
                    row[f"{name}_c"] = pose.conf[i]
                    if pose.world is not None:
                        row[f"{name}_wx"], row[f"{name}_wy"], row[f"{name}_wz"] = pose.world[i]
            rows.append(row)
    finally:
        runner.close()

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    return df


def load_results(videos: list[VideoSpec] | None = None, models: list[ModelSpec] | None = None) -> dict[tuple[str, str], pd.DataFrame]:
    """Loads cached results for (recording, model) pairs. Missing pairs are skipped."""
    results = {}
    for video in videos or VIDEOS:
        for model in models or MODELS:
            path = result_path(video.key, model.key)
            if path.exists():
                results[(video.key, model.key)] = pd.read_csv(path)
    return results


# ---------------------------------------------------------------------------
# Angles and metrics
# ---------------------------------------------------------------------------

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle (0-180°) at vertex B for point arrays of shape (N, D)."""
    ba, bc = a - b, c - b
    cos = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _cols(df: pd.DataFrame, name: str, suffixes: tuple[str, ...]) -> np.ndarray:
    cols = [f"{name}_{s}" for s in suffixes]
    if not all(c in df.columns for c in cols):
        return np.full((len(df), len(suffixes)), np.nan)
    return df[cols].to_numpy(dtype=float)


def joint_angle(df: pd.DataFrame, joint: str, side: str, world: bool = False, min_conf: float = MIN_KP_CONF) -> pd.Series:
    """
    Joint angle for every frame (NaN when any of the 3 keypoints has confidence < min_conf).
    world=True -> 3D angle from world coordinates in meters (MediaPipe only).
    """
    suffixes = ("wx", "wy", "wz") if world else ("x", "y")
    names = [f"{side}_{p}" for p in ANGLE_DEFS[joint]]
    pts = [_cols(df, n, suffixes) for n in names]
    angle = _angle(*pts)
    confs = np.stack([_cols(df, n, ("c",))[:, 0] for n in names], axis=1)
    angle[~(np.nan_to_num(confs, nan=0.0) >= min_conf).all(axis=1)] = np.nan
    return pd.Series(angle, index=df.index)


def pick_side(dfs: list[pd.DataFrame], joint: str = "elbow") -> str:
    """Body side closer to the camera = higher mean confidence of the joint's keypoints across all models."""
    scores = {}
    for side in ("left", "right"):
        cols = [f"{side}_{p}_c" for p in ANGLE_DEFS[joint]]
        vals = [df[cols].to_numpy(dtype=float) for df in dfs if all(c in df.columns for c in cols)]
        scores[side] = float(np.nanmean(np.concatenate(vals))) if vals else 0.0
    return max(scores, key=scores.get)


def exercise_angle(df: pd.DataFrame, video: VideoSpec, side: str, joint: str = "elbow", world: bool = False) -> pd.Series:
    """
    Angle used in the exercise analysis:
    - side='left'/'right' -> that side,
    - side='both' -> mean of left and right (if one side is occluded - the visible one).
    """
    if side != "both":
        return joint_angle(df, joint, side, world=world)
    left = joint_angle(df, joint, "left", world=world)
    right = joint_angle(df, joint, "right", world=world)
    return pd.concat([left, right], axis=1).mean(axis=1, skipna=True)


def jitter(angle: pd.Series) -> float:
    """
    Signal noise: median |θ[t+1] − 2θ[t] + θ[t−1]| (second difference, °/frame²) over consecutive frames.
    Real motion at 30 FPS is smooth, so large values = jumping keypoints.
    """
    a = angle.to_numpy(dtype=float)
    second = a[2:] - 2 * a[1:-1] + a[:-2]
    second = second[~np.isnan(second)]
    return float(np.median(np.abs(second))) if len(second) else np.nan


def latency_stats(df: pd.DataFrame) -> dict[str, float]:
    lat = df["latency_ms"].iloc[WARMUP_FRAMES:]
    return {
        "latency_median_ms": float(lat.median()),
        "latency_p95_ms": float(lat.quantile(0.95)),
        "fps": float(1000.0 / lat.mean()),
    }


# ---------------------------------------------------------------------------
# Rep counting based on the thresholds from docs/exercises/*.md
# ---------------------------------------------------------------------------

REP_RULES = {
    # push-up: start/end = elbow extended >= 165°, trigger = elbow flexed <= 90° (push_up.md)
    "push_up": {"start": "elbow >= 165°", "trigger": "elbow <= 90°"},
    # pull-up: start/end = dead hang, elbows >= 160°; trigger = head above the wrist line (pull_up.md)
    "pull_up": {"start": "elbow >= 160°", "trigger": "głowa nad linią nadgarstków"},
    # OHP: start/end = rack position, elbows <= 80°; trigger = elbows >= 165° and wrists above the head (over_head_press.md)
    "ohp": {"start": "elbow <= 80°", "trigger": "elbow >= 165° i nadgarstki nad głową"},
}


def _head_y(df: pd.DataFrame) -> np.ndarray:
    """Highest (smallest y) of the head keypoints: nose, ears - with confidence >= threshold."""
    ys = []
    for name in ("nose", "left_ear", "right_ear"):
        y = _cols(df, name, ("y",))[:, 0].copy()
        c = np.nan_to_num(_cols(df, name, ("c",))[:, 0], nan=0.0)
        y[c < MIN_KP_CONF] = np.nan
        ys.append(y)
    stacked = np.stack(ys, axis=1)
    out = np.full(len(df), np.nan)
    has = ~np.isnan(stacked).all(axis=1)
    out[has] = np.nanmin(stacked[has], axis=1)
    return out


def _wrist_line_y(df: pd.DataFrame) -> np.ndarray:
    ys = []
    for name in ("left_wrist", "right_wrist"):
        y = _cols(df, name, ("y",))[:, 0].copy()
        c = np.nan_to_num(_cols(df, name, ("c",))[:, 0], nan=0.0)
        y[c < MIN_KP_CONF] = np.nan
        ys.append(y)
    stacked = np.stack(ys, axis=1)
    out = np.full(len(df), np.nan)
    has = ~np.isnan(stacked).all(axis=1)
    out[has] = np.nanmean(stacked[has], axis=1)
    return out


def rep_signals(
    df: pd.DataFrame, video: VideoSpec, elbow: pd.Series, extension_offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (is_start, is_trigger) for every frame. NaN -> False (the state machine does not change).
    extension_offset lowers the elbow EXTENSION thresholds (165°/160°) by the given degrees - for sensitivity analysis.
    """
    e = elbow.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        extended = lambda threshold: e >= threshold - extension_offset  # noqa: E731
        if video.exercise == "push_up":
            return extended(165), e <= 90
        head, wrists = _head_y(df), _wrist_line_y(df)
        if video.exercise == "pull_up":
            return extended(160), head < wrists
        if video.exercise == "ohp":
            return e <= 80, extended(165) & (wrists < head)
    raise ValueError(video.exercise)


def count_reps(is_start: np.ndarray, is_trigger: np.ndarray, segment: np.ndarray | None = None) -> tuple[int, list[int]]:
    """
    State machine: start position -> trigger (full range of motion) -> back to start position = 1 rep.
    The state is reset at segment boundaries (editing cuts).
    Returns (rep count, row indices where a rep was counted).
    """
    state, reps, at = "wait_start", 0, []
    prev_seg = None
    for i, (s, t) in enumerate(zip(is_start, is_trigger)):
        if segment is not None and segment[i] != prev_seg:
            state, prev_seg = "wait_start", segment[i]
        if state == "wait_start" and s:
            state = "ready"
        elif state == "ready" and t:
            state = "triggered"
        elif state == "triggered" and s:
            reps += 1
            at.append(i)
            state = "ready"
    return reps, at


def count_reps_for(df: pd.DataFrame, video: VideoSpec, elbow: pd.Series, extension_offset: float = 0.0) -> tuple[int, list[int]]:
    is_start, is_trigger = rep_signals(df, video, elbow, extension_offset)
    return count_reps(is_start, is_trigger, df["segment"].to_numpy())


# ---------------------------------------------------------------------------
# Skeleton drawing on a frame
# ---------------------------------------------------------------------------

def read_frame(video: VideoSpec, frame_idx: int) -> np.ndarray:
    """Reads frames sequentially (seeking with CAP_PROP_POS_FRAMES in H.264 can be off by a few frames)."""
    cap = cv2.VideoCapture(str(video.path))
    for _ in range(frame_idx):
        cap.grab()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Nie można odczytać klatki {frame_idx} z {video.path}")
    return frame


def draw_skeleton(frame_bgr: np.ndarray, row: pd.Series, color_bgr: tuple[int, int, int], min_conf: float = MIN_KP_CONF) -> np.ndarray:
    """Draws the 17 COCO keypoints from a results row (skips low-confidence keypoints)."""
    out = frame_bgr.copy()
    if not bool(row.get("detected", False)):
        return out
    thick = max(2, int(round(out.shape[0] / 240)))

    def pt(name):
        c = row.get(f"{name}_c", np.nan)
        if pd.isna(c) or c < min_conf:
            return None
        return int(round(row[f"{name}_x"])), int(round(row[f"{name}_y"]))

    for a, b in SKELETON_EDGES:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(out, pa, pb, color_bgr, thick, cv2.LINE_AA)
    for name in COCO_KEYPOINTS:
        p = pt(name)
        if p:
            cv2.circle(out, p, thick + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, p, thick + 1, color_bgr, -1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ekstrakcja punktów dla porównania bibliotek na nagraniach ćwiczeń.")
    parser.add_argument("--download", action="store_true", help="Tylko pobierz nagrania z YouTube do data/.")
    parser.add_argument("--videos", nargs="*", default=None, help=f"Klucze nagrań (domyślnie wszystkie): {list(VIDEOS_BY_KEY)}")
    parser.add_argument("--models", nargs="*", default=None, help=f"Klucze modeli (domyślnie wszystkie): {list(MODELS_BY_KEY)}")
    parser.add_argument("--overwrite", action="store_true", help="Przelicz ponownie, nawet jeśli wynik jest w cache.")
    args = parser.parse_args()

    videos = [VIDEOS_BY_KEY[k] for k in args.videos] if args.videos else VIDEOS
    if args.download:
        download_videos(videos)
        return

    models = [MODELS_BY_KEY[k] for k in args.models] if args.models else MODELS
    for video in videos:
        for model in models:
            t0 = time.perf_counter()
            try:
                df = extract(video, model.key, overwrite=args.overwrite)
            except Exception as err:  # noqa: BLE001
                print(f"[FAIL] {video.key:24s} {model.label:20s} {type(err).__name__}: {err}")
                continue
            print(
                f"[OK]   {video.key:24s} {model.label:20s} {len(df):4d} klatek, "
                f"wykryto {df['detected'].mean():6.1%}, {time.perf_counter() - t0:6.1f} s"
            )


if __name__ == "__main__":
    main()
