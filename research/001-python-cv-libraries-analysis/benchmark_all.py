"""
Unified benchmark runner for all 4 Python Pose Estimation libraries (Task #1: Python CV libraries analysis).

Runs each candidate library in headless mode (--no-display, --speed 1.0, --max-frames 150)
on the bench-press videos (/bench-press/bench-press-01 .. 04) and appends only the missing
tables to benchmark_results.md without overwriting existing results.

Usage:
    python research/001-python-cv-libraries-analysis/benchmark_all.py
    python research/001-python-cv-libraries-analysis/benchmark_all.py --videos /bench-press/bench-press-02 /bench-press/bench-press-03
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DEFAULT_VIDEOS = [
    "/bench-press/bench-press-01",
    "/bench-press/bench-press-02",
    "/bench-press/bench-press-03",
    "/bench-press/bench-press-04",
]

BENCHMARKS = [
    {
        "name": "MediaPipe Pose (Lite)",
        "script": "test_mediapipe.py",
        "extra_args": ["--model", "lite"],
        "keypoints": "33 (2D + 3D World)",
    },
    {
        "name": "MediaPipe Pose (Full)",
        "script": "test_mediapipe.py",
        "extra_args": ["--model", "full"],
        "keypoints": "33 (2D + 3D World)",
    },
    {
        "name": "YOLOv8n-Pose (Nano)",
        "script": "test_yolo_pose.py",
        "extra_args": ["--model", "yolov8n-pose.pt"],
        "keypoints": "17 COCO (2D)",
    },
    {
        "name": "YOLO11n-Pose (Nano)",
        "script": "test_yolo_pose.py",
        "extra_args": ["--model", "yolo11n-pose.pt"],
        "keypoints": "17 COCO (2D)",
    },
    {
        "name": "MoveNet (Lightning)",
        "script": "test_movenet.py",
        "extra_args": ["--model", "lightning"],
        "keypoints": "17 COCO (2D)",
    },
    {
        "name": "MoveNet (Thunder)",
        "script": "test_movenet.py",
        "extra_args": ["--model", "thunder"],
        "keypoints": "17 COCO (2D)",
    },
    {
        "name": "RTMPose (Lightweight)",
        "script": "test_rtmpose.py",
        "extra_args": ["--mode", "lightweight"],
        "keypoints": "17 COCO (2D)",
    },
    {
        "name": "RTMPose (Balanced)",
        "script": "test_rtmpose.py",
        "extra_args": ["--mode", "balanced"],
        "keypoints": "17 COCO (2D)",
    },
]


def parse_metric(pattern: str, text: str, default: str = "N/A") -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default


def run_benchmark_for_video(video_path: str, max_frames: int) -> str:
    """Runs all 8 model benchmarks on a single video and returns its Markdown table section."""
    results_table: list[dict[str, str]] = []

    print("\n" + "=" * 75)
    print(f"RUNNING POSE ESTIMATION BENCHMARK SUITE ON: {video_path}")
    print("=" * 75)

    for item in BENCHMARKS:
        script_path = SCRIPT_DIR / item["script"]
        cmd = [
            sys.executable,
            str(script_path),
            "--video",
            video_path,
            "--speed",
            "1.0",
            "--max-frames",
            str(max_frames),
            "--no-display",
            *item["extra_args"],
        ]

        print(f"\n>>> [{video_path}] Benchmarking: {item['name']} ...")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = proc.stdout + "\n" + proc.stderr

        if proc.returncode != 0:
            print(f"    [WARN] Failed (exit={proc.returncode}):")
            for line in output.strip().splitlines()[-5:]:
                print(f"      {line}")
            results_table.append(
                {
                    "name": item["name"],
                    "keypoints": item["keypoints"],
                    "detected": "ERROR",
                    "avg_ms": "ERROR",
                    "p95_ms": "ERROR",
                    "fps": "ERROR",
                }
            )
            continue

        detected = parse_metric(r"Pose detected in\s*:\s*([^\n]+)", output)
        avg_ms = parse_metric(r"Avg inference time\s*:\s*([0-9.]+\s*ms)", output)
        p95_ms = parse_metric(r"95th percentile\s*:\s*([0-9.]+\s*ms)", output)
        fps = parse_metric(r"Average FPS\s*:\s*([0-9.]+\s*FPS)", output)

        print(f"    Done -> Avg: {avg_ms} | P95: {p95_ms} | FPS: {fps} | Detected: {detected}")

        results_table.append(
            {
                "name": item["name"],
                "keypoints": item["keypoints"],
                "detected": detected,
                "avg_ms": avg_ms,
                "p95_ms": p95_ms,
                "fps": fps,
            }
        )

    md_lines = [
        f"### Quantitative Benchmark Results (`{video_path}`)",
        "",
        "| Model / Library | Keypoints (2D/3D) | Detection Rate | Avg Latency (ms) | P95 Latency (ms) | Avg FPS |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for row in results_table:
        md_lines.append(
            f"| **{row['name']}** | {row['keypoints']} | {row['detected']} | {row['avg_ms']} | {row['p95_ms']} | **{row['fps']}** |"
        )

    return "\n".join(md_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CV pose estimation benchmarks across bench-press videos.")
    parser.add_argument(
        "--videos",
        nargs="+",
        default=DEFAULT_VIDEOS,
        help="List of videos inside data/ folder (default: bench-press-01 through 04).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Max frames to evaluate per model so slow CPU models finish quickly (default: 150 frames, 0 = all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if the video's table is already present in benchmark_results.md.",
    )
    args = parser.parse_args()

    report_path = SCRIPT_DIR / "benchmark_results.md"
    existing_content = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    for video in args.videos:
        section_header = f"### Quantitative Benchmark Results (`{video}`)"
        if section_header in existing_content and not args.force:
            print(f"[SKIP] Table for '{video}' already exists in {report_path.name}. Skipping to avoid overwriting.")
            continue

        table_md = run_benchmark_for_video(video, args.max_frames)

        # Append immediately to benchmark_results.md so progress is never lost
        with open(report_path, "a", encoding="utf-8") as f:
            if existing_content and not existing_content.endswith("\n\n"):
                f.write("\n")
            f.write(table_md + "\n\n")

        existing_content = report_path.read_text(encoding="utf-8")
        print(f"\n[SAVED] Appended benchmark table for '{video}' to {report_path}")

    print("\n" + "=" * 75)
    print(f"ALL BENCHMARKS COMPLETE! Full contents of {report_path.name}:")
    print("=" * 75 + "\n")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
