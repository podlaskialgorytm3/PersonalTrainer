"""
Unified benchmark runner for all 4 Python Pose Estimation libraries (Task #1: Python CV libraries analysis).

Runs each candidate library in headless mode (--no-display, --speed 1.0 so all frames are evaluated)
on the same reference workout video and prints a Markdown comparison table ready for the ADR.

Usage:
    python research/001-python-cv-libraries-analysis/benchmark_all.py
    python research/001-python-cv-libraries-analysis/benchmark_all.py --video /bench-press/bench-press-01
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all CV pose estimation benchmarks on a reference video.")
    parser.add_argument(
        "--video",
        type=str,
        default="/bench-press/bench-press-01",
        help="Reference video inside data/ folder (default: /bench-press/bench-press-01).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=150,
        help="Max frames to evaluate per model so slow CPU models finish quickly (default: 150 frames, 0 = all).",
    )
    args = parser.parse_args()

    results_table: list[dict[str, str]] = []

    print("=" * 75)
    print(f"RUNNING POSE ESTIMATION BENCHMARK SUITE ON: {args.video}")
    print("=" * 75)

    for item in BENCHMARKS:
        script_path = SCRIPT_DIR / item["script"]
        cmd = [
            sys.executable,
            str(script_path),
            "--video",
            args.video,
            "--speed",
            "1.0",
            "--max-frames",
            str(args.max_frames),
            "--no-display",
            *item["extra_args"],
        ]

        print(f"\n>>> Benchmarking: {item['name']} ...")
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

    # Generate Markdown report table
    md_lines = [
        f"### Quantitative Benchmark Results (`{args.video}`)",
        "",
        "| Model / Library | Keypoints (2D/3D) | Detection Rate | Avg Latency (ms) | P95 Latency (ms) | Avg FPS |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for row in results_table:
        md_lines.append(
            f"| **{row['name']}** | {row['keypoints']} | {row['detected']} | {row['avg_ms']} | {row['p95_ms']} | **{row['fps']}** |"
        )

    markdown_output = "\n".join(md_lines)
    report_path = SCRIPT_DIR / "benchmark_results.md"
    report_path.write_text(markdown_output + "\n", encoding="utf-8")

    print("\n" + "=" * 75)
    print("MARKDOWN TABLE (also saved to research/001-python-cv-libraries-analysis/benchmark_results.md):")
    print("=" * 75 + "\n")
    print(markdown_output)
    print("\n" + "=" * 75)


if __name__ == "__main__":
    main()
