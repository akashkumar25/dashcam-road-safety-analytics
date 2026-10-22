"""CPU throughput benchmark for the detection + tracking pipeline.

For each model it measures two things on the same clip:

* detector-only speed (``detect`` per frame), after a warm-up pass that absorbs
  model loading and first-call overhead;
* end-to-end speed of ``analyse_video`` (decode, detect, track, distance, events),
  without writing an annotated video.

Results are reported as frames per second so they can be compared with the
source video's frame rate: above 1.0x real time the pipeline keeps up live.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from pathlib import Path

import cv2
from pydantic import BaseModel

from .pipeline import Detector, analyse_video


class BenchmarkRow(BaseModel):
    model: str
    image_size: int
    frames: int
    detect_fps: float
    pipeline_fps: float
    realtime_factor: float  # pipeline_fps / video fps


def _read_frames(video: Path, limit: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"no frames in {video}")
    return frames, fps


def benchmark_model(
    name: str,
    detector_factory: Callable[[], Detector],
    video: Path,
    frames: int = 60,
    image_size: int = 640,
    warmup: int = 3,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkRow:
    sample, video_fps = _read_frames(video, frames)
    detector = detector_factory()
    for frame in sample[:warmup]:
        detector.detect(frame)

    start = clock()
    for frame in sample:
        detector.detect(frame)
    detect_fps = len(sample) / max(clock() - start, 1e-9)

    start = clock()
    summary = analyse_video(video, detector, max_frames=len(sample))
    pipeline_fps = summary.frames / max(clock() - start, 1e-9)
    return BenchmarkRow(
        model=name,
        image_size=image_size,
        frames=len(sample),
        detect_fps=round(detect_fps, 2),
        pipeline_fps=round(pipeline_fps, 2),
        realtime_factor=round(pipeline_fps / video_fps, 2),
    )


def system_description() -> str:
    import os

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{cpu}, {os.cpu_count()} logical CPUs"


def to_markdown(rows: list[BenchmarkRow], video: str, system: str) -> str:
    lines = [
        f"Clip: `{video}`, {rows[0].frames} frames per run. System: {system}.",
        "",
        "| Model | Input size | Detector FPS | Full pipeline FPS | x real time |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.model} | {r.image_size} | {r.detect_fps:.1f} | {r.pipeline_fps:.1f} "
            f"| {r.realtime_factor:.2f} |"
        )
    return "\n".join(lines)
