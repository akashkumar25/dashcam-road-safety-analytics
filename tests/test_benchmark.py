import cv2
import numpy as np
import pytest
import supervision as sv

from dashcam_analytics import cli
from dashcam_analytics.benchmark import benchmark_model, to_markdown


class EmptyDetector:
    def detect(self, frame):
        return sv.Detections.empty()


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for _ in range(20):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()
    return path


def test_benchmark_row_and_markdown(video):
    row = benchmark_model("fake", EmptyDetector, video, frames=15, warmup=2)
    assert row.frames == 15
    assert row.detect_fps > 0 and row.pipeline_fps > 0
    assert row.realtime_factor == pytest.approx(row.pipeline_fps / 10.0, rel=0.05)
    md = to_markdown([row], "clip.mp4", "test box")
    assert "| fake |" in md and "test box" in md


def test_benchmark_frames_capped_by_video_length(video):
    assert benchmark_model("fake", EmptyDetector, video, frames=500).frames == 20


def test_benchmark_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        benchmark_model("fake", EmptyDetector, tmp_path / "nope.mp4")
    assert cli.main(["benchmark", str(tmp_path / "nope.mp4")]) == 1
