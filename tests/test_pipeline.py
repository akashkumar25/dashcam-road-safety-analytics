import cv2
import numpy as np
import pytest
import supervision as sv

from dashcam_analytics import cli
from dashcam_analytics.pipeline import analyse_video

W, H, N = 320, 240, 30


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (W, H))
    for _ in range(N):
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    writer.release()
    return path


class ScriptedDetector:
    """A car drifting right for the whole clip and a pedestrian that appears for 10 frames."""

    def __init__(self):
        self.i = 0

    def detect(self, frame):
        boxes, classes = [[10 + 3 * self.i, 100, 70 + 3 * self.i, 160]], [2]
        if 10 <= self.i < 20:
            boxes.append([250, 80, 270, 140])
            classes.append(0)
        if self.i == 5:  # one-frame false positive
            boxes.append([150, 20, 180, 50])
            classes.append(7)
        self.i += 1
        return sv.Detections(
            xyxy=np.array(boxes, dtype=float),
            confidence=np.full(len(boxes), 0.9),
            class_id=np.array(classes),
        )


def test_counts_unique_tracks_and_ignores_noise(video):
    summary = analyse_video(video, ScriptedDetector())
    assert summary.frames == N and (summary.width, summary.height) == (W, H)
    assert summary.counts == {"car": 1, "pedestrian": 1}
    assert summary.vehicles == 1 and summary.pedestrians == 1
    car = next(t for t in summary.tracks if t.label == "car")
    assert car.frames_seen >= N - 2


def test_writes_annotated_video(video, tmp_path):
    out = tmp_path / "out" / "annotated.mp4"
    analyse_video(video, ScriptedDetector(), out_video=out)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N
    ok, frame = cap.read()
    assert ok and frame.any()  # boxes were drawn on the black frame
    cap.release()


def test_max_frames(video):
    assert analyse_video(video, ScriptedDetector(), max_frames=7).frames == 7


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyse_video(tmp_path / "nope.mp4", ScriptedDetector())


def test_cli_missing_video_returns_error(tmp_path, capsys):
    rc = cli.main(["analyze", str(tmp_path / "nope.mp4")])
    assert rc == 1
    assert "error" in capsys.readouterr().err
