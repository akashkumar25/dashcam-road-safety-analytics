import math

import cv2
import numpy as np
import supervision as sv

from dashcam_analytics.distance import CameraModel, DistanceEstimator, DistanceReading, classify
from dashcam_analytics.pipeline import analyse_video


def test_focal_length_from_fov():
    cam = CameraModel(width=1000, hfov_deg=90)
    assert math.isclose(cam.focal_px, 500.0)


def test_distance_pinhole():
    cam = CameraModel(width=1000, hfov_deg=90)  # f = 500 px
    # a 1.5 m car whose box is 75 px tall is 500 * 1.5 / 75 = 10 m away
    assert math.isclose(cam.distance_m("car", 75), 10.0)
    assert cam.distance_m("unknown", 75) is None
    assert cam.distance_m("car", 0) is None


def test_closing_speed_and_ttc():
    cam = CameraModel(width=1000, hfov_deg=90)
    est = DistanceEstimator(cam, fps=10)
    reading = None
    # box grows 75 -> 100 px over 10 frames: 10 m -> 7.5 m in 1 s => ~2.5 m/s closing
    for i in range(10):
        h = 75 + i * 25 / 9
        reading = est.update(i, 1, "car", (450, 300, 550, 300 + h))
    assert reading.in_ego_lane
    assert 2.0 < reading.closing_speed_mps < 3.0
    assert 2.5 < reading.ttc_s < 4.0


def test_receding_object_has_no_ttc():
    est = DistanceEstimator(CameraModel(width=1000), fps=10)
    for i in range(6):
        reading = est.update(i, 1, "car", (450, 300, 550, 300 + 100 - 5 * i))
    assert reading.closing_speed_mps < 0 and reading.ttc_s is None


def test_lane_filter_and_warnings():
    est = DistanceEstimator(CameraModel(width=1000), fps=10)
    side = est.update(0, 2, "car", (50, 300, 150, 400))
    assert not side.in_ego_lane and classify(0, side) == []
    near = DistanceReading(track_id=1, label="car", distance_m=5.0, closing_speed_mps=3.0,
                           ttc_s=1.7, in_ego_lane=True)
    kinds = {w.kind for w in classify(3, near)}
    assert kinds == {"tailgating", "collision_risk"}
    ped = near.model_copy(update={"label": "pedestrian", "ttc_s": None})
    assert classify(3, ped) == []  # pedestrians are handled by event detection, not tailgating


class ApproachingCar:
    """A lead car centred in the lane whose box grows every frame."""

    def __init__(self):
        self.i = 0

    def detect(self, frame):
        h = 40 + 3 * self.i
        box = [150, 100, 150 + h, 100 + h]
        self.i += 1
        return sv.Detections(
            xyxy=np.array([box], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([2]),
        )


def test_pipeline_reports_distance_and_warnings(tmp_path):
    path = tmp_path / "c.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
    for _ in range(30):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()
    summary = analyse_video(path, ApproachingCar())
    assert summary.min_distance_m is not None and summary.min_distance_m < 8
    assert any(x.kind == "tailgating" for x in summary.warnings)
