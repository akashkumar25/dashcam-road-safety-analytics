import csv
import io

from dashcam_analytics.distance import DistanceReading
from dashcam_analytics.events import EventDetector, events_to_csv

FPS = 10.0


def reading(tid=1, label="car", dist=20.0, closing=0.0, ttc=None, lane=True):
    return DistanceReading(track_id=tid, label=label, distance_m=dist, closing_speed_mps=closing,
                           ttc_s=ttc, in_ego_lane=lane)


def run(items, **kw):
    det = EventDetector(FPS, **kw)
    for frame, r, offset in items:
        det.update(frame, r, offset)
    return det.finish()


def test_near_miss_merges_consecutive_frames():
    items = [(f, reading(dist=5.0 - 0.1 * f, closing=6.0, ttc=0.8 - 0.01 * f), 0.0) for f in range(5)]
    events = run(items)
    near = [e for e in events if e.kind == "near_miss"]
    assert len(near) == 1
    assert (near[0].start_frame, near[0].end_frame) == (0, 4)
    assert near[0].min_distance_m == 4.6 and near[0].start_s == 0.0 and near[0].end_s == 0.4


def test_gap_splits_events():
    r = reading(dist=4.0, closing=5.0, ttc=0.8)
    events = run([(0, r, 0.0), (1, r, 0.0), (30, r, 0.0)])
    assert [(e.start_frame, e.end_frame) for e in events if e.kind == "near_miss"] == [(0, 1), (30, 30)]


def test_sudden_braking_needs_ego_lane_and_range():
    fast = reading(dist=15.0, closing=5.0, ttc=3.0)
    assert [e.kind for e in run([(0, fast, 0.0)])] == ["sudden_braking"]
    assert run([(0, reading(dist=15.0, closing=5.0, ttc=3.0, lane=False), 0.4)]) == []
    assert run([(0, reading(dist=50.0, closing=5.0, ttc=10.0), 0.0)]) == []


def test_pedestrian_proximity_uses_widened_lane():
    ped = reading(label="pedestrian", dist=6.0)
    assert [e.kind for e in run([(0, ped, 0.3)])] == ["pedestrian_proximity"]
    assert run([(0, ped, 0.45)]) == []
    assert run([(0, reading(label="pedestrian", dist=15.0), 0.0)]) == []


def test_quiet_traffic_yields_no_events():
    assert run([(f, reading(dist=25.0, closing=0.5, ttc=None), 0.0) for f in range(20)]) == []


def test_csv_export_roundtrip():
    events = run([(0, reading(dist=4.0, closing=5.0, ttc=0.8), 0.0)])
    rows = list(csv.DictReader(io.StringIO(events_to_csv(events))))
    assert rows[0]["kind"] == "near_miss" and rows[0]["min_distance_m"] == "4.0"
    assert events_to_csv([]).strip().split(",")[0] == "kind"


def test_pipeline_emits_events_and_csv(tmp_path):
    import cv2
    import numpy as np
    import supervision as sv

    from dashcam_analytics import cli
    from dashcam_analytics.pipeline import analyse_video

    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    for _ in range(25):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()

    class Approaching:
        """Centred car whose box grows fast: distance collapses well inside near-miss range."""

        i = 0

        def detect(self, frame):
            h = 30 + 6 * self.i
            self.i += 1
            return sv.Detections(
                xyxy=np.array([[140.0, 200.0 - h, 180.0, 200.0]]),
                confidence=np.array([0.9]), class_id=np.array([2]),
            )

    summary = analyse_video(path, Approaching())
    assert {e.kind for e in summary.events} >= {"sudden_braking", "near_miss"}
    assert "near_miss" in events_to_csv(summary.events)
    out = tmp_path / "ev.csv"
    assert cli.main(["analyze", str(tmp_path / "nope.mp4"), "--events-csv", str(out)]) == 1
