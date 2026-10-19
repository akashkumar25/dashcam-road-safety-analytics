"""Event detection on top of per-track distance readings.

Rules (all thresholds are parameters, all approximate because distance is monocular):
- ``near_miss``: vehicle in the ego lane with TTC under ``near_miss_ttc_s`` and closer than
  ``near_miss_m``.
- ``sudden_braking``: vehicle in the ego lane closing faster than ``braking_closing_mps``
  within ``braking_range_m``. A proxy: it cannot tell the ego car braking from the lead car
  braking, only that the gap collapsed quickly.
- ``pedestrian_proximity``: pedestrian within ``pedestrian_m`` and inside the ego lane widened
  by ``pedestrian_lane_factor``.

Frames that trigger the same rule for the same track are merged into one event; a gap of
more than ``merge_gap_s`` starts a new event. Pure Python, no OpenCV.
"""

from __future__ import annotations

import csv
import io

from pydantic import BaseModel

from .distance import DistanceReading


class Event(BaseModel):
    kind: str
    track_id: int
    label: str
    start_frame: int
    end_frame: int
    start_s: float
    end_s: float
    min_distance_m: float
    min_ttc_s: float | None = None
    max_closing_speed_mps: float = 0.0


CSV_FIELDS = list(Event.model_fields)


def events_to_csv(events: list[Event]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for e in events:
        writer.writerow(e.model_dump())
    return buf.getvalue()


class EventDetector:
    def __init__(
        self,
        fps: float,
        lane_half_width: float = 0.18,
        near_miss_ttc_s: float = 1.5,
        near_miss_m: float = 6.0,
        braking_closing_mps: float = 4.0,
        braking_range_m: float = 30.0,
        pedestrian_m: float = 10.0,
        pedestrian_lane_factor: float = 2.0,
        merge_gap_s: float = 1.0,
    ):
        self.fps = fps
        self.lane_half_width = lane_half_width
        self.near_miss_ttc_s, self.near_miss_m = near_miss_ttc_s, near_miss_m
        self.braking_closing_mps, self.braking_range_m = braking_closing_mps, braking_range_m
        self.pedestrian_m, self.pedestrian_lane_factor = pedestrian_m, pedestrian_lane_factor
        self.merge_gap_frames = max(1, round(merge_gap_s * fps))
        self._open: dict[tuple[str, int], Event] = {}
        self._done: list[Event] = []

    def _kinds(self, r: DistanceReading, offset: float) -> list[str]:
        """``offset``: horizontal distance of the box centre from frame centre, as a fraction
        of frame width."""
        kinds: list[str] = []
        vehicle = r.label != "pedestrian"
        if vehicle and r.in_ego_lane:
            if r.ttc_s is not None and r.ttc_s < self.near_miss_ttc_s and r.distance_m < self.near_miss_m:
                kinds.append("near_miss")
            if r.closing_speed_mps >= self.braking_closing_mps and r.distance_m < self.braking_range_m:
                kinds.append("sudden_braking")
        if (
            not vehicle
            and r.distance_m < self.pedestrian_m
            and offset <= self.lane_half_width * self.pedestrian_lane_factor
        ):
            kinds.append("pedestrian_proximity")
        return kinds

    def update(self, frame: int, r: DistanceReading, offset: float) -> None:
        for kind in self._kinds(r, offset):
            key = (kind, r.track_id)
            ev = self._open.get(key)
            if ev is not None and frame - ev.end_frame > self.merge_gap_frames:
                self._done.append(ev)
                ev = None
            if ev is None:
                ev = Event(
                    kind=kind, track_id=r.track_id, label=r.label, start_frame=frame,
                    end_frame=frame, start_s=round(frame / self.fps, 2),
                    end_s=round(frame / self.fps, 2), min_distance_m=r.distance_m,
                    min_ttc_s=r.ttc_s, max_closing_speed_mps=r.closing_speed_mps,
                )
                self._open[key] = ev
                continue
            ev.end_frame, ev.end_s = frame, round(frame / self.fps, 2)
            ev.min_distance_m = min(ev.min_distance_m, r.distance_m)
            if r.ttc_s is not None:
                ev.min_ttc_s = r.ttc_s if ev.min_ttc_s is None else min(ev.min_ttc_s, r.ttc_s)
            ev.max_closing_speed_mps = max(ev.max_closing_speed_mps, r.closing_speed_mps)

    def finish(self) -> list[Event]:
        events = self._done + list(self._open.values())
        self._done, self._open = [], {}
        return sorted(events, key=lambda e: (e.start_frame, e.kind, e.track_id))
