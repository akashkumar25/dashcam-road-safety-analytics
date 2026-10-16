"""Monocular distance and time-to-collision estimation.

Pinhole model: ``distance = focal_px * real_height / box_height_px``. The focal
length in pixels comes from the camera's horizontal field of view, and the real
height from a typical size per class. Both are rough, so results are meant for
warnings, not measurements. Pure Python: no OpenCV needed.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque

from pydantic import BaseModel

# Typical real-world heights in metres.
REAL_HEIGHT_M = {"car": 1.5, "motorcycle": 1.4, "bus": 3.2, "truck": 3.0, "pedestrian": 1.7}


class CameraModel(BaseModel):
    width: int
    hfov_deg: float = 90.0

    @property
    def focal_px(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    def distance_m(self, label: str, box_height_px: float) -> float | None:
        height = REAL_HEIGHT_M.get(label)
        if height is None or box_height_px <= 0:
            return None
        return self.focal_px * height / box_height_px


class DistanceReading(BaseModel):
    track_id: int
    label: str
    distance_m: float
    closing_speed_mps: float  # positive = getting closer
    ttc_s: float | None  # None when not closing in
    in_ego_lane: bool


class DistanceEstimator:
    """Per-track distance, closing speed (smoothed over a short window) and TTC."""

    def __init__(
        self,
        camera: CameraModel,
        fps: float,
        lane_half_width: float = 0.18,
        window: int = 8,
    ):
        self.camera, self.fps = camera, fps
        self.lane_half_width = lane_half_width  # fraction of frame width either side of centre
        self.window = window
        self._history: dict[int, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def update(
        self, frame: int, track_id: int, label: str, xyxy: tuple[float, float, float, float]
    ) -> DistanceReading | None:
        dist = self.camera.distance_m(label, xyxy[3] - xyxy[1])
        if dist is None:
            return None
        hist = self._history[track_id]
        hist.append((frame, dist))
        closing = 0.0
        if len(hist) >= 3:
            (f0, d0), (f1, d1) = hist[0], hist[-1]
            dt = (f1 - f0) / self.fps
            if dt > 0:
                closing = (d0 - d1) / dt
        ttc = dist / closing if closing > 0.1 else None
        cx = (xyxy[0] + xyxy[2]) / 2
        in_lane = abs(cx - self.camera.width / 2) <= self.lane_half_width * self.camera.width
        return DistanceReading(
            track_id=track_id,
            label=label,
            distance_m=round(dist, 2),
            closing_speed_mps=round(closing, 2),
            ttc_s=round(ttc, 2) if ttc is not None else None,
            in_ego_lane=in_lane,
        )


class RiskWarning(BaseModel):
    frame: int
    track_id: int
    kind: str  # "tailgating" | "collision_risk"
    distance_m: float
    ttc_s: float | None = None


def classify(
    frame: int,
    reading: DistanceReading,
    tailgate_m: float = 8.0,
    ttc_s: float = 2.5,
) -> list[RiskWarning]:
    """Warnings for a reading. Only vehicles in the ego lane count as tailgating."""
    out: list[RiskWarning] = []
    if reading.label != "pedestrian" and reading.in_ego_lane and reading.distance_m < tailgate_m:
        out.append(
            RiskWarning(frame=frame, track_id=reading.track_id, kind="tailgating",
                     distance_m=reading.distance_m, ttc_s=reading.ttc_s)
        )
    if reading.in_ego_lane and reading.ttc_s is not None and reading.ttc_s < ttc_s:
        out.append(
            RiskWarning(frame=frame, track_id=reading.track_id, kind="collision_risk",
                     distance_m=reading.distance_m, ttc_s=reading.ttc_s)
        )
    return out
