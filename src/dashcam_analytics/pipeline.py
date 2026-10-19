"""Detection + multi-object tracking over a dashcam video.

``analyse_video`` reads frames with OpenCV, runs a detector, tracks objects with
supervision's ByteTrack, counts unique tracked vehicles and pedestrians and can
write an annotated video. The detector is injectable so the pipeline is testable
without model weights; ``YoloDetector`` is the default (Ultralytics, CPU).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import supervision as sv
from pydantic import BaseModel

from .distance import CameraModel, DistanceEstimator, RiskWarning, classify
from .events import Event, EventDetector

# COCO class ids -> report label
CLASS_LABELS = {0: "pedestrian", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_LABELS = {"car", "motorcycle", "bus", "truck"}


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> sv.Detections: ...


class YoloDetector:
    """Ultralytics YOLO on CPU, restricted to road-user classes. Weights load on first use."""

    def __init__(self, weights: str = "yolov8n.pt", image_size: int = 640, confidence: float = 0.35):
        self.weights, self.image_size, self.confidence = weights, image_size, confidence
        self._model = None

    def detect(self, frame: np.ndarray) -> sv.Detections:  # pragma: no cover - needs weights
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        result = self._model.predict(
            frame,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=list(CLASS_LABELS),
            device="cpu",
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)


class TrackRecord(BaseModel):
    track_id: int
    label: str
    first_frame: int
    last_frame: int
    frames_seen: int


class VideoSummary(BaseModel):
    video: str
    frames: int
    fps: float
    width: int
    height: int
    counts: dict[str, int]
    tracks: list[TrackRecord]
    min_distance_m: float | None = None
    warnings: list[RiskWarning] = []
    events: list[Event] = []

    @property
    def vehicles(self) -> int:
        return sum(n for label, n in self.counts.items() if label in VEHICLE_LABELS)

    @property
    def pedestrians(self) -> int:
        return self.counts.get("pedestrian", 0)


def analyse_video(
    video: Path,
    detector: Detector,
    out_video: Path | None = None,
    max_frames: int | None = None,
    min_track_frames: int = 3,
    hfov_deg: float = 90.0,
) -> VideoSummary:
    """Track road users through ``video``; tracks seen in fewer than ``min_track_frames`` frames
    are treated as noise and left out of the counts."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_video is not None:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
    estimator = DistanceEstimator(CameraModel(width=width, hfov_deg=hfov_deg), fps)
    warnings: list[RiskWarning] = []
    event_detector = EventDetector(fps)
    min_distance: float | None = None
    tracker = sv.ByteTrack(frame_rate=max(1, round(fps)))
    box_annotator, label_annotator = sv.BoxAnnotator(), sv.LabelAnnotator()

    labels_by_track: dict[int, str] = {}
    seen: dict[int, list[int]] = defaultdict(list)
    frame_index = 0
    try:
        while max_frames is None or frame_index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame)
            if len(detections):
                keep = np.isin(detections.class_id, list(CLASS_LABELS))
                detections = detections[keep]
            tracked = tracker.update_with_detections(detections)
            texts: list[str] = []
            for tid, cid, box in zip(tracked.tracker_id, tracked.class_id, tracked.xyxy):
                label = CLASS_LABELS[int(cid)]
                labels_by_track[int(tid)] = label
                seen[int(tid)].append(frame_index)
                reading = estimator.update(frame_index, int(tid), label, tuple(box))
                text = f"#{int(tid)} {label}"
                if reading is not None:
                    text += f" {reading.distance_m:.0f}m"
                    lead = reading.in_ego_lane and label != "pedestrian"
                    if lead and (min_distance is None or reading.distance_m < min_distance):
                        min_distance = reading.distance_m
                    warnings.extend(classify(frame_index, reading))
                    offset = abs((box[0] + box[2]) / 2 - width / 2) / width
                    event_detector.update(frame_index, reading, offset)
                texts.append(text)
            if writer is not None:
                frame = box_annotator.annotate(frame.copy(), tracked)
                frame = label_annotator.annotate(frame, tracked, labels=texts)
                writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    tracks = [
        TrackRecord(
            track_id=tid,
            label=labels_by_track[tid],
            first_frame=frames[0],
            last_frame=frames[-1],
            frames_seen=len(frames),
        )
        for tid, frames in sorted(seen.items())
        if len(frames) >= min_track_frames
    ]
    counts: dict[str, int] = defaultdict(int)
    for t in tracks:
        counts[t.label] += 1
    return VideoSummary(
        video=video.name,
        frames=frame_index,
        fps=fps,
        width=width,
        height=height,
        counts=dict(counts),
        tracks=tracks,
        min_distance_m=min_distance,
        warnings=warnings,
        events=event_detector.finish(),
    )
