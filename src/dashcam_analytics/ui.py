"""Gradio UI: upload a dashcam clip, get an annotated video, counts and events.

``AnalysisService`` holds the logic and does not import Gradio, so it is unit-tested
with a scripted detector; ``build_app`` only wires widgets to it.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from .events import Event
from .pipeline import Detector, VideoSummary, analyse_video

EVENT_COLUMNS = ["kind", "track_id", "label", "start_s", "end_s", "min_distance_m", "min_ttc_s"]


@dataclass
class AnalysisResult:
    video_path: str
    summary_text: str
    events: list[list]
    summary_json_path: str


def events_table(events: list[Event]) -> list[list]:
    return [[getattr(e, c) for c in EVENT_COLUMNS] for e in events]


def summary_markdown(summary: VideoSummary) -> str:
    lines = [
        f"**{summary.frames} frames** at {summary.fps:.0f} fps ({summary.width}x{summary.height})",
        f"- Vehicles: {summary.vehicles}, pedestrians: {summary.pedestrians}",
        "- Closest lead vehicle: "
        + (f"{summary.min_distance_m:.1f} m" if summary.min_distance_m is not None else "n/a"),
        f"- Warnings: {len(summary.warnings)}, events: {len(summary.events)}",
        "",
        "_Distances and times are monocular estimates, not measurements._",
    ]
    return "\n".join(lines)


class AnalysisService:
    def __init__(self, detector: Detector, out_dir: Path | None = None):
        self.detector = detector
        self.out_dir = out_dir or Path(tempfile.mkdtemp(prefix="dashcam-ui-"))

    def run(self, video: str | None, hfov_deg: float = 90.0, max_frames: int | None = None) -> AnalysisResult:
        if not video:
            raise ValueError("Upload a video first.")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video).stem
        out_video = self.out_dir / f"{stem}_annotated.mp4"
        try:
            summary = analyse_video(
                Path(video), self.detector, out_video, max_frames or None, hfov_deg=hfov_deg
            )
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        json_path = self.out_dir / f"{stem}_summary.json"
        json_path.write_text(summary.model_dump_json(indent=2))
        return AnalysisResult(
            str(out_video), summary_markdown(summary), events_table(summary.events), str(json_path)
        )


def build_app(service: AnalysisService):  # pragma: no cover - widget wiring
    import gradio as gr

    def handle(video, hfov, max_frames):
        try:
            r = service.run(video, float(hfov), int(max_frames) if max_frames else None)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return r.video_path, r.summary_text, r.events, r.summary_json_path

    with gr.Blocks(title="Dashcam road-safety analytics") as app:
        gr.Markdown("# Dashcam road-safety analytics\nYOLO + tracking on CPU. Long clips are slow: limit the frames to try it out.")
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Dashcam clip")
                hfov = gr.Slider(60, 140, value=90, step=1, label="Camera horizontal FOV (degrees)")
                max_frames = gr.Number(value=300, precision=0, label="Max frames (0 = whole video)")
                go = gr.Button("Analyse", variant="primary")
            with gr.Column():
                out = gr.Video(label="Annotated video", format="mp4")
                summary = gr.Markdown()
        events = gr.Dataframe(headers=EVENT_COLUMNS, label="Events")
        report = gr.File(label="JSON summary")
        go.click(handle, [video, hfov, max_frames], [out, summary, events, report])
    return app
