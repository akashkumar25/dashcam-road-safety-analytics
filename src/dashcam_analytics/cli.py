"""Command line entry point."""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Settings

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def list_videos(settings: Settings) -> list[str]:
    if not settings.videos_dir.is_dir():
        return []
    return sorted(
        p.name for p in settings.videos_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dashcam-analytics", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("status", help="show configuration and discovered videos")
    analyse = sub.add_parser("analyze", help="detect and track road users in a video")
    analyse.add_argument("video", type=Path)
    analyse.add_argument("--out-video", type=Path, help="write an annotated video here")
    analyse.add_argument("--summary", type=Path, help="write the JSON summary here")
    analyse.add_argument("--events-csv", type=Path, help="write detected events as CSV here")
    analyse.add_argument("--max-frames", type=int)
    analyse.add_argument("--hfov", type=float, default=90.0, help="camera horizontal FOV, degrees")
    ui = sub.add_parser("ui", help="launch the Gradio web UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)

    if args.command == "status":
        s = Settings()
        print(f"Model: {s.yolo_model}, image size: {s.image_size}, confidence: {s.confidence}")
        videos = list_videos(s)
        print(f"Videos in {s.videos_dir}: {len(videos)}")
        for name in videos:
            print(f"  - {name}")
        return 0
    if args.command == "analyze":
        try:
            from .pipeline import YoloDetector, analyse_video
        except ImportError:
            print('error: install the vision stack: pip install -e ".[vision]"', file=sys.stderr)
            return 1
        s = Settings()
        detector = YoloDetector(s.yolo_model, s.image_size, s.confidence)
        try:
            summary = analyse_video(args.video, detector, args.out_video, args.max_frames, hfov_deg=args.hfov)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        text = summary.model_dump_json(indent=2)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(text)
        if args.events_csv:
            from .events import events_to_csv

            args.events_csv.parent.mkdir(parents=True, exist_ok=True)
            args.events_csv.write_text(events_to_csv(summary.events))
        print(
            f"{summary.frames} frames, {summary.vehicles} vehicles, "
            f"{summary.pedestrians} pedestrians, {len(summary.warnings)} warnings, "
            f"{len(summary.events)} events"
        )
        return 0
    if args.command == "ui":
        try:
            from .pipeline import YoloDetector
            from .ui import AnalysisService, build_app
        except ImportError:
            print('error: pip install -e ".[vision,ui]"', file=sys.stderr)
            return 1
        s = Settings()
        service = AnalysisService(YoloDetector(s.yolo_model, s.image_size, s.confidence))
        build_app(service).launch(server_name=args.host, server_port=args.port)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
