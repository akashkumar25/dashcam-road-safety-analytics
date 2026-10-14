"""Command line entry point."""

import argparse
import sys

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
    args = parser.parse_args(argv)

    if args.command == "status":
        s = Settings()
        print(f"Model: {s.yolo_model}, image size: {s.image_size}, confidence: {s.confidence}")
        videos = list_videos(s)
        print(f"Videos in {s.videos_dir}: {len(videos)}")
        for name in videos:
            print(f"  - {name}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
