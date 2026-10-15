# dashcam-road-safety-analytics

Dashcam video analytics on CPU: pretrained YOLO detection and tracking (Ultralytics + supervision), vehicle and pedestrian counts, following-distance and tailgating warnings, near-miss events, an annotated video and a summary report.

> **Status: scaffold.** Project structure, configuration, CLI and CI are in place. The pipeline stages are on the roadmap below.

## Design

- **CPU first.** Defaults to the YOLO nano model at 640 px; no GPU needed.
- **Public data only.** Sample video and benchmarks use openly licensed footage.
- **Monocular.** Distance and time-to-collision are estimated from a single camera, so values are approximate and meant as warnings, not measurements.

## Install

```bash
pip install -e ".[dev]"
dashcam-analytics status
```

Add the vision stack (Ultralytics, supervision, OpenCV) when the pipeline lands: `pip install -e ".[vision]"`.

Or with Docker:

```bash
docker compose up --build
```

Copy `.env.example` to `.env` to change the model or thresholds. Put videos in `data/videos/`.

## Analyze a video

```bash
pip install -e ".[vision]"      # ultralytics (CPU torch), supervision, OpenCV
dashcam-analytics analyze data/videos/drive.mp4 --out-video data/output/annotated.mp4 --summary data/output/summary.json
```

YOLO (default `yolov8n.pt`, restricted to person, car, motorcycle, bus, truck) feeds supervision's ByteTrack. The summary lists each track (id, class, first/last frame) and counts unique vehicles and pedestrians; tracks seen in fewer than 3 frames are dropped as noise. The detector is injectable (`analyse_video(video, detector)`), so tests run on a synthetic video with a scripted detector and need only the `video` extra (supervision + OpenCV), not model weights. The real YOLO path needs the weights, which Ultralytics downloads on first use.

## Development

```bash
ruff check .
pytest
```

## Roadmap

- [x] Scaffold: package layout, settings, CLI, tests, CI, Docker
- [x] Detection and multi-object tracking pipeline
- [ ] Monocular distance estimation and time-to-collision warnings
- [ ] Event detection (near-miss, sudden braking proxy, pedestrian proximity) with CSV/JSON export
- [ ] Gradio UI and annotated video output
- [ ] CPU FPS benchmark for nano/small models

## Demo

![demo](docs/demo.gif)

_Demo GIF placeholder, to be recorded once the UI exists._

## License

MIT
