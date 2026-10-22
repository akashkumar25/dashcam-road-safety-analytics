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

## Distance and time-to-collision

Each tracked road user gets a distance from the pinhole model `distance = focal_px * real_height / box_height_px`, with the focal length derived from the camera's horizontal field of view (`--hfov`, default 90 degrees) and a typical height per class (car 1.5 m, truck 3.0 m, pedestrian 1.7 m, ...). Closing speed is the change in distance over a short window and TTC is `distance / closing speed`. Objects whose box centre lies in the central 36% of the frame count as the ego lane. Warnings in the JSON summary (`warnings`, `min_distance_m`): `tailgating` (vehicle in lane closer than 8 m) and `collision_risk` (TTC under 2.5 s). Distances appear in the annotated video labels.

These numbers are approximate (a wrong FOV or an unusual vehicle size scales every distance) and were tested only on synthetic boxes with known geometry, not calibrated against real footage.

## Events and export

```bash
dashcam-analytics analyze data/videos/drive.mp4 --summary out/summary.json --events-csv out/events.csv
```

`events` in the JSON summary (and the CSV, one row per event) lists `near_miss` (ego-lane vehicle, TTC under 1.5 s and closer than 6 m), `sudden_braking` (ego-lane vehicle closing at 4 m/s or more within 30 m; a proxy that cannot tell the ego car braking from the lead car braking) and `pedestrian_proximity` (pedestrian within 10 m inside twice the ego-lane width). Consecutive frames merge into one event with start/end frame and time, minimum distance/TTC and peak closing speed. Thresholds are parameters of `EventDetector`; like the distances they are uncalibrated and only tested on synthetic boxes.

## Web UI

```bash
pip install -e ".[vision,ui]"
dashcam-analytics ui            # http://127.0.0.1:7860
```

Upload a clip, set the camera field of view and a frame limit (default 300, so a first try finishes quickly on a CPU), and get the annotated video, counts, an events table and the JSON summary for download. The analysis logic lives in `AnalysisService` (no Gradio import) and is tested with a scripted detector; the Gradio app is only checked to build, not driven in a browser. The annotated video is written as `mp4v`; Gradio re-encodes it for the browser when `ffmpeg` is installed, otherwise the preview may not play (the file is still valid).

## CPU benchmark

```bash
dashcam-analytics benchmark data/videos/drive.mp4 --models yolov8n.pt yolov8s.pt --frames 60 --image-size 640
```

For each model the command reports detector-only FPS (after a warm-up) and end-to-end FPS of the full pipeline (decode, detect, ByteTrack, distance/TTC, events; no video written), plus the speed relative to the clip's own frame rate (above 1.0 means it keeps up live).

Measured on a 4-vCPU Intel Xeon @ 2.10 GHz (no GPU), Ultralytics YOLOv8 pretrained weights, a 768x432, 12.5 FPS public sample clip (`car-detection.mp4` from [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos)), 60 frames per run:

| Model | Input size | Detector FPS | Full pipeline FPS | x real time |
|---|---|---|---|---|
| yolov8n | 640 | 20.4 | 24.4 | 1.96 |
| yolov8s | 640 | 12.8 | 11.9 | 0.95 |
| yolov8n | 480 | 41.9 | 37.8 | 3.02 |
| yolov8s | 480 | 21.5 | 15.6 | 1.25 |
| yolov8n | 320 | 83.9 | 66.0 | 5.28 |
| yolov8s | 320 | 33.0 | 33.4 | 2.67 |

Treat these as indicative: a shared vCPU varies by a few FPS between runs (the 640 nano row shows detector FPS below pipeline FPS for that reason), and a busier scene adds tracking cost. Takeaways: the nano model at 640 runs faster than a 12.5 FPS clip on four CPU cores; the small model at 640 is about real time; dropping the input size to 480 or 320 trades some accuracy on distant objects for 1.5-3x more throughput, which matters for the distance estimate of far vehicles. Use `DASHCAM_YOLO_MODEL` and `DASHCAM_IMAGE_SIZE` to choose the setting.

## Roadmap

- [x] Scaffold: package layout, settings, CLI, tests, CI, Docker
- [x] Detection and multi-object tracking pipeline
- [x] Monocular distance estimation and time-to-collision warnings
- [x] Event detection (near-miss, sudden braking proxy, pedestrian proximity) with CSV/JSON export
- [x] Gradio UI and annotated video output
- [x] CPU FPS benchmark for nano/small models

## Demo

![demo](docs/demo.gif)

_Demo GIF placeholder, to be recorded once the UI exists._

## License

MIT
