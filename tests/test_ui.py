import json
from pathlib import Path

import pytest

from dashcam_analytics.ui import AnalysisService, build_app
from test_pipeline import ScriptedDetector, video  # noqa: F401  (fixture re-use)


def test_run_produces_video_summary_and_json(video, tmp_path):  # noqa: F811
    service = AnalysisService(ScriptedDetector(), out_dir=tmp_path / "out")
    result = service.run(str(video), hfov_deg=90.0)
    assert Path(result.video_path).is_file() and Path(result.video_path).stat().st_size > 0
    assert "Vehicles: 1, pedestrians: 1" in result.summary_text
    data = json.loads(Path(result.summary_json_path).read_text())
    assert data["counts"] == {"car": 1, "pedestrian": 1}
    assert isinstance(result.events, list)


def test_max_frames_limits_the_analysis(video, tmp_path):  # noqa: F811
    result = AnalysisService(ScriptedDetector(), out_dir=tmp_path).run(str(video), max_frames=10)
    assert json.loads(Path(result.summary_json_path).read_text())["frames"] == 10


def test_missing_video_is_a_clear_error(tmp_path):
    service = AnalysisService(ScriptedDetector(), out_dir=tmp_path)
    with pytest.raises(ValueError, match="Upload"):
        service.run(None)
    with pytest.raises(ValueError, match="cannot open"):
        service.run(str(tmp_path / "nope.mp4"))


def test_app_builds(tmp_path):
    assert build_app(AnalysisService(ScriptedDetector(), out_dir=tmp_path)) is not None
