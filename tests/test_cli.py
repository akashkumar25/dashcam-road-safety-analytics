from dashcam_analytics.cli import list_videos, main
from dashcam_analytics.config import Settings


def test_defaults_are_cpu_friendly():
    s = Settings()
    assert s.yolo_model == "yolov8n.pt" and s.image_size <= 640


def test_list_videos_filters_by_suffix(tmp_path):
    (tmp_path / "b.MP4").write_bytes(b"x")
    (tmp_path / "a.avi").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert list_videos(Settings(videos_dir=tmp_path)) == ["a.avi", "b.MP4"]


def test_list_videos_missing_dir(tmp_path):
    assert list_videos(Settings(videos_dir=tmp_path / "nope")) == []


def test_status_command(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("DASHCAM_VIDEOS_DIR", str(tmp_path))
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Model: yolov8n.pt" in out and "Videos in" in out
