"""Runtime settings, read from environment variables or a .env file.

Defaults target an ordinary laptop CPU: the YOLO nano model at a modest input size.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DASHCAM_", env_file=".env", extra="ignore")

    yolo_model: str = "yolov8n.pt"
    image_size: int = 640
    confidence: float = 0.35
    videos_dir: Path = Path("data/videos")
    output_dir: Path = Path("data/output")
