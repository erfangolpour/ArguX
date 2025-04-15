# backend/settings.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the base directory of the backend application
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # Application settings
    app_name: str = "ArguX API"
    app_description: str = "Real-time public camera monitoring tool"
    log_level: str = "INFO"

    # CORS settings
    cors_allow_origins: list[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list[str] = ["*"]
    cors_allow_headers: list[str] = ["*"]

    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_url: str = f"redis://{redis_host}:{redis_port}/{redis_db}"
    active_tasks_key: str = "argux:active_tasks"
    scan_running_key: str = "argux:scan_running"

    # Celery settings
    celery_broker_url: str = redis_url
    celery_result_backend: str = redis_url

    # YOLO settings
    yolo_base_dir: Path = BASE_DIR / "YOLO"
    yolo_weights_dir: str = str(yolo_base_dir)
    yolo_runs_dir: str = str(yolo_base_dir)
    default_model: str = "yolo11n.pt"  # Changed from yolo11n.pt to yolo11n.pt

    # HTTP Client settings
    http_client_timeout: int = 5  # Increased default timeout slightly
    http_user_agent: str = "Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5"

    # Scan settings
    max_concurrent_tasks: int = 5  # Max concurrent processing tasks

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Create a single settings instance
settings = Settings()

# Ensure YOLO directory exists
settings.yolo_base_dir.mkdir(exist_ok=True)
