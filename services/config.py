"""Server settings, read from environment variables only (documented in server/README.md)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ALLOWED_EXTENSIONS = (".mov", ".mp4", ".mkv", ".webm")
# the ComfyUI graph the manual render package fills in (task 21)
QUALITY_GRAPH = Path(__file__).resolve().parents[1] / "comfyui" / "CueFX_Phase1_Quality.json"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)
    max_upload_mb: int = 500
    look_image: Path | None = None  # None: <data>/look/standin_orb.png, the CLI's default look
    looks_dir: Path | None = None   # None: <data>/looks; the manual render's look images, <look id>.png
    quality_graph: Path = QUALITY_GRAPH
    ltx_overlap_enabled: bool = False
    comfy_url: str | None = None
    comfy_token: str | None = None
    comfy_timeout_s: int = 7200

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def standin_dir(self) -> Path:
        return self.data_dir / "standin"

    @property
    def look_path(self) -> Path:
        return self.look_image or self.data_dir / "look" / "standin_orb.png"

    @property
    def looks(self) -> Path:
        return self.looks_dir or self.data_dir / "looks"

    @property
    def manual_dir(self) -> Path:
        return self.data_dir / "manual"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


def _path(value: str | None) -> Path | None:
    return Path(value).resolve() if value else None


def from_env(env: dict[str, str] | None = None) -> Settings:
    """Settings from CUEFX_* variables (`env` defaults to os.environ); unset or empty means the default."""
    env = os.environ if env is None else env
    get = lambda key, default: env.get(key) or default  # noqa: E731
    return Settings(
        data_dir=Path(get("CUEFX_DATA_DIR", "data")).resolve(),
        host=get("CUEFX_HOST", "127.0.0.1"),
        port=int(get("CUEFX_PORT", "8000")),
        cors_origins=tuple(o.strip() for o in get("CUEFX_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()),
        max_upload_mb=int(get("CUEFX_MAX_UPLOAD_MB", "500")),
        look_image=_path(env.get("CUEFX_LOOK_IMAGE")),
        looks_dir=_path(env.get("CUEFX_LOOKS_DIR")),
        ltx_overlap_enabled=get("CUEFX_LTX_OVERLAP_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on"),
        comfy_url=env.get("CUEFX_COMFY_URL") or None,
        comfy_token=env.get("CUEFX_COMFY_TOKEN") or None,
        comfy_timeout_s=int(get("CUEFX_COMFY_TIMEOUT_S", "7200")),
    )
