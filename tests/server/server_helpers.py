"""Helpers and fixtures for the server tests: golden files, a synthetic clip (frame i is flat grey 10 + 10 i), contract
checks. (Not a conftest.py: it would clash with tests/conftest.py, which the older tests import by name.)"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.cue import api
from services.delivery.video import VideoWriter
from services import config
from app import create_app

GOLDEN = api.CONTRACT_DIR / "golden"
FRAMES = 24


def golden(name: str):
    return json.loads((GOLDEN / name).read_text(encoding="utf-8"))


def grey(i: int) -> int:
    return 10 + 10 * i


def make_clip(path: Path, frames: int = FRAMES, size=(640, 360), fps=24) -> Path:
    with VideoWriter(path, size, fps, crf=12) as w:
        for i in range(frames):
            w.write(np.full((size[1], size[0], 3), grey(i), np.uint8))
    return path


def assert_fits(name: str, doc) -> None:
    assert api.schema_errors(name, doc) == [], api.schema_errors(name, doc)


def assert_error(r, status: int, code: str | None = None) -> dict:
    """The response is the contract error shape with a code from error_codes.json."""
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"code", "message"} and body["code"] in api.codes() and body["message"]
    if code:
        assert body["code"] == code
    return body


@pytest.fixture(scope="session")
def golden_length_clip(tmp_path_factory) -> Path:
    """A synthetic clip as long as the golden track (193 frames), flat grey."""
    path = tmp_path_factory.mktemp("synthetic193") / "synthetic193.mp4"
    with VideoWriter(path, (320, 180), 24, crf=30) as w:
        for _ in range(193):
            w.write(np.full((180, 320, 3), 90, np.uint8))
    return path


SEED_SYNTHETIC_CUE = {"f": 5, "x": 660, "y": 660}
SYNTHETIC_TRANSFORM = {"scale": 3.0, "offset": [0, 0]}  # 640x360 -> 1920x1080 is exactly 3x, no padding
SYNTHETIC_TRACK = [{"frame": f, "x": 600.0 + 15 * f, "y": 660.0, "confidence": 0.9, "uncertain": False,
                    "source": "tracked", "palm_w": 120.0} for f in range(FRAMES)]


def synthetic_cue(clip: str = "synthetic.mp4") -> dict:
    """A small valid cue for the 24-frame synthetic clip (640x360, conformed to the 1920x1080 plate; canvas 512x256 on
    the top 1920x960)."""
    cue = api.new_cue(clip, 24, FRAMES, [1920, 1080], "right", SEED_SYNTHETIC_CUE, SYNTHETIC_TRACK, SYNTHETIC_TRANSFORM)
    cue.update(canvas_size=[512, 256], canvas_rect=[0, 0, 1920, 960], anchor_offset=[0, -90], appear_frame=8,
               release_frame=14, diameter_px=120, vanish={"start": 18, "end": 20}, entrance={"seconds": 0.125},
               route=[{"f": 14, "x": None, "y": None, "diameter_px": 120, "easing_in": "linear"},
                      {"f": 17, "x": 1560, "y": 360, "diameter_px": 90, "easing_in": "ease_in"}])
    return api.normalize(cue)


@pytest.fixture(scope="session")
def clip_file(tmp_path_factory) -> Path:
    return make_clip(tmp_path_factory.mktemp("synthetic") / "synthetic.mp4")


@pytest.fixture
def settings(tmp_path) -> config.Settings:
    return config.Settings(data_dir=tmp_path / "data")


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings), raise_server_exceptions=False)


@pytest.fixture(scope="session")
def uploaded_template(tmp_path_factory, clip_file) -> tuple[Path, dict]:
    """The synthetic clip uploaded (and conformed) once per session: (data dir, clip entry)."""
    data = tmp_path_factory.mktemp("uploaded") / "data"
    c = TestClient(create_app(config.Settings(data_dir=data)))
    r = c.post("/api/clips", files={"file": ("synthetic.mp4", clip_file.read_bytes(), "video/mp4")})
    assert r.status_code == 201, r.text
    return data, r.json()


@pytest.fixture
def uploaded(client, settings, uploaded_template) -> dict:
    """The synthetic clip as uploaded to this test's server: the stored upload and its plate are copied from the
    session's upload (the conform is the slow part; test_conform_upload.py covers it on fresh uploads)."""
    data, info = uploaded_template
    clip_id = info["clip_id"]
    shutil.copytree(data / "uploads" / clip_id, settings.uploads_dir / clip_id)
    plate = settings.cache_dir / clip_id / "plate.mp4"
    plate.parent.mkdir(parents=True)
    shutil.copy2(data / "cache" / clip_id / "plate.mp4", plate)
    assert client.get("/api/clips").json() == [info]
    return info
