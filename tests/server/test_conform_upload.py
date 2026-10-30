"""Task 23: uploads are conformed to the 1920x1080, 24 fps plate; everything downstream reads the plate."""
import io
import subprocess
from pathlib import Path

import numpy as np
import pytest

from server_helpers import assert_error, client, settings  # noqa: F401
from services.cue import api
from services.analysis import conform
from services.make.layout import find_ffmpeg
from services.delivery.video import probe, read_all

RED, BLUE = np.array([0, 0, 255]), np.array([255, 0, 0])  # BGR
SECONDS = 2


def make_source(path: Path, w: int, h: int, fps: int, seconds: float = SECONDS, rotation: int | None = None) -> Path:
    """A Rec.709-tagged clip: top half red, bottom half blue. With `rotation`, the file stores w x h and carries a
    display-rotation tag (a phone clip)."""
    plain = path if rotation is None else path.with_name("unrotated_" + path.name)
    vf = (f"drawbox=x=0:y=ih/2:w=iw:h=ih/2:color=blue:t=fill,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p")
    ffmpeg = find_ffmpeg()
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c=red:s={w}x{h}:r={fps}:d={seconds}",
                    "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709", str(plain)], check=True)
    if rotation is not None:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-display_rotation:v:0", str(rotation), "-i", str(plain),
                        "-c", "copy", str(path)], check=True)
    return path


def upload(client, path: Path):
    return client.post("/api/clips", files={"file": (path.name, path.read_bytes(), "video/mp4")})


def plate_of(client, clip_id: str) -> Path:
    return client.app.state.store.plate(clip_id)


def close(pixel, colour, tol=14) -> bool:
    return bool(np.all(np.abs(pixel.astype(int) - colour) <= tol))


# (name, stored size, fps, rotation tag, displayed size, frames) -- frames = round(SECONDS x 24)
CASES = [
    ("hd720_30", (1280, 720), 30, None, (1280, 720)),
    pytest.param("portrait_30", (1080, 1920), 30, None, (1080, 1920), marks=pytest.mark.slow),
    pytest.param("uhd_25", (3840, 2160), 25, None, (3840, 2160), marks=pytest.mark.slow),
    ("rotated_30", (720, 1280), 30, 90, (1280, 720)),
]


@pytest.mark.parametrize("name, stored, fps, rotation, shown", CASES, ids=["hd720_30", "portrait_30", "uhd_25", "rotated_30"])
def test_upload_is_conformed_to_the_plate(client, tmp_path, name, stored, fps, rotation, shown):
    src = make_source(tmp_path / f"{name}.mp4", *stored, fps, rotation=rotation)
    original = src.read_bytes()
    r = upload(client, src)
    assert r.status_code == 201, r.text
    info = r.json()
    frames_expected = round(SECONDS * 24)
    assert (info["fps"], info["frames"], info["plate_size"]) == (24.0, frames_expected, [1920, 1080])
    assert client.get("/api/clips").json() == [info]

    # the original is stored unchanged; the plate is a separate file in the cache
    store = client.app.state.store
    assert store.path(info["clip_id"]).read_bytes() == original
    plate = plate_of(client, info["clip_id"])
    assert plate == store.s.cache_dir / info["clip_id"] / "plate.mp4"
    p = probe(plate)
    assert (p.width, p.height, p.fps) == (1920, 1080, 24.0)
    frames = read_all(plate)
    assert len(frames) == frames_expected

    # the transform maps the displayed picture onto the plate: content inside its rectangle, black outside
    tf = store.transform(info["clip_id"])
    scale, (sw, sh), (ox, oy) = conform.fit(*shown)
    assert tf == {"scale": round(scale, 6), "offset": [ox, oy]}
    assert abs(sw - shown[0] * tf["scale"]) <= 1 and abs(sh - shown[1] * tf["scale"]) <= 1
    f = frames[frames_expected // 2]
    inner = f[oy + 4: oy + sh - 4, ox + 4: ox + sw - 4]
    assert inner.size
    outside = np.ones(f.shape[:2], bool)
    outside[max(oy - 2, 0): oy + sh + 2, max(ox - 2, 0): ox + sw + 2] = False
    assert outside.sum() == 0 or f[outside].max() <= 20  # padding is black
    cx, cy = ox + sw // 2, oy + sh // 2
    if rotation is None:   # top half red, bottom half blue
        assert close(f[oy + sh // 4, cx], RED) and close(f[oy + 3 * sh // 4, cx], BLUE)
    else:                  # the rotation was applied before scaling: the halves are now left and right
        left, right = f[cy, ox + sw // 4], f[cy, ox + 3 * sw // 4]
        assert {close(left, RED), close(right, BLUE)} == {True} or {close(left, BLUE), close(right, RED)} == {True}

    # the frames route and the proxy video are the plate
    jpg = client.get(f"/api/clips/{info['clip_id']}/frames/{frames_expected - 1}.jpg")
    assert jpg.status_code == 200
    assert_error(client.get(f"/api/clips/{info['clip_id']}/frames/{frames_expected}.jpg"), 400, "input_invalid")
    proxy = client.get(f"/api/clips/{info['clip_id']}/video")
    out = tmp_path / "proxy.mp4"
    out.write_bytes(proxy.content)
    q = probe(out)
    assert (q.width, q.height, q.fps) == (1920, 1080, 24.0) and len(read_all(out)) == frames_expected


def test_a_clip_that_already_is_a_plate_is_used_as_is(client, tmp_path):
    src = make_source(tmp_path / "plate.mp4", 1920, 1080, 24)
    info = upload(client, src).json()
    assert info["plate_size"] == [1920, 1080] and info["fps"] == 24.0
    store = client.app.state.store
    assert plate_of(client, info["clip_id"]) == store.path(info["clip_id"])  # no copy, no re-encode
    assert not (store.s.cache_dir / info["clip_id"] / "plate.mp4").exists()
    assert store.transform(info["clip_id"]) == {"scale": 1.0, "offset": [0, 0]}


def test_the_plate_is_cached_by_content(client, tmp_path):
    src = make_source(tmp_path / "hd.mp4", 1280, 720, 30)
    info = upload(client, src).json()
    plate = plate_of(client, info["clip_id"])
    stamp = plate.stat().st_mtime_ns
    again = upload(client, src)
    assert again.status_code == 200 and again.json() == info
    assert plate_of(client, info["clip_id"]).stat().st_mtime_ns == stamp
    # a new app on the same data folder finds it
    from fastapi.testclient import TestClient
    from app import create_app

    c2 = TestClient(create_app(client.app.state.store.s))
    assert c2.get("/api/clips").json() == [info] and plate.stat().st_mtime_ns == stamp


def test_analyze_and_the_cue_use_the_plate(client, tmp_path, monkeypatch):
    seen = []

    def fake(clip_path, seed, model=None, **kw):
        seen.append((clip_path, probe(clip_path)))
        pts = [{"frame": f, "x": 960.0, "y": 540.0, "confidence": 0.9, "uncertain": False, "source": "tracked",
                "palm_w": 100.0} for f in range(48)]
        return api.analyze_result(api.points_from_track(pts))

    monkeypatch.setattr(api, "analyze", fake)
    info = upload(client, make_source(tmp_path / "portrait.mp4", 1080, 1920, 30)).json()
    r = client.post("/api/cue/new", json={"clip_id": info["clip_id"], "hand": "right",
                                          "hand_seed": {"frame": 5, "x": 960, "y": 540}})
    assert r.status_code == 200, r.text
    cue = r.json()
    assert seen[0][0] == plate_of(client, info["clip_id"]) and (seen[0][1].width, seen[0][1].height) == (1920, 1080)
    assert (cue["plate_size"], cue["fps"], cue["frames"], cue["clip"]) == ([1920, 1080], 24, 48, "portrait.mp4")
    assert cue["transform"] == {"scale": 0.5625, "offset": [656, 0]}
    assert api.schema_errors("cue", cue) == []


def test_the_frame_limit_is_checked_after_conforming(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MAX_CLIP_FRAMES", 40)
    src = make_source(tmp_path / "long.mp4", 1280, 720, 30)           # 60 frames -> 48 conformed
    store = client.app.state.store
    assert_error(upload(client, src), 400, "clip_too_long")           # refused from the duration, before conforming
    assert not store.s.cache_dir.exists() or not any(store.s.cache_dir.iterdir())
    monkeypatch.setattr(conform, "estimated_frames", lambda s: 0)     # as if the duration was unknown
    body = assert_error(upload(client, src), 400, "clip_too_long")
    assert "48" in body["message"]                                    # the conformed count, not the 60 of the file
    assert client.get("/api/clips").json() == []                      # nothing is kept
    assert not any(store.s.uploads_dir.iterdir()) and not (store.s.cache_dir / "plate").exists()
    monkeypatch.setattr(api, "MAX_CLIP_FRAMES", 48)                   # exactly the limit is accepted
    assert upload(client, src).status_code == 201


def test_a_file_that_is_not_a_video_is_clip_unreadable(client):
    r = client.post("/api/clips", files={"file": ("junk.mp4", io.BytesIO(b"not a video" * 100), "video/mp4")})
    assert_error(r, 400, "clip_unreadable")
