"""Shared synthetic cue: a hand that stays at (1000, 600), 200 frames, every field of cue v2.1 valid.

The entrance is off (`entrance.seconds = 0`), so every test written before the entrance existed checks exactly the
old behaviour; test_entrance.py covers the entrance and the 0.5 s default of a cue without the block.
"""
from dataclasses import replace

import pytest

from services.cue.cue import (Behind, Cue, EffectRange, Entrance, Grade, HandSeed, RoutePoint, TrackPoint, Transform, Vanish)

PALM_W = 100.0


def make_cue(**changes) -> Cue:
    cue = Cue(
        clip="synthetic.mov", fps=24, frames=200, plate_size=(1920, 1080), hand="right", hand_seed=HandSeed(60, 1000, 600),
        track=[TrackPoint(f, 1000.0, 600.0, "tracked", PALM_W, 0.9) for f in range(200)],
        transform=Transform(1.0, (0, 0)),
        effect_range=EffectRange(32, 153), canvas_size=(1024, 576), canvas_rect=(0, 0, 1920, 1080),
        anchor_offset=(0, -70), appear_frame=40, release_frame=107, diameter_px=110,
        route=[RoutePoint(107, None, None, 110), RoutePoint(116, 700, 420, 60), RoutePoint(130, 420, 360, 25)],
        vanish=Vanish(128, 131), behind=[Behind(80, 91, "left_hand")], look="energy_orb", seed=42,
        grade=Grade("#9fd4ff", 1.0, 0.6, 0.4, 1.0, 3), entrance=Entrance(0.0),
    )
    for k, v in changes.items():
        setattr(cue, k, v)
    return cue


@pytest.fixture
def cue() -> Cue:
    return make_cue()


def moving_track(speed: float = 0.0) -> list[TrackPoint]:
    """The synthetic hand, moving `speed` px per frame along x (starting at 1000)."""
    return [TrackPoint(f, 1000.0 + speed * f, 600.0, "tracked", PALM_W, 0.9) for f in range(200)]


def with_lost_gap(track: list[TrackPoint], a: int, b: int, jump: float = 900.0) -> list[TrackPoint]:
    """Frames [a, b) are interpolated across a jump of `jump` px: what a lost, reacquired gap looks like."""
    out = list(track)
    for f in range(b, len(out)):
        out[f] = replace(out[f], x=out[f].x - jump)
    for f in range(a, b):
        t = (f - a + 1) / (b - a + 1)
        out[f] = replace(out[f], x=out[a - 1].x + t * (out[b].x - out[a - 1].x), source="interpolated", lost=True)
    return out
