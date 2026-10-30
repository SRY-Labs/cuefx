"""Task 36 stop rule: the real 71319 cue's Make keys (stand-in look, energy_orb look) and the golden key do not change.
The cue and the look images are in data/ (not in git): the test is skipped where they are not."""
from pathlib import Path

import pytest

from services.cue.cue import load_cue
from services.cue.make_key import make_key

ROOT = Path(__file__).resolve().parents[1]
CUE = ROOT / "data" / "out" / "interaction" / "71319" / "cue.json"
LOOKS = {"standin_orb": (ROOT / "data" / "look" / "standin_orb.png", "5a7d756a5335"),
         "energy_orb": (ROOT / "data" / "looks" / "energy_orb.png", "02aa1dacc873")}


@pytest.mark.parametrize("name", LOOKS)
def test_the_real_cues_make_key_is_what_it_was(name):
    look, key = LOOKS[name]
    if not (CUE.is_file() and look.is_file()):
        pytest.skip("data/out/interaction/71319/cue.json or the look image is not here")
    assert make_key(load_cue(CUE), look) == key
