"""The scored ladder (episode 8): fresh arenas, no overlap between the two sides, picks up where it
stopped; my saved rounds replay from the same start as their AI twin."""

import json
from pathlib import Path

import numpy as np
import pytest

from ailearns.games.tag import TEST_SEEDS
from ailearns.play_tag import LADDER, OUT, PER_RUNG, PRACTICE, ladder


def test_ladder_uses_the_tables_arenas_once_each():
    rounds = ladder("robot", []) + ladder("blob", [])
    assert len(rounds) == 2 * len(LADDER["robot"]) * PER_RUNG
    arenas = [r["arena"] for r in rounds]
    assert len(set(arenas)) == len(arenas)  # every scored round in its own arena
    assert min(arenas) >= TEST_SEEDS[PRACTICE] and max(arenas) <= TEST_SEEDS[-1]  # never a practice arena
    assert all(r["arena"] == TEST_SEEDS[PRACTICE + r["index"]] for r in rounds)  # the table's round, start and all


def test_ladder_resumes_after_played_rounds():
    first = ladder("robot", [])[0]
    log = [{"as": "robot", "vs": first["vs"], "index": first["index"], "scored": True, "won": True},
           {"as": "robot", "vs": first["vs"], "index": first["index"] + 1, "won": False}]  # practice: ignored
    todo = [r for r in ladder("robot", log) if r["result"] is None]
    assert todo[0]["index"] == first["index"] + 1
    assert all(r["result"] is None for r in ladder("blob", log))


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "runs" / "tag-r4").exists(), reason="needs the trained runs (ailearns restore)")
def test_my_round_and_its_ai_twin_start_from_the_same_spot():
    from ailearns import record_tag as rt
    log = [json.loads(line) for line in (OUT / "rounds.jsonl").read_text().splitlines()]
    r = next(r for r in log if r.get("scored") and r["as"] == "robot")
    me = rt.rollout(rt.ME, r["vs"], r["arena"])
    twin = rt.rollout("runs/tag-r4:190", r["vs"], r["arena"], table_start=True)
    assert np.allclose(me["pos"][0], twin["pos"][0])
    assert me["caught"] == r["caught"] and me["steps"] == round(r["seconds"] / me["cfg"].dt)
