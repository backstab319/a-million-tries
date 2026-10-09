"""The race car game: drivable tracks, sane physics, lasers that measure the right thing."""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

import pytest

from ailearns.games.car import CarConfig, CarGame, make_track, make_tracks

CFG = replace(CarConfig(), time_limit=120.0)  # long enough for a slow, careful lap
SEEDS = list(range(8)) + [100_000, 100_001]


@pytest.fixture(scope="module")
def game():
    return CarGame(make_tracks(SEEDS, CFG), CFG)


@pytest.mark.parametrize("seed", SEEDS)
def test_tracks_are_drivable(seed):
    t = make_track(seed, CFG)
    turn = (np.roll(t["heading"], -1) - t["heading"] + np.pi) % (2 * np.pi) - np.pi
    assert 1 / np.abs(turn / (t["length"] / CFG.points)).max() >= 0.8 * CFG.width  # no impossible corner
    for wall in ("left", "right"):  # each wall stays a track-width away from the centreline
        d = np.linalg.norm(t[wall][:, None] - t["center"][None], axis=2).min(axis=1)
        assert d.min() > 0.45 * CFG.width


def test_lasers_see_the_walls(game):
    st = game.init(jax.random.PRNGKey(0), track=1, random_start=False)
    d = np.asarray(game.lasers(st))
    side = CFG.ray_angles.index(90), CFG.ray_angles.index(-90)
    assert all(abs(d[i] - CFG.width / 2) < 1.5 for i in side)  # on the centreline: half a width each way
    assert (d > 0).all() and (d <= CFG.ray_range).all()


def _drive(game, track, policy, steps=3000):
    st = game.init(jax.random.PRNGKey(1), track=track, random_start=False)
    step, c = jax.jit(game.step), np.asarray(game.tracks.center[track])
    total = 0.0
    for _ in range(steps):
        st, r = step(st, jnp.asarray(policy(st, c), jnp.float32))
        total += float(r)
        if bool(st.done):
            break
    return st, total


def _pursuit(speed):
    def policy(st, c):  # aim at a point a few metres ahead on the centreline
        tgt = c[(int(st.idx) + 5) % len(c)] - np.asarray(st.pos)
        err = (np.arctan2(tgt[1], tgt[0]) - float(st.heading) + np.pi) % (2 * np.pi) - np.pi
        return [np.clip(2 * err, -1, 1), np.clip((speed - float(st.speed)) * 0.5, -1, 1)]
    return policy


@pytest.mark.parametrize("track", [0, 1, 8])
def test_a_careful_driver_finishes(game, track):
    st, total = _drive(game, track, _pursuit(10.0))
    assert not bool(st.crashed)
    assert float(st.progress) > 0.9 * float(game.tracks.length[track])  # most of a lap in 120 s
    assert total > 0


def test_flat_out_without_steering_crashes(game):
    st, total = _drive(game, 0, lambda st, c: [0.0, 1.0])
    assert bool(st.crashed) and total < 0


def test_too_fast_for_the_hairpin_crashes(game):
    """Grip is limited: the same careful line at 3x the speed can't make the corners."""
    st, _ = _drive(game, 0, _pursuit(30.0))
    assert bool(st.crashed)
