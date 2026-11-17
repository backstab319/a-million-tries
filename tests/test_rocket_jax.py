"""The JAX booster must fly exactly like the numpy reference (starts synced, same actions)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax import enable_x64  # noqa: E402

from ailearns.agents.rocket_classic import act  # noqa: E402
from ailearns.games.rocket import FLYING, LANDED, Boosters, RocketConfig, entry, world  # noqa: E402
from ailearns.games.rocket_jax import JaxBoosters, State  # noqa: E402

FIELDS = State._fields


def _sync(ref: Boosters, dtype) -> State:
    return State(**{f: jnp.asarray(getattr(ref, f), jnp.int32 if f in ("ignitions", "outcome") else
                                   (bool if f == "lit" else dtype)) for f in FIELDS})


def _actions(ref: Boosters, rng, mode: str) -> np.ndarray:
    if mode == "random":
        a = rng.uniform(-1, 1, (ref.n, 3 + (ref.cfg.fin_area > 0)))
        a[:, 0] = np.where(rng.random(ref.n) < 0.9, np.abs(a[:, 0]), -np.abs(a[:, 0]))  # mostly lit, some cuts
        return a
    if entry(ref.cfg):  # (11) from 70 km: the hand-written entry, then the classic
        from ailearns.agents.rocket_entry import act as entry_act
        return entry_act(ref)
    return act(ref)  # no noise: a throttle nudged below 0 shuts the engine for good (one ignition)


@pytest.mark.parametrize("where", ["pad", "storm", "storm+fins", "storm+fins+fluid", "storm+fins+faults", "entry"])
@pytest.mark.parametrize("mode", ["classic", "random"])
def test_jax_booster_matches_numpy(mode, where):
    n = 64
    ref, game = Boosters(n, config=world(where), seed=1), JaxBoosters(world(where))
    rng = np.random.default_rng(3)
    ends = 0
    # the storm: the wind's power law (y ** 0.14) rounds differently in numpy and XLA in the last bit, and a gusty
    # minute of flight grows that to ~1e-9; anything real would be far bigger
    tol = 1e-9 if where == "pad" else 1e-7
    steps = int(ref.cfg.max_time / ref.cfg.dt) + 5
    if where == "entry":  # (11) body lift makes it unstable in attitude, and the JAX copy flies the numpy copy's
        steps = 2000      # actions open-loop, so last-bit rounding grows ~1000x in 10 s once the landing burn starts
        # (agrees to 1e-10 until then): compare the entry burn and the glide, the first 100 s
    with enable_x64():
        step = jax.jit(jax.vmap(game.step))
        observe = jax.jit(jax.vmap(game.observe))
        st = _sync(ref, jnp.float64)
        for _ in range(steps):
            a = _actions(ref, rng, mode)
            _, r1, d1 = ref.step(a)
            st, r2, d2 = step(st, jnp.asarray(a))
            np.testing.assert_allclose(np.asarray(r2), r1, rtol=tol, atol=tol)
            np.testing.assert_array_equal(np.asarray(d2), d1)
            for f in FIELDS:
                np.testing.assert_allclose(np.asarray(getattr(st, f)), getattr(ref, f), rtol=tol, atol=tol, err_msg=f)
            # atol 1e-6: the square-root senses (7b) magnify 1e-12 near zero (sqrt(1e-12) = 1e-6); states are 1e-9
            np.testing.assert_allclose(np.asarray(observe(st)), ref.obs(), rtol=1e-9, atol=1e-6)
            ends += int((d1 & (ref.outcome != FLYING)).sum())
            if d1.all():
                break
    if where == "entry":
        return
    assert d1.all() and ends > 0
    if mode == "classic":
        # the comparison covered real landings, not only crashes (the classic lands ~15% in the storm)
        assert (ref.outcome == LANDED).mean() > (0.5 if where == "pad" else 0.0)


def test_float32_lands_like_float64():
    """Training runs in float32: the classic controller must land the same burns there (chaos aside)."""
    n = 200
    ref, game = Boosters(n, seed=5), JaxBoosters()
    step = jax.jit(jax.vmap(game.step))
    st = _sync(ref, jnp.float32)
    shadow = Boosters(n, seed=0)  # a numpy copy of the float32 state, so the controller can read it
    for _ in range(int(ref.cfg.max_time / ref.cfg.dt) + 5):
        for f in FIELDS:
            getattr(shadow, f)[:] = np.asarray(getattr(st, f))
        ref.step(act(ref))
        st, _, done = step(st, jnp.asarray(act(shadow)))
        if done.all() and (ref.outcome != FLYING).all():
            break
    agree = (np.asarray(st.outcome) == ref.outcome).mean()
    assert agree >= 0.95, agree


def test_reset_draws_from_the_start_box():
    cfg, game = RocketConfig(), JaxBoosters()
    st = jax.vmap(game.init)(jax.random.split(jax.random.key(0), 1000))
    for f, (lo, hi) in [("x", cfg.start_x), ("y", cfg.start_height), ("vx", cfg.start_vx),
                        ("tilt", cfg.start_tilt), ("fuel", cfg.start_fuel)]:
        v = np.asarray(getattr(st, f))
        assert lo <= v.min() and v.max() <= hi and v.std() > 0, f
    assert (-np.asarray(st.vy) >= cfg.start_fall[0]).all() and not np.asarray(st.lit).any()
    done = jnp.arange(1000) % 2 == 0
    st2 = jax.vmap(game.reset_if)(st, done, jax.random.split(jax.random.key(1), 1000))
    same = np.asarray(st2.x) == np.asarray(st.x)
    assert same[1::2].all() and not same[::2].any()
