"""Race day (episode 15): what a pit stop costs. One car alone on test circuits, the cautious scripted driver (no AI),
3 laps without a stop and 3 laps with one (called in on lap 2: the lane, the limiter, the box, out again).
(ep16 step 9b) `--pace vsc`: the whole race at the VSC's minimum time (every car on the track that slow, so the stop's
cost is what the lane loses against it); `--pace sc`: at the Safety Car's speed (`sc_share` x the minimum's).

    uv run --group jax python scripts/race_pitstop.py [--n 8] [--pace green|vsc|sc]
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.games import race, race_control, race_field
from ailearns.games import race_car as rc
from ailearns.games.race_checks import look_ahead, test_driver

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--pace", choices=["green", "vsc", "sc"], default="green")
a = ap.parse_args()

cfg = race.RaceConfig(time_limit=900.0, pit_lane=True)
seeds = list(race.TEST_SEEDS)[:a.n]
F = race_field.RaceField(race.make_tracks(seeds, cfg), n_cars=1, laps=3, cfg=cfg, contact=False, slipstream=False)
g, K = F.game, len(seeds)
ctl = race_control.RaceControl(F)
share = {"green": 0.0, "vsc": 1.0, "sc": ctl.sc_share}[a.pace]  # x the minimum time's speed (0: no cap)


def driver(f, since):
    """The scripted driver; called in, it takes the lane through the pit zone and stops on its box mark."""
    st = jax.tree.map(lambda x: x[0], f.cars)
    s = g.from_pit(st.track, st.idx)
    out_lap = (f.stops[0] > 0) & (since < cfg.pit_exit - F.box_s[0])  # `since`: metres driven since the stop
    lane = ((f.box_call[0] & (s > -cfg.pit_entry) & (s < cfg.pit_exit))  # called in: into the lane ...
            | out_lap)                                                   # ... and stay in it to the exit
    ease = jnp.clip(jnp.minimum((s + cfg.pit_entry) / 60.0, (cfg.pit_exit - s) / 100.0), 0.0, 1.0)  # in, out
    v = jnp.hypot(st.car.vx, st.car.vy)
    to_box = F.box_s[0] - s
    stop_v = jnp.sqrt(2 * 4.0 * jnp.maximum(to_box, 0.0))  # ease to a stop on the mark (4 m/s^2)
    stopping = f.box_call[0] & g.in_pit_lane(st) & (to_box < 60)
    act = test_driver(g, st, offset=jnp.where(lane, F.pit_line(to_box - look_ahead(st.car), stopping) * ease, 0.0))
    entry_v = jnp.sqrt(cfg.pit_limit ** 2 + 2 * 6.0 * jnp.maximum(-cfg.pit_entry + 30 - s, 0.0))  # 80 by the entry
    braking = f.box_call[0] & (s < -cfg.pit_entry + 30) & (s > -cfg.pit_entry - 600)
    pedal = jnp.where(braking, jnp.minimum(act[1], jnp.clip(0.5 * (entry_v - v), -0.5, 1.0)), act[1])
    pedal = jnp.where(stopping, jnp.clip(0.5 * (stop_v - v), -0.5, 0.3), pedal)
    if share:  # off the pit lane: no faster than the minimum time (the delta, as race control counts it)
        cap = share / ctl.min_pace[st.track, st.idx]
        pedal = jnp.where(g.in_pit_lane(st), pedal, jnp.minimum(pedal, jnp.clip(0.5 * (cap - v), -0.5, 1.0)))
    return jnp.stack([act[0], pedal])[None]


def race_one(track, stop):
    f = F.grid(jax.random.PRNGKey(0), track, jnp.asarray(rc.F1_LIKE[0])[None])

    def body(carry, _):
        f, at_stop = carry
        s = g.from_pit(f.cars.track[0], f.cars.idx[0])
        before = (s > -cfg.pit_entry - 700) & (s < -cfg.pit_entry - 400)  # called in well before the entry
        call = stop & (F.laps_done(f)[0] == 1) & (f.stops[0] == 0) & before  # on lap 2
        f = f._replace(box_call=f.box_call | call)
        new = F.step(f, driver(f, f.cars.progress[0] - at_stop))[0]
        at_stop = jnp.where(new.stops[0] > f.stops[0], new.cars.progress[0], at_stop)
        return (new, at_stop), None
    return jax.lax.scan(body, (f, jnp.asarray(jnp.inf)), None, int(cfg.time_limit / cfg.dt))[0][0]


run = jax.jit(jax.vmap(race_one, in_axes=(0, None)))
tr = jnp.arange(K)
plain, pitted = run(tr, False), run(tr, True)
t0, t1 = np.asarray(plain.finish_t[:, 0]), np.asarray(pitted.finish_t[:, 0])
stops, out = np.asarray(pitted.stops[:, 0]), np.asarray(pitted.cars.crashed[:, 0] | plain.cars.crashed[:, 0])
ok = np.isfinite(t0) & np.isfinite(t1) & (stops == 1) & ~out
print(f"scripted driver, F1-like car, {K} test circuits, 3 laps, pace: {a.pace}")
for k in range(K):
    print(f"  circuit {seeds[k]}: {t0[k]:.1f} s, with a stop {t1[k]:.1f} s (stops {stops[k]}, out {out[k]})")
loss = t1[ok] - t0[ok]
print(f"a stop costs {loss.mean():.1f} s (range {loss.min():.1f}-{loss.max():.1f}) on {ok.sum()}/{K} circuits; "
      f"{F.tyre_s:.1f} s of it in the box")
