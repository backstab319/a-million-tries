"""Race day (episode 15), step 7: measure first. Eight copies of the ep14 driver (`runs/race-v4-1`, trained alone)
on a race grid, F1-like cars on mediums, a full race (25 laps) on each test circuit: who goes out, when, and how
(into the wall, or wrecked by contact), the pile-ups, how far they get. Nothing trained here.

Two ways to drive it, since it was never shown another car:
- `blind`: its lasers see walls only (what it was trained on); other cars don't exist for it.
- `lasers`: its lasers stop on other cars' bodies too (race day's senses), so a car ahead looks like a wall.
Either way it gets its own view only (the obs it was trained on), shown new tyres like `race_compounds.py` (it backs
off when it sees worn ones; the physics still wears them).

    uv run --group jax python scripts/race_grid.py [--driver runs/race-v4-1] [--n 8] [--laps 25] [--save]
    --save: every car's position, heading and status, every 0.25 s -> out/ep15/grid-<mode>.npz (footage)
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_car import forward, load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race, race_field
from ailearns.games import race_car as rc

ap = argparse.ArgumentParser()
ap.add_argument("--driver", default="runs/race-v4-1")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--laps", type=int, default=25)
ap.add_argument("--minutes", type=float, default=40.0, help="time cap")
ap.add_argument("--save", action="store_true")
a = ap.parse_args()

cfg = race.RaceConfig(time_limit=a.minutes * 60, pit_lane=True)
seeds = list(race.TEST_SEEDS)[:a.n]
F = race_field.RaceField(race.make_tracks(seeds, cfg), laps=a.laps, cfg=cfg)
params, _ = load(pick_checkpoint(a.driver, "last"))
K, N, steps = len(seeds), F.n, int(cfg.time_limit / cfg.dt)
designs = jnp.tile(jnp.asarray(rc.F1_LIKE[0])[None], (N, 1))
RAYS = len(cfg.ray_angles)
WEAR = RAYS + 3 * len(cfg.ahead) + 9 + 12  # where the tyre wear sits in its own view
EVERY = 5  # saved frames: every 0.25 s


def run(mode):
    def drive(f):
        own = jax.vmap(F.game.observe)(f.cars, f.setup)
        if mode == "lasers":
            own = own.at[:, :RAYS].set(jnp.minimum(own[:, :RAYS], F.car_lasers(f) / cfg.ray_range))
        mean, _, _ = forward(params, own.at[:, WEAR:WEAR + 4].set(0.0))
        return jnp.clip(mean, -1, 1)

    def body(f, _):
        new, _ = F.step(f, drive(f))
        new = jax.tree.map(lambda x, y: jnp.where(f.done, x, y), f, new)  # the race is over: hold it
        c = new.cars
        return new, ((new.bump > 0) & ~f.done, c.crashed & ~new.wrecked, new.wrecked, F.laps_done(new), new.finished,
                     c.car.pos, c.car.yaw, new.damage.wing_front)

    def one(track, key):
        f = F.grid(key, track, designs)
        return jax.lax.scan(body, f, None, length=steps)

    f, out = jax.jit(jax.vmap(one))(jnp.arange(K), jax.random.split(jax.random.PRNGKey(0), K))
    return f, [np.asarray(x) for x in out]


def first(mask, axis=1):
    """Seconds at which each mask first turns true (nan: never)."""
    t = (np.argmax(mask, axis) + 1) * cfg.dt
    return np.where(mask.any(axis), t, np.nan)


for mode in ("blind", "lasers"):
    f, (bump, walled, wrecked, laps, finished, pos, yaw, wing) = run(mode)
    out = walled | wrecked  # (K, steps, N)
    t_out = first(out)        # (K, N)
    how_wreck = wrecked[:, -1]
    t_contact = first(bump.any(-1), 1)  # (K,) first contact in each race
    print(f"\n== {mode}: {N} x {a.driver} (F1-like, mediums), {a.laps} laps, {K} test circuits ==")
    piles = 0
    for k in range(K):
        L = float(F.tracks.length[k])
        ts = np.sort(t_out[k][~np.isnan(t_out[k])])
        groups = np.split(ts, np.where(np.diff(ts) > 5.0)[0] + 1) if len(ts) else []
        pile = [g for g in groups if len(g) >= 2]
        piles += len(pile)
        lead = laps[k, -1].max()
        print(f"circuit {seeds[k]} ({L / 1000:.1f} km): out {int(out[k, -1].sum())}/{N} "
              f"(wall {int(walled[k, -1].sum())}, contact wreck {int(how_wreck[k].sum())}), "
              f"first contact {t_contact[k]:.1f} s, outs at {', '.join(f'{x:.0f}' for x in ts) or '-'} s, "
              f"pile-ups {len(pile)} ({'+'.join(str(len(g)) for g in pile) or '-'} cars), "
              f"most laps {lead:.0f}, finished {int(finished[k, -1].sum())}, in contact {bump[k].any(-1).sum() * cfg.dt:.0f} s")
    o = out[:, -1]
    print(f"all: {o.mean():.0%} out, median out at {np.nanmedian(t_out) / 60:.1f} min, {(t_out < 30).mean():.0%} out in "
          f"the first 30 s, {how_wreck.mean():.0%} wrecked by contact, {finished[:, -1].mean():.0%} finished, "
          f"{piles} pile-ups")
    if a.save:
        dest = Path("out/ep15")
        dest.mkdir(parents=True, exist_ok=True)
        s = slice(EVERY - 1, None, EVERY)
        np.savez_compressed(dest / f"grid-{mode}.npz", seeds=np.array(seeds), dt=cfg.dt * EVERY,
                            pos=pos[:, s], yaw=yaw[:, s], walled=walled[:, s], wrecked=wrecked[:, s],
                            laps=laps[:, s], finished=finished[:, s], wing_front=wing[:, s], bump=bump[:, s])
        print(f"-> {dest / f'grid-{mode}.npz'}")
