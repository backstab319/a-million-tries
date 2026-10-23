"""Physics-bug audit (episode 14): is a driver getting distance from the sim rather than from driving? Each finalist
with its own driver on test circuits, 60 s from a standing start, every step checked:
  - progress vs distance actually driven (> ~1.05 = cutting the circuit; a racing line clips apexes, so ~1.0-1.04),
    and the biggest single-step jump (progress per step vs speed x dt: a teleport shows up here)
  - tyres on grass (the run-off), and on which side: the inside of a bend (cutting) or the outside (running wide)
  - g forces (lateral, accelerating, braking) and engine power used (kinetic + height energy gained vs the engine's
    watts; the mass is the car's as it burns fuel, plus the wheels' spin as extra mass; the engine's own spin is left
    out, so a real power figure reads a little under 1)
  - bottoming out (skid plank on the ground)

    uv run --group jax python scripts/race_audit.py [--first 40 --n 8] name=runs/<driver>:<design.json> ...
"""

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_car import forward, load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race
from ailearns.games import race_car as rc

G = 9.81
ap = argparse.ArgumentParser()
ap.add_argument("finalists", nargs="+", help="name=runs/<driver>[@<update>]:<design.json>")
ap.add_argument("--first", type=int, default=40)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--seconds", type=float, default=60.0)
a = ap.parse_args()

cfg = race.RaceConfig()
seeds = list(race.TEST_SEEDS)[a.first:a.first + a.n]
game = race.RaceGame(race.make_tracks(seeds, cfg), cfg)
K, steps, grass = len(seeds), int(a.seconds / cfg.dt), cfg.grass_surface[0]


def height(st):
    """The car's height: the centreline's, plus the banking's rise across the road at its offset."""
    t = game.tracks
    return t.z[st.track, st.idx] + st.lat * t.side[st.track, st.idx]


def audit(params, design):
    p = rc.derive(jnp.asarray(design), game.spec)
    keys = jax.random.split(jax.random.PRNGKey(0), K)
    st = jax.vmap(lambda k, t: game.init(k, design=jnp.asarray(design), track=t, random_start=False))(keys, jnp.arange(K))
    st = st._replace(car=st.car._replace(vx=jnp.zeros_like(st.car.vx)))

    def body(st, _):
        mean, _, _ = forward(params, jax.vmap(game.observe)(st))
        new, _ = jax.vmap(game.step)(st, jnp.clip(mean, -1, 1))
        live = ~st.done
        v0, v1 = jnp.hypot(st.car.vx, st.car.vy), jnp.hypot(new.car.vx, new.car.vy)
        m = p.mass - new.car.fuel_used
        z0, z1 = height(st), height(new)
        row = {"live": live, "v": v1, "moved": new.progress - st.progress, "driven": (v0 + v1) / 2 * cfg.dt,
               "grass": jnp.sum(jnp.isclose(new.surface, grass), -1), "lat_g": jnp.abs(new.car.ay) / G,
               "lon_g": new.car.ax / G, "power": ((m + p.spin_wheels) * (v1 ** 2 - v0 ** 2) / 2 + m * G * (z1 - z0)) / cfg.dt / p.power,
               "plank": jnp.sum(new.plank, -1) > 0,  # inside of the bend: same side as the corner's centre
               "inside": new.lat * game.tracks.curvature[new.track, new.idx] > 0}
        keep = st.done
        new = jax.tree.map(lambda x, y: jnp.where(keep.reshape(keep.shape + (1,) * (x.ndim - 1)), x, y), st, new)
        return new, row

    st, rows = jax.lax.scan(body, st, None, length=steps)
    r = {k: np.asarray(v) for k, v in rows.items()}
    live = r["live"]
    jump = np.where(live & (r["driven"] > 0.5), r["moved"] / np.maximum(r["driven"], 1e-6), 0)
    return {"metres": float(np.asarray(st.progress).mean()), "crash": float(np.asarray(st.crashed).mean()),
            "prog/driven": float(r["moved"][live].sum() / r["driven"][live].sum()),
            "worst step": float(jump.max()),
            "extra m/run": float(np.where(live, np.maximum(r["moved"] - r["driven"], 0), 0).sum() / K),
            "extra on grass m/run": float(np.where(live & (r["grass"] >= 2), np.maximum(r["moved"] - r["driven"], 0),
                                                   0).sum() / K),
            "extra on inside grass m/run": float(np.where(live & (r["grass"] >= 2) & r["inside"],
                                                          np.maximum(r["moved"] - r["driven"], 0), 0).sum() / K),
            "2+ off inside %": 100 * float((r["grass"][live] >= 2).sum() and
                                           (r["inside"][live] & (r["grass"][live] >= 2)).sum()
                                           / (r["grass"][live] >= 2).sum()),
            "any grass %": 100 * float((r["grass"][live] > 0).mean()),
            "2+ off %": 100 * float((r["grass"][live] >= 2).mean()),
            "4 off %": 100 * float((r["grass"][live] == 4).mean()),
            "top km/h": 3.6 * float(r["v"][live].max()),
            "max lat g": float(r["lat_g"][live].max()), "max accel g": float(r["lon_g"][live].max()),
            "max brake g": float(-r["lon_g"][live].min()),
            "max power x": float(r["power"][live & (r["v"] > 5)].max()),
            "plank %": 100 * float(r["plank"][live].mean())}


print(f"physics audit on test circuits {seeds[0]}-{seeds[-1]}, {a.seconds:.0f} s from a standing start\n")
for f in a.finalists:
    name, _, rest = f.partition("=")
    driver, _, design_file = rest.rpartition(":")
    path, _, ck = driver.replace("@", ":").partition(":")
    params, _ = load(pick_checkpoint(path, ck or "last"))
    d = np.array(json.load(open(design_file))["history"][-1]["design"], np.float32)
    out = audit(params, d)
    print(f"{name}: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items()))
