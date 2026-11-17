"""Record one booster flight step by step, for the renderers (ep18 step 12: the 2D view and Blender both read it).

Pure bookkeeping on top of games.rocket: flies a pilot (a PPO checkpoint, the hand-written hoverslam or the
hand-written entry) from a seed in a world, and keeps every step's state as plain arrays. Never draws.

    rec = record("runs/rocket-ppo25/checkpoints/u15258.pkl", seed=7, where="entry")
    save(rec, "out/rocket/flights/ppo25-s7.npz")
"""

import pickle
from pathlib import Path

import numpy as np

from ailearns.games.rocket import CRASHED, LANDED, TIMEOUT, Boosters, RocketConfig, entry, world

OUTCOMES = {LANDED: "landed", CRASHED: "crashed", TIMEOUT: "timed out"}
# per step: what the renderers need (fin_state 0 working, 1 stuck, 2 dry and limp; engines = how many are lit)
FIELDS = ("time", "x", "y", "vx", "vy", "tilt", "spin", "thrust", "gimbal", "fin", "fin_state", "engines",
          "fuel", "heat", "q", "deck_y", "wind")


def pilot(name: str):
    """'classic' (hand-written hoverslam), 'entry-script' (hand-written entry), or a checkpoint path (mean action).
    -> (policy(b) -> actions, the world the checkpoint trained in or None)."""
    if name == "classic":
        from ailearns.agents.rocket_classic import act
        return act, None
    if name == "entry-script":
        from ailearns.agents.rocket_entry import act
        return act, world("entry")
    import jax.numpy as jnp

    from ailearns.agents.ppo_rocket import forward
    c = pickle.loads(Path(name).read_bytes())
    params = {k: jnp.asarray(v) for k, v in c["params"].items()}
    n_in, layers = params["pi0.w"].shape[0], len(c["config"]["hidden"])
    trained = RocketConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in c["game"].items()})
    return (lambda b: np.clip(np.asarray(forward(params, jnp.asarray(b.obs()[:, :n_in], jnp.float32), layers)[0],
                                         np.float64), -1, 1)), trained


def record(name: str, seed: int, where: str | None = None) -> dict:
    """Fly one booster; -> {field: array per step, 'outcome', 'pilot', 'seed', 'world', 'config'}."""
    act, trained = pilot(name)
    cfg = world(where) if where else (trained or RocketConfig())
    b = Boosters(1, cfg, seed=seed)
    rows, senses, acts = [], [], []  # senses: what a pilot sees each step (b.obs()), for the brain view
    # (render/rocket_brain); acts: what it chose (engine, gimbal, thrusters[, fins]), for the sound (rocket_audio)

    def row():
        senses.append(b.obs()[0])
        c = b.cfg
        state = 1 if b.time[0] >= b.fin_fail_t[0] else 2 if (c.fin_fluid[1] > 0 and b.fluid[0] <= 0) else 0
        used = c.ignitions - b.ignitions[0]
        engines = c.light_engines[min(max(used - 1, 0), len(c.light_engines) - 1)] if b.lit[0] else 0
        speed2 = b.vx[0] ** 2 + b.vy[0] ** 2
        rows.append((b.time[0], b.x[0], b.y[0], b.vx[0], b.vy[0], b.tilt[0], b.spin[0], b.thrust[0], b.gimbal[0],
                     b.fin[0], state, engines, b.fuel[0], b.heat[0], 0.5 * b.air_density(b.y)[0] * speed2,
                     b.deck_y()[0], b.wind_at(b.y, b.time)[0]))
    while True:
        row()
        a = np.asarray(act(b), np.float64)
        acts.append(a[0])
        _, _, done = b.step(a)
        if done[0]:
            row()
            acts.append(a[0])
            break
    a = np.array(rows, dtype=np.float64)
    rec = {f: a[:, i] for i, f in enumerate(FIELDS)}
    rec["obs"] = np.array(senses, dtype=np.float64)
    rec["action"] = np.array(acts, dtype=np.float64)
    rec.update(outcome=OUTCOMES.get(int(b.outcome[0]), "?"), pilot=name, seed=seed, world=where or "",
               dt=float(cfg.dt), pad_radius=float(cfg.pad_radius), length=float(cfg.length),
               diameter=float(cfg.diameter), fins=bool(cfg.fin_area > 0), entry=entry(cfg),
               heat_limit=float(cfg.heat_limit), q_limit=float(cfg.q_limit), bend_limit=float(cfg.bend_limit),
               ship=bool(cfg.pad_radius < RocketConfig().pad_radius))  # the drone ship's deck is smaller than the pad
    return rec


def save(rec: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in rec.items()})
    return path


def load(path) -> dict:
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


if __name__ == "__main__":  # python -m ailearns.games.rocket_flight <pilot> <seed> [world] [out.npz]
    import sys
    args = sys.argv[1:]
    r = record(args[0], int(args[1]), args[2] if len(args) > 2 and args[2] != "-" else None)
    out = args[3] if len(args) > 3 else f"out/rocket/flights/{Path(args[0]).stem}-s{args[1]}.npz"
    print(save(r, out), r["outcome"], f"{r['time'][-1]:.1f} s, {len(r['time'])} steps")
