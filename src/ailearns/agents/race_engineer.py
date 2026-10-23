"""The engineer (episode 14): judging car designs with the driver, and evolving them.

`judge` is how every design is scored: the frozen driver drives it on several circuits from a standing start for a
fixed time, best guesses only (no exploration noise), all designs and circuits at once; the score is the average
distance along the track (a crash ends that run where it happened, so it scores what it managed). The same driver
judges every design, so a design scores well only if it's fast *and* this driver can handle it (plan section 8,
risk 2: the final check gives each finalist its own fine-tuned driver).

`evolve` is the engineer: evolution strategies over the 13 dials, scored by `judge` minus a penalty for overspending.
"""

import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_car import forward
from ailearns.games import race
from ailearns.games import race_car as rc


@partial(jax.jit, static_argnames=("game", "steps"))
def _drive_all(params, game: race.RaceGame, designs, tracks, steps: int):
    P, K = designs.shape[0], tracks.shape[0]
    d = jnp.repeat(designs, K, axis=0)                     # (P*K, D): each design on every circuit
    tr = jnp.tile(tracks, P)
    keys = jax.random.split(jax.random.PRNGKey(0), P * K)
    st = jax.vmap(lambda k, dd, t: game.init(k, design=dd, track=t, random_start=False))(keys, d, tr)
    st = st._replace(car=st.car._replace(vx=jnp.zeros_like(st.car.vx)))  # standing start whatever cfg.start_speed

    def body(st, _):
        mean, _, _ = forward(params, jax.vmap(game.observe)(st))
        new, _ = jax.vmap(game.step)(st, jnp.clip(mean, -1, 1))
        keep = st.done  # finished runs stay where they ended
        return jax.tree.map(lambda a, b: jnp.where(keep.reshape(keep.shape + (1,) * (a.ndim - 1)), a, b), st, new), None

    st, _ = jax.lax.scan(body, st, None, length=steps)
    return st.progress.reshape(P, K), st.crashed.reshape(P, K)


def judge(params, game: race.RaceGame, designs, tracks=None, seconds: float = 60.0) -> dict:
    """designs (P, len(DESIGN)) -> {"metres": (P,) average distance, "crash": (P,) share of circuits crashed on,
    "per_track": (P, K) metres}. tracks: indices into game.tracks (default: all of them)."""
    designs = jnp.asarray(designs, jnp.float32)
    tracks = jnp.arange(game.tracks.length.shape[0]) if tracks is None else jnp.asarray(tracks)
    metres, crashed = _drive_all(params, game, designs, tracks, int(seconds / game.cfg.dt))
    metres, crashed = np.asarray(metres), np.asarray(crashed)
    return {"metres": metres.mean(1), "crash": crashed.mean(1), "per_track": metres}


# ── evolution ────────────────────────────────────────────────────────────────

def to_design(u):
    """[-1, 1] per dial -> design numbers."""
    return rc.DESIGN_LOW + (jnp.clip(u, -1, 1) + 1) / 2 * (rc.DESIGN_HIGH - rc.DESIGN_LOW)


def score(params, game, designs, tracks, seconds, overspend_m: float) -> tuple[np.ndarray, dict]:
    """The engineer's score for each design: metres the driver covers, minus overspend_m per $1k over budget."""
    out = judge(params, game, designs, tracks, seconds)
    cost = np.array([float(rc.cost_k(d)) for d in np.asarray(designs)])
    return out["metres"] - overspend_m * np.maximum(cost - rc.BUDGET_K, 0.0), {**out, "cost": cost}


def evolve(params, game, out_dir, gens: int = 60, pop: int = 32, sigma: float = 0.15, lr: float = 0.05,
           circuits: int = 8, seconds: float = 60.0, overspend_m: float = 10.0, seed: int = 0, start=None,
           anneal: float = 1.0) -> list:
    """OpenAI-ES over the design (each dial in [-1, 1]): mirrored pairs around the current design, all judged on the
    same circuits that generation (a fresh draw each generation, so it can't tune to a few circuits), ranked, and the
    design moves along the ranked changes (Adam). Writes history.json (every generation's design, its score, cost and
    the best member) to out_dir after each generation: the morph footage comes from it. anneal < 1: the search width and
    the step size shrink linearly to that share of their start by the last generation (steadier near the end)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    D = len(rc.DESIGN)
    mu = np.zeros(D) if start is None else np.asarray(start, float)
    m, v = np.zeros(D), np.zeros(D)
    n_tracks = game.tracks.length.shape[0]
    history = []
    sigma0, lr0 = sigma, lr
    for g in range(gens):
        shrink = 1 - (1 - anneal) * g / max(gens - 1, 1)
        sigma, lr = sigma0 * shrink, lr0 * shrink
        eps = rng.standard_normal((pop // 2, D))
        members = np.concatenate([mu + sigma * eps, mu - sigma * eps, mu[None]])  # last one: the design itself
        tracks = rng.choice(n_tracks, circuits, replace=False)
        designs = np.asarray(to_design(jnp.asarray(members)))
        s, info = score(params, game, designs, tracks, seconds, overspend_m)
        ranks = np.argsort(np.argsort(s[:-1])) / (len(s) - 2) - 0.5          # members only, centred ranks
        r_plus, r_minus = ranks[: pop // 2], ranks[pop // 2:]
        grad = ((r_plus - r_minus)[:, None] * eps).sum(0) / (pop * sigma)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        mu = np.clip(mu + lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8), -1, 1)
        best = int(np.argmax(s[:-1]))
        history.append({"gen": g, "design": designs[-1].round(4).tolist(), "score": round(float(s[-1]), 1),
                        "metres": round(float(info["metres"][-1]), 1), "crash": float(info["crash"][-1]),
                        "cost_k": round(float(info["cost"][-1]), 1), "members_mean": round(float(s[:-1].mean()), 1),
                        "best": {"design": designs[best].round(4).tolist(), "score": round(float(s[best]), 1)},
                        "tracks": tracks.tolist()})
        (out_dir / "history.json").write_text(json.dumps({"dials": rc.DESIGN_NAMES, "budget_k": rc.BUDGET_K,
                                                          "history": history}, indent=1))
        h = history[-1]
        print(f"gen {g:3d}  design: {h['metres']:6.0f} m  ${h['cost_k']:5.0f}k  crash {h['crash']:.2f}  "
              f"members avg {h['members_mean']:6.0f}  best {h['best']['score']:6.0f}", flush=True)
    return history


if __name__ == "__main__":
    import argparse
    from ailearns.agents.ppo_car import load
    from ailearns.checkpoints import pick_checkpoint
    ap = argparse.ArgumentParser(description="the engineer: evolve a car design for a driver (episode 14)")
    ap.add_argument("driver", help="runs/<driver> or runs/<driver>:<checkpoint>")
    ap.add_argument("out", help="where to write history.json, e.g. runs/eng-1")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--circuits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anneal", type=float, default=1.0, help="search width and step end at this share (1 = fixed)")
    ap.add_argument("--start-from", default="", help="start from another run's last design (its history.json)")
    a = ap.parse_args()
    path, _, ck = a.driver.partition(":")
    params, meta = load(pick_checkpoint(path, ck or "last"))
    cfg = race.RaceConfig()
    game = race.RaceGame(race.make_tracks(list(race.TRAIN_SEEDS), cfg), cfg)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    meta_out = {**vars(a), "driver_steps": meta["steps"], "game": "race-engineer"}
    (Path(a.out) / "config.json").write_text(json.dumps(meta_out))
    start = None
    if a.start_from:  # the last design of an earlier run, back in [-1, 1] per dial
        d = np.array(json.loads(Path(a.start_from).read_text())["history"][-1]["design"])
        start = 2 * (d - rc.DESIGN_LOW) / (rc.DESIGN_HIGH - rc.DESIGN_LOW) - 1
    evolve(params, game, a.out, gens=a.gens, pop=a.pop, circuits=a.circuits, seed=a.seed, start=start,
           anneal=a.anneal)
