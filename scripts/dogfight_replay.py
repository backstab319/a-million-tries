"""Fly a dogfight brain greedily and record fights (ep21): a summary, and replays for the 3D page.

    uv run python scripts/dogfight_replay.py runs/dogfight-chase1/checkpoints/u00500.pkl --fights 256
    uv run python scripts/dogfight_replay.py <ckpt> --record 4 --name chase1-u500 --level 1

Summary: kill / crash / draw rates, how long fights last, rounds fired, and how the learner flew (range, the angle
of its nose off the target, speed, g). Replays (`--record K`): the first K fights, every 2nd step, into
`web/dogfight/replays/<name>-<k>.json` (positions, nose and lift directions, g, health, bullets in flight).
"""

import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_dogfight import DogfightPPOConfig, sample_action
from ailearns.agents.ppo_rocket import forward
from ailearns.games.dogfight import ALIVE, CRASHED, SHOT, FighterConfig, Fighters


def load(ckpt: str):
    """-> (params, n_layers, policy config, game config) of a checkpoint."""
    ck = pickle.loads(Path(ckpt).read_bytes())
    known = DogfightPPOConfig.__dataclass_fields__
    pcfg = DogfightPPOConfig(**{k: v for k, v in ck["config"].items() if k in known})
    if "trigger" not in ck["config"]:
        pcfg.trigger = "gauss"  # runs before step 2g
    return jax.tree.map(jnp.asarray, ck["params"]), len(ck["config"]["hidden"]), pcfg, FighterConfig(**ck["game"])


def run(ckpt: str, fights: int, level: float, seed: int, bullets: bool = False, sample: bool = False,
        vs: str | None = None):
    """vs: a second brain flies plane 1 (sampled like plane 0) instead of the scripted target."""
    params, n_layers, pcfg, gcfg = load(ckpt)
    game = Fighters(gcfg)
    T = int(game.cfg.time_limit / game.cfg.dt)
    other = load(vs) if vs else None

    def pilot(p, nl, pc, st, i, k):
        mean, _ = forward(p, game.observe(st, i)[None], nl)
        return sample_action(p, mean[0], k, pc) if sample else mean[0]  # greedy: fires iff p(fire) > 0.5

    def act(st, k):
        k0, k1 = jax.random.split(k)
        a1 = pilot(*other[:3], st, 1, k1) if other else game.scripted_action(st, 1)
        return jnp.stack([pilot(params, n_layers, pcfg, st, 0, k0), a1])

    def one(key):
        st0 = game.init(key, level)

        def body(carry, k):
            st, done = carry
            nose, _ = game.nose(st)
            d = st.pos[1] - st.pos[0]
            off = jnp.arccos(jnp.clip(jnp.dot(nose[0], d) / jnp.linalg.norm(d), -1, 1))
            frame = {"pos": st.pos, "nose": nose, "up": st.up, "load": st.load, "health": st.health,
                     "speed": jnp.linalg.norm(st.vel, axis=1), "range": jnp.linalg.norm(d), "off": off,
                     "alive": ~done}
            if bullets:  # big: only for the few fights recorded
                frame.update(ppos=st.ppos, live=st.prounds > 0)
            new, _, d_now = game.step(st, act(st, k))
            st = jax.tree.map(lambda a, b: jnp.where(done, a, b), st, new)
            return (st, done | d_now), frame
        (st, _), frames = jax.lax.scan(body, (st0, jnp.bool_(False)), jax.random.split(jax.random.fold_in(key, 1), T))
        return st, frames

    keys = jax.random.split(jax.random.key(seed), fights)
    ends, frames = jax.jit(jax.vmap(one))(keys)
    ends, frames = jax.device_get(ends), jax.device_get(frames)
    return game, ends, frames


def summary(game, ends, frames):
    c0, c1 = ends.cause[:, 0], ends.cause[:, 1]
    head = {"won": float(((c1 != ALIVE) & (c0 == ALIVE)).mean()), "lost": float(((c0 != ALIVE) & (c1 == ALIVE)).mean()),
            "both_down": float(((c0 != ALIVE) & (c1 != ALIVE)).mean())}
    alive = frames["alive"]
    n = len(c0)
    w = alive.astype(float)
    mean = lambda x: float((x * w).sum() / w.sum())  # noqa: E731
    return {
        "fights": n, "kill": float((c1 == SHOT).mean()), "target_crashed": float((c1 == CRASHED).mean()),
        "crashed": float((c0 == CRASHED).mean()), "draw": float(((c0 == ALIVE) & (c1 == ALIVE)).mean()),
        "fight_s": float(alive.sum(1).mean() * game.cfg.dt), "rounds_fired": float((game.cfg.rounds - ends.ammo[:, 0]).mean()),
        "dealt": float((1 - ends.health[:, 1]).mean()), "range_m": mean(frames["range"]),
        "off_deg": mean(np.degrees(frames["off"])), "speed": mean(frames["speed"][..., 0]),
        "g": mean(frames["load"][..., 0]),
        "share_within_1km_30deg": mean((frames["range"] < 1000) & (np.degrees(frames["off"]) < 30)), **head,
    }


def save_replays(game, ends, frames, k: int, name: str, every: int, out: Path, planes=("learner", "target")):
    out.mkdir(parents=True, exist_ok=True)
    r = lambda a, d=1: np.round(np.asarray(a, float), d).tolist()  # noqa: E731
    for i in range(min(k, len(ends.cause))):
        alive = frames["alive"][i]
        last = int(alive.sum())
        idx = list(range(0, last, every)) + [last - 1]
        steps = []
        for t in idx:
            live = frames["live"][i, t]
            steps.append({"t": round(t * game.cfg.dt, 2), "pos": r(frames["pos"][i, t]), "nose": r(frames["nose"][i, t], 3),
                          "up": r(frames["up"][i, t], 3), "g": r(frames["load"][i, t]), "hp": r(frames["health"][i, t], 2),
                          "v": r(frames["speed"][i, t], 0),
                          "bullets": [r(frames["ppos"][i, t, p][live[p]], 0) for p in (0, 1)]})
        rec = {"name": f"{name}-{i}", "dt": game.cfg.dt * every, "cause": np.asarray(ends.cause[i]).tolist(),
               "planes": list(planes), "steps": steps}
        (out / f"{name}-{i}.json").write_text(json.dumps(rec, separators=(",", ":")))
        print(f"wrote {out / f'{name}-{i}.json'} ({len(steps)} frames, causes {rec['cause']})")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ckpt")
    p.add_argument("--fights", type=int, default=256)
    p.add_argument("--level", type=float, default=None, help="start box (default: the checkpoint's ladder level)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--record", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--vs", default=None, help="a second brain flies the red jet (head to head) instead of the script")
    p.add_argument("--greedy", action="store_true", help="fly the policy's mean action, no noise (default: sampled, as "
                   "in training: the trigger is a threshold on a noisy control, so the mean almost never fires; step 2e)")
    a = p.parse_args()
    ck = pickle.loads(Path(a.ckpt).read_bytes())
    level = a.level if a.level is not None else float(ck.get("level", 1.0))
    game, ends, frames = run(a.ckpt, a.fights, level, a.seed, sample=not a.greedy, vs=a.vs)
    print(json.dumps({"ckpt": a.ckpt, "level": level, "greedy": a.greedy, **{k: round(v, 3) for k, v in summary(game, ends, frames).items()}}))
    if a.record:
        game, ends, frames = run(a.ckpt, a.record, level, a.seed, bullets=True, sample=not a.greedy, vs=a.vs)
        save_replays(game, ends, frames, a.record, a.name or Path(a.ckpt).parent.parent.name, a.every,
                     Path("web/dogfight/replays"), planes=["learner", "ai" if a.vs else "target"])


if __name__ == "__main__":
    main()
