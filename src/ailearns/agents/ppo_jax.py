"""PPO again, in JAX: same algorithm and settings as ppo.py, but compiled end to end.

One whole update (play T moves in every game, work out advantages, a few
epochs of minibatch learning) is a single compiled function, so the chip never
waits for Python. It runs data-parallel over every device it finds (the 8
cores of a TPU, or both GPUs of Kaggle's "T4 x2"): each device plays its share
of the games and the gradients are averaged.

The network is read off `ppo.SnakeNet` (same layers, same parameter names,
same initial weights), so checkpoints are ordinary torch files: rendering and
analysis load them exactly like torch-trained ones.
"""

import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ailearns.agents.ppo import PPOConfig, SnakeNet, _git_version, obs_shape
from ailearns.games.snake import SnakeConfig
from ailearns.games.snake_jax import JaxSnake


# ── the network ─────────────────────────────────────────────────────────────

def net_spec(net: SnakeNet) -> tuple:
    """SnakeNet's body as a list of layers (kind, torch parameter name, stride, padding)."""
    import torch.nn as nn
    spec = []
    for i, m in enumerate(net.body):
        if isinstance(m, nn.Conv2d):
            spec.append(("conv", f"body.{i}", m.stride[0], m.padding[0]))
        elif isinstance(m, nn.Linear):
            spec.append(("linear", f"body.{i}", 0, 0))
        elif isinstance(m, nn.ReLU):
            spec.append(("relu", "", 0, 0))
        elif isinstance(m, nn.Flatten):
            spec.append(("flatten", "", 0, 0))
        else:
            raise ValueError(m)
    return tuple(spec)


def forward(spec: tuple, params: dict, x: jax.Array):
    """Same maths as SnakeNet.forward. x: (batch, channels, h, w). Returns (logits, value)."""
    for kind, name, stride, pad in spec:
        if kind == "conv":
            x = lax.conv_general_dilated(x, params[name + ".weight"], (stride, stride), [(pad, pad), (pad, pad)],
                                         dimension_numbers=("NCHW", "OIHW", "NCHW"))
            x = x + params[name + ".bias"][None, :, None, None]
        elif kind == "linear":
            x = x @ params[name + ".weight"].T + params[name + ".bias"]
        elif kind == "relu":
            x = jax.nn.relu(x)
        else:
            x = x.reshape(x.shape[0], -1)
    logits = x @ params["policy.weight"].T + params["policy.bias"]
    value = (x @ params["value.weight"].T + params["value.bias"])[:, 0]
    return logits, value


def to_torch(params: dict) -> dict:
    import torch
    return {k: torch.from_numpy(np.array(v)) for k, v in params.items()}


# ── Adam, as torch.optim.Adam does it ───────────────────────────────────────

def adam_init(params: dict) -> dict:
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": jax.tree.map(jnp.zeros_like, params), "t": jnp.zeros((), jnp.int32)}


def adam_step(params, opt, grads, lr, b1=0.9, b2=0.999, eps=1e-5):
    t = opt["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, opt["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g * g, opt["v"], grads)
    c1, c2 = 1 - b1 ** t, 1 - b2 ** t
    params = jax.tree.map(lambda p, m, v: p - lr / c1 * m / (jnp.sqrt(v) / jnp.sqrt(c2) + eps), params, m, v)
    return params, {"m": m, "v": v, "t": t}


# ── training ────────────────────────────────────────────────────────────────

def make_update(cfg: PPOConfig, spec: tuple, game: JaxSnake, n_local: int, n_dev: int):
    """The per-device update; shard_map runs it on every device with its share of the games."""
    T, view = cfg.n_steps, cfg.view
    mb = cfg.minibatch // n_dev
    n_mb = (T * n_local) // mb
    vstep, vreset = jax.vmap(game.step), jax.vmap(game.reset_if)
    vobs = jax.vmap(partial(game.observe, view=view))
    start_len = game.cfg.start_length

    def loss_fn(params, b):
        logits, value = forward(spec, params, b["obs"])
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, b["act"][:, None], 1)[:, 0]
        ratio = jnp.exp(logp - b["logp"])
        a = b["adv"]
        a = (a - a.mean()) / (a.std(ddof=1) + 1e-8)
        pg = jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
        vl = 0.5 * ((value - b["ret"]) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
        return loss, {"pg_loss": pg, "v_loss": vl, "entropy": ent,
                      "clipfrac": (jnp.abs(ratio - 1) > cfg.clip).mean()}

    def update(params, opt, env, obs, key, lr):
        key, k_roll, k_learn = jax.random.split(key, 3)
        dev = lax.axis_index("d")
        k_roll, k_learn = jax.random.fold_in(k_roll, dev), jax.random.fold_in(k_learn, dev)

        # 1. Play T moves in every game.
        def play(carry, k):
            env, obs = carry
            logits, value = forward(spec, params, obs)
            ka, ks, kr = jax.random.split(k, 3)
            action = jax.random.categorical(ka, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
            env, reward, done, won = vstep(env, action, jax.random.split(ks, n_local))
            final, length = env.length - start_len, env.steps  # read before the reset
            env = vreset(env, done, jax.random.split(kr, n_local))
            return (env, vobs(env)), {"obs": obs, "act": action, "logp": logp, "val": value, "rew": reward,
                                      "done": done, "won": won, "final": final, "len": length}
        (env, obs_next), tr = lax.scan(play, (env, obs), jax.random.split(k_roll, T))

        # 2. Advantages (GAE), walking backwards in time.
        _, next_value = forward(spec, params, obs_next)

        def gae(carry, x):
            last, nv = carry
            nonterminal = 1.0 - x["done"].astype(jnp.float32)
            delta = x["rew"] + cfg.gamma * nv * nonterminal - x["val"]
            last = delta + cfg.gamma * cfg.lam * nonterminal * last
            return (last, x["val"]), last
        _, adv = lax.scan(gae, (jnp.zeros_like(next_value), next_value), tr, reverse=True)
        batch = {"obs": tr["obs"], "act": tr["act"], "logp": tr["logp"], "adv": adv, "ret": adv + tr["val"]}
        batch = jax.tree.map(lambda x: x.reshape(T * n_local, *x.shape[2:]), batch)

        # 3. Learn: a few epochs of shuffled minibatches; gradients averaged over devices.
        def epoch(carry, k):
            def step(carry, idx):
                params, opt = carry
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params,
                                                                            jax.tree.map(lambda x: x[idx], batch))
                grads = lax.pmean(grads, "d")
                norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
                scale = jnp.minimum(1.0, cfg.max_grad_norm / (norm + 1e-6))
                params, opt = adam_step(params, opt, jax.tree.map(lambda g: g * scale, grads), lr)
                return (params, opt), aux
            perm = jax.random.permutation(k, T * n_local)[: n_mb * mb].reshape(n_mb, mb)
            return lax.scan(step, carry, perm)
        (params, opt), aux = lax.scan(epoch, (params, opt), jax.random.split(k_learn, cfg.epochs))

        # 4. Game stats, summed over devices.
        ended = tr["done"]
        m = {"ended": ended.sum(), "score_sum": jnp.where(ended, tr["final"], 0).sum(),
             "len_sum": jnp.where(ended, tr["len"], 0).sum(), "wins": tr["won"].sum()}
        m = jax.tree.map(lambda x: lax.psum(x, "d"), m)
        m["score_max"] = lax.pmax(jnp.where(ended, tr["final"], -1).max(), "d")
        m.update(jax.tree.map(lambda x: lax.pmean(x.mean(), "d"), aux))
        return params, opt, env, obs_next, key, m

    return update


def train(cfg: PPOConfig, run_dir: str | Path, resume: bool = False, max_hours: float | None = None) -> None:
    """Like ppo.train (same files, same metrics), on every JAX device available."""
    import torch
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    n_dev = len(devices)
    if cfg.n_envs % n_dev or cfg.minibatch % n_dev:
        raise SystemExit(f"--envs and minibatch must divide by the {n_dev} devices")
    sha = _git_version()
    state = None
    if resume:
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        if state.get("backend") != "jax":
            raise SystemExit("this run was trained with the torch backend; resume it with that")
        meta = json.loads((run_dir / "config.json").read_text())
        meta["total_steps"] = cfg.total_steps
        meta.setdefault("resumed", []).append({"update": state["update"], "git": sha, "device": str(devices[0])})
        (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    else:
        if (run_dir / "metrics.jsonl").exists():
            raise SystemExit(f"{run_dir} already has a run; use --resume or a new directory")
        (run_dir / "config.json").write_text(json.dumps(
            {**asdict(cfg), "git": sha, "backend": "jax", "device": f"{n_dev} x {devices[0].device_kind}"}, indent=2))
    first = state["update"] + 1 if state else 1
    print(f"JAX on {n_dev} x {devices[0].device_kind}", flush=True)

    import torch.nn  # noqa: F401  (SnakeNet is a torch module: the source of layers + initial weights)
    torch.manual_seed(cfg.seed)
    tnet = SnakeNet(obs_shape(cfg.size, cfg.view), cfg.arch)
    spec = net_spec(tnet)
    src = state["model"] if state else tnet.state_dict()
    params = {k: jnp.asarray(v.numpy()) for k, v in src.items()}
    opt = jax.tree.map(jnp.asarray, state["optimizer"]) if state else adam_init(params)

    mesh = Mesh(np.array(devices), ("d",))
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    game = JaxSnake(SnakeConfig(size=cfg.size))
    T, N = cfg.n_steps, cfg.n_envs
    update = make_update(cfg, spec, game, N // n_dev, n_dev)
    update = jax.jit(jax.shard_map(update, mesh=mesh, in_specs=(P(), P(), P("d"), P("d"), P(), P()),
                                   out_specs=(P(), P(), P("d"), P("d"), P(), P())),
                     donate_argnums=(0, 1, 2, 3))
    key = jax.random.key(cfg.seed + first - 1)
    key, k_env = jax.random.split(key)
    env = jax.jit(jax.vmap(game.init), out_shardings=shard)(jax.random.split(k_env, N))
    obs = jax.jit(jax.vmap(partial(game.observe, view=cfg.view)), out_shardings=shard)(env)
    params, opt, key = jax.device_put((params, opt, key), rep)

    n_updates = cfg.total_steps // (T * N)
    games_played = state["games"] if state else 0
    wins_total = state["wins_total"] if state else 0
    best_score = state["best_score"] if state else 0
    start = time.time() - (state["elapsed"] if state else 0.0)
    session_start = time.time()
    log = open(run_dir / "metrics.jsonl", "a")
    from ailearns.agents import tracking
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))

    def save(update_no: int, stats: dict) -> None:
        torch.save({"model": to_torch(params), "config": asdict(cfg), "update": update_no,
                    "steps": update_no * T * N, "games": games_played, "stats": stats, "backend": "jax"},
                   run_dir / "checkpoints" / f"u{update_no:05d}.pt")

    def save_latest(update_no: int) -> None:
        torch.save({"model": to_torch(params), "optimizer": jax.tree.map(np.asarray, opt), "config": asdict(cfg),
                    "update": update_no, "games": games_played, "wins_total": wins_total, "best_score": best_score,
                    "elapsed": time.time() - start, "backend": "jax"}, run_dir / "latest.pt.tmp")
        (run_dir / "latest.pt.tmp").replace(run_dir / "latest.pt")

    if not state:
        save(0, {})
    for update_no in range(first, n_updates + 1):
        lr = jnp.float32(cfg.lr * (1 - (update_no - 1) / n_updates))  # linear decay, as in ppo.py
        params, opt, env, obs, key, m = update(params, opt, env, obs, key, lr)
        m = jax.device_get(m)
        ended = int(m["ended"])
        games_played += ended
        wins_total += int(m["wins"])
        if ended:
            best_score = max(best_score, int(m["score_max"]))
        stats = {
            "update": update_no, "steps": update_no * T * N, "games": games_played,
            "time": round(time.time() - start, 1),
            "score_mean": float(m["score_sum"]) / ended if ended else None,
            "score_max": int(m["score_max"]) if ended else None,
            "best_score": best_score,
            "ep_len_mean": float(m["len_sum"]) / ended if ended else None,
            "wins": int(m["wins"]), "wins_total": wins_total,
            "pg_loss": float(m["pg_loss"]), "v_loss": float(m["v_loss"]),
            "entropy": float(m["entropy"]), "clipfrac": float(m["clipfrac"]),
        }
        log.write(json.dumps(stats) + "\n")
        log.flush()
        tracking.log(tracker, stats)
        out_of_time = max_hours is not None and time.time() - session_start > max_hours * 3600
        if update_no in cfg.early_checkpoints or update_no % cfg.checkpoint_every == 0 or update_no == n_updates \
                or out_of_time:
            save(update_no, stats)
            save_latest(update_no)
        if update_no % 10 == 0 or update_no == first:
            sps = (update_no - first + 1) * T * N / (time.time() - session_start)
            sm = stats["score_mean"]
            print(f"update {update_no:5d}/{n_updates}  steps {stats['steps']:>11,}  games {games_played:>9,}  "
                  f"score {sm if sm is None else round(sm, 2)!s:>6}  best {best_score:3d}  "
                  f"wins {wins_total}  ent {stats['entropy']:.3f}  {sps:,.0f} steps/s", flush=True)
        if out_of_time:
            print(f"stopped after {max_hours} h at update {update_no}; carry on with --resume", flush=True)
            break
    log.close()
    tracking.finish(tracker)
