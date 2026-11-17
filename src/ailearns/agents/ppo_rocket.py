"""PPO for the booster's landing burn, in JAX, compiled end to end.

The same algorithm as `ppo_jax.py` (one update = play T steps in every burn, GAE, a few epochs of minibatches,
gradients averaged over every device), but for continuous controls: the policy is a Gaussian over the three
actions (engine, gimbal, thrusters), and the network is a small pure-JAX MLP with separate policy and value
trunks. The sim clips actions to [-1, 1]; the log-probability is of the unclipped sample, as usual.

Checkpoints are pickled dicts of numpy arrays (`checkpoints/u00050.pkl`), with the config, so anything can load
them without JAX. Run (the output folder must be under runs/ so a Kaggle script job brings it home):

    python src/ailearns/agents/ppo_rocket.py --out runs/rocket-ppo1 --steps 200_000_000 --hours 11
"""

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_jax import adam_init, adam_step
from ailearns.games.rocket import CRASHED, LANDED, TIMEOUT, RocketConfig, entry, n_actions, n_obs, world
from ailearns.agents.rocket_demos import FIELDS as DEMO_FIELDS
from ailearns.games.rocket_jax import JaxBoosters, State


@dataclass
class RocketPPOConfig:
    total_steps: int = 200_000_000
    n_envs: int = 4096
    n_steps: int = 64
    epochs: int = 4
    minibatch: int = 16384
    lr: float = 3e-4
    gamma: float = 0.995  # a burn is up to 1200 steps; 0.995 looks ~10 s ahead
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: tuple = (128, 128)
    log_std_init: float = -0.5
    seed: int = 0
    early_checkpoints: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75])
    checkpoint_every: int = 50
    # the backward ladder (ep18 6k): below level 1 a burn starts part-way along one of the classic controller's
    # landings (rocket_demos): level 0 just before touchdown, 0.5 halfway through, 1 the real start box. It climbs
    # by level_step each time the smoothed land rate reaches level_up_at, and never goes down.
    level_start: float = 1.0
    level_step: float = 0.05
    level_up_at: float = 0.5
    # (9d) this share of burns starts from the real start box at every level: the classic lights its engine 0.7 s
    # in, so the recordings are 96% lit and the light-or-not decision went untrained until level 1 (ppo10 then
    # stopped lighting at all, for good)
    fresh_share: float = 0.2
    sharp_senses: bool = True  # False: the four square-root senses (ep18 7b) read 0, for an A/B
    max_std: float = 1.0  # (12m) cap on the exploration spread: under the bending limit, noise of the whole control
    # range throws the booster across the airflow and every training flight breaks up (ppo26 collapsed so)


# ── the network ─────────────────────────────────────────────────────────────

def init_params(cfg: RocketPPOConfig, key, game_cfg: RocketConfig | None = None) -> dict:
    """Two tanh MLPs (policy, value), orthogonal weights; small policy head so it starts near 'do nothing'.
    Sized for the booster: grid fins (10a) add a sense and a control."""
    orth = jax.nn.initializers.orthogonal
    game_cfg = game_cfg or RocketConfig()
    n_act = n_actions(game_cfg)
    params, sizes = {}, (n_obs(game_cfg), *cfg.hidden)
    for trunk, out, gain in (("pi", n_act, 0.01), ("v", 1, 1.0)):
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            key, k = jax.random.split(key)
            params[f"{trunk}{i}.w"] = orth(np.sqrt(2))(k, (a, b))
            params[f"{trunk}{i}.b"] = jnp.zeros(b)
        key, k = jax.random.split(key)
        params[f"{trunk}.out.w"] = orth(gain)(k, (sizes[-1], out))
        params[f"{trunk}.out.b"] = jnp.zeros(out)
    params["log_std"] = jnp.full(n_act, cfg.log_std_init)
    return params


def forward(params: dict, obs: jax.Array, n_layers: int):
    """obs (batch, n_obs) -> (action mean, value)."""
    def trunk(name):
        x = obs
        for i in range(n_layers):
            x = jnp.tanh(x @ params[f"{name}{i}.w"] + params[f"{name}{i}.b"])
        return x @ params[f"{name}.out.w"] + params[f"{name}.out.b"]
    return trunk("pi"), trunk("v")[:, 0]


LOG_STD_MAX = 0.0  # a spread of 1: every control's whole range. Wider means nothing (the sim clips actions to
                   # [-1, 1]) but cost nothing either, and with an entropy bonus it grew to 1e20 (ep18 rocket-ppo2)
LOG_STD_MIN = -5.0


def std_of(params, max_std: float = 1.0):
    """The policy's log spread, capped to where it still means something (and at most log(max_std): 12m)."""
    return jnp.clip(params["log_std"], LOG_STD_MIN, min(LOG_STD_MAX, float(np.log(max_std))))


def log_prob(mean, log_std, action):
    z = (action - mean) / jnp.exp(log_std)
    return (-0.5 * z * z - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)


# ── training ────────────────────────────────────────────────────────────────

def make_start(game: JaxBoosters, demos: dict | None, fresh_share: float = 0.0):
    """-> start(key, level): a fresh burn. Level >= 1 (or no demos): the real start box. Below: a random recorded
    landing, entered at (1 - level) of the way along it, give or take one rung."""
    if demos is None:
        return lambda key, level: game.init(key)
    states, lens = jnp.asarray(demos["states"], jnp.float32), jnp.asarray(demos["lens"])

    def start(key, level):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        fresh = game.init(k1)
        i = jax.random.randint(k2, (), 0, lens.shape[0])
        at = jnp.clip(level - 0.05 * jax.random.uniform(k3), 0.0, 1.0)  # a little behind the rung, never past it
        row = states[i, jnp.clip(jnp.round((1.0 - at) * (lens[i] - 1)).astype(jnp.int32), 0, lens[i] - 1)]
        demo = fresh._replace(**{f: row[j].astype(getattr(fresh, f).dtype) for j, f in enumerate(DEMO_FIELDS)})
        use_fresh = (level >= 1.0) | (jax.random.uniform(k4) < fresh_share)
        return jax.tree.map(lambda a, b: jnp.where(use_fresh, a, b), fresh, demo)
    return start


def senses(cfg: RocketPPOConfig, game: JaxBoosters):
    if cfg.sharp_senses:
        return game.observe
    return lambda st: game.observe(st).at[12:].set(0.0)


def make_update(cfg: RocketPPOConfig, game: JaxBoosters, n_local: int, n_dev: int, demos: dict | None = None):
    T, L = cfg.n_steps, len(cfg.hidden)
    mb = cfg.minibatch // n_dev
    n_mb = (T * n_local) // mb
    vstep = jax.vmap(game.step)
    vobs = jax.vmap(senses(cfg, game))
    start = make_start(game, demos, cfg.fresh_share)

    def reset_if(st, done, key, level):
        return jax.tree.map(lambda a, b: jnp.where(done, a, b), start(key, level), st)
    vreset = jax.vmap(reset_if, in_axes=(0, 0, 0, None))

    def loss_fn(params, b):
        mean, value = forward(params, b["obs"], L)
        logp = log_prob(mean, std_of(params, cfg.max_std), b["act"])
        ratio = jnp.exp(logp - b["logp"])
        a = b["adv"]
        a = (a - a.mean()) / (a.std(ddof=1) + 1e-8)
        pg = jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
        vl = 0.5 * ((value - b["ret"]) ** 2).mean()
        ent = (std_of(params, cfg.max_std) + 0.5 * np.log(2 * np.pi * np.e)).sum()
        loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
        return loss, {"pg_loss": pg, "v_loss": vl, "entropy": ent,
                      "clipfrac": (jnp.abs(ratio - 1) > cfg.clip).mean()}

    def update(params, opt, env, obs, key, lr, level):
        key, k_roll, k_learn = jax.random.split(key, 3)
        dev = lax.axis_index("d")
        k_roll, k_learn = jax.random.fold_in(k_roll, dev), jax.random.fold_in(k_learn, dev)

        # 1. Fly T steps in every burn; a finished burn restarts at once.
        def play(carry, k):
            env, obs = carry
            mean, value = forward(params, obs, L)
            ka, kr = jax.random.split(k)
            action = mean + jnp.exp(std_of(params, cfg.max_std)) * jax.random.normal(ka, mean.shape)
            logp = log_prob(mean, std_of(params, cfg.max_std), action)
            env, reward, done = vstep(env, action)
            end = {"outcome": env.outcome, "vy": env.vy, "vx": env.vx, "x": env.x, "fuel": env.fuel,
                   "time": env.time}  # read before the reset
            env = vreset(env, done, jax.random.split(kr, n_local), level)
            return (env, vobs(env)), {"obs": obs, "act": action, "logp": logp, "val": value, "rew": reward,
                                      "done": done, **end}
        (env, obs_next), tr = lax.scan(play, (env, obs), jax.random.split(k_roll, T))

        # 2. Advantages (GAE), walking backwards in time.
        _, next_value = forward(params, obs_next, L)

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

        # 4. Burn stats, summed over devices.
        d = tr["done"]
        landed, crashed = d & (tr["outcome"] == LANDED), d & (tr["outcome"] == CRASHED)
        m = {"ended": d.sum(), "landed": landed.sum(), "crashed": crashed.sum(),
             "timeouts": (d & (tr["outcome"] == TIMEOUT)).sum(),
             "len_sum": jnp.where(d, tr["time"], 0).sum(), "rew_sum": tr["rew"].sum(),
             "impact_sum": jnp.where(d, jnp.abs(tr["vy"]), 0).sum(),
             "slide_sum": jnp.where(d, jnp.abs(tr["vx"]), 0).sum(),
             "miss_sum": jnp.where(d, jnp.abs(tr["x"]), 0).sum(),
             "fuel_landed_sum": jnp.where(landed, tr["fuel"], 0).sum()}
        m = jax.tree.map(lambda x: lax.psum(x, "d"), m)
        m.update(jax.tree.map(lambda x: lax.pmean(x.mean(), "d"), aux))
        return params, opt, env, obs_next, key, m

    return update


def train(cfg: RocketPPOConfig, run_dir: str | Path, game_cfg: RocketConfig | None = None, resume: bool = False,
          max_hours: float | None = None, init_from: str | None = None) -> None:
    """init_from: a checkpoint whose network starts the run (12m: relearn under new physics from a solved pilot);
    the optimizer and the ladder start fresh."""
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    n_dev = len(devices)
    if cfg.n_envs % n_dev or cfg.minibatch % n_dev:
        raise SystemExit(f"--envs and minibatch must divide by the {n_dev} devices")
    game_cfg = game_cfg or RocketConfig()
    state = None
    if resume:
        state = pickle.loads((run_dir / "latest.pkl").read_bytes())
    elif (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already has a run; use --resume or a new directory")
    else:
        (run_dir / "config.json").write_text(json.dumps(
            {**asdict(cfg), "game": asdict(game_cfg), "git": _git_version(), "backend": "jax", "init_from": init_from,
             "device": f"{n_dev} x {devices[0].device_kind}"}, indent=2, default=float))
    first = state["update"] + 1 if state else 1
    print(f"JAX on {n_dev} x {devices[0].device_kind}", flush=True)

    key = jax.random.key(cfg.seed + first - 1)
    key, k_init, k_env = jax.random.split(key, 3)
    params = jax.tree.map(jnp.asarray, state["params"]) if state else init_params(cfg, k_init, game_cfg)
    if init_from and not state:
        start_params = pickle.loads(Path(init_from).read_bytes())["params"]
        if jax.tree.structure(start_params) != jax.tree.structure(params) or any(
                np.shape(start_params[k]) != np.shape(params[k]) for k in params):
            raise SystemExit(f"{init_from}: a different network or senses than this run's")
        params = jax.tree.map(jnp.asarray, start_params)
        print(f"network from {init_from}", flush=True)
    opt = jax.tree.map(jnp.asarray, state["optimizer"]) if state else adam_init(params)

    mesh = Mesh(np.array(devices), ("d",))
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    game = JaxBoosters(game_cfg)
    T, N = cfg.n_steps, cfg.n_envs
    demos = None
    if cfg.level_start < 1.0:
        from ailearns.agents.rocket_demos import record
        # (10a) recorded without grid fins: the classic holds fins straight, which costs it landings (storm: 15% ->
        # 5.5%), so with fins its ladder would be thin. The same starts as the finless runs; fins then fly from there.
        # (11c) except from 70 km: the entry script flies the fins to reach the ship at all
        fins = game_cfg.fin_area if entry(game_cfg) else 0.0
        demos = record(config=replace(game_cfg, fin_area=fins, fault_p=0.0))  # (10e) and on healthy hardware
        print(f"ladder: {len(demos['lens'])} classic landings to start along", flush=True)
    update = make_update(cfg, game, N // n_dev, n_dev, demos)
    update = jax.jit(jax.shard_map(update, mesh=mesh, in_specs=(P(), P(), P("d"), P("d"), P(), P(), P()),
                                   out_specs=(P(), P(), P("d"), P("d"), P(), P())),
                     donate_argnums=(0, 1, 2, 3))
    level = state.get("level", cfg.level_start) if state else cfg.level_start
    land_ema = state.get("land_ema", 0.0) if state else 0.0
    env = jax.jit(jax.vmap(make_start(game, demos, cfg.fresh_share), in_axes=(0, None)), out_shardings=shard)(
        jax.random.split(k_env, N), jnp.float32(level))
    obs = jax.jit(jax.vmap(senses(cfg, game)), out_shardings=shard)(env)
    params, opt, key = jax.device_put((params, opt, key), rep)

    n_updates = cfg.total_steps // (T * N)
    burns = state["burns"] if state else 0
    landings = state["landings"] if state else 0
    start = time.time() - (state["elapsed"] if state else 0.0)
    session_start = time.time()
    log = open(run_dir / "metrics.jsonl", "a")
    from ailearns.agents import tracking
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    np_tree = lambda t: jax.tree.map(np.asarray, t)  # noqa: E731

    def save(update_no: int, stats: dict) -> None:
        (run_dir / "checkpoints" / f"u{update_no:05d}.pkl").write_bytes(pickle.dumps(
            {"params": np_tree(params), "config": asdict(cfg), "game": asdict(game_cfg), "update": update_no,
             "steps": update_no * T * N, "burns": burns, "stats": stats}))

    def save_latest(update_no: int) -> None:
        (run_dir / "latest.pkl.tmp").write_bytes(pickle.dumps(
            {"params": np_tree(params), "optimizer": np_tree(opt), "update": update_no, "burns": burns,
             "landings": landings, "elapsed": time.time() - start, "level": level, "land_ema": land_ema}))
        (run_dir / "latest.pkl.tmp").replace(run_dir / "latest.pkl")

    if not state:
        save(0, {})
    for update_no in range(first, n_updates + 1):
        lr = jnp.float32(cfg.lr * (1 - (update_no - 1) / n_updates))  # linear decay
        params, opt, env, obs, key, m = update(params, opt, env, obs, key, lr, jnp.float32(level))
        m = jax.device_get(m)
        ended = int(m["ended"])
        burns += ended
        landings += int(m["landed"])
        per = lambda k: float(m[k]) / ended if ended else None  # noqa: E731
        burn_level = level
        if ended:
            land_ema = 0.9 * land_ema + 0.1 * per("landed")
        if level < 1.0 and land_ema >= cfg.level_up_at:  # climb a rung; the new rung starts from a clean slate
            level, land_ema = min(1.0, round(level + cfg.level_step, 4)), 0.0
        stats = {
            "update": update_no, "steps": update_no * T * N, "burns": burns, "time": round(time.time() - start, 1),
            "level": burn_level, "land_rate": per("landed"), "crash_rate": per("crashed"), "timeout_rate": per("timeouts"),
            "landings": landings, "burn_s": per("len_sum"), "impact_vy": per("impact_sum"),
            "impact_vx": per("slide_sum"), "miss_x": per("miss_sum"),
            "fuel_left_landed": float(m["fuel_landed_sum"]) / int(m["landed"]) if int(m["landed"]) else None,
            "reward_per_burn": per("rew_sum"),
            "pg_loss": float(m["pg_loss"]), "v_loss": float(m["v_loss"]),
            "entropy": float(m["entropy"]), "clipfrac": float(m["clipfrac"]),
            "std": [round(float(s), 3) for s in np.exp(np.asarray(std_of(params, cfg.max_std)))],
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
            f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
            print(f"update {update_no:5d}/{n_updates}  level {burn_level:.2f}  steps {stats['steps']:>12,}  burns {burns:>9,}  "
                  f"land {f(stats['land_rate'], 3)}  impact {f(stats['impact_vy'], 1)} m/s  "
                  f"slide {f(stats['impact_vx'], 1)}  miss {f(stats['miss_x'], 0)} m  "
                  f"burn {f(stats['burn_s'], 1)} s  std {stats['std']}  {sps:,.0f} steps/s", flush=True)
        if out_of_time:
            print(f"stopped after {max_hours} h at update {update_no}; carry on with --resume", flush=True)
            break
    log.close()
    tracking.finish(tracker)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="run folder, under runs/")
    p.add_argument("--steps", type=float, default=RocketPPOConfig.total_steps)
    p.add_argument("--envs", type=int, default=RocketPPOConfig.n_envs)
    p.add_argument("--n-steps", type=int, default=RocketPPOConfig.n_steps)
    p.add_argument("--minibatch", type=int, default=RocketPPOConfig.minibatch)
    p.add_argument("--lr", type=float, default=RocketPPOConfig.lr)
    p.add_argument("--gamma", type=float, default=RocketPPOConfig.gamma)
    p.add_argument("--ent-coef", type=float, default=RocketPPOConfig.ent_coef,
                   help="entropy bonus: keeps the policy's spread from collapsing before it finds a landing")
    p.add_argument("--level-start", type=float, default=RocketPPOConfig.level_start,
                   help="0: start a moment before the classic's touchdowns and climb back to the real start (1: no ladder)")
    p.add_argument("--no-sharp-senses", action="store_true", help="blank the square-root senses (an A/B)")
    p.add_argument("--world", default="pad", help="where it lands: pad, ship, storm; +fins for grid fins (games.rocket.world)")
    p.add_argument("--fresh-share", type=float, default=RocketPPOConfig.fresh_share,
                   help="share of ladder burns that start from the real start box anyway")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hours", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-std", type=float, default=RocketPPOConfig.max_std, help="cap on the exploration spread")
    p.add_argument("--init-from", default=None, help="start from this checkpoint's network (fresh optimizer, ladder)")
    a = p.parse_args()
    cfg = RocketPPOConfig(total_steps=int(a.steps), n_envs=a.envs, n_steps=a.n_steps, minibatch=a.minibatch,
                          lr=a.lr, gamma=a.gamma, ent_coef=a.ent_coef, level_start=a.level_start,
                          sharp_senses=not a.no_sharp_senses,
                          fresh_share=a.fresh_share, seed=a.seed, max_std=a.max_std)
    train(cfg, a.out, game_cfg=world(a.world), resume=a.resume, max_hours=a.hours, init_from=a.init_from)


if __name__ == "__main__":
    main()
