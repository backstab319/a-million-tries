"""Proximal Policy Optimization, from scratch, for the Snake games.

The loop, in plain words:
  1. Let the current policy play many games at once for a little while.
  2. For every move, work out how much better it turned out than the
     critic expected (the "advantage").
  3. Nudge the policy toward moves that beat expectations, but never too far
     in one update (that's the "proximal" part: the clipped ratio).
  4. Repeat. Save a checkpoint now and then so we can film every stage.
"""

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ailearns.games.snake import N_ACTIONS, SnakeConfig
from ailearns.games.snake_torch import TorchSnakeGames


class SnakeNet(nn.Module):
    """Sees the board, outputs move preferences (policy) and a score guess (value).

    arch="mlp": every cell wired to every neuron. Cheap; trains fast on a CPU.
    arch="cnn": small conv net. Sees shapes, but ~8x slower on this laptop.
    arch="cnn-deep": 5 conv layers; meant for the GPU.
    """

    def __init__(self, obs_shape: tuple[int, int, int], arch: str = "mlp", hidden: int = 256):
        super().__init__()
        c, p, _ = obs_shape
        if arch == "mlp":
            self.body = nn.Sequential(
                nn.Flatten(), nn.Linear(c * p * p, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
        elif arch == "cnn":
            q = (p + 1) // 2
            self.body = nn.Sequential(
                nn.Conv2d(c, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(32 * q * q, hidden), nn.ReLU(),
            )
        elif arch == "cnn-deep":  # more layers = each neuron sees further, for judging how big a gap is
            q = (p + 1) // 2
            self.body = nn.Sequential(
                nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.Flatten(), nn.Linear(64 * q * q, hidden), nn.ReLU(),
            )
        else:
            raise ValueError(arch)
        self.policy = nn.Linear(hidden, N_ACTIONS)
        self.value = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.policy.weight, 0.01)
        nn.init.zeros_(self.policy.bias)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.policy(h), self.value(h).squeeze(-1)


@dataclass
class PPOConfig:
    size: int = 10
    view: str = "ego2"  # "board", "ego" or "ego2" (see SnakeGames.observe)
    arch: str = "mlp"
    total_steps: int = 20_000_000
    n_envs: int = 128
    n_steps: int = 128
    epochs: int = 4
    minibatch: int = 2048
    lr: float = 2.5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 0
    # Checkpoints: dense early (the funny, terrible phase), then regular.
    early_checkpoints: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75])
    checkpoint_every: int = 50


def obs_shape(size: int, view: str) -> tuple[int, int, int]:
    if view == "board":
        return (4, size + 2, size + 2)
    return (3 if view == "ego" else 4, 2 * size - 1, 2 * size - 1)


def load_policy(path: str | Path) -> tuple[SnakeNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg.setdefault("view", "board")  # runs from before the ego view existed
    net = SnakeNet(obs_shape(cfg["size"], cfg["view"]), cfg.get("arch", "mlp"))
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt


@torch.no_grad()
def act(net: SnakeNet, obs: np.ndarray, greedy: bool = True) -> np.ndarray:
    logits, _ = net(torch.from_numpy(obs))
    if greedy:
        return logits.argmax(-1).numpy()
    return torch.distributions.Categorical(logits=logits).sample().numpy()


def _git_version() -> str:
    """The code version this run was trained with. Cloud jobs have no git repo; they pass it in."""
    if os.environ.get("AILEARNS_GIT"):
        return os.environ["AILEARNS_GIT"]
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return "unknown"


def train(cfg: PPOConfig, run_dir: str | Path, resume: bool = False, max_hours: float | None = None,
          device: str = "cpu") -> None:
    """Train, saving checkpoints into run_dir.

    resume: continue from run_dir/latest.pt (same config; --steps may be raised).
    max_hours: stop cleanly (and resumably) after this long; cloud sessions have time limits.
    """
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)
    sha = _git_version()
    state = None
    if resume:
        state = torch.load(run_dir / "latest.pt", map_location=dev, weights_only=False)
        old = {k: v for k, v in state["config"].items() if k != "total_steps"}
        new = {k: v for k, v in asdict(cfg).items() if k != "total_steps"}
        if old != new:
            raise SystemExit(f"config differs from the run being resumed: "
                             f"{ {k: (old.get(k), v) for k, v in new.items() if old.get(k) != v} }")
        meta = json.loads((run_dir / "config.json").read_text())
        meta["total_steps"] = cfg.total_steps
        meta.setdefault("resumed", []).append({"update": state["update"], "git": sha, "device": device})
        (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    else:
        if (run_dir / "metrics.jsonl").exists():
            raise SystemExit(f"{run_dir} already has a run; use --resume or a new directory")
        (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "git": sha, "device": device}, indent=2))
    first = state["update"] + 1 if state else 1
    torch.manual_seed(cfg.seed + first - 1)  # a resumed run doesn't replay the same random numbers

    games = TorchSnakeGames(cfg.n_envs, SnakeConfig(size=cfg.size), seed=cfg.seed + first - 1, device=device)
    shape = obs_shape(cfg.size, cfg.view)
    net = SnakeNet(shape, cfg.arch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    if state:
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])

    T, N = cfg.n_steps, cfg.n_envs
    buf_obs = torch.zeros((T, N, *shape), device=dev)
    buf_act = torch.zeros((T, N), dtype=torch.long, device=dev)
    buf_logp = torch.zeros((T, N), device=dev)
    buf_val = torch.zeros((T, N), device=dev)
    buf_rew = torch.zeros((T, N), device=dev)
    buf_done = torch.zeros((T, N), device=dev)
    buf_final = torch.zeros((T, N), dtype=torch.long, device=dev)  # score / length of games that just ended
    buf_len = torch.zeros((T, N), dtype=torch.long, device=dev)
    buf_won = torch.zeros((T, N), dtype=torch.bool, device=dev)

    n_updates = cfg.total_steps // (T * N)
    games_played = state["games"] if state else 0
    wins_total = state["wins_total"] if state else 0
    best_score = state["best_score"] if state else 0
    start = time.time() - (state["elapsed"] if state else 0.0)
    session_start = time.time()
    log = open(run_dir / "metrics.jsonl", "a")
    from ailearns.agents import tracking
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))

    def save(update: int, stats: dict) -> None:
        torch.save({"model": net.state_dict(), "config": asdict(cfg), "update": update,
                    "steps": update * T * N, "games": games_played, "stats": stats},
                   run_dir / "checkpoints" / f"u{update:05d}.pt")

    def save_latest(update: int) -> None:  # everything needed to carry on (optimizer included)
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "config": asdict(cfg),
                    "update": update, "games": games_played, "wins_total": wins_total,
                    "best_score": best_score, "elapsed": time.time() - start},
                   run_dir / "latest.pt.tmp")
        (run_dir / "latest.pt.tmp").replace(run_dir / "latest.pt")  # never leave a half-written file

    obs = games.observe(cfg.view)
    if not state:
        save(0, {})
    for update in range(first, n_updates + 1):
        for g in opt.param_groups:  # linear learning-rate decay
            g["lr"] = cfg.lr * (1 - (update - 1) / n_updates)

        # 1. Play. (Everything stays on the device; game stats are read once per update.)
        for t in range(T):
            with torch.no_grad():
                logits, value = net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            buf_obs[t], buf_act[t], buf_logp[t], buf_val[t] = obs, action, dist.log_prob(action), value

            reward, done, won = games.step(action)
            buf_final[t], buf_len[t], buf_won[t] = games.score, games.steps, won
            games.reset(done)
            buf_rew[t] = reward
            buf_done[t] = done.float()
            obs = games.observe(cfg.view)
        ended = buf_done.bool()
        finished_scores = buf_final[ended].tolist()
        finished_lengths = buf_len[ended].tolist()
        wins = int(buf_won.sum())

        # 2. Advantages (GAE): how much better than the critic expected.
        with torch.no_grad():
            _, next_value = net(obs)
        adv = torch.zeros((T, N), device=dev)
        last = torch.zeros(N, device=dev)
        for t in reversed(range(T)):
            nv = next_value if t == T - 1 else buf_val[t + 1]
            nonterminal = 1.0 - buf_done[t]
            delta = buf_rew[t] + cfg.gamma * nv * nonterminal - buf_val[t]
            last = delta + cfg.gamma * cfg.lam * nonterminal * last
            adv[t] = last
        returns = adv + buf_val

        # 3. Learn, a few passes over the batch in shuffled minibatches.
        b_obs = buf_obs.reshape(T * N, *shape)
        b_act, b_logp = buf_act.reshape(-1), buf_logp.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), returns.reshape(-1)
        clipfracs, pg_losses, v_losses, entropies = [], [], [], []
        for _ in range(cfg.epochs):
            for mb in torch.randperm(T * N, device=dev).split(cfg.minibatch):
                logits, value = net(b_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_act[mb])
                ratio = (logp - b_logp[mb]).exp()
                a = b_adv[mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * (value - b_ret[mb]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                clipfracs.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.item())

        # 4. Log and checkpoint.
        games_played += len(finished_scores)
        wins_total += wins
        if finished_scores:
            best_score = max(best_score, max(finished_scores))
        stats = {
            "update": update, "steps": update * T * N, "games": games_played,
            "time": round(time.time() - start, 1),
            "score_mean": float(np.mean(finished_scores)) if finished_scores else None,
            "score_max": max(finished_scores) if finished_scores else None,
            "best_score": best_score,
            "ep_len_mean": float(np.mean(finished_lengths)) if finished_lengths else None,
            "wins": wins, "wins_total": wins_total,
            "pg_loss": float(np.mean(pg_losses)), "v_loss": float(np.mean(v_losses)),
            "entropy": float(np.mean(entropies)), "clipfrac": float(np.mean(clipfracs)),
        }
        log.write(json.dumps(stats) + "\n")
        log.flush()
        tracking.log(tracker, stats)
        out_of_time = max_hours is not None and time.time() - session_start > max_hours * 3600
        if update in cfg.early_checkpoints or update % cfg.checkpoint_every == 0 or update == n_updates or out_of_time:
            save(update, stats)
            save_latest(update)
        if update % 10 == 0 or update == 1:
            sps = update * T * N / (time.time() - start)
            sm = stats["score_mean"]
            print(f"update {update:5d}/{n_updates}  steps {stats['steps']:>11,}  games {games_played:>9,}  "
                  f"score {sm if sm is None else round(sm, 2)!s:>6}  best {best_score:3d}  "
                  f"wins {wins_total}  ent {stats['entropy']:.3f}  {sps:,.0f} steps/s", flush=True)
        if out_of_time:
            print(f"stopped after {max_hours} h at update {update}; carry on with --resume", flush=True)
            break
    log.close()
    tracking.finish(tracker)
