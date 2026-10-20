"""PPO for CQB (games/cqb.py): one side learns, the other is a script (episode 9 starts with
Assault vs the camper) or, later, frozen brains.

The brain is tag's (two tanh layers of 256, a value head), with a new kind of output: three moves
(forward/back, sideways, turn) are Gaussian, like tag's, and three on/off choices (fire, interact,
run) are each a coin with a learned bias (Bernoulli); a fourth, throw, for a side that has flashbangs
(ep9 3w). Its best guess: the mean moves, and a choice is on when its chance is over a half.

Everything (all rounds playing T decisions, advantages, learning) is one compiled function,
data-parallel over every device (both T4s on Kaggle). Checkpoints are torch files of the weights.
"""

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_jax import adam_init, adam_step, to_torch
from ailearns.games import cqb
from ailearns.games import cqb_scripts as S
from ailearns.games.cqb import ASSAULT, DEFENSE, TRAIN_SEEDS, CQBConfig, CQBGame, make_buildings

N_MOVE, N_SWITCH = 3, 3   # fwd, side, turn | fire, interact, run (cqb.FWD..RUN order); + throw with flashbangs
SWITCHES = ("fire", "interact", "run", "throw", "reload")
THROW_BIAS = -4.0          # a widened brain throws on ~2% of decisions: it tries, without emptying its pockets at once


@dataclass
class CQBPPOConfig:
    total_steps: int = 200_000_000
    side: str = "assault"        # who learns
    opponent: str = "camper"     # a script (cqb_scripts.SCRIPTS), a brain "runs/a:175", or a pool "camper,runs/a:175"
    explore_reward: float = 0.0  # Assault: reward per new "seen" map cell (a hint; 0 = the round result only)
    time_cost: float = 0.0       # Assault pays this per second (the game's CQBConfig.time_cost); Defense earns it
    flashbangs: tuple = (0, 0)   # per round (Assault, Defense): the game's CQBConfig.flashbangs (ep9 3t)
    magazine: int = 0            # rounds per magazine: the game's CQBConfig.magazine (0: no magazines; ep9 3am)
    spare_mags: int = 3          # magazines carried besides the loaded one: the game's CQBConfig.spare_mags (ep12: 1)
    time_limit: float = 45.0     # seconds a round lasts: the game's CQBConfig.time_limit (ep12: 90 in big houses)
    blind_away: float = 1.0      # flashbang blindness when it wasn't in your view, x blind_time (ep12: 0.25)
    hostage: bool = False        # the game's CQBConfig.hostage (ep12: secure the hostage to win too)
    hostage_deep: float = 0.0    # the game's CQBConfig.hostage_deep
    extract: bool = False        # the game's CQBConfig.extract
    hostage_noise: float = 0.0   # the game's CQBConfig.hostage_noise
    through_drywall: float = 0.0  # (ep13) the game's CQBConfig.through_drywall (homes: shots through inside walls)
    unseen_hit: float = 0.5       # (ep13) the game's CQBConfig.unseen_hit
    hostage_hit: bool = False     # (ep13) the game's CQBConfig.hostage_hit (the side that hits it loses)
    health: float = 0.0          # the game's CQBConfig.health (0: one hit and you're down; ep9 3ar)
    react: float = 0.0           # the game's CQBConfig.react (0: no reaction time; ep9 3aw)
    teams: tuple = (1, 1)        # the game's CQBConfig.team_sizes (ep10): one brain plays every operator of its side
    bump: float = 0.0            # the game's CQBConfig.bump (loud collisions; ep10 step 9)
    ready: float = 0.0           # the game's CQBConfig.ready (rifle up after running; ep10 step 9)
    stun: float = 0.0            # the game's CQBConfig.stun (a hard collision stuns; ep10 9f)
    share_sight: bool = True     # the game's CQBConfig.share_sight (False: no automatic callouts; ep11)
    radio: tuple = (0, 0)        # the game's CQBConfig.radio: words per side (ep11); the brain gets a word head
    split_entry: bool = False    # the game's CQBConfig.split_entry (the pair starts at two doors; ep11)
    relay: float = 0.0           # the game's CQBConfig.relay (fresh callouts only; ep11)
    intel: int = 0               # the game's CQBConfig.intel (Assault operators on a perfect drone feed; ep11)
    feed: int = 0                # the game's CQBConfig.feed (the drone feed as its own inputs; ep11 10m)
    route: bool = False          # the game's CQBConfig.route (the way to the enemy it knows of; ep11 10d)
    pilot: bool = False          # the game's CQBConfig.pilot (Assault operator 1 is a drone pilot who only talks; ep11)
    callout: bool = False        # the game's CQBConfig.callout (the pilot speaks a fixed protocol; ep11 10zc)
    callout_way: bool = False    # the game's CQBConfig.callout_way (the protocol calls the way, not the bearing)
    house: str = ""              # a building preset, cqb.HOUSES ("big": 8-12 rooms; ep11 10j)
    listen_coef: float = 0.0     # (radio, ep11) a nudge to let the word heard change what it does: how far its choices
                                 # move between the word it heard and nothing heard (capped per decision). 0: none
    signal_coef: float = 0.0     # (radio) a nudge to say different words in different situations: the words' spread
                                 # over a minibatch minus their spread in each situation. 0: none (Eccles et al. 2019)
    listen_cap: float = 1.0      # the listening nudge stops counting past this much difference per decision
    guess_coef: float = 0.0      # (ep12) the brain also guesses where the nearest enemy really is (the 8 numbers of
                                 # CQBGame.privileged) from its own view: a side head on its body, scored against the
                                 # truth in training only (an auxiliary task). Play never reads the guess. 0: off
    critic_priv: bool = False    # (ep11 10zh) an asymmetric critic: a separate value network that also reads where the
                                 # nearest enemy really is (CQBGame.privileged), for training only; the brain plays from
                                 # its own numbers as before. False: the value head on the brain's body, as always
    vf_warmup: int = 0           # (ep11 10u) the first this-many updates train only the value head (the policy is held):
                                 # a brain whose body was changed elsewhere (imitation) has a value head that no longer
                                 # fits it, and PPO's first steps on those advantages wreck the policy. 0: none
    stacked: float = 0.0         # this share of training rounds are entry drills (CQBGame.init_stacked; ep9 3y)
    memory: int = 0              # a memory (GRU) of this size between the two layers (0: none; ep9 3ad)
    plateau: int = 0             # stop once the 25-update average win rate hasn't set a new best (by 0.5 points) for
                                 # this many updates (0: never; ep9 3af: most runs were flat long before the end)
    mem_lr: float = 0.1          # with memory: the memory's own weights learn at this share of lr (they start fresh
                                 # next to a trained brain; at full speed they knocked it over: ep9 3ag-3ai)
    mem_chunk: int = 32          # with memory: learning replays pieces of this many decisions, each from the memory
                                 # it had in play (short pieces: the replay is one decision after another; 3ah)
    init_from: str = ""          # start from another run's brain, e.g. "runs/cqb-v3-dummy:38" (one from before
                                 # flashbangs is widened: see widen)
    reset_std: bool = False      # with init_from: experiment again (move noise back to exp(init_log_std))
    n_envs: int = 2048
    n_steps: int = 128
    epochs: int = 4
    minibatch: int = 8192
    lr: float = 3e-4
    gamma: float = 0.995         # rounds are 450 decisions: value ~20 s ahead
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 256
    init_log_std: float = -0.5
    seed: int = 0
    buildings: int = len(TRAIN_SEEDS)
    early_checkpoints: tuple = (0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 75)
    checkpoint_every: int = 25


# ── the brain ────────────────────────────────────────────────────────────────

def init_params(key, obs_size: int, hidden: int, init_log_std: float, n_switch: int = N_SWITCH) -> dict:
    def dense(k, n_in, n_out, scale):
        return jax.nn.initializers.orthogonal(scale)(k, (n_out, n_in)), jnp.zeros(n_out)
    k = jax.random.split(key, 4)
    p = {}
    p["body.0.weight"], p["body.0.bias"] = dense(k[0], obs_size, hidden, np.sqrt(2))
    p["body.2.weight"], p["body.2.bias"] = dense(k[1], hidden, hidden, np.sqrt(2))
    p["policy.weight"], p["policy.bias"] = dense(k[2], hidden, N_MOVE + n_switch, 0.01)  # moves ~0, switches ~50/50
    p["value.weight"], p["value.bias"] = dense(k[3], hidden, 1, 1.0)
    p["log_std"] = jnp.full(N_MOVE, init_log_std)
    return p


def obs_blocks(cfg: CQBConfig) -> list:
    """The observation's parts, in CQBGame.observe's order, as (name, size); they add up to cfg.obs_size."""
    c = cqb
    out = [("base", cfg.n_rays + 10 + 9 + 7 + 8 + cfg.seen_grid[0] * cfg.seen_grid[1] + 1)]
    out += [("bang", c.BANG_OBS)] if any(cfg.flashbangs) else []
    out += [("gun", c.GUN_OBS)] if cfg.magazine else []
    out += [("health", c.HEALTH_OBS)] if cfg.health else []
    out += [("react", c.REACT_OBS)] if cfg.react else []
    out += [("teams", cfg.team_obs)] if cfg.team_obs else []
    out += [("ready", c.READY_OBS)] if cfg.ready else []
    out += [("stun", c.STUN_OBS)] if cfg.stun else []
    out += [("radio", c.radio_obs(max(cfg.radio)))] if any(cfg.radio) else []
    out += [("route", c.ROUTE_OBS)] if cfg.route else []
    out += [("feed", c.FEED_OBS)] if cfg.feed else []
    out += [("hostage", c.HOSTAGE_OBS)] if cfg.hostage else []
    out += [("extract", c.EXTRACT_OBS)] if cfg.hostage and cfg.extract else []
    out += [("hostage_sound", c.HOSTAGE_SOUND_OBS)] if cfg.hostage and cfg.extract and cfg.hostage_noise else []
    assert sum(n for _, n in out) == cfg.obs_size, (out, cfg.obs_size)
    return out


def transplant(params: dict, src: CQBConfig, dst: CQBConfig) -> dict:
    """(ep11 10x) A brain trained under `src`'s observation, ready for `dst`'s: each part of what it reads moves to
    where `dst` puts it (a 1v1 brain's route and feed numbers sit after the team numbers in 2v1), parts `dst` adds read
    zero weights. So it sees and decides exactly as before on everything it knew. Parts must not change size."""
    w = params["body.0.weight"]
    assert w.shape[1] == src.obs_size, "transplant needs a brain that reads all of src's numbers"
    where, at = {}, 0
    for name, n in obs_blocks(src):
        where[name] = (at, n)
        at += n
    cols, at = [], 0
    for name, n in obs_blocks(dst):
        if name == "teams" and name in where and where[name][1] != n:   # (ep12) more operators a side: more slots
            a, m = where[name]
            per = cqb.ENEMY_OBS + cqb.MATE_OBS
            se, de = (m - 1) // per, (n - 1) // per                       # slots in src and dst
            assert de >= se, "transplant can add team slots, not remove them"
            z = lambda k: jnp.zeros((w.shape[0], k), w.dtype)            # noqa: E731
            E, M = cqb.ENEMY_OBS, cqb.MATE_OBS
            cols += [w[:, a:a + se * E], z((de - se) * E),                  # the enemy slots it knew, then new ones
                     w[:, a + se * E:a + se * (E + M)], z((de - se) * M),   # the same for teammates
                     w[:, a + m - 1:a + m]]                                 # which one of my team I am
        elif name in where:
            a, m = where[name]
            assert m == n, f"{name}: {m} numbers in src, {n} in dst"
            cols.append(w[:, a:a + n])
        else:
            cols.append(jnp.zeros((w.shape[0], n), w.dtype))
        at += n
    return {**params, "body.0.weight": jnp.concatenate(cols, 1)}


def fit(params: dict, ck: dict, dst: CQBConfig, side: int) -> dict:
    """(ep12) A saved brain (its params and checkpoint) ready to play `side` in a game with `dst`'s rules: its inputs
    moved to where dst puts them (more team slots read zero weights) and a throw switch added if dst gives that side
    flashbangs it never had. Plays exactly as before on everything it knew."""
    src = CQBConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cqb"].items()})
    if params["body.0.weight"].shape[1] == src.obs_size and obs_blocks(src) != obs_blocks(dst):
        params = transplant(params, src, dst)
    return widen(params, dst.obs_size, switches_for(dst, side))


def widen(params: dict, obs_size: int, n_switch: int) -> dict:
    """A brain from before flashbangs, ready for a game with them: zero weights on the new inputs (at the end of the
    observation) and a throw switch that starts with zero weights and a bias of THROW_BIAS. It plays exactly as before,
    plus the odd throw, and learns from there."""
    p = dict(params)
    w = p["body.0.weight"]
    if w.shape[1] < obs_size:
        p["body.0.weight"] = jnp.concatenate([w, jnp.zeros((w.shape[0], obs_size - w.shape[1]), w.dtype)], 1)
    extra = N_MOVE + n_switch - p["policy.weight"].shape[0]
    if extra > 0:
        p["policy.weight"] = jnp.concatenate([p["policy.weight"], jnp.zeros((extra, p["policy.weight"].shape[1]))])
        p["policy.bias"] = jnp.concatenate([p["policy.bias"], jnp.full(extra, THROW_BIAS)])
    return p


MEMORY_KEYS = ("gru.wi", "gru.wh", "gru.bi", "gru.bh", "memory.weight")


def has_memory(params: dict) -> bool:
    return "gru.wi" in params


def memory_size(params: dict) -> int:
    return params["gru.wh"].shape[1] if has_memory(params) else 0


def add_memory(params: dict, size: int, key) -> dict:
    """Give a brain a memory (ep9 3ad): a GRU of `size` reads the first layer's output every decision and keeps a
    state from decision to decision (wiped when a round ends); what it remembers is added to the first layer's output
    through `memory.weight`, which starts at zero. So the brain plays exactly as before until it learns to use it."""
    hidden = params["body.0.weight"].shape[0]
    k1, k2 = jax.random.split(key)
    p = dict(params)
    p["gru.wi"] = jax.nn.initializers.orthogonal(1.0)(k1, (3 * size, hidden))
    p["gru.wh"] = jax.nn.initializers.orthogonal(1.0)(k2, (3 * size, size))
    p["gru.bi"], p["gru.bh"] = jnp.zeros(3 * size), jnp.zeros(3 * size)
    p["memory.weight"] = jnp.zeros((hidden, size))
    return p


def n_words(params: dict) -> int:
    """Radio words this brain can say (0: no word head)."""
    return params["say.bias"].shape[0] - 1 if "say.bias" in params else 0


PRIV = cqb.FEED_OBS   # the privileged numbers an asymmetric critic reads besides the observation


def add_critic(params: dict, obs_size: int, key, hidden: int = 256) -> dict:
    """(ep11 10zh) A separate value network over (observation, privileged numbers), two tanh layers: the asymmetric
    critic. The brain's own value head stays in the params (unused while this is there)."""
    def dense(k, n_in, n_out, scale):
        w = jax.random.orthogonal(k, max(n_in, n_out))[:n_out, :n_in] * scale
        return w.astype(jnp.float32), jnp.zeros(n_out, jnp.float32)
    k = jax.random.split(key, 3)
    p = dict(params)
    p["critic.0.weight"], p["critic.0.bias"] = dense(k[0], obs_size + PRIV, hidden, np.sqrt(2))
    p["critic.2.weight"], p["critic.2.bias"] = dense(k[1], hidden, hidden, np.sqrt(2))
    p["critic.out.weight"], p["critic.out.bias"] = dense(k[2], hidden, 1, 1.0)
    return p


def add_guess(params: dict, key, hidden: int = 256) -> dict:
    """(ep12) The guessing head: from the brain's body (its last hidden layer) to the PRIV numbers."""
    p = dict(params)
    w = jax.nn.initializers.orthogonal(0.1)(key, (PRIV, params["body.2.weight"].shape[0]))
    p["guess.weight"], p["guess.bias"] = w, jnp.zeros(PRIV)
    return p


def guess(params: dict, obs):
    """The brain's guess of where the nearest enemy is: (batch, obs) -> (batch, PRIV). Never used in play."""
    x = jnp.tanh(obs @ params["body.0.weight"].T + params["body.0.bias"])
    x = jnp.tanh(x @ params["body.2.weight"].T + params["body.2.bias"])
    return x @ params["guess.weight"].T + params["guess.bias"]


def critic_value(params: dict, obs, priv):
    """The asymmetric critic's value: (batch, obs), (batch, PRIV) -> (batch,)."""
    x = jnp.tanh(jnp.concatenate([obs, priv], -1) @ params["critic.0.weight"].T + params["critic.0.bias"])
    x = jnp.tanh(x @ params["critic.2.weight"].T + params["critic.2.bias"])
    return (x @ params["critic.out.weight"].T + params["critic.out.bias"])[:, 0]


def add_radio(params: dict, words: int) -> dict:
    """Give a brain a word head (ep11): silence or one of `words`, a choice among words + 1 read from the second
    layer. Zero weights: at first every choice is equally likely whatever it sees (the words mean nothing yet)."""
    hidden = params["body.2.weight"].shape[0]
    return {**params, "say.weight": jnp.zeros((words + 1, hidden)), "say.bias": jnp.zeros(words + 1)}


def _split(logits, words: int):
    """Logits -> (the switches', the words' (or None)): a brain with a radio has words + 1 more at the end."""
    return (logits, None) if not words else (logits[..., :-(words + 1)], logits[..., -(words + 1):])


def _gru(params: dict, x, h):
    """One step of the memory: x (batch, hidden), h (batch, size) -> the new h (torch's GRU equations)."""
    ir, iz, i_n = jnp.split(x @ params["gru.wi"].T + params["gru.bi"], 3, -1)
    hr, hz, h_n = jnp.split(h @ params["gru.wh"].T + params["gru.bh"], 3, -1)
    r, z = jax.nn.sigmoid(ir + hr), jax.nn.sigmoid(iz + hz)
    return (1 - z) * jnp.tanh(i_n + r * h_n) + z * h


def forward_mem(params: dict, obs: jax.Array, h: jax.Array):
    """forward with the memory state h (batch, size; size 0 without memory) -> forward's outputs and the new h."""
    x = jnp.tanh(obs @ params["body.0.weight"].T + params["body.0.bias"])
    if has_memory(params):
        h = _gru(params, x, h)
        x = x + h @ params["memory.weight"].T
    x = jnp.tanh(x @ params["body.2.weight"].T + params["body.2.bias"])
    out = x @ params["policy.weight"].T + params["policy.bias"]
    if "say.weight" in params:   # the radio's words, after the switches (see _split)
        out = jnp.concatenate([out, x @ params["say.weight"].T + params["say.bias"]], -1)
    value = (x @ params["value.weight"].T + params["value.bias"])[:, 0]
    return out[:, :N_MOVE], out[:, N_MOVE:], params["log_std"], value, h


def replay(params: dict, obs, done, h0):
    """A brain with memory, run again over each round's decisions in order (for learning): obs (T, envs, obs_size),
    done (T, envs): the round ended at that decision (its memory is wiped after it), h0 (envs, size): the memory at
    the first one. -> forward's outputs, flattened to (T * envs, ...) in that order."""
    def one(h, x):
        mean, logits, _, value, h2 = forward_mem(params, x[0], h)
        return jnp.where(x[1][:, None], 0.0, h2), (mean, logits, value)
    _, (mean, logits, value) = lax.scan(one, h0, (obs, done))
    flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
    return flat(mean), flat(logits), params["log_std"], flat(value)


def to_chunks(x, L: int):
    """(T, n, ...) -> (L, T // L * n, ...): the rollout cut into pieces of L decisions, piece c of round i at
    column c * n + i (the order h[::L].reshape(-1, size) gives the memories they start from)."""
    T, n = x.shape[:2]
    x = jnp.swapaxes(x.reshape(T // L, L, n, *x.shape[2:]), 0, 1)
    return x.reshape(L, T // L * n, *x.shape[3:])


def forward(params: dict, obs: jax.Array):
    """obs (batch, obs_size) -> move means (batch, 3), switch logits (batch, 3 or 4), log std (3,), value (batch,).
    (A brain with memory: as at the first decision of a round, nothing remembered yet.)"""
    return forward_mem(params, obs, jnp.zeros((obs.shape[0], memory_size(params))))[:4]


def sample(key, mean, logits, log_std, words: int = 0):
    """A try around the brain's guess: noisy moves, coin-flip switches (and with a radio, a word drawn from its
    choices, 0 = silence, in the last column). -> (moves, switches 0/1 [+ word])."""
    km, ks = jax.random.split(key)
    moves = mean + jnp.exp(log_std) * jax.random.normal(km, mean.shape)
    sl, wl = _split(logits, words)
    if words:
        ks, kw = jax.random.split(ks)
    switches = (jax.random.uniform(ks, sl.shape) < jax.nn.sigmoid(sl)).astype(jnp.float32)
    if words:
        switches = jnp.concatenate([switches, jax.random.categorical(kw, wl)[..., None].astype(jnp.float32)], -1)
    return moves, switches


def log_prob(mean, logits, log_std, moves, switches, live=None, words: int = 0):
    """live (batch, switches [+ 1]), 0/1: switches that can do something right now (throw: a flashbang in hand, none
    in the air; the word: up and not mid-word). A dead one is left out: its coin flip changes nothing, so it can't be
    credited or blamed (ep9 3z)."""
    g = (-0.5 * ((moves - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    sl, wl = _split(logits, words)
    n = sl.shape[-1]
    sw = switches[..., :n]
    b = sw * jax.nn.log_sigmoid(sl) + (1 - sw) * jax.nn.log_sigmoid(-sl)
    out = g + (b if live is None else b * live[..., :n]).sum(-1)
    if words:
        w = switches[..., -1].astype(jnp.int32)
        lw = jnp.take_along_axis(jax.nn.log_softmax(wl), w[..., None], -1)[..., 0]
        out = out + (lw if live is None else lw * live[..., -1])
    return out


def entropy(logits, log_std, live=None, words: int = 0):
    g = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
    sl, wl = _split(logits, words)
    n = sl.shape[-1]
    p = jax.nn.sigmoid(sl)
    b = -(p * jax.nn.log_sigmoid(sl) + (1 - p) * jax.nn.log_sigmoid(-sl))
    out = g + (b if live is None else b * live[..., :n]).sum(-1).mean()
    if words:
        hw = -(jax.nn.softmax(wl) * jax.nn.log_softmax(wl)).sum(-1)
        out = out + (hw if live is None else hw * live[..., -1]).mean()
    return out


def switches_for(game_cfg: CQBConfig, me: int) -> int:
    """How many on/off outputs a brain for this side and these rules has: fire, interact, run, then throw (with
    flashbangs, or as a never-live placeholder when there are magazines but no flashbangs), then reload."""
    guns = game_cfg.magazine > 0
    return N_SWITCH + int(game_cfg.flashbangs[me] > 0 or guns) + int(guns)


def live_switches(game: CQBGame, st, op: int, n_switch: int, words: int = 0):
    """(rounds, n_switch [+ 1]) 0/1 for log_prob, for operator `op`: fire, interact and run always count; throw only
    with a flashbang in hand and none in the air; reload only when the magazine isn't full, there are rounds to load
    and no reload on; (with a radio, last) the word only when up and not mid-word."""
    live = jnp.ones(st.alive.shape[:1] + (n_switch,))
    if n_switch > N_SWITCH:
        live = live.at[:, N_SWITCH].set(((st.bangs[:, op] > 0) & (st.bang_timer[:, op] == 0)).astype(jnp.float32))
    if n_switch > N_SWITCH + 1:
        can = (st.mag[:, op] < game.cfg.magazine) & (st.reserve[:, op] > 0) & (st.reload_timer[:, op] == 0)
        live = live.at[:, N_SWITCH + 1].set(can.astype(jnp.float32))
    if words:
        talk = st.alive[:, op] & (st.said_age[:, op] + 1 >= game.word_steps)
        live = jnp.concatenate([live, talk[:, None].astype(jnp.float32)], 1)
    return live


def listen_kl(mean, logits, log_std, mean_m, logits_m, live, words: int):
    """Per decision: how far the choices (moves, live switches, the word) are from the same decision's choices with
    nothing heard (KL divergence, theirs from the muted ones; ep11's listening nudge)."""
    g = (0.5 * ((mean - mean_m) / jnp.exp(log_std)) ** 2).sum(-1)
    sl, wl = _split(logits, words)
    sm, wm = _split(logits_m, words)
    n = sl.shape[-1]
    p, q = jax.nn.sigmoid(sl), jax.nn.sigmoid(sm)
    b = p * (jax.nn.log_sigmoid(sl) - jax.nn.log_sigmoid(sm)) + (1 - p) * (jax.nn.log_sigmoid(-sl) - jax.nn.log_sigmoid(-sm))
    out = g + (b * live[..., :n]).sum(-1)
    if words:
        lp, lq = jax.nn.log_softmax(wl), jax.nn.log_softmax(wm)
        out = out + (jnp.exp(lp) * (lp - lq)).sum(-1) * live[..., -1]
    return out


def to_action(moves, switches, words: int = 0):
    """The game's 6 numbers (7 with throw, ...): moves as they are (the game clips them), switches as +1 / -1; with a
    radio, the word (last column of switches) goes in the game's SAY slot (the switches it lacks: off)."""
    if not words:
        return jnp.concatenate([moves, 2.0 * switches - 1.0], -1)
    pad = -jnp.ones(moves.shape[:-1] + (cqb.SAY - N_MOVE - (switches.shape[-1] - 1),))
    return jnp.concatenate([moves, 2.0 * switches[..., :-1] - 1.0, pad, switches[..., -1:]], -1)


def _best(mean, logits, words: int, key=None):
    """Best-guess moves and switches; the word: the likeliest choice, or (key given) drawn from the choices."""
    sl, wl = _split(logits, words)
    sw = (sl > 0).astype(jnp.float32)
    if words:
        w = jnp.argmax(wl, -1) if key is None else jax.random.categorical(key, wl)
        sw = jnp.concatenate([sw, w[..., None].astype(jnp.float32)], -1)
    return to_action(jnp.clip(mean, -1, 1), sw, words)


def best_guess(params: dict, obs):
    mean, logits, _, _ = forward(params, obs)
    return _best(mean, logits, n_words(params))


def brain(params: dict, who: int, sample_moves: bool = False, sample_words: bool = False):
    """A brain as a player (game, state, who, key) -> action, like the scripts (`who` defaults to the one given here;
    with teams, one brain plays every operator of its side): its best guess, or
    (sample_moves) a try around it like in training. A brain with memory is a player with a `memory` size that takes
    and returns its state: (game, state, who, key, h) -> (action, h); play_rounds keeps it, from zeros each round.
    sample_words (a brain with a radio, ep11): best-guess moves, but the word drawn from its choices (the likeliest
    choice alone can be the same word, or silence, every time even when the brain talks a lot)."""
    n_in = params["body.0.weight"].shape[1]
    W = n_words(params)
    if has_memory(params):
        def remember(game: CQBGame, st, _who=who, key=None, h=None):
            obs = game.observe(st, _who)[None, :n_in]
            mean, logits, log_std, _, h = forward_mem(params, obs, h[None])
            if sample_moves:
                act = to_action(*sample(key, mean, logits, log_std, W), W)
            else:
                act = _best(mean, logits, W, key if sample_words else None)
            return cqb.full_action(act[0]), h[0]
        remember.memory = memory_size(params)
        return remember

    def play(game: CQBGame, st, _who=who, key=None):
        obs = game.observe(st, _who)[None, :n_in]  # a brain from before flashbangs reads the first 107 (the same ones)
        if not sample_moves:
            if sample_words and W:
                mean, logits, _, _ = forward(params, obs)
                return cqb.full_action(_best(mean, logits, W, key)[0])
            return cqb.full_action(best_guess(params, obs)[0])
        mean, logits, log_std, _ = forward(params, obs)
        return cqb.full_action(to_action(*sample(key, mean, logits, log_std, W), W)[0])
    return play


def load(path) -> tuple[dict, dict]:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: jnp.asarray(v.numpy()) for k, v in ck["model"].items()}, ck


def _frozen(spec: str, who: int, game_cfg: CQBConfig | None):
    """A saved brain "runs/a:175" as a player; with game_cfg, fitted to that game first (`fit`)."""
    from ailearns.checkpoints import pick_checkpoint
    run, _, ck = spec.partition(":")
    params, meta = load(pick_checkpoint(run, ck or "last"))
    return brain(fit(params, meta, game_cfg, who) if game_cfg is not None else params, who)


def pool_members(spec: str, who: int, game_cfg: CQBConfig | None = None) -> list:
    """The players in a pool spec, "camper,runs/a:175,runs/b:381" (scripts and frozen brains of any size); with
    game_cfg, each brain is fitted to that game (ep12: a 2v2 brain in a 3v2 game)."""
    members = []
    for m in (x.strip() for x in spec.split(",")):
        if m in S.SCRIPTS:
            members.append(S.SCRIPTS[m])
        else:
            members.append(_frozen(m, who, game_cfg))
            if getattr(members[-1], "memory", 0):
                raise SystemExit(f"{m} has a memory: training against one isn't supported yet")
    return members


def pool_player(spec: str, who: int):
    """A pool of opponents as one player (game, state, who, key, slot) -> action: training round `slot` always plays
    member slot % K, so every member gets the same share of practice (tag's pool, episode 7). Every member plays and
    one is kept: the plain version (training uses pool_actions, which gives the same actions)."""
    members = pool_members(spec, who)
    if len(members) == 1:
        return lambda game, st, who, key, slot: cqb.full_action(members[0](game, st, who, key))

    def play(game, st, who, key, slot):
        acts = jnp.stack([cqb.full_action(m(game, st, who, key)) for m in members])
        return acts[slot % len(members)]
    return play


def pool_actions(members: list, game: CQBGame, env, who: int, keys):
    """pool_player's actions for a whole batch of rounds (round i plays member i % K), with each member working
    only on its own rounds: K times less work than every member playing every round (ep9 3af)."""
    n, K = keys.shape[0], len(members)
    acts = jnp.zeros((n, cqb.ACTION_SIZE))
    for m, player in enumerate(members):
        idx = np.arange(m, n, K)
        sub = jax.tree.map(lambda x: x[idx], env)
        acts = acts.at[idx].set(jax.vmap(lambda st, k: cqb.full_action(player(game, st, who, k)))(sub, keys[idx]))
    return acts


def opponent_player(spec: str, who: int, game_cfg: CQBConfig | None = None):
    """A script name, or frozen brains "runs/a:175" (the first one plays) -> player (fitted to game_cfg if given)."""
    if spec in S.SCRIPTS:
        return S.SCRIPTS[spec]
    return _frozen(spec.split(",")[0].strip(), who, game_cfg)


# ── training ────────────────────────────────────────────────────────────────

def start_rounds(game: CQBGame, key, n: int, n_local: int | None = None, n_drill: int = 0):
    """n rounds, each with a random part of its clock already gone. Only the first round of each slot is
    shorter; without this all n start together and nearly all time out together, so every update sees the
    same stretch of every round (ep9 3f). n_drill: the first n_drill slots of each device's n_local are drills."""
    k_init, k_t = jax.random.split(key)
    keys = jax.random.split(k_init, n)
    env = jax.vmap(game.init)(keys)
    if n_drill:
        drill = (jnp.arange(n) % n_local) < n_drill
        stacked = jax.vmap(game.init_stacked)(keys)
        env = jax.tree.map(lambda a, b: jnp.where(drill.reshape(-1, *[1] * (a.ndim - 1)), b, a), env, stacked)
    return env._replace(t=jax.random.randint(k_t, (n,), 0, game.T, dtype=env.t.dtype))


def drill_slots(cfg: CQBPPOConfig, n_local: int) -> int:
    return int(round(cfg.stacked * n_local))


def make_update(cfg: CQBPPOConfig, game: CQBGame, n_local: int, n_dev: int):
    T = cfg.n_steps
    me = ASSAULT if cfg.side == "assault" else DEFENSE
    mine, theirs_ops = game.side_ops(me), game.side_ops(1 - me)   # operators: mine share the brain
    K = len(mine)
    rows = n_local * K                             # samples per decision: each of my operators in each round
    mb = cfg.minibatch // n_dev
    n_mb = (T * rows) // mb
    sign = 1.0 if me == ASSAULT else -1.0          # the game's rewards are Assault's
    W = int(game.words[mine[0]])                   # radio words (ep11; 0: none)
    members = pool_members(cfg.opponent, 1 - me, game.cfg)
    vobs = jax.vmap(lambda s: jnp.stack([game.observe(s, op) for op in mine]))   # (rounds, K, obs)
    vpriv = jax.vmap(lambda s: jnp.stack([game.privileged(s, op) for op in mine]))   # (rounds, K, PRIV): critic only
    crit = cfg.critic_priv
    gss = cfg.guess_coef > 0
    need_priv = crit or gss
    flat = lambda x: x.reshape(rows, *x.shape[2:])                                 # noqa: E731
    n_drill = drill_slots(cfg, n_local)
    vreset = jax.vmap(game.reset_if)
    vreset_drill = jax.vmap(lambda st, k: jax.tree.map(lambda a, b: jnp.where(st.done, a, b), game.init_stacked(k), st))
    slots = jnp.arange(n_local)  # which pool member each round plays (the same on every device)
    drill = slots < n_drill      # these slots are entry drills (each kind reset on its own: no wasted work)

    def reset(env, keys):
        if not n_drill:
            return vreset(env, keys)
        if n_drill == n_local:                                   # drills only
            return vreset_drill(env, keys)
        cut = lambda a, b: jax.tree.map(lambda x: x[a:b], env)  # noqa: E731
        head, tail = vreset_drill(cut(0, n_drill), keys[:n_drill]), vreset(cut(n_drill, None), keys[n_drill:])
        return jax.tree.map(lambda a, b: jnp.concatenate([a, b]), head, tail)

    def play(st, mine_acts, theirs_acts):
        """mine_acts (K, 8), theirs_acts (K', 8) -> the next state and each of my operators' reward (K,): the team's
        result, plus (a hint) the cells it saw for the first time."""
        acts = [None] * game.N
        for k, op in enumerate(mine):
            acts[op] = cqb.full_action(mine_acts[k])
        for k, op in enumerate(theirs_ops):
            acts[op] = theirs_acts[k]
        new, reward = game.step(st, jnp.stack(acts))
        new_cells = jnp.stack([(new.cleared[op] & ~st.cleared[op]).sum() for op in mine])
        return new, sign * reward + cfg.explore_reward * new_cells * ~st.done
    vplay = jax.vmap(play)

    memory = cfg.memory > 0

    def loss_fn(params, b):
        if memory:
            mean, logits, log_std, value = replay(params, b["obs"], b["done"], b["h0"])
            b = {k: v.reshape(-1, *v.shape[2:]) for k, v in b.items() if k not in ("h0", "done", "obs")}
        else:
            mean, logits, log_std, value = forward(params, b["obs"])
        if crit:
            value = critic_value(params, b["obs"], b["priv"])
        logp = log_prob(mean, logits, log_std, b["moves"], b["switches"], b["live"], W)
        ratio = jnp.exp(logp - b["logp"])
        a = b["adv"]
        a = (a - a.mean()) / (a.std(ddof=1) + 1e-8)
        pg = jnp.maximum(-a * ratio, -a * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip))
        if game.teams_on:   # a teammate who's down decides nothing: its samples don't teach the moves
            pg = (pg * b["acting"]).sum() / jnp.maximum(b["acting"].sum(), 1.0)
        else:
            pg = pg.mean()
        vl = 0.5 * ((value - b["ret"]) ** 2).mean()
        # the bonus and the stats count only operators who are up (ep11 step 8: a downed one's choices change nothing,
        # so the bonus drifted them to coin flips, and p_throw showed 0.13 while standing operators threw at 0.002)
        live = b["live"] * b["acting"][:, None] if game.teams_on else b["live"]
        ent = entropy(logits, log_std, live, W)
        loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
        gstat = {}
        if gss:   # the guess, scored where the operator is up (a downed one's view says nothing)
            mask = b["acting"] if game.teams_on else jnp.ones(b["obs"].shape[0])
            err = ((guess(params, b["obs"]) - b["priv"]) ** 2).mean(-1)
            g_err = (err * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            loss = loss + cfg.guess_coef * g_err
            gstat = {"guess_err": g_err}
        bias = {}
        if W and cfg.listen_coef:   # the same decisions with nothing heard: how much did the word heard change them?
            K1 = W + 1
            heard = slice(b["obs"].shape[-1] - 2 * K1, b["obs"].shape[-1] - K1)
            fresh = b["obs"][:, heard.stop - 1] > 0                                  # heard something lately
            muted = b["obs"].at[:, heard].set(0.0)
            mean_m, logits_m, _, _ = jax.lax.stop_gradient(forward(params, muted))
            kl = listen_kl(mean, logits, log_std, mean_m, logits_m, b["live"], W)
            on = fresh.astype(jnp.float32) * (b["acting"] if game.teams_on else 1.0)
            listen = (jnp.minimum(kl, cfg.listen_cap) * on).sum() / jnp.maximum(on.sum(), 1.0)
            loss = loss - cfg.listen_coef * listen
            bias["listen_kl"] = listen
        if W and cfg.signal_coef:   # words that depend on the situation: spread of all words minus spread in each case
            lw = b["live"][:, -1]
            pw = jax.nn.softmax(_split(logits, W)[1])
            avg = (pw * lw[:, None]).sum(0) / jnp.maximum(lw.sum(), 1.0)
            h_avg = -(avg * jnp.log(avg + 1e-8)).sum()
            h_each = (-(pw * jnp.log(pw + 1e-8)).sum(-1) * lw).sum() / jnp.maximum(lw.sum(), 1.0)
            loss = loss - cfg.signal_coef * (h_avg - h_each)
            bias["word_info"] = h_avg - h_each
        sl, wl = _split(logits, W)
        n = sl.shape[-1]
        p = (jax.nn.sigmoid(sl) * live[:, :n]).sum(0) / jnp.maximum(live[:, :n].sum(0), 1.0)  # where it counts
        words = {}
        if W:   # how likely each choice is, where the brain can talk: silence, then the words
            lw = live[:, -1:]
            pw = (jax.nn.softmax(wl) * lw).sum(0) / jnp.maximum(lw.sum(), 1.0)
            words = {f"p_say{k}": pw[k] for k in range(W + 1)}
        return loss, {"pg_loss": pg, "v_loss": vl, "entropy": ent, "std": jnp.exp(log_std).mean(),
                      "clipfrac": (jnp.abs(ratio - 1) > cfg.clip).mean(),
                      **{f"p_{n}": p[i] for i, n in enumerate(SWITCHES[:p.shape[0]])}, **words, **bias, **gstat}

    def update(params, opt, env, obs, key, lr, pol, h):
        key, k_roll, k_learn = jax.random.split(key, 3)
        dev = lax.axis_index("d")
        k_roll, k_learn = jax.random.fold_in(k_roll, dev), jax.random.fold_in(k_learn, dev)

        def tick(carry, k):
            env, obs, h = carry
            mean, logits, log_std, value, h_next = forward_mem(params, obs, h)
            if need_priv:
                priv = flat(vpriv(env))
            if crit:
                value = critic_value(params, obs, priv)
            ka, ko, kr = jax.random.split(k, 3)
            moves, switches = sample(ka, mean, logits, log_std, W)
            n_sw = logits.shape[-1] - (W + 1 if W else 0)
            live = flat(jnp.stack([live_switches(game, env, op, n_sw, W) for op in mine], 1))
            logp = log_prob(mean, logits, log_std, moves, switches, live, W)
            up = env.alive[:, jnp.asarray(mine)]
            if game.pilot_on and game.cfg.callout:   # (ep11 10zc) the pilot's moves do nothing and with callout its words
                up = up.at[:, 0].set(False)   # are fixed: its samples teach nothing, left out like a downed operator's.
                # Without callout (10zk) they stay in: its word choices are what it has to learn
            acting = flat(up).astype(jnp.float32)
            theirs = jnp.stack([pool_actions(members, game, env, op, jax.random.split(
                ko if n == 0 else jax.random.fold_in(ko, n), n_local)) for n, op in enumerate(theirs_ops)], 1)
            env, reward = vplay(env, to_action(moves, switches, W).reshape(n_local, K, -1), theirs)
            spoke = ((env.said_age[:, jnp.asarray(mine)] == 0) & (env.said[:, jnp.asarray(mine)] > 0)).sum()
            # read before the reset
            a_out, d_out = game.sides_out(env.alive)
            out = {"done": env.done, "won": env.done & (env.outcome == (1 if me == ASSAULT else -1)),
                   "trade": env.done & a_out & d_out,
                   "timeout": env.done & ~a_out & ~d_out, "seconds": env.t * game.cfg.dt,
                   "met": env.done & (env.since[:, mine[0]] < game.T).any(-1),
                   "cells": jnp.where(env.done, jnp.stack([env.cleared[:, op].mean(1) for op in mine]).mean(0), 0.0),
                   "spoke": spoke}
            term = jnp.repeat(env.done, K)                                # each of my operators' samples
            if memory:
                h_next = jnp.where(term[:, None], 0.0, h_next)          # a new round: nothing remembered
            env = reset(env, jax.random.split(kr, n_local))
            return (env, flat(vobs(env)), h_next), {"obs": obs, "moves": moves, "switches": switches, "live": live,
                                                    "logp": logp, "val": value, "rew": flat(reward), "h": h,
                                                    "term": term, "acting": acting, **out,
                                                    **({"priv": priv} if need_priv else {})}
        (env, obs_next, h_end), tr = lax.scan(tick, (env, obs, h), jax.random.split(k_roll, T))

        next_value = (critic_value(params, obs_next, flat(vpriv(env))) if crit
                      else forward_mem(params, obs_next, h_end)[3])

        def gae(carry, x):
            last, nv = carry
            nonterminal = 1.0 - x["term"].astype(jnp.float32)
            delta = x["rew"] + cfg.gamma * nv * nonterminal - x["val"]
            last = delta + cfg.gamma * cfg.lam * nonterminal * last
            return (last, x["val"]), last
        _, adv = lax.scan(gae, (jnp.zeros_like(next_value), next_value), tr, reverse=True)
        batch = {"obs": tr["obs"], "moves": tr["moves"], "switches": tr["switches"], "live": tr["live"],
                 "logp": tr["logp"], "adv": adv, "ret": adv + tr["val"]}
        if game.teams_on:
            batch["acting"] = tr["acting"]
        if need_priv:
            batch["priv"] = tr["priv"]
        if memory:   # pieces of mem_chunk decisions, each replayed in order from the memory it had in play
            L = cfg.mem_chunk
            batch = {k: to_chunks(v, L) for k, v in {**batch, "done": tr["term"]}.items()}
            batch["h0"] = tr["h"][::L].reshape(-1, tr["h"].shape[-1])
        else:
            batch = jax.tree.map(lambda x: x.reshape(T * rows, *x.shape[2:]), batch)

        def epoch(carry, k):
            def step(carry, idx, mb_batch):
                params, opt = carry
                b = jax.tree.map(lambda x: x[idx], batch) if mb_batch is None else mb_batch
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
                grads = lax.pmean(grads, "d")
                grads = {k: g if k.startswith(("value.", "critic.")) else g * pol
                         for k, g in grads.items()}                                   # pol 0: value only
                norm = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
                scale = jnp.minimum(1.0, cfg.max_grad_norm / (norm + 1e-6))
                new, opt = adam_step(params, opt, jax.tree.map(lambda g: g * scale, grads), lr)
                if memory and cfg.mem_lr != 1.0:   # the memory's weights take a smaller step
                    new = {k: params[k] + cfg.mem_lr * (v - params[k]) if k in MEMORY_KEYS else v for k, v in new.items()}
                params = new
                return (params, opt), aux
            if memory:
                n_seq, S = max(mb // cfg.mem_chunk, 1), batch["h0"].shape[0]   # pieces per minibatch, pieces
                perm = jax.random.permutation(k, S)[: (S // n_seq) * n_seq].reshape(-1, n_seq)
                pick = lambda idx: {k: (v[idx] if k == "h0" else v[:, idx]) for k, v in batch.items()}  # noqa: E731
                return lax.scan(lambda c, idx: step(c, None, pick(idx)), carry, perm)
            perm = jax.random.permutation(k, T * rows)[: n_mb * mb].reshape(n_mb, mb)
            return lax.scan(lambda c, idx: step(c, idx, None), carry, perm)
        (params, opt), aux = lax.scan(epoch, (params, opt), jax.random.split(k_learn, cfg.epochs))

        m = {"ended": tr["done"].sum(), "won": tr["won"].sum(), "trade": tr["trade"].sum(),
             "timeout": tr["timeout"].sum(), "met": tr["met"].sum(), "cells": tr["cells"].sum(),
             "win_seconds": jnp.where(tr["won"], tr["seconds"], 0).sum(), "reward": tr["rew"].sum(),
             "drill_ended": (tr["done"] & drill).sum(), "drill_won": (tr["won"] & drill).sum(),
             "spoke": tr["spoke"].sum()}
        m = jax.tree.map(lambda x: lax.psum(x, "d"), m)
        m.update(jax.tree.map(lambda x: lax.pmean(x.mean(), "d"), aux))
        return params, opt, (env, h_end), obs_next, key, m

    def update_with(params, opt, env_h, obs, key, lr, pol):
        env, h = env_h
        return update(params, opt, env, obs, key, lr, pol, h)

    return update_with


def train(cfg: CQBPPOConfig, run_dir: str | Path, resume: bool = False, max_hours: float | None = None,
          game_cfg: CQBConfig = CQBConfig()) -> None:
    import torch
    game_cfg = replace(game_cfg, **cqb.HOUSES[cfg.house])
    game_cfg = replace(game_cfg, time_cost=cfg.time_cost, flashbangs=tuple(cfg.flashbangs), magazine=cfg.magazine,
                       spare_mags=cfg.spare_mags, time_limit=cfg.time_limit, blind_away=cfg.blind_away, hostage=cfg.hostage, hostage_deep=cfg.hostage_deep, extract=cfg.extract, hostage_noise=cfg.hostage_noise, health=cfg.health, react=cfg.react, team_sizes=tuple(cfg.teams), bump=cfg.bump,
                       through_drywall=cfg.through_drywall, unseen_hit=cfg.unseen_hit, hostage_hit=cfg.hostage_hit,
                       ready=cfg.ready, stun=cfg.stun, share_sight=cfg.share_sight, radio=tuple(cfg.radio),
                       split_entry=cfg.split_entry, relay=cfg.relay, intel=cfg.intel,
                       route=cfg.route, feed=cfg.feed, pilot=cfg.pilot, callout=cfg.callout,
                       callout_way=cfg.callout_way)
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    devices = jax.devices()
    n_dev = len(devices)
    if cfg.n_envs % n_dev or cfg.minibatch % n_dev:
        raise SystemExit(f"--envs and minibatch must divide by the {n_dev} devices")
    if (cfg.listen_coef or cfg.signal_coef) and (cfg.memory or not any(cfg.radio)):
        raise SystemExit("the listening / signalling nudges need a radio and a brain without memory")
    if cfg.memory and cfg.n_steps % cfg.mem_chunk:
        raise SystemExit(f"n_steps ({cfg.n_steps}) must divide into pieces of mem_chunk ({cfg.mem_chunk})")
    sha = _git_version()
    state = None
    if resume:
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        meta = json.loads((run_dir / "config.json").read_text())
        meta["total_steps"] = cfg.total_steps
        meta.setdefault("resumed", []).append({"update": state["update"], "git": sha, "device": str(devices[0])})
        (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    else:
        if (run_dir / "metrics.jsonl").exists():
            raise SystemExit(f"{run_dir} already has a run; use --resume or a new directory")
        (run_dir / "config.json").write_text(json.dumps(
            {**asdict(cfg), "game": "cqb", "cqb": asdict(game_cfg), "git": sha, "backend": "jax",
             "device": f"{n_dev} x {devices[0].device_kind}"}, indent=2))
    first = state["update"] + 1 if state else 1
    print(f"JAX on {n_dev} x {devices[0].device_kind}; building {cfg.buildings} buildings …", flush=True)

    t_b = time.time()
    game = CQBGame(make_buildings(list(TRAIN_SEEDS)[: cfg.buildings], game_cfg), game_cfg)
    print(f"buildings ready in {time.time() - t_b:.1f} s", flush=True)
    me = ASSAULT if cfg.side == "assault" else DEFENSE
    n_switch = switches_for(game_cfg, me)
    if state:
        params = {k: jnp.asarray(v.numpy()) for k, v in state["model"].items()}
        opt = jax.tree.map(jnp.asarray, state["optimizer"])
    elif cfg.init_from:
        from ailearns.checkpoints import pick_checkpoint
        run, _, ck = cfg.init_from.partition(":")
        params, src_ck = load(pick_checkpoint(run, ck or "last"))
        if "cqb" in src_ck:   # (ep11 10zb) a brain from another observation layout: each part to its new place
            src = CQBConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in src_ck["cqb"].items()})
            if params["body.0.weight"].shape[1] == src.obs_size and obs_blocks(src) != obs_blocks(game_cfg):
                params = transplant(params, src, game_cfg)
        params = widen(params, game_cfg.obs_size, n_switch)
        if cfg.reset_std:
            params["log_std"] = jnp.full_like(params["log_std"], cfg.init_log_std)
    else:
        params = init_params(jax.random.key(cfg.seed), game_cfg.obs_size, cfg.hidden, cfg.init_log_std, n_switch)
    if not state:
        words = int(game.words[game.side_ops(me)[0]])
        if words and n_words(params) != words:
            params = add_radio(params, words)
        if cfg.critic_priv:
            if cfg.memory:
                raise SystemExit("critic_priv: not with a memory brain")
            if params.get("critic.0.weight", jnp.zeros((0, 0))).shape[1:] != (game_cfg.obs_size + PRIV,):
                params = {k: v for k, v in params.items() if not k.startswith("critic.")}
                params = add_critic(params, game_cfg.obs_size, jax.random.key(cfg.seed + 11), cfg.hidden)
        if cfg.guess_coef > 0:
            if cfg.memory:
                raise SystemExit("guess: not with a memory brain")
            if "guess.weight" not in params:
                params = add_guess(params, jax.random.key(cfg.seed + 13))
        if cfg.memory and not has_memory(params):
            params = add_memory(params, cfg.memory, jax.random.key(cfg.seed + 7))
        opt = adam_init(params)

    mesh = Mesh(np.array(devices), ("d",))
    rep, shard = NamedSharding(mesh, P()), NamedSharding(mesh, P("d"))
    T, N = cfg.n_steps, cfg.n_envs
    update = make_update(cfg, game, N // n_dev, n_dev)
    update = jax.jit(jax.shard_map(update, mesh=mesh, in_specs=(P(), P(), P("d"), P("d"), P(), P(), P()),
                                   out_specs=(P(), P(), P("d"), P("d"), P(), P())),
                     donate_argnums=(0, 1, 2, 3))
    key = jax.random.key(cfg.seed + 1000 + first)
    key, k_env = jax.random.split(key)
    n_local = N // n_dev
    env = jax.jit(lambda k: start_rounds(game, k, N, n_local, drill_slots(cfg, n_local)), out_shardings=shard)(k_env)
    mine = game.side_ops(me)                     # with teams the brain plays each of these: K samples per round
    K = len(mine)
    obs = jax.jit(lambda e: jax.vmap(lambda s: jnp.stack([game.observe(s, op) for op in mine]))(e).reshape(N * K, -1),
                  out_shardings=shard)(env)
    # each operator's memory travels with it (zeros: nothing remembered; also after a resume)
    env = (env, jax.jit(lambda: jnp.zeros((N * K, memory_size(params))), out_shardings=shard)())
    params, opt, key = jax.device_put((params, opt, key), rep)

    n_updates = cfg.total_steps // (T * N)
    rounds = state["rounds"] if state else 0
    start = time.time() - (state["elapsed"] if state else 0.0)
    session_start = time.time()
    log = open(run_dir / "metrics.jsonl", "a")
    from ailearns.agents import tracking
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    meta = {"config": asdict(cfg), "cqb": asdict(game_cfg), "backend": "jax", "game": "cqb"}

    def save(update_no: int, stats: dict) -> None:
        torch.save({"model": to_torch(params), **meta, "update": update_no, "steps": update_no * T * N,
                    "rounds": rounds, "stats": stats}, run_dir / "checkpoints" / f"u{update_no:05d}.pt")

    def save_latest(update_no: int) -> None:
        torch.save({"model": to_torch(params), "optimizer": jax.tree.map(np.asarray, opt), **meta,
                    "update": update_no, "rounds": rounds, "elapsed": time.time() - start},
                   run_dir / "latest.pt.tmp")
        (run_dir / "latest.pt.tmp").replace(run_dir / "latest.pt")

    if not state:
        save(0, {})
    wins = [x["win_rate"] for x in map(json.loads, open(run_dir / "metrics.jsonl"))] if state else []
    best, best_at = -1.0, first - 1
    for update_no in range(first, n_updates + 1):
        lr = jnp.float32(cfg.lr * (1 - (update_no - 1) / n_updates))
        pol = jnp.float32(0.0 if update_no <= cfg.vf_warmup else 1.0)
        params, opt, env, obs, key, m = update(params, opt, env, obs, key, lr, pol)
        m = jax.device_get(m)
        ended = int(m["ended"])
        rounds += ended
        rate = lambda k: float(m[k]) / ended if ended else None  # noqa: E731
        stats = {
            "update": update_no, "steps": update_no * T * N, "rounds": rounds, "time": round(time.time() - start, 1),
            "win_rate": rate("won"), "trade_rate": rate("trade"), "timeout_rate": rate("timeout"),
            "met_rate": rate("met"), "cells_seen": rate("cells"),
            "win_seconds": float(m["win_seconds"]) / int(m["won"]) if int(m["won"]) else None,
            "reward_per_step": float(m["reward"]) / (T * N),
            **{k: float(m[k]) for k in ("pg_loss", "v_loss", "entropy", "std", "clipfrac", "guess_err") if k in m},
            **{f"p_{n}": float(m[f"p_{n}"]) for n in SWITCHES if f"p_{n}" in m},
        }
        if any(game_cfg.radio):   # words started per operator-second, and how likely each choice is (0: silence)
            stats["words_per_s"] = float(m["spoke"]) / (T * N * K * game_cfg.dt)
            stats.update({k: float(m[k]) for k in m if k.startswith("p_say") or k in ("listen_kl", "word_info")})
        if cfg.stacked:   # drills and full rounds apart (win_rate is all of them)
            d_end, d_won = int(m["drill_ended"]), int(m["drill_won"])
            stats["drill_win_rate"] = d_won / d_end if d_end else None
            stats["full_win_rate"] = (int(m["won"]) - d_won) / (ended - d_end) if ended - d_end else None
        log.write(json.dumps(stats) + "\n")
        log.flush()
        tracking.log(tracker, stats)
        out_of_time = max_hours is not None and time.time() - session_start > max_hours * 3600
        wins.append(stats["win_rate"])
        recent = [w for w in wins[-25:] if w is not None]
        if len(wins) >= 25 and recent and np.mean(recent) > best + 0.005:
            best, best_at = float(np.mean(recent)), update_no
        plateaued = cfg.plateau > 0 and best >= 0 and update_no - best_at >= cfg.plateau   # (counting once averaged)
        if update_no in cfg.early_checkpoints or update_no % cfg.checkpoint_every == 0 or update_no == n_updates \
                or out_of_time or plateaued:
            save(update_no, stats)
            save_latest(update_no)
        if update_no % 10 == 0 or update_no == first:
            sps = (update_no - first + 1) * T * N / (time.time() - session_start)
            f = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
            print(f"update {update_no:5d}/{n_updates}  steps {stats['steps']:>11,}  rounds {rounds:>9,}  "
                  f"won {f(stats['win_rate'])}"
                  + (f" (drills {f(stats['drill_win_rate'])}, full {f(stats['full_win_rate'])})" if cfg.stacked else "")
                  + f"  met {f(stats['met_rate'])}  seen {f(stats['cells_seen'])}  "
                  f"fire {stats['p_fire']:.2f} run {stats['p_run']:.2f}"
                  + (f" throw {stats['p_throw']:.3f}" if "p_throw" in stats else "")
                  + f"  std {stats['std']:.2f}"
                  + (f"  guess err {stats['guess_err']:.3f}" if "guess_err" in stats else "")
                  + f"  {sps:,.0f} steps/s",
                  flush=True)
        if out_of_time:
            print(f"stopped after {max_hours} h at update {update_no}; carry on with --resume", flush=True)
            break
        if plateaued:
            print(f"stopped at update {update_no}: no new best 25-update win rate ({best:.3f} at update {best_at}) "
                  f"in {cfg.plateau} updates", flush=True)
            break
    log.close()
    tracking.finish(tracker)


# ── evaluation ──────────────────────────────────────────────────────────────

def evaluate(run: str, checkpoint: str = "last", first: int = 0, n: int = 64, seed: int = 5000,
             opponent: str | None = None) -> dict:
    """Best-guess play, one round in each test building first..first+n (never trained in). Choose
    checkpoints on 0-63, report from 64-255 (playbook: never report from the rounds you picked with)."""
    from ailearns.checkpoints import pick_checkpoint
    params, ck = load(pick_checkpoint(run, str(checkpoint)))
    cfg, game_cfg = ck["config"], CQBConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["cqb"].items()})
    seeds = list(cqb.TEST_SEEDS)[first:first + n]
    game = CQBGame(make_buildings(seeds, game_cfg), game_cfg)
    me = ASSAULT if cfg["side"] == "assault" else DEFENSE
    mine, theirs = brain(params, me), opponent_player(opponent or cfg["opponent"], 1 - me)
    a, d = (mine, theirs) if me == ASSAULT else (theirs, mine)
    r = S.play_rounds(game, a, d, np.arange(n), seed=seed)
    won = r["outcome"] == (1 if me == ASSAULT else -1)
    a_out, d_out = (np.asarray(x) for x in game.sides_out(jnp.asarray(r["alive"])))
    mine = game.side_ops(me)
    return {"run": str(run), "update": ck["update"], "moves": ck["steps"], "rounds": n,
            "win_rate": float(won.mean()), "trade_rate": float((a_out & d_out).mean()),
            "timeout_rate": float((~a_out & ~d_out).mean()),
            "met_rate": float((r["first_seen"][:, mine] >= 0).any(1).mean()), "result": r}
