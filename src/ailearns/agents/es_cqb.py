"""Evolution strategies for CQB (ep9 3az): can evolution find the flashbang, where trial and error (PPO) never did?

PPO explores with noise on every decision, so its throws come at random moments and random moves follow them: they
blind the thrower (3x-3ab) and it learns not to throw. Evolution explores in the weights instead (OpenAI-ES): each
member of a population is the brain plus a small random change, plays whole rounds with its best guess, and is
scored on the result only. A member that throws, throws in the same kind of situation every round, so a sequence
that works (throw from beside the door, stay out of line, go in) can show up whole and be selected.

- Mirrored pairs (brain + change, brain - change), every member on the same rounds that generation (the same
  buildings, starts and luck), scores ranked; the brain moves along the ranked changes (Adam).
- Only what decides play changes (the two layers and the policy outputs; not the value head or the move noise).
- The throw output explores boldly (`sigma_throw`): a brain that learned not to throw sits ~16 below the throw line,
  and small changes would never make a member throw, so there would be nothing to select. Everything else, the
  movement around a throw included, changes in small steps (`sigma`). Where it throws is still the members' doing.
- Logged per generation: the brain's own score on that generation's rounds, the members' throws per round, and per
  throw whether it blinded the enemy or the thrower.
Checkpoints and config.json are in the PPO runs' format, so cqb_eval's report and the footage tools work as they are.
"""

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import ppo_cqb as A
from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_jax import adam_init, adam_step, to_torch
from ailearns.games import cqb
from ailearns.games.cqb import ASSAULT, DEFENSE, TRAIN_SEEDS, CQBConfig, CQBGame, make_buildings

EVOLVED = ("body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias", "policy.weight", "policy.bias")


def _evolved(params: dict) -> tuple:
    """What evolution changes: EVOLVED, and a radio's word head when the brain has one (ep11 10x)."""
    return EVOLVED + (("say.weight", "say.bias") if "say.weight" in params else ())


@dataclass
class CQBESConfig:
    side: str = "assault"
    init_from: str = ""          # the brain evolution starts from (runs/x:update)
    opponent: str = "camper"     # a pool, as in PPO: the rounds are shared out among its members
    time_cost: float = 0.0       # the game's rules, as in CQBPPOConfig
    flashbangs: tuple = (0, 0)
    magazine: int = 0
    spare_mags: int = 3          # the game's CQBConfig.spare_mags (ep12: 1)
    time_limit: float = 45.0     # the game's CQBConfig.time_limit
    blind_away: float = 1.0      # the game's CQBConfig.blind_away (ep12: 0.25)
    hostage: bool = False        # the game's CQBConfig.hostage (ep12)
    hostage_deep: float = 0.0    # the game's CQBConfig.hostage_deep
    extract: bool = False        # the game's CQBConfig.extract (the coaching then counts the way out too)
    hostage_noise: float = 0.0   # the game's CQBConfig.hostage_noise
    through_drywall: float = 0.0  # (ep13) the game's CQBConfig.through_drywall (homes: shots through inside walls)
    unseen_hit: float = 0.5       # (ep13) the game's CQBConfig.unseen_hit
    hostage_hit: bool = False     # (ep13) the game's CQBConfig.hostage_hit (the side that hits it loses)
    hostage_bonus: float = 0.0   # (ep12, coaching like aim_bonus: the ranking only) up to this a round for how far an
                                 # attacker got securing the hostage (its best stretch beside it, / secure_time); the
                                 # defenders lose the same
    sigma_hostage: float = 0.0   # (ep12) a member's change to the weights on the hostage numbers (0: as `sigma`): they
                                 # start at zero, and at sigma 0.005 evolution never grew them (e8)
    health: float = 0.0
    react: float = 0.0
    pop: int = 256               # members a generation (mirrored pairs)
    rounds: int = 48             # rounds each member plays a generation (the same rounds for all of them)
    sigma: float = 0.02          # the size of a member's change to the brain...
    sigma_throw: float = 0.5     # ...and to its throw output (see the module docstring; at 1.0 every member throws)
    lr: float = 0.01             # Adam, on the ranked changes: every weight moves ~lr a generation (e1)...
    lr_rel: float = 0.05         # ...or, if set, ~lr_rel x its own sigma: small steps where it explores in small
                                 # ones (e1's 0.01 everywhere wore the brain down: ep9 3az)
    generations: int = 200
    buildings: int = 1024
    seed: int = 0
    checkpoint_every: int = 10
    method: str = "es"           # (so a run's config says what made it)
    stacked: float = 0.0         # this share of each opponent's rounds are entry drills (CQBGame.init_stacked)
    # coaching (e3, the owner's call: "in the real world a soldier is shown the ropes"): a member's score also gets
    # this much for each of its throws that blinded the enemy, and loses this much for each that blinded itself. Only
    # the score evolution ranks by: the game, the opponents and the reports (wins only) don't change
    blind_bonus: float = 0.0
    self_blind_cost: float = 0.0
    guard: int = -1              # undo a step that loses more than this many of the generation's rounds (-1: off)
    # (ep11 10v, the owner's call: "reward generations with better aim") a member's score also gets up to this much a
    # round for having its rifle on him when they first see each other (either way): aim_bonus x (1 - off / 90°),
    # nothing at 90° or more, nothing in rounds where they never meet. Only the ranking score, like blind_bonus
    aim_bonus: float = 0.0
    # (ep13 8.13, the owner's call: less robotic in the bodycam) a member's score loses this much a round, x the share of
    # its operators' time up spent running inside the building (faster than 1.1 x a walk, inside the footprint): real
    # teams clear a home at a walk. Only the ranking score, like aim_bonus
    run_cost: float = 0.0
    # (ep13 8.15, the owner: "lots of spinning" in the bodycam) a member's score loses this much a round, x its operators'
    # back-and-forths per 10 s up: a swing of 60+ degrees one way straight after one of 60+ the other way (a swing ends
    # when the facing comes back 30 degrees; scripts/cqb_turning.py measures the same). Only the ranking score
    turn_cost: float = 0.0
    # (ep13 8.16, the owner: "they open/close doors constantly in a loop") a member's score loses this much a round, x
    # its operators' door loops per 10 s up: moving the same door again within 3 s of moving it. Only the ranking score
    door_cost: float = 0.0
    house: str = ""              # the game's settings, as in CQBPPOConfig (ep11)
    route: bool = False
    feed: int = 0
    teams: tuple = (1, 1)        # (ep11 10x) with teams the brain plays every operator of its side
    bump: float = 0.0
    ready: float = 0.0
    stun: float = 0.0
    share_sight: bool = True
    split_entry: bool = False
    radio: tuple = (0, 0)
    pilot: bool = False


def _sigmas(cfg: CQBESConfig, params: dict, throw_row: int) -> dict:
    """Per weight: how big a member's change is (bold on the throw output's row and bias, small elsewhere)."""
    sig = {k: jnp.full(params[k].shape, cfg.sigma) for k in _evolved(params)}
    sig["policy.weight"] = sig["policy.weight"].at[throw_row].set(cfg.sigma_throw)
    sig["policy.bias"] = sig["policy.bias"].at[throw_row].set(cfg.sigma_throw)
    if cfg.hostage and cfg.sigma_hostage:   # the hostage numbers (and the extraction's) are the observation's last
        from ailearns.games.cqb import EXTRACT_OBS, HOSTAGE_OBS, HOSTAGE_SOUND_OBS
        n = HOSTAGE_OBS + (EXTRACT_OBS if cfg.extract else 0) + (HOSTAGE_SOUND_OBS if cfg.extract and cfg.hostage_noise else 0)
        sig["body.0.weight"] = sig["body.0.weight"].at[:, -n:].set(cfg.sigma_hostage)
    return sig


def _rollout(game: CQBGame, me: int, params: dict, opp, bis, keys, drill):
    """Rounds in buildings bis (the same starts and luck for every member: keys; drill: entry-drill starts),
    `params` playing its best guess as every operator of side `me` against player `opp`. -> per round: outcome,
    throws, throws that blinded an enemy, that blinded one of mine, side counts (decisions played; my crashes; decisions
    an enemy heard my footsteps; mine downed by mine; words said: for watching, not scored; (ep13) decisions x my operators
    up, and of those running inside the footprint: run_cost; back-and-forths: turn_cost; door loops: door_cost), and per operator of mine how
    far its aim was off the nearest standing enemy (radians) when it first saw one or was first seen (pi: never met)."""
    n_in = params["body.0.weight"].shape[1]
    mine, theirs = game.side_ops(me), game.side_ops(1 - me)
    mo, eo = jnp.asarray(mine), jnp.asarray(theirs)

    def one(key, bi, d):
        st = jax.tree.map(lambda a, b: jnp.where(d, a, b), game.init_stacked(key, bi), game.init(key, bi))

        def tick(carry, k):
            st, thrown, hit_them, hit_me, side, off, prog, sw, dl = carry
            acts = [None] * game.N
            for op in mine:
                acts[op] = cqb.full_action(A.best_guess(params, game.observe(st, op)[None, :n_in])[0])
            ks = [k] if len(theirs) == 1 else list(jax.random.split(k, len(theirs)))
            for op, kk in zip(theirs, ks):
                acts[op] = cqb.full_action(opp(game, st, op, kk))
            new, _ = game.step(st, jnp.stack(acts))
            bang = new.bang_now[mo].any()
            offs = []
            for i, op in enumerate(mine):
                near = jnp.where(new.alive[eo], jnp.linalg.norm(new.pos[eo] - new.pos[op], axis=1), jnp.inf)
                d = new.pos[eo[jnp.argmin(near)]] - new.pos[op]
                first = (new.seen[op, eo] | new.seen[eo, op]).any() & (off[i] >= jnp.pi) & ~st.done
                offs.append(jnp.where(first, jnp.abs(cqb._wrap(jnp.arctan2(d[1], d[0]) - new.face[op])), off[i]))
            on = ~st.done
            fell = st.alive[mo] & ~new.alive[mo]
            p = st.pos[mo]
            inside = ((p >= game.b.foot_lo[bi]) & (p <= game.b.foot_hi[bi])).all(-1)
            ran = jnp.linalg.norm(new.pos[mo] - p, axis=-1) > game.cfg.walk * 1.1 * game.cfg.dt
            sw, back = _swing(sw, cqb._wrap(new.face[mo] - st.face[mo]), on & st.alive[mo])
            dl, loops = _door_loops(dl, new.door_event[mo] & on, new.door_at[mo])
            side = side + on * jnp.stack([
                1, new.bumped[mo].sum(), (new.sound_age[eo, 0] == 0).any(),
                (fell & jnp.isin(new.hit_by[mo], mo)).sum(),
                ((new.said_age[mo] == 0) & (new.said[mo] > 0)).sum(),
                st.alive[mo].sum(), (st.alive[mo] & inside & ran).sum(), back, loops]).astype(jnp.int32)
            prog = jnp.maximum(prog, game.hostage_progress(new))   # (ep12) how far the hostage got (secured, out)
            return (new, thrown + (new.bangs[mo] < st.bangs[mo]).sum(),
                    hit_them + (bang & (new.blind[eo] > st.blind[eo]).any()),
                    hit_me + (bang & (new.blind[mo] > st.blind[mo]).any()), side, jnp.stack(offs), prog, sw, dl), None

        z = jnp.asarray(0)
        sw0 = tuple(jnp.zeros(len(mine), t) for t in (jnp.float32,) * 5 + (bool,))
        dl0 = (jnp.full((len(mine), 2), jnp.inf, jnp.float32), jnp.full(len(mine), DOOR_LOOP + 1, jnp.int32))
        (last, thrown, hit_them, hit_me, side, off, prog, _, _), _ = jax.lax.scan(
            tick, (st, z, z, z, jnp.zeros(9, jnp.int32), jnp.full(len(mine), jnp.pi, jnp.float32), jnp.asarray(0.0), sw0,
                   dl0),
            jax.random.split(jax.random.fold_in(key, 1), game.T))
        return last.outcome, thrown, hit_them, hit_me, side, off, prog

    return jax.vmap(one)(keys, bis, drill)


SWING_HOLD, SWING_BIG = np.radians(30), np.radians(60)


def _swing(sw, dface, up):
    """(ep13 8.15) Each operator's swings, a decision at a time: sw = (facing turned so far, swing start, its furthest
    point, its direction, the last swing's size, counted), dface: the turn this decision. -> (sw, back-and-forths now:
    the swing reaching 60 degrees with the one before it 60+ the other way)."""
    f, start, ext, dirn, prev, counted = sw
    f = f + jnp.where(up, dface, 0.0)
    begin = (dirn == 0) & (f != start)
    dirn = jnp.where(begin, jnp.sign(f - start), dirn)
    further = (f - ext) * dirn >= 0
    turn_back = ~further & (jnp.abs(f - ext) > SWING_HOLD)
    prev = jnp.where(turn_back, jnp.abs(ext - start), prev)
    start = jnp.where(turn_back, ext, start)
    dirn = jnp.where(turn_back, -dirn, dirn)
    counted = counted & ~turn_back
    ext = jnp.where(further | turn_back, f, ext)
    back = up & ~counted & (prev >= SWING_BIG) & (jnp.abs(ext - start) >= SWING_BIG)
    return (f, start, ext, dirn, prev, counted | back), back.sum()


DOOR_LOOP = 30   # decisions (3 s): the same door moved again by the same operator this soon is a loop


def _door_loops(dl, event, at):
    """(ep13 8.16) dl = (the door each operator last moved (its middle), decisions since), event: started moving a door
    now, at: which. -> (dl, loops now: moving the door he moved DOOR_LOOP decisions ago or less)."""
    last_at, since = dl
    loop = event & (since <= DOOR_LOOP) & (jnp.abs(at - last_at).max(-1) < 0.05)
    last_at = jnp.where(event[:, None], at, last_at)
    since = jnp.where(event, 0, jnp.minimum(since + 1, DOOR_LOOP + 1))
    return (last_at, since), loop.sum()


def centered_ranks(f):
    """Scores -> ranks spread over -0.5..0.5, tied scores sharing their rank (scores are win counts: ties are
    common, and breaking them at random would only add noise)."""
    less = (f[None, :] < f[:, None]).sum(1)
    same = (f[None, :] == f[:, None]).sum(1)
    return (less + (same - 1) / 2) / (f.shape[0] - 1) - 0.5


def _met_off(off):
    """Mean aim off him at first sight (degrees) over the rounds where they met."""
    met = off < jnp.pi
    return jnp.degrees(jnp.where(met, off, 0.0).sum() / jnp.maximum(met.sum(), 1))


def make_generation(cfg: CQBESConfig, game: CQBGame, me: int, opponents: list, sig: dict):
    """One generation as one compiled function: (brain, key, rounds) -> (gradient estimate, stats)."""
    pairs = cfg.pop // 2
    sign = 1.0 if me == ASSAULT else -1.0       # outcomes are Assault's

    names = tuple(sig)                          # EVOLVED (+ the word head)

    def noise(k):
        ks = jax.random.split(k, len(names))
        return {name: jax.random.normal(kk, sig[name].shape) for name, kk in zip(names, ks)}

    def play(params, groups):
        outs = [_rollout(game, me, params, opp, *g) for opp, g in zip(opponents, groups)]
        outcome, thrown, hit_them, hit_me, side, off, prog = (jnp.concatenate(x) for x in zip(*outs))
        return sign * outcome.astype(jnp.float32), thrown, hit_them, hit_me, side, off, prog

    def member(center, k, s, groups):
        eps = noise(k)
        p = dict(center, **{n: center[n] + s * sig[n] * eps[n] for n in names})
        score, thrown, hit_them, hit_me, side, off, prog = play(p, groups)
        coached = score + cfg.blind_bonus * hit_them - cfg.self_blind_cost * hit_me
        coached = coached - cfg.run_cost * side[:, 6] / jnp.maximum(side[:, 5], 1)   # (ep13) running inside
        coached = coached - cfg.turn_cost * side[:, 7] * 100 / jnp.maximum(side[:, 5], 1)   # back-and-forths a 10 s
        coached = coached - cfg.door_cost * side[:, 8] * 100 / jnp.maximum(side[:, 5], 1)   # door loops a 10 s
        coached = coached + cfg.aim_bonus * jnp.maximum(0.0, 1.0 - off / (jnp.pi / 2)).mean(-1)   # mean: my operators
        coached = coached + cfg.hostage_bonus * prog * sign   # (ep12) the furthest the hostage got secured: attackers
                                                             # gain it, defenders lose it (d19 never learned to guard)
        return (coached.mean(), thrown.mean(), hit_them.sum(), hit_me.sum(), thrown.sum(), (score > 0).mean(),
                _met_off(off))

    def generation(center, key, groups):
        ks = jax.random.split(key, pairs)
        signs = jnp.concatenate([jnp.ones(pairs), -jnp.ones(pairs)])           # each pair: + change, - change
        f, thrown, hit_them, hit_me, n_thrown, won, m_off = jax.vmap(lambda k, s: member(center, k, s, groups))(
            jnp.concatenate([ks, ks]), signs)
        ranks = centered_ranks(f)
        w = ranks[:pairs] - ranks[pairs:]

        def add(acc, kw):
            k, wi = kw
            eps = noise(k)
            return {n: acc[n] + wi * eps[n] for n in names}, None
        g, _ = jax.lax.scan(add, {n: jnp.zeros_like(center[n]) for n in names}, (ks, w))
        grad = {n: -g[n] / (pairs * sig[n]) for n in names}      # minus: adam_step descends
        own, own_thrown, own_them, own_me, side, own_off, own_prog = play(center, groups)
        drill = jnp.concatenate([g[2] for g in groups])
        throws = n_thrown.sum()
        stats = {"score": f.mean(), "score_best": f.max(), "won_members": won.mean(),
                 "score_brain": own.mean(), "won_brain": (own > 0).mean(),
                 "won_brain_full": ((own > 0) & ~drill).sum() / jnp.maximum((~drill).sum(), 1),
                 "won_brain_drill": ((own > 0) & drill).sum() / jnp.maximum(drill.sum(), 1),
                 "throws_per_round": thrown.mean(), "members_throwing": (thrown > 0).mean(),
                 "blinded_enemy_per_throw": hit_them.sum() / jnp.maximum(throws, 1),
                 "blinded_self_per_throw": hit_me.sum() / jnp.maximum(throws, 1),
                 "brain_throws_per_round": own_thrown.mean(),
                 "brain_blinded_enemy": own_them.sum() / jnp.maximum(own_thrown.sum(), 1),
                 "brain_aim_off_deg": _met_off(own_off), "members_aim_off_deg": m_off.mean(),
                 "brain_met": (own_off < jnp.pi).mean(), "brain_secure_progress": own_prog.mean(),
                 # (ep11 10y) watched, not scored: per round, the brain's side
                 "brain_blinded_mine": own_me.mean(), "brain_crashes": side[:, 1].mean(),
                 "brain_heard_share": side[:, 2].sum() / jnp.maximum(side[:, 0].sum(), 1),
                 "brain_friendly_downs": side[:, 3].mean(), "brain_words": side[:, 4].mean(),
                 "brain_running_inside": side[:, 6].sum() / jnp.maximum(side[:, 5].sum(), 1),
                 "brain_back_forth_per_min": side[:, 7].sum() * 600 / jnp.maximum(side[:, 5].sum(), 1),
                 "brain_door_loops_per_min": side[:, 8].sum() * 600 / jnp.maximum(side[:, 5].sum(), 1)}
        return grad, stats

    return generation


def make_judge(game: CQBGame, me: int, opponents: list):
    """(brain, rounds) -> how many of them it wins (the guard's yardstick: wins only, no coaching)."""
    sign = 1.0 if me == ASSAULT else -1.0

    def judge(params, groups):
        outs = [_rollout(game, me, params, opp, *g)[0] for opp, g in zip(opponents, groups)]
        return (sign * jnp.concatenate(outs) > 0).sum()
    return judge


def train(cfg: CQBESConfig, run_dir: str | Path, max_hours: float | None = None,
          game_cfg: CQBConfig = CQBConfig(), resume: bool = False) -> None:
    """resume: carry on from latest.pt (a job restarted after its machine was taken away: Modal preempts)."""
    import torch
    from ailearns.agents import tracking
    from ailearns.checkpoints import pick_checkpoint
    game_cfg = replace(game_cfg, **cqb.HOUSES[cfg.house])
    game_cfg = replace(game_cfg, time_cost=cfg.time_cost, flashbangs=tuple(cfg.flashbangs), magazine=cfg.magazine,
                       spare_mags=cfg.spare_mags, time_limit=cfg.time_limit, blind_away=cfg.blind_away, hostage=cfg.hostage,
                       hostage_deep=cfg.hostage_deep, extract=cfg.extract, hostage_noise=cfg.hostage_noise,
                       through_drywall=cfg.through_drywall, unseen_hit=cfg.unseen_hit, hostage_hit=cfg.hostage_hit,
                       health=cfg.health, react=cfg.react, route=cfg.route, feed=cfg.feed,
                       team_sizes=tuple(cfg.teams), bump=cfg.bump, ready=cfg.ready, stun=cfg.stun,
                       share_sight=cfg.share_sight, split_entry=cfg.split_entry, radio=tuple(cfg.radio),
                       pilot=cfg.pilot)
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    state = None
    if resume:
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    elif (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already has a run")
    K = len(cfg.opponent.split(","))                    # checked before the (minutes-long) build
    if cfg.rounds % K:
        raise SystemExit(f"rounds ({cfg.rounds}) must divide among the {K} opponents")
    dev = jax.devices()[0]
    if not state:
        (run_dir / "config.json").write_text(json.dumps(
            {**asdict(cfg), "game": "cqb", "cqb": asdict(game_cfg), "git": _git_version(), "backend": "jax",
             "device": f"1 x {dev.device_kind}"}, indent=2))
    print(f"ES on {dev.device_kind}; building {cfg.buildings} buildings …", flush=True)
    game = CQBGame(make_buildings(list(TRAIN_SEEDS)[: cfg.buildings], game_cfg), game_cfg)
    me = ASSAULT if cfg.side == "assault" else DEFENSE
    n_switch = A.switches_for(game_cfg, me)
    run, _, ck = cfg.init_from.partition(":")
    params, src_ck = A.load(pick_checkpoint(run, ck or "last"))
    src = CQBConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in src_ck["cqb"].items()})
    if params["body.0.weight"].shape[1] == src.obs_size and A.obs_blocks(src) != A.obs_blocks(game_cfg):
        params = A.transplant(params, src, game_cfg)       # (ep11 10x) e.g. a 1v1 brain into a 2v1 game with a radio
    params = A.widen(params, game_cfg.obs_size, n_switch)
    if game_cfg.radio[me] and "say.weight" not in params:
        params = A.add_radio(params, game_cfg.radio[me])
    if state:
        params = {k: jnp.asarray(v.numpy()) for k, v in state["model"].items()}
    throw_row = A.N_MOVE + A.SWITCHES.index("throw")
    sig = _sigmas(cfg, params, throw_row)
    opponents = A.pool_members(cfg.opponent, 1 - me, game_cfg)
    generation = jax.jit(make_generation(cfg, game, me, opponents, sig))
    judge = jax.jit(make_judge(game, me, opponents))
    names = _evolved(params)
    opt = jax.tree.map(jnp.asarray, state["optimizer"]) if state else adam_init({n: params[n] for n in names})
    first = state["update"] + 1 if state else 1
    if state:
        print(f"resuming at generation {first}", flush=True)
    meta = {"config": asdict(cfg), "cqb": asdict(game_cfg), "backend": "jax", "game": "cqb"}

    def save(g: int, stats: dict) -> None:
        torch.save({"model": to_torch(params), **meta, "update": g, "stats": stats},
                   run_dir / "checkpoints" / f"u{g:05d}.pt")
        torch.save({"model": to_torch(params), "optimizer": jax.tree.map(np.asarray, opt), **meta, "update": g},
                   run_dir / "latest.pt")

    if not state:
        save(0, {})
    log = open(run_dir / "metrics.jsonl", "a")
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    start = time.time()
    for g in range(first, cfg.generations + 1):   # each generation's luck from its number (the same after a resume)
        k_gen, k_rounds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(cfg.seed + 2000), g))
        bis = jax.random.randint(k_rounds, (K, cfg.rounds // K), 0, cfg.buildings)
        n = cfg.rounds // K
        drill = jnp.arange(n) < int(round(cfg.stacked * n))
        groups = [(bis[i], jax.random.split(jax.random.fold_in(k_rounds, i + 1), n), drill) for i in range(K)]
        grad, m = generation(params, k_gen, groups)
        if cfg.lr_rel:   # Adam's step is linear in lr: take it at lr 1, then scale it per weight
            new, opt_new = adam_step({n: params[n] for n in names}, opt, grad, 1.0)
            new = {n: params[n] + cfg.lr_rel * sig[n] * (new[n] - params[n]) for n in names}
        else:
            new, opt_new = adam_step({n: params[n] for n in names}, opt, grad, cfg.lr)
        kept = True
        if cfg.guard >= 0:   # the step stands only if the brain still wins these rounds (within `guard` of them)
            before = int(round(float(m["won_brain"]) * K * (cfg.rounds // K)))
            after = int(judge(dict(params, **new), groups))
            kept = after >= before - cfg.guard
        if kept:
            params, opt = dict(params, **new), opt_new
        stats = {"update": g, "time": round(time.time() - start, 1), "step_kept": kept,
                 **{k: float(v) for k, v in jax.device_get(m).items()}}
        log.write(json.dumps(stats) + "\n")
        log.flush()
        tracking.log(tracker, stats)
        if g % cfg.checkpoint_every == 0 or g == cfg.generations:
            save(g, stats)
        if g == 1 or g % 5 == 0:
            print(f"gen {g:4d}/{cfg.generations}  brain {stats['score_brain']:+.2f} (won {stats['won_brain']:.2f}: "
                  f"full {stats['won_brain_full']:.2f}, drills {stats['won_brain_drill']:.2f}; "
                  f"throws {stats['brain_throws_per_round']:.2f}/round)  members {stats['score']:+.2f} "
                  f"(best {stats['score_best']:+.2f}; {stats['members_throwing']:.0%} throw, "
                  f"{stats['throws_per_round']:.2f}/round; per throw: enemy blinded {stats['blinded_enemy_per_throw']:.2f}, "
                  f"self {stats['blinded_self_per_throw']:.2f}; members won {stats['won_members']:.2f}; "
                  f"aim off at first sight: brain {stats['brain_aim_off_deg']:.0f} deg, members "
                  f"{stats['members_aim_off_deg']:.0f}; brain a round: crashes {stats['brain_crashes']:.1f}, heard "
                  f"{stats['brain_heard_share']:.0%}, friendly downs {stats['brain_friendly_downs']:.2f}, words "
                  f"{stats['brain_words']:.1f}, own side blinded {stats['brain_blinded_mine']:.2f})"
                  + ("" if kept else "  step undone") + f"  {stats['time']:.0f} s", flush=True)
        if max_hours is not None and time.time() - start > max_hours * 3600:
            save(g, stats)
            print(f"stopped after {max_hours} h at generation {g}", flush=True)
            break
    log.close()
    tracking.finish(tracker)
