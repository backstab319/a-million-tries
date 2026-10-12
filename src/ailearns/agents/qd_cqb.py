"""A gallery of styles, by evolution (ep10, owner: "see if any interesting strategy emerges"): MAP-Elites for a CQB side.

Evolution that keeps only the best brain (ES, ep9 3az) pulls everything towards one answer. MAP-Elites keeps the best
brain *of each style*: a grid of cells over a few numbers that describe how a brain plays, each cell holding the best
brain found that plays that way. New brains are random changes of brains in the grid; one that plays in an empty
style moves in, one that beats the holder of its style (on the same rounds) replaces it. Styles that lose at first
(throwing flashbangs, splitting up) aren't crowded out by the brain that wins most: they compete only with themselves.

Styles for a team (two numbers, measured over each brain's rounds):
- spread: how far apart the team's first two operators stay, on average, while both are up (stack <-> split);
- flashbangs: how many the team throws a round.
Score: the share of rounds won (best guess, the same rounds for every brain in a generation: same buildings, starts,
luck). What changes: the two layers and the policy outputs, in small steps (`sigma`), boldly on the throw output
(`sigma_throw`: as ES, a brain that learned not to throw sits far below the throw line).

A run can start from another's saved gallery (`seed_gallery`): its brains move in, their records start over.

Saved: runs/<run>/gallery.json (each cell: its win rate over every generation it was judged, its style, when it
moved in) and each cell's brain as runs/<run>/cells/<spread>-<bangs>/checkpoints/u00000.pt (PPO format: the footage
tools and reports read it as "runs/<run>/cells/1-0:0"); the best cell (judged in 5+ generations) is also saved as
runs/<run>/checkpoints/u<generation>.pt, so cqb_eval's report works on the run as on any other.
"""

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import ppo_cqb as A
from ailearns.agents.es_cqb import EVOLVED
from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_jax import to_torch
from ailearns.games import cqb
from ailearns.games.cqb import ASSAULT, DEFENSE, TRAIN_SEEDS, CQBConfig, CQBGame, make_buildings


@dataclass
class CQBQDConfig:
    side: str = "assault"
    init_from: str = ""            # the brain the gallery starts from (runs/x:update)
    opponent: str = "camper"       # a pool: the rounds are shared out among its members
    time_cost: float = 0.0         # the game's rules, as in CQBPPOConfig
    flashbangs: tuple = (0, 0)
    magazine: int = 0
    health: float = 0.0
    react: float = 0.0
    teams: tuple = (1, 1)
    bump: float = 0.0
    ready: float = 0.0
    stun: float = 0.0
    children: int = 128            # new brains a generation
    rounds: int = 48               # rounds every brain plays a generation (the same rounds for all)
    sigma: float = 0.02            # the size of a child's change...
    sigma_throw: float = 0.5       # ...and on the throw output
    spread_edges: tuple = (1.5, 3.0, 5.0)   # metres: cells stack | close | apart | split
    bang_edges: tuple = (0.0, 0.25, 1.0)    # flashbangs a round: cells none | rare | some | many
    generations: int = 400
    seed_gallery: str = ""         # start from a saved gallery (runs/x: its cells' brains; their records start over)
    buildings: int = 1024
    seed: int = 0
    checkpoint_every: int = 25
    method: str = "qd"


def _play(game: CQBGame, mine: list, theirs: list, params: dict, opp, bi, key):
    """One round, `params` playing its best guess for every operator in `mine`, `opp` for `theirs`. -> outcome, the
    summed distance between mine[0] and mine[1] over decisions both were up, how many such decisions, flashbangs my
    side threw, my side's operators downed by my side."""
    n_in = params["body.0.weight"].shape[1]
    st = game.init(key, bi)

    def tick(carry, k):
        st, dist, n = carry
        acts = [None] * game.N
        for op in mine:
            acts[op] = cqb.full_action(A.best_guess(params, game.observe(st, op)[None, :n_in])[0])
        ks = jax.random.split(k, len(theirs))
        for j, op in enumerate(theirs):
            acts[op] = cqb.full_action(opp(game, st, op, ks[j]))
        new, _ = game.step(st, jnp.stack(acts))
        if len(mine) > 1:
            both = st.alive[mine[0]] & st.alive[mine[1]] & ~st.done
            d = jnp.linalg.norm(st.pos[mine[0]] - st.pos[mine[1]])
            dist, n = dist + jnp.where(both, d, 0.0), n + both
        return (new, dist, n), None

    (last, dist, n), _ = jax.lax.scan(tick, (st, jnp.asarray(0.0), jnp.asarray(0)),
                                      jax.random.split(jax.random.fold_in(key, 1), game.T))
    m = jnp.asarray(mine)
    thrown = (game.cfg.flashbangs[int(game.team[mine[0]])] * len(mine) - last.bangs[m].sum())
    friendly = ((last.hit_by[m][:, None] == m[None, :]).any(1) & ~last.alive[m]).sum()
    return last.outcome, dist, n, thrown, friendly


def make_judge(game: CQBGame, me: int, opponents: list):
    """(stacked brains, rounds) -> per brain: rounds won, spread, flashbangs a round, friendly-fire downs a round."""
    mine, theirs = game.side_ops(me), game.side_ops(1 - me)
    sign = 1 if me == ASSAULT else -1

    def one(params, groups):
        outs = [jax.vmap(lambda b, k, opp=opp: _play(game, mine, theirs, params, opp, b, k))(*g)
                for opp, g in zip(opponents, groups)]
        outcome, dist, n, thrown, friendly = (jnp.concatenate(x) for x in zip(*outs))
        return {"won": ((sign * outcome) > 0).mean(), "spread": dist.sum() / jnp.maximum(n.sum(), 1),
                "bangs": thrown.mean(), "friendly": friendly.mean()}

    return jax.vmap(one, in_axes=(0, None))


def _cell(cfg: CQBQDConfig, spread: float, bangs: float) -> tuple:
    return (int(np.searchsorted(cfg.spread_edges, spread, side="right")),
            int(np.searchsorted(cfg.bang_edges, bangs - 1e-9, side="left")))


def train(cfg: CQBQDConfig, run_dir: str | Path, max_hours: float | None = None,
          game_cfg: CQBConfig = CQBConfig(), resume: bool = False) -> None:
    import torch
    from ailearns.agents import tracking
    from ailearns.checkpoints import pick_checkpoint
    if resume:
        raise SystemExit("gallery runs don't resume (yet)")
    game_cfg = replace(game_cfg, time_cost=cfg.time_cost, flashbangs=tuple(cfg.flashbangs), magazine=cfg.magazine,
                       health=cfg.health, react=cfg.react, team_sizes=tuple(cfg.teams), bump=cfg.bump,
                       ready=cfg.ready, stun=cfg.stun)
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already has a run")
    dev = jax.devices()[0]
    (run_dir / "config.json").write_text(json.dumps(
        {**asdict(cfg), "game": "cqb", "cqb": asdict(game_cfg), "git": _git_version(), "backend": "jax",
         "device": f"1 x {dev.device_kind}"}, indent=2))
    print(f"MAP-Elites on {dev.device_kind}; building {cfg.buildings} buildings …", flush=True)
    game = CQBGame(make_buildings(list(TRAIN_SEEDS)[: cfg.buildings], game_cfg), game_cfg)
    me = ASSAULT if cfg.side == "assault" else DEFENSE
    run, _, ck = cfg.init_from.partition(":")
    start = A.widen(A.load(pick_checkpoint(run, ck or "last"))[0], game_cfg.obs_size, A.switches_for(game_cfg, me))
    throw_row = A.N_MOVE + A.SWITCHES.index("throw")
    sig = {k: jnp.full(start[k].shape, cfg.sigma) for k in EVOLVED}
    if start["policy.bias"].shape[0] > throw_row:
        sig["policy.weight"] = sig["policy.weight"].at[throw_row].set(cfg.sigma_throw)
        sig["policy.bias"] = sig["policy.bias"].at[throw_row].set(cfg.sigma_throw)
    opponents = A.pool_members(cfg.opponent, 1 - me)
    K = len(opponents)
    if cfg.rounds % K:
        raise SystemExit(f"rounds ({cfg.rounds}) must divide among the {K} opponents")
    judge = jax.jit(make_judge(game, me, opponents))
    rows, cols = len(cfg.spread_edges) + 1, len(cfg.bang_edges) + 1
    C = rows * cols
    fixed = {k: v for k, v in start.items() if k not in EVOLVED}     # the value head and move noise: shared

    @jax.jit
    def breed(archive, parents, key):
        ks = jax.random.split(key, len(EVOLVED))
        return {n: archive[n][parents] + sig[n] * jax.random.normal(k, (parents.shape[0], *sig[n].shape))
                for n, k in zip(EVOLVED, ks)}

    def full(stacked):
        n = next(iter(stacked.values())).shape[0]
        return {**stacked, **{k: jnp.broadcast_to(v, (n, *v.shape)) for k, v in fixed.items()}}

    def brain(archive, c):
        return {**{n: archive[n][c] for n in EVOLVED}, **fixed}

    meta = {"config": asdict(cfg), "cqb": asdict(game_cfg), "backend": "jax", "game": "cqb"}

    def save_brain(params, path: Path, g: int, stats: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": to_torch(params), **meta, "update": g, "stats": stats}, path)

    # the gallery: per cell its brain, and its record (filled, wins over all generations judged, style, history)
    archive = {n: jnp.broadcast_to(start[n], (C, *start[n].shape)) for n in EVOLVED}
    filled = np.zeros(C, bool)
    rec = [None] * C
    if cfg.seed_gallery:     # (the cells' value heads and move noise are the starting brain's: only EVOLVED differ)
        for d in sorted((Path(cfg.seed_gallery) / "cells").iterdir()):
            i, j = (int(x) for x in d.name.split("-"))
            c = i * cols + j
            p = A.load(d / "checkpoints" / "u00000.pt")[0]
            archive = {n: archive[n].at[c].set(p[n]) for n in EVOLVED}
            filled[c] = True
            rec[c] = {"cell": [i, j], "born": -1, "parent": -1, "judged": 0, "won_sum": 0.0, "spread_sum": 0.0,
                      "bangs_sum": 0.0, "friendly_sum": 0.0}
        print(f"seeded with {int(filled.sum())} cells from {cfg.seed_gallery}", flush=True)
    log = open(run_dir / "metrics.jsonl", "a")
    tracker = tracking.start(run_dir, json.loads((run_dir / "config.json").read_text()))
    t0 = time.time()
    save_brain(start, run_dir / "checkpoints" / "u00000.pt", 0, {})
    for g in range(cfg.generations + 1):    # generation 0: the starting brain and its first children
        k_gen, k_rounds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(cfg.seed + 3000), g))
        n = cfg.rounds // K
        bis = jax.random.randint(k_rounds, (K, n), 0, cfg.buildings)
        groups = [(bis[i], jax.random.split(jax.random.fold_in(k_rounds, i + 1), n)) for i in range(K)]
        cells = np.flatnonzero(filled)
        # every generation judges the same number of brains (all C rows of the gallery, filled or not, then the
        # children), so it compiles once: the first version judged only the filled rows and recompiled each time a
        # new style moved in (~18 s a generation instead of a few)
        parents = np.asarray(jax.random.choice(k_gen, jnp.asarray(cells if len(cells) else [0]), (cfg.children,)))
        kids = breed(archive, jnp.asarray(parents), jax.random.fold_in(k_gen, 1))
        members = {n_: jnp.concatenate([archive[n_], kids[n_]]) for n_ in EVOLVED}
        out = jax.device_get(judge(full(members), groups))
        e = C
        if g == 0 and not len(cells):                                  # the starting brain (every row) is a candidate
            kids = {n_: jnp.concatenate([archive[n_][:1], kids[n_]]) for n_ in EVOLVED}
            parents = np.concatenate([[-1], parents])
            out = {k: np.concatenate([v[:1], v[C:]]) for k, v in out.items()}
            e = 0                                                      # (no holders yet: all are candidates)
        for c in cells:                                                # the holders' records over generations
            j = c
            r = rec[c]
            r["judged"] += 1
            r["won_sum"] += float(out["won"][j])
            r["spread_sum"] += float(out["spread"][j])
            r["bangs_sum"] += float(out["bangs"][j])
            r["friendly_sum"] += float(out["friendly"][j])
        holder_now = {int(c): float(out["won"][c]) for c in cells}
        moved_in = 0
        for i in np.argsort(-out["won"][e:]):                          # children, best first
            won, spread, bangs = float(out["won"][e + i]), float(out["spread"][e + i]), float(out["bangs"][e + i])
            c = int(np.ravel_multi_index(_cell(cfg, spread, bangs), (rows, cols)))
            if filled[c] and won <= holder_now[c]:
                continue
            archive = {n_: archive[n_].at[c].set(kids[n_][i]) for n_ in EVOLVED}
            filled[c] = True
            holder_now[c] = won
            rec[c] = {"cell": list(_cell(cfg, spread, bangs)), "born": g, "parent": int(parents[i]),
                      "judged": 1, "won_sum": won, "spread_sum": spread, "bangs_sum": bangs,
                      "friendly_sum": float(out["friendly"][e + i])}
            moved_in += 1
        means = {c: rec[c]["won_sum"] / rec[c]["judged"] for c in np.flatnonzero(filled)}
        best = max(means, key=lambda c: (rec[c]["judged"] >= 5, means[c]))
        stats = {"update": g, "time": round(time.time() - t0, 1), "cells": int(filled.sum()), "moved_in": moved_in,
                 "best_cell": list(rec[best]["cell"]), "best_won": means[best],
                 "children_won": float(out["won"][e:].mean()), "children_best": float(out["won"][e:].max()),
                 "children_bangs": float(out["bangs"][e:].mean()), "children_spread": float(out["spread"][e:].mean())}
        log.write(json.dumps(stats) + "\n")
        log.flush()
        tracking.log(tracker, stats)
        last = g == cfg.generations or (max_hours is not None and time.time() - t0 > max_hours * 3600)
        if g % 10 == 0 or last:
            grid = np.full((rows, cols), "  -  ", dtype=object)
            for c in np.flatnonzero(filled):
                r = rec[c]
                grid[r["cell"][0], r["cell"][1]] = f"{means[c]:4.0%}{'*' if r['judged'] >= 5 else ' '}"
            print(f"gen {g:4d}/{cfg.generations}  cells {int(filled.sum())}/{C}  moved in {moved_in:3d}  "
                  f"children won {stats['children_won']:.2f} (best {stats['children_best']:.2f}, "
                  f"{stats['children_bangs']:.2f} bangs, spread {stats['children_spread']:.1f} m)  {stats['time']:.0f} s\n"
                  + "\n".join("    spread " + ["stack", "close", "apart", "split"][min(i, 3)].ljust(6) + " | "
                              + " ".join(grid[i]) for i in range(rows))
                  + "\n    (columns: flashbangs none | rare | some | many; * = judged in 5+ generations)", flush=True)
        if g and (g % cfg.checkpoint_every == 0 or last):
            save_brain(brain(archive, best), run_dir / "checkpoints" / f"u{g:05d}.pt", g, stats)
            gallery = []
            for c in np.flatnonzero(filled):
                r = rec[c]
                name = f"{r['cell'][0]}-{r['cell'][1]}"
                save_brain(brain(archive, c), run_dir / "cells" / name / "checkpoints" / "u00000.pt", g, r)
                gallery.append({"cell": name, "born": r["born"], "judged": r["judged"],
                                **{k: r[k + "_sum"] / r["judged"] for k in ("won", "spread", "bangs", "friendly")}})
            (run_dir / "gallery.json").write_text(json.dumps({"generation": g, "spread_edges": cfg.spread_edges,
                                                              "bang_edges": cfg.bang_edges, "cells": gallery}, indent=1))
        if last:
            break
    log.close()
    tracking.finish(tracker)
