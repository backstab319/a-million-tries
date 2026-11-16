"""Many ecosystem worlds at once on a GPU (plan section 4, port step 6), or the same worlds in numpy to compare.

Each world is one seed (`--seed0`, `--seed0 + 1`, ...): its own land and first creatures, exactly as `eco_kills.py`
makes them (1000 grass-eaters with diet 0-0.1). Every `--every` days each world reports one line (and a JSON line
in `<out>/report.jsonl`), then one line across all worlds. Both engines measure the same things the same way, so
`scripts/eco_jax_compare.py` can put numpy and JAX side by side.

    # JAX, 32 worlds, 50 years, the standard world, on Modal (stops cleanly before Modal's 2 h; run again to go on)
    ailearns cloud script "scripts/eco_jax_run.py --engine jax --worlds 32 --days 18000 --rain --thirst
        --out runs/eco-jax-1 --hours 1.8" --on modal --accelerator T4
    # numpy, the same 10 worlds for a year, on Modal's CPUs
    ailearns cloud script "scripts/eco_jax_run.py --engine numpy --worlds 10 --days 360 --rain --thirst
        --out runs/eco-np-ref" --on modal --accelerator cpu

`--control-at DAY` (JAX): on that day, every world is copied with random brains and both run 5 days (9c).
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ailearns.games.ecosystem import WATER, World, WorldConfig
from ailearns.games.ecosystem_life import Life, LifeConfig

CAUSES = ("killed", "thirst", "starved", "old")
FOODS = ("fruit", "carcass", "green", "dead", "milk")


def make(seed, a):
    world = World.make(seed, WorldConfig(size=a.size, rain=a.rain))
    life = Life(world, 0, seed, LifeConfig(thirst=a.thirst, gut=a.gut, cold=a.cold, cold_breath=a.cold_breath))
    life.add(a.n, diet=(0.0, 0.1))
    return life


def line(rec):
    """One world's report, for people."""
    food = np.array([rec["food"][k] for k in FOODS])
    share = 100 * food / max(food.sum(), 1e-9)
    s = (f"day {rec['day']:6.0f} world {rec['seed']:3d}: alive {rec['alive']:5d} births {rec['births']:6d} deaths "
         + "/".join(str(rec["dead"][k]) for k in CAUSES) + " (killed/thirst/starved/old); food "
         + " ".join(f"{k} {v:.0f}%" for k, v in zip(FOODS, share))
         + f"; drink {rec['drink']:.0f}% of steps at thirst {rec['drink_thirst']:.2f}, thirst {rec['thirst']:.2f},"
           f" {rec['speed']:.2f} m/s, calls {rec['calls']:.0f}%, attacks {rec['attacks']:.1f}%, green "
           f"{rec['green']:.2f}x; diet {rec['diet']:.2f}, meat-eaters {rec['meat_eaters']}, gen {rec['gen_max']}, "
           f"mass {rec['mass']:.1f} kg")
    if "crowded" in rec:
        s += f" [crowded {rec['crowded']} full {rec['full']} capped {rec['capped']} books {rec['error']:.0e}]"
    return s


def summary(day, recs):
    alive = np.array([r["alive"] for r in recs])
    return (f"day {day:6.0f} ALL {len(recs)} worlds: alive median {np.median(alive):.0f} (min {alive.min()}, max "
            f"{alive.max()}), extinct {(alive == 0).sum()}; gen max {max(r['gen_max'] for r in recs)}; diet "
            f"{np.mean([r['diet'] for r in recs if r['alive']] or [0]):.2f}; meat-eaters "
            f"{sum(r['meat_eaters'] for r in recs)}")


def state_fields(seed, day, alive, diet, gen, adult):
    return dict(seed=seed, day=day, alive=int(alive.sum()), diet=float(diet[alive].mean()) if alive.any() else 0.0,
                meat_eaters=int((alive & (diet >= 0.5)).sum()), gen_max=int(gen[alive].max()) if alive.any() else 0,
                mass=float(adult[alive].mean()) if alive.any() else 0.0)


def probe_fields(steps, drink, drink_thirst, thirst, moved, calls, attacks, food, dead, green_here, green_land, births,
                 dt):
    steps, drink, drink_thirst, thirst, moved, calls, attacks, green_here, green_land = (
        float(v) for v in (steps, drink, drink_thirst, thirst, moved, calls, attacks, green_here, green_land))
    s = max(steps, 1)
    return dict(births=int(births), dead=dict(zip(CAUSES, (int(v) for v in dead))),
                food=dict(zip(FOODS, (float(v) for v in food))), drink=100 * drink / s,
                drink_thirst=drink_thirst / max(drink, 1), thirst=thirst / s, speed=moved / s / dt,
                calls=100 * calls / s, attacks=100 * attacks / s, green=green_here / max(green_land, 1e-9))


# --- numpy ----------------------------------------------------------------------------------------------------------
def numpy_world(args):
    """One numpy world, the whole run; reports as the JAX engine does (same measures)."""
    seed, a = args
    life = make(seed, a)
    cfg, w = life.cfg, life.world
    per_day = round(cfg.day_s / cfg.dt)
    land = w.biome.reshape(-1) != WATER
    per_cell = w.cfg.plant_kj_m2 * w.cfg.cell_m ** 2
    acc = {}

    def reset():
        acc.update(steps=0, drink=0, drink_thirst=0.0, thirst=0.0, moved=0.0, calls=0, attacks=0, food=np.zeros(5),
                   dead=np.zeros(4, int), green_here=0.0, green_land=0.0, births=0)
    reset()
    seen = {}

    def trace(lf, i, attack, meat, plant, fruit, milk, call, x, **_):
        seen.update(i=i, x=x)
        acc["calls"] += int(call.sum())
        acc["attacks"] += int(attack.sum())
        acc["food"][0] += fruit.sum()
        acc["food"][1] += meat.sum()
        acc["food"][4] += milk.sum()
    life.trace = trace
    orig = life._deaths

    def deaths(i, killed=None):   # the cause of each death, before the bodies are cleared (one each, as JAX)
        k = np.zeros(len(i), bool) if killed is None else killed
        parched, starved = life.parched(i), life.reserve[i] < 0
        old = life.age[i] > life.lifespan(life.adult[i])
        acc["dead"] += [k.sum(), (~k & parched).sum(), (~k & ~parched & starved).sum(),
                        (~k & ~parched & ~starved & old).sum()]
        return orig(i, killed)
    life._deaths = deaths
    graze = w.graze

    def graze_seen(cells, want, split=False):   # green and dead grass apart, as the JAX probe has them
        got = graze(cells, want, split=True)
        seen["graze"] = got
        return got if split else got[0] + got[1]
    w.graze = graze_seen
    out, recs = Path(a.out), []
    for t in range(int(a.days * per_day)):
        i = np.flatnonzero(life.alive)
        p0 = life.pos[i].copy()
        food = w.food().reshape(-1)
        avg = food[land].mean() / per_cell
        b0 = life.books.births
        diet = life.diet[i].copy()
        life.step()
        acc["births"] += life.books.births - b0
        if len(i):
            x = seen["x"]
            acc["steps"] += len(i)
            th = x[:, 32]
            acc["thirst"] += th.sum()
            if cfg.thirst:
                acc["drink"] += int(life.drank.sum())
                acc["drink_thirst"] += th[life.drank].sum()
            acc["moved"] += np.hypot(*(life.pos[i] - p0).T).sum()
            g = diet < 0.5
            acc["green_here"] += x[g, 0].sum()
            acc["green_land"] += g.sum() * avg
            kp = cfg.digest_plant * (1 - diet ** cfg.plant_curve)
            green, dead = seen["graze"]
            acc["food"][2] += (kp * green).sum()
            acc["food"][3] += (kp * w.cfg.dead_quality * dead).sum()
        if (t + 1) % round(a.every * per_day) == 0:
            day = (t + 1) / per_day
            a_ = life.alive
            rec = state_fields(seed, day, a_, life.diet, life.gen, life.adult)
            rec.update(probe_fields(acc["steps"], acc["drink"], acc["drink_thirst"], acc["thirst"], acc["moved"],
                                    acc["calls"], acc["attacks"], acc["food"], acc["dead"], acc["green_here"],
                                    acc["green_land"], acc["births"], cfg.dt))
            rec["error"] = life.error()
            recs.append(rec)
            reset()
    return recs


def run_numpy(a):
    seeds = list(range(a.seed0, a.seed0 + a.worlds))
    with Pool(min(len(seeds), os.cpu_count() or 1)) as pool:
        per_world = pool.map(numpy_world, [(s, a) for s in seeds])
    out = Path(a.out)
    with open(out / "report.jsonl", "w") as f:
        for k in range(len(per_world[0])):
            recs = [w[k] for w in per_world]
            for r in recs:
                print(line(r))
                f.write(json.dumps(dict(r, engine="numpy")) + "\n")
            print(summary(recs[0]["day"], recs), flush=True)


# --- JAX ------------------------------------------------------------------------------------------------------------
def run_jax(a):
    import jax
    import jax.numpy as jnp
    from ailearns.games import ecosystem_jax_life as jl
    print("devices:", jax.devices(), flush=True)
    out = Path(a.out)
    seeds = a.keep or list(range(a.seed0, a.seed0 + a.worlds))
    a.worlds = len(seeds)   # (10n) --keep: only these worlds (the rest died out; 16 at cap 65536 don't fit a T4)
    lives = [make(s, a) for s in seeds]
    wcfg, cfg = lives[0].world.cfg, lives[0].cfg
    pairs = [jl.from_numpy(life, a.cap) for life in lives]
    del lives
    stack = lambda xs: jax.tree.map(lambda *v: jnp.stack(v), *xs)
    st, wd = stack([p[0] for p in pairs]), stack([p[1] for p in pairs])
    del pairs
    leaves, tree = jax.tree.flatten(wd)
    period = 0
    ckpt = out / "state.npz"
    src = ckpt
    if not ckpt.exists() and a.init_state:   # (ep17 10n) start from another run's saved world (a glob: Kaggle inputs)
        import glob
        hits = sorted(glob.glob(a.init_state, recursive=True))
        if not hits:
            raise SystemExit(f"--init-state matches nothing: {a.init_state}")
        src = Path(hits[0])
    if src.exists():   # carry on (Modal's 2 h, Kaggle's 12 h, or a preemption)
        z = np.load(src)
        saved = [int(v) for v in z["seeds"]] if "seeds" in z.files else \
            list(range(a.seed0, a.seed0 + z["a0"].shape[0]))
        pick = [saved.index(s) for s in seeds]   # (10n) the kept worlds' rows
        got = []
        for k, fresh in enumerate(leaves):
            old = z[f"a{k}"][pick]
            if old.shape != fresh.shape:   # (10n) saved with fewer creature slots: the new ones start empty (dead)
                assert old.shape[0] == fresh.shape[0] and old.shape[2:] == fresh.shape[2:] \
                    and old.shape[1] < fresh.shape[1], (k, old.shape, fresh.shape)
                pad = np.array(fresh)
                pad[:, :old.shape[1]] = old
                old = pad
            got.append(jnp.asarray(old))
        wd = jax.tree.unflatten(tree, got)
        period = int(z["period"])
        print(f"resumed at day {period * a.every:.0f} from {src} (cap {a.cap})", flush=True)
    per_day = round(cfg.day_s / cfg.dt)
    steps = round(a.every * per_day / jl.plant_every(wcfg, cfg))
    near = jl.grid_for(wcfg, a.per_bucket)
    run = jax.jit(jax.vmap(lambda s, w, k: jl.run(wcfg, cfg, s, w, k, steps, nearest=near, probe=jl.no_probe())))
    ctl_steps = round(5 * per_day / jl.plant_every(wcfg, cfg))
    ctl = jax.jit(jax.vmap(lambda s, w, k: jl.run(wcfg, cfg, s, w, k, ctl_steps, nearest=near,
                                                  probe=jl.no_probe())))
    err = jax.jit(jax.vmap(lambda w: jl.error(w, cfg)))
    base = jax.random.key(a.seed0 * 7919 + 17)
    t0 = time.perf_counter()
    report = open(out / "report.jsonl", "a")
    last = jax.tree.map(np.asarray, wd.books)
    total_periods = int(round(a.days / a.every))
    while period < total_periods:
        keys = jax.random.split(jax.random.fold_in(base, period), a.worlds)
        wd, pr = run(st, wd, keys)
        period += 1
        day = period * a.every
        c = jax.tree.map(np.asarray, wd.c)
        pr = jax.tree.map(np.asarray, pr)
        books = jax.tree.map(np.asarray, wd.books)
        e = np.asarray(err(wd))
        recs = []
        for k, s in enumerate(seeds):
            rec = state_fields(s, day, c.alive[k], c.diet[k], c.gen[k], c.adult[k])
            rec.update(probe_fields(*(getattr(pr, f)[k] for f in jl.Probe._fields), cfg.dt))
            rec.update(crowded=int(books.crowded[k] - last.crowded[k]), full=int(books.full[k] - last.full[k]),
                       capped=int(books.capped[k] - last.capped[k]), error=float(e[k]))
            recs.append(rec)
            if a.worlds <= a.print_worlds:
                print(line(rec))
            report.write(json.dumps(dict(rec, engine="jax")) + "\n")
        last = books
        hrs = (time.perf_counter() - t0) / 3600
        print(summary(day, recs) + f" [{hrs:.2f} h]", flush=True)
        report.flush()
        if any(abs(d - day) < 1e-6 for d in a.control_at):
            control(a, jl, jax, jnp, ctl, st, wd, base, day)
        if a.hours and hrs > a.hours or period == total_periods or period % a.save_every == 0:
            np.savez(ckpt, period=period, seeds=np.array(seeds), **{f"a{k}": np.asarray(v) for k, v in enumerate(jax.tree.leaves(wd))})
        if a.snap_every and day % a.snap_every < a.every / 2:   # (11d) a copy of the world now, for footage later
            np.savez(out / f"snap-{day:05.0f}.npz", period=period, seeds=np.array(seeds),
                     **{f"a{k}": np.asarray(v) for k, v in enumerate(jax.tree.leaves(wd))})
        if a.hours and hrs > a.hours and period < total_periods:
            print(f"stopped at day {day:.0f} after {hrs:.2f} h; run the same command again to go on", flush=True)
            return


def control(a, jl, jax, jnp, ctl, st, wd, base, day):
    """(9c) Every world twice from this moment: its own brains v random ones, 5 days each."""
    k = jax.random.fold_in(base, 10_000 + int(day))
    rnd = jax.vmap(jl.scramble)(wd, jax.random.split(k, a.worlds))
    keys = jax.random.split(jax.random.fold_in(k, 1), a.worlds)
    res = {}
    for tag, w in (("evolved", wd), ("random", rnd)):
        w2, pr = ctl(st, w, keys)
        pr = jax.tree.map(np.asarray, pr)
        n0, n1 = np.asarray(w.c.alive.sum(1)), np.asarray(w2.c.alive.sum(1))
        res[tag] = dict(kept=n1 / np.maximum(n0, 1), green=pr.green_here / np.maximum(pr.green_land, 1e-9),
                        drink=100 * pr.drink / np.maximum(pr.steps, 1), speed=pr.moved / np.maximum(pr.steps, 1))
    live = np.asarray(wd.c.alive.sum(1)) > 0
    ev, rn = res["evolved"], res["random"]
    better = (ev["kept"] > rn["kept"])[live]
    print(f"day {day:6.0f} CONTROL ({live.sum()} living worlds, 5 days): evolved brains keep "
          f"{100 * np.median(ev['kept'][live]):.0f}% alive v random {100 * np.median(rn['kept'][live]):.0f}% "
          f"(evolved better in {better.sum()}/{live.sum()}); green {np.median(ev['green'][live]):.2f}x v "
          f"{np.median(rn['green'][live]):.2f}x; drink {np.median(ev['drink'][live]):.0f}% v "
          f"{np.median(rn['drink'][live]):.0f}%; {np.median(ev['speed'][live]):.2f} v "
          f"{np.median(rn['speed'][live]):.2f} m/s", flush=True)
    with open(Path(a.out) / "control.jsonl", "a") as f:
        f.write(json.dumps({"day": day, **{f"{t}_{m}": [float(v) for v in res[t][m]] for t in res
                                            for m in res[t]}}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("jax", "numpy"), default="jax")
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--keep", type=int, nargs="*", default=None, help="(JAX) run only these seeds' worlds; a saved "
                    "state holding more is cut down to them")
    ap.add_argument("--days", type=float, default=360)
    ap.add_argument("--every", type=float, default=30)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--cap", type=int, default=2048, help="(JAX) creature slots a world")
    ap.add_argument("--per-bucket", type=int, default=16,
                    help="(JAX) creatures a sight bucket lists (more are missed, counted as crowded; ep17 10j)")
    ap.add_argument("--rain", action="store_true")
    ap.add_argument("--thirst", action="store_true")
    ap.add_argument("--gut", action="store_true", help="(11a) a gut that fits the body: small ones pass fibre unfermented")
    ap.add_argument("--cold", action="store_true", help="(11b) keeping warm costs: heat lost ~mass^0.5 x (37 C - air)")
    ap.add_argument("--cold-breath", action="store_true", help="(11d) keeping warm costs only the water breathed out")
    ap.add_argument("--control-at", type=float, action="append", default=[])
    ap.add_argument("--init-state", default="", help="(JAX) if --out has no state.npz: start from this one (a glob); "
                    "a smaller cap's state is padded with empty slots")
    ap.add_argument("--snap-every", type=float, default=0.0, help="(JAX) also keep the world every this many days "
                    "(snap-<day>.npz)")
    ap.add_argument("--hours", type=float, default=0.0, help="(JAX) stop cleanly after this long; run again to go on")
    ap.add_argument("--save-every", type=int, default=10, help="(JAX) save the state every this many reports")
    ap.add_argument("--print-worlds", type=int, default=16, help="(JAX) a line per world up to this many worlds")
    ap.add_argument("--out", default="runs/eco-jax")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print(" ".join(sys.argv), flush=True)
    (run_jax if a.engine == "jax" else run_numpy)(a)


if __name__ == "__main__":
    main()
