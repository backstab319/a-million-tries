"""Race day 2 (episode 16), step 9g: a real ES generation's pieces, timed. `race_speed.py` drives every car with one
brain; training (`es_field.train`) runs, one after another, every generation:

- the members: 128 brains x 24 rounds = 3072 races at once (`make_generation`)
- own: the centre brain on the same 24 rounds (inside `make_generation`, after the members)
- judge: the new brain on those 24 rounds (the guard)
- bench: the brain on 32 fixed rounds (its curve)

A small batch barely loads the GPU but takes as many steps as a big one. Built as `es_field.train` builds them,
from a run's config, with races cut to `--cap` seconds (every race runs to the cap, so time scales with it), each
scaled to the run's own cap. Also timed: `play` at 129 x 24 races (own folded into the members) and 56 (judge +
bench at once).

    uv run --group jax python scripts/gen_speed.py [--run runs/race-fin-7:50] [--cap 10]
"""

import argparse
import dataclasses
import time

import jax
import jax.numpy as jnp

from ailearns.agents import es_field
from ailearns.agents.ppo_car import load
from ailearns.checkpoints import pick_checkpoint
from ailearns.games import race
from ailearns.profiling import profile

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/race-fin-7:50", help="a checkpoint: its run's config, and the brain")
ap.add_argument("--cap", type=float, default=10.0, help="seconds of race (the run's own cap is scaled to)")
ap.add_argument("--race-control", action="store_true", help="race control on (race-fin-7 trained without it)")
ap.add_argument("--lasers", default="", help="with --quick: comma-separated race.LASER_KERNEL modes (jax, warp), each timed")
ap.add_argument("--brains", default="", help="with --quick: comma-separated 32 / 16 (es_field brain16), each timed")
ap.add_argument("--quick", action="store_true", help="only the generation (members + centre), timed")
ap.add_argument("--profile", action="store_true", help="only the generation, under the profiler (GPU time per kernel)")
ap.add_argument("--small", action="store_true", help="pop 4, rounds 2, bench 2, 2 circuits (a smoke test)")
a = ap.parse_args()

import torch  # noqa: E402

path, _, ck = a.run.partition(":")
ckpt = pick_checkpoint(path, ck or "last")
conf = torch.load(ckpt, map_location="cpu", weights_only=False)["config"]  # (checkpoints carry their run's config)
names = {f.name for f in dataclasses.fields(es_field.FieldESConfig)}
cfg = es_field.FieldESConfig(**{k: v for k, v in conf.items() if k in names})
real_cap = cfg.finish_cap_s
cfg = dataclasses.replace(cfg, finish_cap_s=a.cap, chunk=cfg.chunk or cfg.pop,
                          race_control=a.race_control or cfg.race_control)
if a.small:
    cfg = dataclasses.replace(cfg, pop=4, chunk=4, rounds=2, bench=2, tracks=2)
scale = real_cap / a.cap
print(f"{jax.devices()[0].device_kind}: {a.run} (pop {cfg.pop}, rounds {cfg.rounds}, bench {cfg.bench}, guard "
      f"{cfg.guard}, race control {cfg.race_control}), {a.cap:.0f} s races, x{scale:.0f} to its {real_cap:.0f} s",
      flush=True)

school = es_field.make_school(cfg, list(race.TRAIN_SEEDS)[: cfg.tracks])
rival = es_field._fit(load(ckpt)[0], a.run, school)  # (the run's own brain as centre and rivals)
params = dict(rival)
sig = es_field._sigmas(cfg, params, school.field.obs_size)
play = es_field.make_play_finish(cfg, school, rival)
generation = jax.jit(es_field.make_generation(cfg, play, sig))
keys = jax.random.split(jax.random.PRNGKey(1), cfg.rounds)
bench_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 3000), max(cfg.bench, 1))
played = jax.jit(lambda p, k: {n: v.sum() for n, v in play(p, k).items()})


def timed(fn, *args):
    jax.block_until_ready(fn(*args))  # compile
    t0 = time.time()
    jax.block_until_ready(fn(*args))
    return (time.time() - t0) * scale


def many(n):  # n brains x the 24 rounds, at once (as the members are played)
    stack = jax.tree.map(lambda v: jnp.broadcast_to(v, (n,) + v.shape), params)
    return jax.jit(lambda s: jax.vmap(lambda p: {k: v.sum() for k, v in play(p, keys).items()})(s)), stack


if a.profile:
    profile(generation, params, jax.random.PRNGKey(2), keys, detail=15)
    raise SystemExit
rows = {}
rows["generation (members + own)"] = timed(generation, params, jax.random.PRNGKey(2), keys)
if a.quick:
    print(f"  generation (members + own) {rows['generation (members + own)']:7.1f} s a generation "
          f"(lasers: {race.LASER_KERNEL})", flush=True)
    for mode in a.lasers.split(",") if a.lasers else ():  # the same generation, the wall lasers each way
        race.LASER_KERNEL = mode  # (read when the generation is traced: a new jit each)
        gen = jax.jit(es_field.make_generation(cfg, play, sig))
        print(f"  generation, lasers {mode:5s}      {timed(gen, params, jax.random.PRNGKey(2), keys):7.1f} s a generation",
              flush=True)
    race.LASER_KERNEL = "auto"
    for b in a.brains.split(",") if a.brains else ():  # the same generation, the brains in 32-bit or 16-bit
        cb = dataclasses.replace(cfg, brain16=b == "16")
        gen = jax.jit(es_field.make_generation(cb, es_field.make_play_finish(cb, school, rival), sig))
        print(f"  generation, brains {b}-bit        {timed(gen, params, jax.random.PRNGKey(2), keys):7.1f} s a generation",
              flush=True)
    raise SystemExit
rows[f"  own alone ({cfg.rounds} races)"] = timed(played, params, keys)
rows["judge"] = rows[f"  own alone ({cfg.rounds} races)"]  # the same play on the same rounds
rows[f"bench ({cfg.bench} races)"] = timed(played, params, bench_keys)
f, s = many(cfg.pop)
rows[f"  members alone ({cfg.pop} x {cfg.rounds})"] = timed(f, s)
f, s = many(cfg.pop + 1)
rows[f"members + own at once ({cfg.pop + 1} x {cfg.rounds})"] = timed(f, s)
both = jnp.concatenate([keys, bench_keys])
rows[f"judge + bench at once ({len(both)} races)"] = timed(played, params, both)
for name, t in rows.items():
    print(f"  {name:36s} {t:7.1f} s a generation", flush=True)
now = rows["generation (members + own)"] + rows["judge"] + rows[f"bench ({cfg.bench} races)"]
after = rows[f"members + own at once ({cfg.pop + 1} x {cfg.rounds})"] + rows[f"judge + bench at once ({len(both)} races)"]
print(f"a generation now ~{now:.0f} s; with own folded in and judge + bench together ~{after:.0f} s", flush=True)
