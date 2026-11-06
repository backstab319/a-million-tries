"""Race day 2 (episode 16), step 12: D1, driving under race control by demonstration. "We showed it how to drive under
the flags" (say so on screen).

Episode 15's turn-in by demonstration (`bc_field`), now with race control on and the scripted experts of step 9d
(`race_experts.experts`: yellow, VSC, Safety Car, the new box, round a stopped car) as the teachers. DAgger rounds
(Ross et al. 2011): drive, label every state an expert is in charge of with what it would do, learn, drive again with
more of the brain's own driving, so it also learns to recover from its own mistakes.

- **The brain:** race-fin-7:50 widened to race control's 127 senses (`ppo_field.widen_senses`: zero weights on the
  12 new ones), so on day one it drives exactly as episode 15's. Only the driving changes (two layers and the
  policy, the new senses' weights included), by plain regression on the actions (MSE).
- **Where the experts are in charge** (any of their zones): their inputs are the label. Everywhere else the label is
  the starting brain's own driving (self-distillation; `keep_other` of those steps kept), so it keeps its racing.
- **Stretches:** the race school's mid-race stretches with race control on (episode 16's pit lane, random box calls),
  and in `incident` of them a breakdown early on (`breakdown`, as `race_expert_check.inject`: a car stops dead where it
  is, or its engine dies and it coasts off), so most stretches have a yellow, a VSC or a Safety Car to learn from;
  naturally only about one race in two has one at all.
Logged per round, on the round's own stretches (the brain alone from half way): steps in each expert's zone, cars out
(not the broken-down ones), penalties, the races neutralised. The test is the acceptance test (plan section 6).
Checkpoints are in `ppo_field`'s format.

    uv run --group jax python -m ailearns.agents.bc_control runs/race-d1-1 [--field '{"rounds": 2}']
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents import race_experts as rx
from ailearns.agents.bc_field import EVOLVED
from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_car import forward, load
from ailearns.agents.ppo_field import widen_senses
from ailearns.agents.ppo_jax import adam_init, adam_step, to_torch
from ailearns.games import race, race_control, race_field, race_school


@dataclass
class ControlBCConfig:
    init_from: str = "runs/race-fin-7:50"
    teacher_from: str = ""       # (step 15t) whose racing is the label outside the experts' zones (default: init_from).
                                 # Each run from the last one copied a copy: cars out 63 (D1-2, taught by race-fin-7)
                                 # -> 80 (D1-6, by D1-2) -> 103 (D1-7, by D1-6)
    fields: int = 32             # races driven at once (8 cars each, every car learns)
    stretch_s: float = 90.0      # long enough for the Safety Car to come out, catch the leader and queue the field
    rounds: int = 8              # DAgger rounds: the experts drive their zones with chance `beta` (1 -> 0)
    stretches: int = 4           # stretches each field drives a round
    epochs: int = 4
    batch: int = 4096
    lr: float = 3e-4
    keep_other: float = 0.1      # share of the brain's own (out-of-zone) steps kept as labels
    keep_zone: float = 0.5       # share of in-zone steps kept (a Safety Car stretch is nearly all zone: memory)
    incident: float = 0.8        # stretches with a breakdown early on
    incident_s: tuple = (2.0, 15.0)  # ... at a random second in this window
    call_rate_s: float = 300.0   # the school's random box calls (the box expert's lessons)
    slow_incident: float = 0.0   # (step 15d) stretches with a second breakdown, dead, `slow_after_s` into a VSC or
                                 # Safety Car: the field is slow and close, so the car behind goes round a wreck from
                                 # close up at low speed (D1-2 parked behind one for 104 s, step 15c)
    slow_after_s: float = 20.0
    slow_clear_m: float = 25.0   # ... with no car closer behind than this (at Safety Car pace one can still stop)
    dodge_school: float = 0.0    # (step 15h) stretches that start with a car dead on the track and another stopped
    dodge_gaps: tuple = (6.0, 20.0)  # this many metres (centre to centre) behind it on its line: going round a wreck
                                 # from a standstill, both sides (15f: the brains passed on the left 0-41%)
    dodge_weight: int = 1        # (step 15i) each kept step in the dodge zone counts this many times (repeated in
    reverse_weight: int = 1      # the data), x this again where the label backs up (D1-4 selected reverse but
                                 # didn't give the throttle: the rare move averaged away among ordinary steps)
    tracks: int = len(race.TRAIN_SEEDS)
    seed: int = 0
    method: str = "bc_control"


def make_school(cfg: ControlBCConfig, seeds) -> race_school.RaceSchool:
    rcfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)   # episode 16's pit lane
    field = race_field.RaceField(race.make_tracks(list(seeds), rcfg), n_cars=8, laps=25, cfg=rcfg)
    return race_school.RaceSchool(field, race_school.SchoolConfig(
        stretch_s=cfg.stretch_s, grid_share=0.0, call_rate_s=cfg.call_rate_s, pit_share=0.0, called=0.15),
        race_control.RaceControl(field))


def breakdown(F: race_field.RaceField, f: race_field.Field, key, allowed=None, clear_m=100.0, dead_only=False):
    """One car still racing on the track breaks down: dead where it is (only with no car within 100 m behind: nobody
    avoids a car stopping dead from speed just ahead, and real cars don't), or its engine (it coasts off). `allowed`:
    (n_cars,) the cars it may be (es_field's incident school: rivals only); `clear_m`, `dead_only`: step 15d's slow
    breakdowns. -> the field, (n_cars,) which car."""
    N, lane = F.n, jax.vmap(F.game.in_pit_lane)
    ku, kk = jax.random.split(key)
    ok = ~(f.cars.crashed | f.wrecked | f.finished | F.dry_out(f)) & ~lane(f.cars)
    ok = ok if allowed is None else ok & allowed
    kind = jax.random.bernoulli(kk) | dead_only
    L = F.tracks.length[f.cars.track[0]]
    behind = (f.cars.lap_pos[:, None] - f.cars.lap_pos[None, :]) % L  # (i, j) metres j is behind i
    clear = ~((ok[None, :] & ~jnp.eye(N, dtype=bool) & (behind < clear_m)).any(1))
    ok = ok & (clear | ~kind)
    k = jnp.argmax(jnp.where(ok, jax.random.uniform(ku, (N,)), -1.0))
    one = (jnp.arange(N) == k) & ok.any()
    dead = one & kind
    car = f.cars.car._replace(vx=jnp.where(dead, 0.0, f.cars.car.vx), vy=jnp.where(dead, 0.0, f.cars.car.vy),
                              fuel_used=jnp.where(one & ~dead, f.setup.fuel, f.cars.car.fuel_used))
    return f._replace(cars=f.cars._replace(car=car, crashed=f.cars.crashed | dead, done=f.cars.done | dead)), one


def dodge_setup(F: race_field.RaceField, f: race_field.Field, key, gaps=(6.0, 20.0)):
    """(step 15h) One car racing on the track dies where it is, moved up to `side` m off its line, and another is put
    stopped `gap` m behind it on that line (its tyres, fuel and damage kept), as a car that pulled up behind a wreck.
    `gap` is drawn from `gaps`, `side` from -2..2 m. -> the field, (n_cars,) the dead car."""
    N, lane = F.n, jax.vmap(F.game.in_pit_lane)
    ka, kb, kg, ks, kp = jax.random.split(key, 5)
    ok = ~(f.cars.crashed | f.wrecked | f.finished | F.dry_out(f)) & ~lane(f.cars) & (jnp.abs(f.cars.lat) < 3.0)
    a = jnp.argmax(jnp.where(ok, jax.random.uniform(ka, (N,)), -1.0))
    b = jnp.argmax(jnp.where(ok & (jnp.arange(N) != a), jax.random.uniform(kb, (N,)), -1.0))
    go = ok.sum() >= 2
    lo, hi = gaps
    gap = jax.random.uniform(kg, (), minval=lo, maxval=hi)
    side = jax.random.uniform(ks, (), minval=-2.0, maxval=2.0)   # the wreck: anywhere across the middle of the road
    cars = f.cars
    tr = cars.track[a]
    wa = F.place(kp, tr, cars.design[a], cars.lap_pos[a], side)
    wb = F.place(kp, tr, cars.design[b], cars.lap_pos[a] - gap, side)

    def put(cars, i, st, progress, dead):
        car = cars.car
        new = jax.tree.map(lambda x, y: x.at[i].set(y), car, st.car._replace(
            temp=car.temp[i], wear=car.wear[i], fuel_used=car.fuel_used[i]))
        return cars._replace(car=new, idx=cars.idx.at[i].set(st.idx), lat=cars.lat.at[i].set(st.lat),
                             lap_pos=cars.lap_pos.at[i].set(st.lap_pos), progress=cars.progress.at[i].set(progress),
                             crashed=cars.crashed.at[i].set(cars.crashed[i] | dead),
                             done=cars.done.at[i].set(cars.done[i] | dead))

    moved = put(put(cars, a, wa, cars.progress[a], True), b, wb, cars.progress[a] - gap, False)
    cars = jax.tree.map(lambda x, y: jnp.where(go, x, y), moved, cars)
    return f._replace(cars=cars), (jnp.arange(N) == a) & go



def make_rollout(cfg: ControlBCConfig, school: race_school.RaceSchool, teacher: dict):
    """(brain, key, beta) -> per car-step: obs, label, kept; and the stretch's counts (zone steps, out, penalties,
    its highest race status)."""
    F, ctl = school.field, school.control
    T = int(round(cfg.stretch_s / F.cfg.dt))
    lo, hi = (int(round(s / F.cfg.dt)) for s in cfg.incident_s)

    def one(params, key, beta):
        k0, k1, k2, k3 = jax.random.split(key, 4)
        les = school.start(k0)
        at = jnp.where(jax.random.uniform(k2) < cfg.incident, jax.random.randint(k3, (), lo, hi), -1)
        slow = jax.random.uniform(jax.random.fold_in(k2, 1)) < cfg.slow_incident
        school_d = jax.random.uniform(jax.random.fold_in(k2, 2)) < cfg.dodge_school

        def tick(carry, k):
            les, on, hit, zsteps, top, neut_s, done2 = carry
            ka, kb, kc, ks, ki = jax.random.split(k, 5)
            f, now_hit = jax.lax.cond(f_t(les) == at, lambda f: breakdown(F, f, ki),
                                      lambda f: (f, jnp.zeros(F.n, bool)), les.field)
            st = les.control.status
            neut_s = jnp.where((st == race_control.VSC) | (st == race_control.SC), neut_s + F.cfg.dt, 0.0)
            second = slow & ~done2 & (neut_s >= cfg.slow_after_s)
            f, hit2 = jax.lax.cond(second, lambda f: breakdown(F, f, jax.random.fold_in(ki, 1), clear_m=cfg.slow_clear_m,
                                                               dead_only=True),
                                   lambda f: (f, jnp.zeros(F.n, bool)), f)
            now_hit, done2 = now_hit | hit2, done2 | second
            f, hit3 = jax.lax.cond(school_d & (f_t(les) == 1), lambda f: dodge_setup(F, f, jax.random.fold_in(ki, 2), tuple(cfg.dodge_gaps)),
                                   lambda f: (f, jnp.zeros(F.n, bool)), f)
            now_hit = now_hit | hit3
            les = les._replace(field=f)
            obs = school.observe(les)
            mine = forward(params, obs)[0]
            own = forward(teacher, obs)[0]  # the starting brain: the label outside the zones, its racing inside them
            ex, zones = rx.experts(ctl, f, les.control, jnp.clip(own, -1, 1))
            zone = zones.any(-1)
            label = jnp.where(zone[:, None], ex, own)
            drive = jnp.where((zone & (jax.random.uniform(ka, zone.shape) < beta))[:, None], ex, mine)
            racing = school.racing(f)
            keep = on & racing & jnp.where(zone, jax.random.uniform(kc, zone.shape) < cfg.keep_zone,
                                           jax.random.uniform(kb, zone.shape) < cfg.keep_other)
            zsteps = zsteps + (on & racing)[:, None] * zones
            les, _, _, _, over = school.step(les, jnp.clip(drive, -1, 1), ks)
            top = jnp.maximum(top, jnp.where(on, les.control.status, 0))
            return (les, on & ~over, hit | now_hit, zsteps, top, neut_s, done2), (obs, label, keep, zones[:, rx.DODGE])

        def f_t(les):
            return les.field.t - les.t0

        init = (les, jnp.asarray(True), jnp.zeros(F.n, bool), jnp.zeros((F.n, len(rx.ZONES)), jnp.int32),
                jnp.asarray(0), jnp.asarray(0.0), jnp.asarray(False))
        (les, _, hit, zsteps, top, _, _), (obs, label, keep, dodge) = jax.lax.scan(tick, init, jax.random.split(k1, T))
        f, c = les.field, les.control
        out = (f.cars.crashed | f.wrecked) & ~hit
        return obs, label, keep, dodge, zsteps.sum(0), out.sum(), ctl.penalty_s(c).sum(), top

    return jax.jit(jax.vmap(one, in_axes=(None, 0, None)))


def train(cfg: ControlBCConfig, run_dir: str | Path) -> None:
    import torch
    from ailearns.checkpoints import pick_checkpoint
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already has a run")
    print(f"D1 on {jax.devices()[0].device_kind}; building {cfg.tracks} circuits …", flush=True)
    school = make_school(cfg, list(race.TRAIN_SEEDS)[: cfg.tracks])
    F = school.field
    (run_dir / "config.json").write_text(json.dumps(
        {**asdict(cfg), "game": "race_field", "race": asdict(F.cfg), "school": asdict(school.cfg),
         "obs_size": school.obs_size, "race_control": True, "git": _git_version(), "backend": "jax"}, indent=2))
    path, _, ck = cfg.init_from.partition(":")
    params = widen_senses(load(pick_checkpoint(path, ck or "last"))[0], school.obs_size)
    tp, _, tk = (cfg.teacher_from or cfg.init_from).partition(":")
    teacher = widen_senses(load(pick_checkpoint(tp, tk or "last"))[0], school.obs_size)
    opt = adam_init({n: params[n] for n in EVOLVED})
    rollout = make_rollout(cfg, school, teacher)
    meta = {"config": asdict(cfg), "race": asdict(F.cfg), "backend": "jax", "game": "race_field",
            "race_control": True}

    def loss(p, frozen, obs, label):
        mean = forward(dict(frozen, **p), obs)[0]
        return ((mean - label) ** 2).sum(-1).mean()

    @jax.jit
    def learn(p, frozen, opt, obs, label):
        g = jax.grad(loss)(p, frozen, obs, label)
        return adam_step(p, opt, g, cfg.lr)

    torch.save({"model": to_torch(params), **meta, "update": 0}, run_dir / "checkpoints" / "u00000.pt")
    data_obs, data_label = [], []
    log = open(run_dir / "metrics.jsonl", "a")
    start = time.time()
    rng = np.random.default_rng(cfg.seed)
    for r in range(1, cfg.rounds + 1):
        beta = max(0.0, 1.0 - (r - 1) / max(cfg.rounds // 2, 1))  # the experts' share in their zones: 1 -> 0
        zs, outs, pen, tops, n_st = np.zeros(len(rx.ZONES), np.int64), 0, 0.0, [], 0
        for i in range(cfg.stretches):
            keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(cfg.seed), r * 1000 + i), cfg.fields)
            obs, label, keep, dodge, z, o, p_s, top = jax.device_get(rollout(params, keys, beta))
            keep = keep.astype(bool)
            times = np.where(dodge[keep], cfg.dodge_weight * np.where(label[keep][:, 2] > 0.5, cfg.reverse_weight, 1), 1)
            data_obs.append(np.repeat(obs[keep], times, axis=0))
            data_label.append(np.repeat(label[keep], times, axis=0))
            zs += z.sum(0)
            outs, pen, n_st = outs + int(o.sum()), pen + float(p_s.sum()), n_st + len(top)
            tops += list(top)
            print(f"  round {r} stretch {i + 1}: kept {int(keep.sum()):,} steps", flush=True)
        X, Y = np.concatenate(data_obs), np.concatenate(data_label)
        frozen = {k: v for k, v in params.items() if k not in EVOLVED}
        p = {n: params[n] for n in EVOLVED}
        losses = []
        for _ in range(cfg.epochs):
            order = rng.permutation(len(X))
            for b in range(0, len(X) - cfg.batch + 1, cfg.batch):
                idx = order[b: b + cfg.batch]
                p, opt = learn(p, frozen, opt, jnp.asarray(X[idx]), jnp.asarray(Y[idx]))
            losses.append(float(loss(p, frozen, jnp.asarray(X[idx]), jnp.asarray(Y[idx]))))
        params = dict(params, **p)
        tops = np.asarray(tops)
        cars = n_st * F.n
        stats = {"update": r, "time": round(time.time() - start, 1), "beta": beta, "samples": len(X),
                 "loss": losses[-1], "out_per_car": outs / cars, "penalty_s_per_car": pen / cars,
                 **{f"zone_{k}_s": float(zs[q] * F.cfg.dt / cars) for q, k in enumerate(rx.ZONES)},
                 "vsc_share": float((tops == race_control.VSC).mean()),
                 "sc_share": float(((tops == race_control.SC) | (tops == race_control.SC_IN)).mean()),
                 "red_share": float((tops == race_control.RED).mean())}
        log.write(json.dumps(stats) + "\n")
        log.flush()
        torch.save({"model": to_torch(params), **meta, "update": r, "stats": stats},
                   run_dir / "checkpoints" / f"u{r:05d}.pt")
        print(f"round {r}/{cfg.rounds}  beta {beta:.2f}  out/car {stats['out_per_car']:.3f}  penalty/car "
              f"{stats['penalty_s_per_car']:.1f} s  zones/car (s) " + " ".join(
                  f"{k} {stats[f'zone_{k}_s']:.1f}" for k in rx.ZONES)
              + f"  VSC/SC/red {stats['vsc_share']:.2f}/{stats['sc_share']:.2f}/{stats['red_share']:.2f}  "
              f"samples {len(X):,}  loss {losses[-1]:.4f}  {stats['time']:.0f} s", flush=True)
    log.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="D1: driving under race control by demonstration (episode 16, step 12)")
    ap.add_argument("run")
    ap.add_argument("--field", default="", help='ControlBCConfig fields as JSON, e.g. \'{"rounds": 2}\'')
    a = ap.parse_args()
    train(ControlBCConfig(**json.loads(a.field or "{}")), a.run)
