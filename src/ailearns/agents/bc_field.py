"""Race day (episode 15), step 8s: the turn-in by demonstration. "We showed it how to enter the pits" (say so).

PPO (steps 8d-8i) and evolution (race-es-1..3) never found the turn-in: called from the racing line, brake from race
speed and move ~10 m right into the 100 m opening before the pit wall, then stop on the mark. So a scripted pit
driver shows it (`expert`: the step 7 test driver's line-following, aimed down the lane, braking to 80 km/h by the
entry and to a stop on the box; `scripts/race_pitstop.py` measured a stop with it at ~19 s), and the brain imitates
it, in DAgger rounds (Ross et al. 2011): drive, label every state the expert is in charge of with what the expert
would do, learn, drive again with more of the brain's own driving, so it also learns to recover from its own mistakes.

- **Where the expert is in charge** (the "zone"): a car called in, from `approach_m` before the pit entry to the
  exit, and any car in the pit lane (the out-lap down the lane). Everywhere else the label is the starting brain's
  own driving, so it keeps its racing (self-distillation; `keep_other` of those steps are kept).
- Every stretch is a pit lesson called from the racing line before the entry (the race school's turn-in lessons).
- Only the driving changes (two layers and the policy), by plain regression on the actions (MSE).
Logged per round, on the round's own stretches (the brain alone in the last rounds): called cars that started outside
the lane, how many turned in, and stops. The test is `scripts/race_box_calls.py` on unseen circuits.
Checkpoints are in `ppo_field`'s format.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_car import forward, load
from ailearns.agents.ppo_jax import adam_init, adam_step, to_torch
from ailearns.games import race, race_field, race_school
from ailearns.games.race_checks import look_ahead, test_driver

EVOLVED = ("body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias", "policy.weight", "policy.bias")


@dataclass
class FieldBCConfig:
    init_from: str = "runs/race-field-5"
    fields: int = 64             # races driven at once (8 cars each, every car learns)
    stretch_s: float = 40.0
    rounds: int = 8              # DAgger rounds: the expert drives its zone with chance `beta` (1 -> 0)
    stretches: int = 4           # stretches each field drives a round
    epochs: int = 4
    batch: int = 4096
    lr: float = 3e-4
    keep_other: float = 0.2      # share of the brain's own (out-of-zone) steps kept as labels
    pit_share: float = 0.5       # stretches that are turn-in lessons; the rest race mid-race with `called` of the cars
    called: float = 0.3          # called in from wherever they are (race-bc-1: lessons only, so a car called far from
                                 # the pits had never been seen, and on test calls from anywhere it hit the wall 153/256)
    approach_m: float = 400.0    # called: the expert takes over this far before the pit entry
    tracks: int = len(race.TRAIN_SEEDS)
    seed: int = 0
    method: str = "bc"


def expert(F: race_field.RaceField, f: race_field.Field, approach_m: float, brain_act):
    """(n_cars, 3) the pit expert's inputs and (n_cars,) where it's in charge (see the module docstring).

    A hybrid (step 8s): on the same 169 called starts the scripted test driver hit the wall 27 times, the brain 11 (it
    brakes into corners from race speed worse), so the brain (`brain_act`: its own inputs) keeps racing on the approach
    and the script only adds the pit work: a speed cap to 80 km/h by the entry, the line into the lane from
    `turn_m` before the entry, the stop on the mark, and the lane to the exit."""
    g, cfg = F.game, F.cfg
    turn_m = 50.0

    def one(st, called, box_s, mine):
        s = g.from_pit(st.track, st.idx)
        in_lane = g.in_pit_lane(st)
        # into the lane only through the opening before the pit wall (ep16 step 9d: a car called while already beside
        # the pits, past the entry, was steered across into the wall; left to itself the ep15 brain dived for the lane
        # and hit the wall too), and one called inside the opening after the braking ends turned in at 120 km/h on the
        # throttle into the lane's outer wall. Past the braking, only a car already slowed (`slow`) carries on into the
        # lane (one turning in isn't "in the lane" yet there: holding every car back from that point made them miss the
        # entry lap after lap); the rest are held on the centreline to the pit exit: the next entry
        v0 = jnp.hypot(st.car.vx, st.car.vy)
        slow = v0 < 1.3 * cfg.pit_limit
        # (run 3: a car the brain had already turned off the tarmac towards the lane was pulled back across the track at
        # 30 m/s into the far wall: one off the tarmac on the lane's side carries on in, the limiter slows it)
        half = cfg.width / 2
        committed = st.lat < -half
        # (step 9f) and a slowed car only if it's already heading for the lane: under the Safety Car every car slows,
        # and cars called just past the entry, still on the racing line, were let in from there and crossed 10 m at
        # 13-15 m/s into the lane's outer edge (3 of the 5 pit-entry crashes in 48 races)
        late = (slow & (st.lat < -1.0)) | committed
        opening = (s > -cfg.pit_entry - turn_m) & ((s < -cfg.pit_entry + 30) | (late & (s < -cfg.pit_wall_len - 20.0)))
        follow = (called & opening) | in_lane
        beside = called & ~follow & (s >= -cfg.pit_entry + 30) & (s < cfg.pit_exit)
        ease = jnp.clip(jnp.minimum((s + cfg.pit_entry) / 60.0, (cfg.pit_exit - s) / 100.0), 0.0, 1.0)  # in, out
        v = jnp.hypot(st.car.vx, st.car.vy)
        to_box = box_s - s
        stop_v = jnp.sqrt(2 * 4.0 * jnp.maximum(to_box, 0.0))  # ease to a stop on the mark (4 m/s^2)
        stopping = called & in_lane & (to_box < 60)
        line = F.pit_line(to_box - look_ahead(st.car), stopping)  # down the fast lane, across into the box (ep16's working lane)
        act = test_driver(g, st, offset=jnp.where(follow, line * ease, 0.0))
        # 80 km/h before the turn-in starts, and no more than a breath of throttle through it to the lane (traced:
        # turning in at 100-120 km/h on the throttle spun the car, power oversteer)
        turn_at = -cfg.pit_entry - turn_m
        entry_v = jnp.sqrt(cfg.pit_limit ** 2 + 2 * 6.0 * jnp.maximum(turn_at - s, 0.0))
        braking = called & (((s < -cfg.pit_entry + 30) & (s > -cfg.pit_entry - approach_m)) | (follow & ~in_lane))
        turning = called & ~in_lane & (s >= turn_at) & (s < -cfg.pit_entry + 30)
        # held: eased towards the centreline, aiming at 70% of its offset (run 5: aimed at the centreline itself, a car
        # 4-5 m off it at 25-30 m/s got full lock and swerved into the cars beside it, all called at once as the
        # Safety Car came out; aimed at its own line, run 3, it drifted wide in a corner)
        held = test_driver(g, st, offset=jnp.clip(0.7 * st.lat, -half + 1.5, half - 1.5))
        # (at the script's racing speed: the called brain lifted for the lane it can't take,
        steer = jnp.where(follow, act[0], jnp.where(beside, held[0], mine[0]))  # and at 58 km/h a car leaving the
        own = jnp.where(follow, act[1], jnp.where(beside, held[1], mine[1]))  # pits hit it)
        pedal = jnp.where(braking, jnp.minimum(own, jnp.clip(0.5 * (entry_v - v), -0.5, 1.0)), own)
        pedal = jnp.where(turning, jnp.minimum(pedal, 0.1), pedal)
        pedal = jnp.where(stopping, jnp.clip(0.5 * (stop_v - v), -0.5, 0.3), pedal)
        # back onto the tarmac after the pit exit (run 3: a car still off it on the lane's side 110 m past the exit was
        # steered into the wall by the brain)
        rejoin = ~in_lane & (s >= cfg.pit_exit - 20.0) & (s < cfg.pit_exit + 200.0) & (st.lat < -half + 1.0)
        steer = jnp.where(rejoin, test_driver(g, st, offset=-half + 2.0)[0], steer)
        zone = (called & (s > -cfg.pit_entry - approach_m) & (s < cfg.pit_exit)) | in_lane | rejoin
        return jnp.stack([steer, pedal, -1.0]), zone  # reverse off

    act, zone = jax.vmap(one)(f.cars, f.box_call, F.box_s, brain_act)
    # (ep16 step 9f) behind the car ahead in its line, in the zone: under the Safety Car almost the whole field boxes
    # on the same lap, and following only the line and its speed they touched in the lane 67 times in 352 stops.
    # Slowing at 3 m/s^2 behind it, half a second back at speed (the red flag script's rule, step 9e)
    cars = f.cars
    v = jnp.hypot(cars.car.vx, cars.car.vy)
    L = F.tracks.length[cars.track[0]]
    gap = (cars.lap_pos[None, :] - cars.lap_pos[:, None]) % L  # (i, j) metres car j is ahead of car i
    there = ~(cars.crashed | f.finished | f.wrecked)
    ahead = (there[None, :] & ~jnp.eye(F.n, dtype=bool) & (gap < 80.0)
             & (jnp.abs(cars.lat[None, :] - cars.lat[:, None]) < 2.4))
    room = jnp.maximum(gap - 7.0 - 0.5 * v[:, None], 0.0)
    keep = jnp.where(ahead, v[None, :] + jnp.sqrt(2 * 3.0 * room), 999.0).min(1)
    pedal = jnp.where(zone, jnp.minimum(act[:, 1], jnp.clip(0.5 * (keep - v), -0.5, 1.0)), act[:, 1])
    return act.at[:, 1].set(pedal), zone


def make_school(cfg: FieldBCConfig, seeds) -> race_school.RaceSchool:
    rcfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
    field = race_field.RaceField(race.make_tracks(list(seeds), rcfg), n_cars=8, laps=25, cfg=rcfg)
    return race_school.RaceSchool(field, race_school.SchoolConfig(
        stretch_s=cfg.stretch_s, grid_share=0.0, call_rate_s=1e9, pit_share=cfg.pit_share, pit_car=0.5,
        pit_approach=1.0, called=cfg.called))


def make_rollout(cfg: FieldBCConfig, school: race_school.RaceSchool, teacher: dict):
    """(brain, key, beta) -> per car-step: obs, label, kept, and the stretch's turn-in counts."""
    F = school.field
    T = int(round(cfg.stretch_s / F.cfg.dt))
    lane = jax.vmap(F.game.in_pit_lane)

    def one(params, key, beta):
        k0, k1 = jax.random.split(key)
        les = school.start(k0)
        called_out = les.field.box_call & ~lane(les.field.cars)

        def tick(carry, k):
            les, on, entered = carry
            ka, kb, ks = jax.random.split(k, 3)
            f = les.field
            obs = F.observe(f)
            mine = forward(params, obs)[0]
            own = forward(teacher, obs)[0]  # the starting brain: the label outside the zone, its racing inside it
            ex, zone = expert(F, f, cfg.approach_m, own)
            label = jnp.where(zone[:, None], ex, own)
            drive = jnp.where((zone & (jax.random.uniform(ka, zone.shape) < beta))[:, None], ex, mine)
            keep = on & school.racing(f) & (zone | (jax.random.uniform(kb, zone.shape) < cfg.keep_other))
            les, _, _, _, over = school.step(les, drive, ks)
            entered = entered | (on & lane(les.field.cars))
            return (les, on & ~over, entered), (obs, label, keep)

        (les, _, entered), (obs, label, keep) = jax.lax.scan(
            tick, (les, jnp.asarray(True), jnp.zeros(F.n, bool)), jax.random.split(k1, T))
        return obs, label, keep, called_out.sum(), (called_out & entered).sum(), les.field.stops.sum()

    return jax.jit(jax.vmap(one, in_axes=(None, 0, None)))


def train(cfg: FieldBCConfig, run_dir: str | Path) -> None:
    import torch
    from ailearns.checkpoints import pick_checkpoint
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already has a run")
    print(f"BC on {jax.devices()[0].device_kind}; building {cfg.tracks} circuits …", flush=True)
    school = make_school(cfg, list(race.TRAIN_SEEDS)[: cfg.tracks])
    F = school.field
    (run_dir / "config.json").write_text(json.dumps(
        {**asdict(cfg), "game": "race_field", "race": asdict(F.cfg), "school": asdict(school.cfg),
         "obs_size": F.obs_size, "git": _git_version(), "backend": "jax"}, indent=2))
    path, _, ck = cfg.init_from.partition(":")
    teacher, _ = load(pick_checkpoint(path, ck or "last"))
    params = dict(teacher)
    opt = adam_init({n: params[n] for n in EVOLVED})
    rollout = make_rollout(cfg, school, teacher)
    meta = {"config": asdict(cfg), "race": asdict(F.cfg), "backend": "jax", "game": "race_field"}

    def loss(p, obs, label, w):
        mean = forward(dict(params_frozen, **p), obs)[0]
        return (w * ((mean - label) ** 2).sum(-1)).sum() / jnp.maximum(w.sum(), 1.0)

    @jax.jit
    def learn(p, opt, obs, label, w):
        g = jax.grad(loss)(p, obs, label, w)
        return adam_step(p, opt, g, cfg.lr)

    torch.save({"model": to_torch(params), **meta, "update": 0}, run_dir / "checkpoints" / "u00000.pt")
    data_obs, data_label = [], []
    log = open(run_dir / "metrics.jsonl", "a")
    start = time.time()
    rng = np.random.default_rng(cfg.seed)
    for r in range(1, cfg.rounds + 1):
        beta = max(0.0, 1.0 - (r - 1) / max(cfg.rounds // 2, 1))  # the expert's share in its zone: 1 -> 0 by half way
        called = turned = stops = 0
        for i in range(cfg.stretches):
            keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(cfg.seed), r * 1000 + i), cfg.fields)
            obs, label, keep, c, t_in, st = jax.device_get(rollout(params, keys, beta))
            keep = keep.astype(bool)
            data_obs.append(obs[keep])
            data_label.append(label[keep])
            called, turned, stops = called + int(c.sum()), turned + int(t_in.sum()), stops + int(st.sum())
        X, Y = np.concatenate(data_obs), np.concatenate(data_label)
        params_frozen = {k: v for k, v in params.items() if k not in EVOLVED}
        p = {n: params[n] for n in EVOLVED}
        losses = []
        for _ in range(cfg.epochs):
            order = rng.permutation(len(X))
            for b in range(0, len(X) - cfg.batch + 1, cfg.batch):
                idx = order[b: b + cfg.batch]
                p, opt = learn(p, opt, jnp.asarray(X[idx]), jnp.asarray(Y[idx]), jnp.ones(len(idx)))
            losses.append(float(loss(p, jnp.asarray(X[idx]), jnp.asarray(Y[idx]), jnp.ones(len(idx)))))
        params = dict(params, **p)
        stats = {"update": r, "time": round(time.time() - start, 1), "beta": beta, "samples": len(X),
                 "called_out": called, "turned_in": turned, "stops": stops, "loss": losses[-1]}
        log.write(json.dumps(stats) + "\n")
        log.flush()
        torch.save({"model": to_torch(params), **meta, "update": r, "stats": stats},
                   run_dir / "checkpoints" / f"u{r:05d}.pt")
        print(f"round {r}/{cfg.rounds}  beta {beta:.2f} (this round's driving: called from the track {called}, "
              f"turned in {turned}, stops {stops})  samples {len(X):,}  loss {losses[-1]:.4f}  {stats['time']:.0f} s",
              flush=True)
    log.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="the turn-in by demonstration (episode 15, step 8s)")
    ap.add_argument("run")
    ap.add_argument("--field", default="", help='FieldBCConfig fields as JSON, e.g. \'{"rounds": 2}\'')
    a = ap.parse_args()
    train(FieldBCConfig(**json.loads(a.field or "{}")), a.run)
