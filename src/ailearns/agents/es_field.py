"""Race day (episode 15), step 8j: evolution for the race field. Can it find the turn-in, where PPO never did?

PPO (`ppo_field`) explores with a little noise on every 0.05 s input; braking from race speed and moving 7 m right
into the pit lane takes ~3 s of inputs that agree, and that noise never added up to it (step 8i: 0 of 128 called
cars turned in). Evolution explores in the weights instead (OpenAI-ES, as `es_cqb`): each member is the brain plus a
small random change, drives whole stretches with its best guess, and is ranked on what it scored. A member that drifts
right and brakes when called does it every time it's called, so the turn-in can show up whole and be selected.

- Mirrored pairs (brain + change, brain - change), every member on the same stretches that generation (the same
  circuits, packs, calls and luck); scores ranked; the brain moves along the ranked changes (Adam, `lr_rel`).
- A member drives the learner's cars (the first `n_learn` of each field); the rest are the starting brain, best guess.
- Only what decides driving changes (the two layers and the policy; not the value head or `log_std`). The weights on
  the pit senses (the last `PIT_OBS` inputs) explore boldly (`sigma_pit`), the rest in small steps (`sigma`)
  (playbook: a new goal, a bold sigma on the new part).
- Stretches are pit lessons (`SchoolConfig`'s pit lessons and turn-in lessons), so every one has called cars.
- The score is the race school's reward over the stretch (metres, the wall, the team's stop payment), summed over the
  learner's cars, plus ONE coaching term (`coach`, disclosed): the pit coach's potential where each car ended up minus
  where it started (shaping summed over a whole stretch; a car that leaves the race keeps its last potential, so a
  crash isn't counted as "arrived"), and `lane_bonus` once for each car called from the track that gets into the
  pit lane (step 8n). Only the ranking uses it: the logged race reward and the tests don't.

**Finish mode** (`mode="finish"`, step 8k; owner: "favour the drivers who are the fastest, wreck less even though they
might have contact, and can actually finish races"): each round is the last `last_laps` laps of a race, from a
mid-race state (tyres worn, fuel burned to match), and a member is ranked on its cars' **race time** to the flag, no
coach. Not finishing (the wall, a wreck, running dry, too slow for `finish_cap_s`) costs the cap + `dnf_s` + the
distance left at `v_ref`; contact costs only what it does to the car. The team calls a car in when its tyres reach
`call_wear` (a stand-in for the strategist), and a stop made when called is paid back `stop_pay_s` (the team covers
the stop, owner's call). Fuel isn't scripted: the driver sees its fuel and the race left, and saving fuel to finish
is its own business (as F1 drivers lift and coast).
Checkpoints are in `ppo_field`'s format (the `model` weights), so `scripts/race_box_calls.py` reads them as they are.
"""

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ailearns.agents.es_cqb import centered_ranks
from ailearns.agents.ppo import _git_version
from ailearns.agents.ppo_car import forward, forward16, half, load
from ailearns.agents.ppo_jax import adam_init, adam_step, to_torch
from ailearns.agents.bc_control import breakdown
from ailearns.agents.ppo_field import widen_senses
from ailearns.games import race, race_control, race_field, race_school

EVOLVED = ("body.0.weight", "body.0.bias", "body.2.weight", "body.2.bias", "policy.weight", "policy.bias")
PIT_OBS = 5  # the observation's last numbers: called in, in the lane, its box, the entry ahead, the stop's time left


@dataclass
class FieldESConfig:
    init_from: str = "runs/race-field-5"  # the brain evolution starts from (runs/x:update)
    pop: int = 256               # members a generation (mirrored pairs)
    rounds: int = 32             # stretches each member drives a generation (the same for all of them)
    chunk: int = 0               # members driven at once (memory: chunk x rounds fields; 0: all; step 8r: on an L4,
                                 # 3,072 races at once cost 13% less per race than 768)
    sigma: float = 0.005         # the size of a member's change to the brain...
    sigma_pit: float = 0.05      # ...and to the weights on the pit senses
    lr_rel: float = 0.1          # Adam's step, ~lr_rel x each weight's own sigma
    generations: int = 100
    n_cars: int = 8
    n_learn: int = 4
    laps: int = 25
    stretch_s: float = 40.0      # a turn-in and a stop take ~30 s from 300 m before the entry
    pit_share: float = 1.0       # race_school.SchoolConfig: every stretch a pit lesson...
    pit_car: float = 0.5         # ...half the cars in it (the others race)
    pit_approach: float = 0.5    # ...half of them called from the racing line before the pit entry
    call_rate_s: float = 1e9     # no random calls during a stretch (the lessons' calls only: less noise to rank)
    coach: float = 1.0           # the pit coach's weight (the one coaching term; 0: off)
    gamma: float = 0.995         # the coach's potential uses the PPO driver's discount (as it was taught)
    lane_bonus: float = 0.0      # (step 8n, part of the coach) a car called from the track that gets into the pit lane
                                 # earns this once (race-es-2: a turn-in without a clean stop only lost time, so
                                 # evolution learned not to try; ~5 outweighs the lane's ~4 points of lost metres)
    tracks: int = len(race.TRAIN_SEEDS)
    seed: int = 0
    checkpoint_every: int = 10
    method: str = "es"           # (so a run's config says what made it)
    guard: float = -1.0          # undo a step that drops the brain's score on the generation's rounds by more than this
                                 # (-1: off)
    mode: str = "pit"            # "pit": pit lessons, race-school reward + coach; "finish": race ends, race time
    last_laps: int = 2           # finish: the leader this many laps from the flag
    finish_cap_s: float = 450.0  # finish: a round's time cap (2 laps of the longest circuit, 6.8 km, + a stop: ~6 min)
    call_wear: float = 0.7       # finish: the team calls a car in at this mean tyre wear (the cliff is 0.8)
    plan: int = 2                # finish: the compound fitted (rc.COMPOUNDS: hard)
    dnf_s: float = 60.0          # finish: not finishing costs this on top of the cap...
    v_ref: float = 50.0          # ...and the metres left at this speed (m/s)
    fuel_spread: float = 0.01    # finish: fuel burned so far = the race done's share +- this. A team misses its plan by
                                 # ~1-2%; race-fin-1's 0.15 and race-fin-2's 0.05 left about half the cars too dry for
                                 # the last 2 laps however they drove (+-0.05 of the burn is +-4.6% of the tank vs ~8.7%
                                 # needed)
    bench: int = 32              # the brain alone on these fixed rounds every generation (its curve; 0: off)
    # (step 9f) Grand Prix tyres: wear x gp_km / the race's distance x tyre_cal (race_field.RaceField; 0: off), and the
    # team calls a car in on its most worn tyre (step 9e: on the mean of four, a worn front went off first)
    gp_km: float = 0.0
    tyre_cal: float = 1.0
    call_worst: bool = False
    # (step 9f, owner: "variable drivers") the rival cars' drivers: each rival car drawn per round from this pool
    # (comma-separated runs/x[:N], same senses); empty: init_from drives them all
    rivals: str = ""
    # (ep16 step 8b) race control in every round: flags, the director, the Safety Car, the stewards. Finish mode scores
    # the race time after penalties, and a disqualified car (one compound) as not finishing. A 115-sense brain is
    # widened to race control's 127 (zero weights on the new senses), and those get `sigma_rc`
    race_control: bool = False
    sigma_rc: float = 0.05
    # (ep16 step 13, plan section 4: the incident school) finish mode: in this share of rounds one rival car (never a
    # learner's) breaks down at a random second in `incident_s` (`bc_control.breakdown`: dead on the spot, or a dead
    # engine), so most races have a yellow, a VSC or a Safety Car to learn from, not one in two
    incident: float = 0.0
    incident_s: tuple = (60.0, 1200.0)
    # (step 9g, owner) the brains' multiplies in 16-bit while training (ppo_car.forward16: a T4's brains 2.9x faster,
    # ~15% of a generation; actions within 0.0022 of 32-bit). Evaluation, footage and the browser games stay 32-bit.
    brain16: bool = True


_forward32 = forward


def make_school(cfg: FieldESConfig, seeds) -> race_school.RaceSchool:
    rcfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
    field = race_field.RaceField(race.make_tracks(list(seeds), rcfg), n_cars=cfg.n_cars, laps=cfg.laps, cfg=rcfg,
                                 gp_km=cfg.gp_km, tyre_cal=cfg.tyre_cal)
    ctl = race_control.RaceControl(field) if cfg.race_control else None
    if cfg.mode == "finish":  # the end of a race: no lessons, no calls but the team's (in the rollout)
        return race_school.RaceSchool(field, race_school.SchoolConfig(
            stretch_s=cfg.finish_cap_s, grid_share=0.0, call_rate_s=1e9, pit_share=0.0, called=0.0,
            last_laps=cfg.last_laps, fuel_spread=cfg.fuel_spread), ctl)
    # coach 0 in the school: the rollout scores the coach itself (whole-stretch potential, see the docstring)
    return race_school.RaceSchool(field, race_school.SchoolConfig(
        stretch_s=cfg.stretch_s, grid_share=0.0, call_rate_s=cfg.call_rate_s, pit_share=cfg.pit_share,
        pit_car=cfg.pit_car, pit_approach=cfg.pit_approach, coach_gamma=cfg.gamma), ctl)


def _sigmas(cfg: FieldESConfig, params: dict, field_obs: int) -> dict:
    """Each weight's sigma: `sigma`, the pit senses (the field's last `PIT_OBS`) `sigma_pit`, and race control's senses
    (after the field's) `sigma_rc`."""
    sig = {k: jnp.full(params[k].shape, cfg.sigma) for k in EVOLVED}
    w = sig["body.0.weight"].at[:, field_obs - PIT_OBS:field_obs].set(cfg.sigma_pit)
    sig["body.0.weight"] = w.at[:, field_obs:].set(cfg.sigma_rc)
    return sig


def make_play(cfg: FieldESConfig, school: race_school.RaceSchool, rival: dict):
    """(brain, stretch keys) -> per stretch: the race reward, the coach's term, and what we watch (summed over the
    learner's cars)."""
    F, K = school.field, cfg.n_learn
    T = int(round(cfg.stretch_s / F.cfg.dt))
    forward = forward16 if cfg.brain16 else _forward32
    ready = half if cfg.brain16 else (lambda p: p)  # 16-bit weights made once a round, not every decision
    rival = ready(rival)
    lane = jax.vmap(F.game.in_pit_lane)

    def one(params, key):
        k0, k1 = jax.random.split(key)
        params = ready(params)
        les = school.start(k0)
        f0 = les.field
        phi0 = school.potential(f0)[:K]
        called_out = (f0.box_call & ~lane(f0.cars))[:K]  # called in, not yet in the lane: the turn-in to find

        def tick(carry, k):
            les, on, phi, entered, rew = carry
            obs = school.observe(les)
            mine = forward(params, obs[:K])[0]
            theirs = forward(rival, obs[K:])[0]
            racing = school.racing(les.field)[:K]
            les, r, _, _, over = school.step(les, jnp.concatenate([mine, theirs]), k)
            f = les.field
            still = school.racing(f)[:K]
            phi = jnp.where(on & still, school.potential(f)[:K], phi)  # out of the race: its last potential stays
            entered = entered | (on & racing & lane(f.cars)[:K])
            return (les, on & ~over, phi, entered, rew + jnp.where(on, r[:K], 0.0)), None

        z = jnp.zeros(K)
        (les, _, phi, entered, rew), _ = jax.lax.scan(
            tick, (les, jnp.asarray(True), phi0, jnp.zeros(K, bool), z), jax.random.split(k1, T))
        f = les.field
        return {"reward": rew.sum(),
                "coach": cfg.coach * (phi - phi0).sum() + cfg.lane_bonus * (called_out & entered).sum(),
                "metres": (f.cars.progress - les.start)[:K].sum(), "stops": f.stops[:K].sum(),
                "called_out": called_out.sum(), "turned_in": (called_out & entered).sum(),
                "wall": (f.cars.crashed & ~f.wrecked)[:K].sum(), "wrecked": f.wrecked[:K].sum()}

    return lambda params, keys: jax.vmap(lambda k: one(params, k))(keys)


def make_play_finish(cfg: FieldESConfig, school: race_school.RaceSchool, rival: dict):
    """(brain, round keys) -> per round: minus the learner's cars' race time (summed), and what we watch. `rival`: one
    brain, or a pool (every weight with a leading pool axis, `rival["pool"]` set): each rival car draws its driver."""
    F, K = school.field, cfg.n_learn
    T = int(round(cfg.finish_cap_s / F.cfg.dt))
    forward = forward16 if cfg.brain16 else _forward32
    ready = half if cfg.brain16 else (lambda p: p)  # 16-bit weights made once a round, not every decision
    lane = jax.vmap(F.game.in_pit_lane)

    def one(params, key):
        params = ready(params)
        k0, k1 = jax.random.split(key)
        les = school.start(k0)
        f0 = les.field
        ki, ka, kt = jax.random.split(jax.random.fold_in(key, 11), 3)
        lo, hi = (int(round(s / F.cfg.dt)) for s in cfg.incident_s)
        at = jnp.where(jax.random.uniform(ka) < cfg.incident, jax.random.randint(kt, (), lo, hi), -1)
        rivals = jnp.arange(F.n) >= K
        D = F.laps * F.tracks.length[f0.cars.track[:K]]  # race distance
        if "pool" in rival:  # (step 9f) every rival car its own driver from the pool, for this round
            n_pool = rival["body.0.weight"].shape[0]
            pick = jax.random.randint(jax.random.fold_in(key, 9), (F.n - K,), 0, n_pool)
            # (ep16 step 13) each driver's 16-bit weights made on its own: `half` on the stacked pool transposed the
            # pool axis into the weights (the pool, step 9f, came before 16-bit brains, 9g)
            theirs = jax.vmap(ready)({n: v[pick] for n, v in rival.items() if n != "pool"})
            drive_rivals = lambda o: jax.vmap(lambda p, x: forward(p, x[None])[0][0])(theirs, o)  # noqa: E731
        else:
            mates = ready(rival)
            drive_rivals = lambda o: forward(mates, o)[0]  # noqa: E731

        def tick(carry, k):
            les, on, asked, entered, paid = carry
            f = jax.lax.cond(les.field.t - les.t0 == at, lambda f: breakdown(F, f, ki, rivals)[0], lambda f: f,
                             les.field)
            les = les._replace(field=f)
            wear = f.cars.car.wear.max(-1) if cfg.call_worst else f.cars.car.wear.mean(-1)
            call = school.racing(f) & ~f.box_call & (wear >= cfg.call_wear)  # the team
            f = f._replace(box_call=f.box_call | call, plan=jnp.where(call, cfg.plan, f.plan))
            asked = asked | (on & call[:K] & ~lane(f.cars)[:K])  # called from the track: it has to turn in
            obs = school.observe(les._replace(field=f))
            acts = jnp.concatenate([forward(params, obs[:K])[0], drive_rivals(obs[K:])])
            les, _, _, _, over = school.step(les._replace(field=f), acts, k)
            new = les.field
            entered = entered | (on & asked & lane(new.cars)[:K])
            paid = paid + (on & (new.stops > f.stops) & f.box_call)[:K]
            return (les, on & ~over, asked, entered, paid), None

        zb = jnp.zeros(K, bool)
        (les, _, asked, entered, paid), _ = jax.lax.scan(
            tick, (les, jnp.asarray(True), zb, zb, jnp.zeros(K, jnp.int32)), jax.random.split(k1, T))
        f = les.field
        prog = f.cars.progress[:K]
        left = jnp.maximum(D - prog, 0.0)
        ctl, c = school.control, les.control
        pen = ctl.penalty_s(c)[:K] if ctl else jnp.zeros(K)  # the stewards (ep16 step 7)
        dsq = ctl.disqualified(f, c)[:K] if ctl else jnp.zeros(K, bool)
        done = f.finished[:K] & ~dsq  # disqualified: as if it hadn't finished
        race_s = jnp.where(done, f.finish_t[:K] - f0.t * F.cfg.dt + left / cfg.v_ref,
                           cfg.finish_cap_s + cfg.dnf_s + left / cfg.v_ref) + pen - F.stop_pay_s * paid
        dry = (f.cars.car.fuel_used >= f.setup.fuel)[:K]
        z = jnp.asarray(0)
        return {"reward": -race_s.sum(), "coach": jnp.asarray(0.0), "race_s": race_s.sum(), "finished": done.sum(),
                "penalty_s": pen.sum(), "dsq": dsq.sum(), "vsc": c.vsc_n if ctl else z, "sc": c.sc_n if ctl else z,
                "red": c.red_n if ctl else z,
                "metres": (prog - les.start[:K]).sum(), "stops": f.stops[:K].sum(), "called_out": asked.sum(),
                "turned_in": (asked & entered).sum(), "wall": (f.cars.crashed & ~f.wrecked)[:K].sum(),
                "wrecked": f.wrecked[:K].sum(), "dry": dry.sum(),
                "dry_out": (dry & ~done & ~f.wrecked[:K]).sum(),  # dry and that's why it didn't finish
                "fuel_had": (f0.setup.fuel - f0.cars.car.fuel_used)[:K].sum(),  # kg in the tanks at the start
                "fuel_burned": (f.cars.car.fuel_used - f0.cars.car.fuel_used)[:K].sum(),
                "to_go": (D - les.start[:K]).sum()}  # metres to the flag at the start

    return lambda params, keys: jax.vmap(lambda k: one(params, k))(keys)


def make_generation(cfg: FieldESConfig, play, sig: dict):
    """One generation as one compiled function: (brain, key, stretch keys) -> (gradient estimate, stats)."""
    pairs = cfg.pop // 2
    names = tuple(sig)

    def noise(k):
        ks = jax.random.split(k, len(names))
        return {n: jax.random.normal(kk, sig[n].shape) for n, kk in zip(names, ks)}

    def member(center, ks, keys):
        k, s = ks
        eps = noise(k)
        return play(dict(center, **{n: center[n] + s * sig[n] * eps[n] for n in names}), keys)

    # (step 9g) all members at once: the centre brain's own rounds ride along as one more member with no noise (0 x
    # the noise: exactly the centre), not as a batch of 24 races after them, which took 41 s of a T4's ~710 s
    fold = cfg.chunk >= cfg.pop

    def generation(center, key, keys):
        ks = jax.random.split(key, pairs)
        signs = jnp.concatenate([jnp.ones(pairs), -jnp.ones(pairs)])
        if fold:
            outs = jax.vmap(lambda x: member(center, x, keys))((jnp.concatenate([ks, ks, ks[:1]]),
                                                                 jnp.concatenate([signs, jnp.zeros(1)])))
            own = jax.tree.map(lambda v: v[-1], outs)
            outs = jax.tree.map(lambda v: v[:-1], outs)
        else:
            outs = jax.lax.map(lambda x: member(center, x, keys), (jnp.concatenate([ks, ks]), signs),
                               batch_size=cfg.chunk)
            own = play(center, keys)
        m = {"score": (outs["reward"] + outs["coach"]).mean(1), "reward": outs["reward"].mean(1),
             "stops": outs["stops"].sum(1), "turned_in": outs["turned_in"].sum(1)}
        ranks = centered_ranks(m["score"])
        w = ranks[:pairs] - ranks[pairs:]

        def add(acc, kw):
            k, wi = kw
            eps = noise(k)
            return {n: acc[n] + wi * eps[n] for n in names}, None
        g, _ = jax.lax.scan(add, {n: jnp.zeros_like(center[n]) for n in names}, (ks, w))
        grad = {n: -g[n] / (pairs * sig[n]) for n in names}  # minus: adam_step descends
        R = keys.shape[0]
        stats = {"score": m["score"].mean(), "score_best": m["score"].max(), "reward_members": m["reward"].mean(),
                 "members_stops_per_stretch": m["stops"].mean() / R,
                 "members_turned_in_per_stretch": m["turned_in"].mean() / R,
                 "members_stopping": (m["stops"] > 0).mean(),
                 "brain_score": (own["reward"] + own["coach"]).mean(), "brain_reward": own["reward"].mean(),
                 "brain_coach": own["coach"].mean(), "brain_metres_per_car": own["metres"].sum() / (R * cfg.n_learn),
                 **{f"brain_{k}": v.sum() for k, v in own.items() if k not in ("reward", "coach", "metres")}}
        if "race_s" in own:  # finish mode: per car
            stats["brain_race_s_per_car"] = own["race_s"].sum() / (R * cfg.n_learn)
        return grad, stats

    return generation


def _fit(brain: dict, name: str, school: race_school.RaceSchool) -> dict:
    """A brain for the school's senses: a field brain widened to race control's (zero weights on the new senses)."""
    have, want = brain["body.0.weight"].shape[1], school.obs_size
    if have == school.field.obs_size and want > have:
        return widen_senses(brain, want)
    if have != want:
        raise SystemExit(f"{name} has {have} senses, the school {want}")
    return brain


def train(cfg: FieldESConfig, run_dir: str | Path, max_hours: float | None = None, resume: bool = False) -> None:
    """resume: carry on from latest.pt."""
    import torch
    from ailearns.agents import tracking
    from ailearns.checkpoints import pick_checkpoint
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    state = None
    if resume:
        state = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    elif (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir} already has a run")
    cfg = replace(cfg, chunk=cfg.chunk or cfg.pop)
    if cfg.pop % 2 or cfg.pop % cfg.chunk:
        raise SystemExit(f"pop ({cfg.pop}) must be even and divide into chunks of {cfg.chunk}")
    dev = jax.devices()[0]
    print(f"ES on {dev.device_kind}; building {cfg.tracks} circuits …", flush=True)
    school = make_school(cfg, list(race.TRAIN_SEEDS)[: cfg.tracks])
    F = school.field
    if not state:
        (run_dir / "config.json").write_text(json.dumps(
            {**asdict(cfg), "game": "race_field", "race": asdict(F.cfg), "school": asdict(school.cfg),
             "obs_size": school.obs_size, "git": _git_version(), "backend": "jax", "device": f"1 x {dev.device_kind}"},
            indent=2))
    path, _, ck = cfg.init_from.partition(":")
    rival = _fit(load(pick_checkpoint(path, ck or "last"))[0], cfg.init_from, school)
    params = {k: jnp.asarray(v.numpy()) for k, v in state["model"].items()} if state else dict(rival)
    sig = _sigmas(cfg, params, F.obs_size)
    if cfg.rivals:  # (step 9f) a pool of drivers for the rival cars
        if cfg.mode != "finish":
            raise SystemExit("rivals: finish mode only")
        pool = []
        for r in cfg.rivals.split(","):
            rp, _, rk = r.strip().partition(":")
            b = _fit(load(pick_checkpoint(rp, rk or "last"))[0], r, school)
            if b["body.0.weight"].shape != rival["body.0.weight"].shape:
                raise SystemExit(f"{r} isn't shaped like {cfg.init_from}")
            pool.append(b)
        rival = {n: jnp.stack([b[n] for b in pool]) for n in rival} | {"pool": jnp.asarray(len(pool))}
    play = (make_play_finish if cfg.mode == "finish" else make_play)(cfg, school, rival)
    generation = jax.jit(make_generation(cfg, play, sig))
    bench_keys = jax.random.split(jax.random.PRNGKey(cfg.seed + 3000), max(cfg.bench, 1))
    # (step 9g) the guard's rounds and the bench's in one play of the new brain: two small batches one after the
    # other took 137 s of a T4's ~710 s generation, together 83. A step turned down leaves the brain as it was, so
    # its bench is the last one.
    judge = jax.jit(lambda p, keys: play(p, jnp.concatenate([keys, bench_keys]) if cfg.bench else keys))
    bench = jax.jit(lambda p: {k: v.sum() for k, v in play(p, bench_keys).items()})
    opt = jax.tree.map(jnp.asarray, state["optimizer"]) if state else adam_init({n: params[n] for n in EVOLVED})
    first = state["update"] + 1 if state else 1
    meta = {"config": asdict(cfg), "race": asdict(F.cfg), "backend": "jax", "game": "race_field"}

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
    best_bench = -float("inf")  # the best bench score so far (saved when beaten)
    last_bench = None
    for g in range(first, cfg.generations + 1):  # each generation's luck from its number (the same after a resume)
        k_gen, k_rounds = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(cfg.seed + 2000), g))
        keys = jax.random.split(k_rounds, cfg.rounds)
        grad, m = generation(params, k_gen, keys)
        new, opt_new = adam_step({n: params[n] for n in EVOLVED}, opt, grad, 1.0)  # linear in lr: scale per weight
        new = dict(params, **{n: params[n] + cfg.lr_rel * sig[n] * (new[n] - params[n]) for n in EVOLVED})
        kept, b, R = True, None, cfg.rounds
        if cfg.guard >= 0:  # the step stands only if the brain still scores on these rounds (within `guard`)
            o = jax.device_get(judge(new, keys))
            kept = float((o["reward"][:R] + o["coach"][:R]).mean()) >= float(m["brain_score"]) - cfg.guard
            if cfg.bench and kept:
                b = {k: v[R:].sum() for k, v in o.items()}
        if kept:
            params, opt = new, opt_new
        stats = {"update": g, "time": round(time.time() - start, 1), "step_kept": kept,
                 **{k: float(v) for k, v in jax.device_get(m).items()}}
        if cfg.bench:  # the brain after this generation, on the same rounds every time: its real curve
            if b is None:
                b = last_bench if not kept and last_bench is not None else jax.device_get(bench(params))
            last_bench = b
            per_car = cfg.bench * cfg.n_learn
            stats.update({f"bench_{k}": float(v) for k, v in b.items()})
            stats["bench_score"] = float(b["reward"] + b["coach"]) / cfg.bench
            if "race_s" in b:
                stats["bench_race_s_per_car"] = float(b["race_s"]) / per_car
        log.write(json.dumps(stats) + "\n")
        log.flush()
        tracking.log(tracker, stats)
        # (ep16 13e) and whenever the bench sets a new best: d2-3's best (gen 66) came between saves and was lost
        new_best = cfg.bench and stats["bench_score"] > best_bench
        if new_best:
            best_bench = stats["bench_score"]
        if g % cfg.checkpoint_every == 0 or g == cfg.generations or new_best:
            save(g, stats)
        s = stats
        if cfg.mode == "finish":
            print(f"gen {g:4d}/{cfg.generations}  brain: {s['brain_race_s_per_car']:.1f} s/car, finished "
                  f"{s['brain_finished']:.0f}/{cfg.rounds * cfg.n_learn}, stops {s['brain_stops']:.0f}, turned in "
                  f"{s['brain_turned_in']:.0f}/{s['brain_called_out']:.0f}, wall {s['brain_wall']:.0f}, wrecked "
                  f"{s['brain_wrecked']:.0f}, dry {s['brain_dry']:.0f}"
                  + (f", penalties {s['brain_penalty_s']:.0f} s, dsq {s['brain_dsq']:.0f}, VSC {s['brain_vsc']:.0f}, "
                     f"SC {s['brain_sc']:.0f}, red {s['brain_red']:.0f}" if cfg.race_control else "")
                  + f"  members: score {s['score']:.1f} (best "
                  f"{s['score_best']:.1f})" + ("" if kept else "  step undone")
                  + (f"  bench: {s['bench_race_s_per_car']:.1f} s/car, finished {s['bench_finished']:.0f}/"
                     f"{cfg.bench * cfg.n_learn}, dry {s['bench_dry']:.0f}, wrecked {s['bench_wrecked']:.0f}, stops "
                     f"{s['bench_stops']:.0f}" if cfg.bench else "") + f"  {s['time']:.0f} s", flush=True)
        else:
            print(f"gen {g:4d}/{cfg.generations}  brain: reward {s['brain_reward']:.2f}/stretch (coach "
                  f"{s['brain_coach']:+.2f}), {s['brain_metres_per_car']:.0f} m/car, stops {s['brain_stops']:.0f}, "
                  f"turned in {s['brain_turned_in']:.0f}/{s['brain_called_out']:.0f}, wall {s['brain_wall']:.0f}, "
                  f"wrecked {s['brain_wrecked']:.0f}  members: score {s['score']:.2f} (best {s['score_best']:.2f}), "
                  f"{s['members_stopping']:.0%} stop, turned in {s['members_turned_in_per_stretch']:.2f}/stretch"
                  + ("" if kept else "  step undone") + f"  {s['time']:.0f} s", flush=True)
        if max_hours is not None and time.time() - start > max_hours * 3600:
            save(g, stats)
            print(f"stopped after {max_hours} h at generation {g}", flush=True)
            break
    log.close()
    tracking.finish(tracker)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="evolution in the race school (episode 15, step 8j)")
    ap.add_argument("run")
    ap.add_argument("--field", default="", help='FieldESConfig fields as JSON, e.g. \'{"pop": 64}\'')
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    train(FieldESConfig(**json.loads(a.field or "{}")), a.run, max_hours=a.hours, resume=a.resume)
