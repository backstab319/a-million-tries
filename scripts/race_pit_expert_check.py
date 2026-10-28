"""Race day (episode 15), step 8s: how well the pit expert (agents/bc_field.expert) answers calls from the racing line,
with and without contact, worn tyres and damage (16 training circuits, 64 turn-in lessons, everyone else race-field-5).

    uv run --group jax python scripts/race_pit_expert_check.py
"""
import sys, jax, jax.numpy as jnp, numpy as np
from ailearns.agents import bc_field as B
from ailearns.agents.ppo_car import forward, load
from ailearns.games import race, race_field, race_school
p, _ = load("runs/race-field-5/checkpoints/u03051.pt")
rcfg = race.RaceConfig(time_limit=3600.0, pit_lane=True)
tracks = race.make_tracks(list(race.TRAIN_SEEDS)[:16], rcfg)
def diag(name, contact=True, **school):
    F = race_field.RaceField(tracks, n_cars=8, laps=25, cfg=rcfg, contact=contact)
    sc = dict(stretch_s=40.0, grid_share=0.0, call_rate_s=1e9, pit_share=1.0, pit_car=0.5, pit_approach=1.0) | school
    S = race_school.RaceSchool(F, race_school.SchoolConfig(**sc))
    lane = jax.vmap(F.game.in_pit_lane)
    def one(key):
        les = S.start(key); f0 = les.field; co = f0.box_call & ~lane(f0.cars)
        def tick(c, k):
            les, ent = c; f = les.field
            own = forward(p, F.observe(f))[0]
            ex, zone = B.expert(F, f, 400.0, own)
            act = jnp.where(zone[:, None], ex, own)
            les = S.step(les, act, k)[0]
            return (les, ent | lane(les.field.cars)), None
        (les, ent), _ = jax.lax.scan(tick, (les, jnp.zeros(8, bool)), jax.random.split(key, 800))
        f = les.field
        return co, ent, f.cars.crashed & ~f.wrecked, f.wrecked, f.stops > 0
    co, ent, wall, wreck, stop = map(np.asarray, jax.jit(jax.vmap(one))(jax.random.split(jax.random.key(1), 64)))
    n = co.sum()
    print(f"{name:34s} called {n:4d}  turned in {(co&ent).sum()/n:5.0%}  stopped {(co&stop).sum()/n:5.0%}  "
          f"wall {(co&wall).sum()/n:5.0%}  wrecked {(co&wreck).sum()/n:5.0%}", flush=True)
diag("1-2 called, no contact, fresh+intact", contact=False, pit_car=0.2, worn=0.0, damaged=0.0)
diag("1-2 called, no contact", contact=False, pit_car=0.2)
diag("1-2 cars called (contact, worn)", pit_car=0.2)
