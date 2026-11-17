"""Fly a trained rocket checkpoint from the real start box on unseen seeds, the same way the classic was measured
(scripts/rocket_classic.py), and say how its burns end. Mean action, no exploration noise; numpy sim.

    PYTHONPATH=src python scripts/rocket_eval.py runs/rocket-ppo4/checkpoints/u07629.pkl [n] [seed] [world]

world: fly it somewhere else than it trained (games.rocket.world: pad, ship, ...).
"""

import pickle
import sys

import jax.numpy as jnp
import numpy as np

from ailearns.agents.ppo_rocket import forward
from ailearns.games.rocket import CRASHED, LANDED, TIMEOUT, Boosters, RocketConfig, world


def main(ckpt, n=2000, seed=7, where=None):
    c = pickle.loads(open(ckpt, "rb").read())
    params = {k: jnp.asarray(v) if not isinstance(v, dict) else {kk: jnp.asarray(vv) for kk, vv in v.items()}
              for k, v in c["params"].items()}
    layers = len(c["config"]["hidden"])
    cfg = world(where) if where else RocketConfig(**{k: tuple(v) if isinstance(v, list) else v
                                                     for k, v in c["game"].items()})
    b = Boosters(int(n), config=cfg, seed=int(seed))
    vy, vx, x, fuel = (np.zeros(b.n) for _ in range(4))
    dry_y = np.full(b.n, np.nan)  # (10c) height where open-loop fins ran out of fluid
    n_in = np.asarray(c["params"]["pi0.w"]).shape[0]  # checkpoints from before the extra senses (7b) see 12
    keep = np.ones(n_in)
    if not c["config"].get("sharp_senses", True):  # trained with the square-root senses blanked (an A/B)
        keep[12:] = 0.0
    obs = b.obs()
    while True:
        flying = b.outcome == 0
        mean, _ = forward(params, jnp.asarray(obs[:, :n_in] * keep, jnp.float32), layers)
        obs, _, done = b.step(np.clip(np.asarray(mean, np.float64), -1, 1))
        now = flying & done
        newly = flying & np.isnan(dry_y) & (b.fluid <= 0) & (b.cfg.fin_fluid[1] > 0)
        dry_y[newly] = b.y[newly]
        vy[now], vx[now], x[now], fuel[now] = b.vy[now], b.vx[now], b.x[now], b.fuel[now]
        if done.all():
            break
    print(f"{ckpt} (update {c['update']}), {b.n} burns from the real start box, seed {seed}:")
    for name, code in (("landed", LANDED), ("crashed", CRASHED), ("timed out", TIMEOUT)):
        print(f"  {name:10s} {np.mean(b.outcome == code):6.1%}")
    ok = b.outcome == LANDED
    if ok.any():
        wear = 0.5 * ((vy[ok] / b.cfg.land_vy) ** 2 + (vx[ok] / b.cfg.land_vx) ** 2)  # Boosters.leg_wear
        print(f"  touchdown {np.mean(-vy[ok]):.2f} m/s mean, miss {np.mean(np.abs(x[ok])):.1f} m, "
              f"fuel left {np.mean(fuel[ok]):.0f} kg, leg wear {np.mean(wear):.0%}")
    crash = b.outcome == CRASHED
    if crash.any():
        print(f"  crashes: {np.mean(-vy[crash]):.1f} m/s down, {np.mean(np.abs(vx[crash])):.1f} sideways, "
              f"{np.mean(np.abs(x[crash])):.0f} m off")
    if b.cfg.fin_fluid[1] > 0:  # open-loop fins: who ran dry, where, and how they did
        dry = ~np.isnan(dry_y)
        print(f"  fins ran dry: {np.mean(dry):.1%} of burns (median {np.nanmedian(dry_y) if dry.any() else 0:.0f} m up); "
              f"landed {np.mean(ok[dry]) if dry.any() else 0:.1%} of those, {np.mean(ok[~dry]) if (~dry).any() else 0:.1%} of the rest")
    if b.cfg.fault_p > 0:  # (10e) how it does with each fault
        fin_s, gim_s, weak = b.fin_fail_t < 1e9, b.gimbal_fail_t < 1e9, b.engine_eff < 1
        for name, sel in (("no fault", ~(fin_s | gim_s | weak)), ("fins stuck", fin_s), ("gimbal jammed", gim_s),
                          ("weak engine", weak)):
            print(f"  {name:14s} landed {np.mean(ok[sel]) if sel.any() else 0:6.1%}  ({sel.sum()} burns)")
    if b.cfg.wind_mean[1] > 0:  # weather: how it does by wind at the deck (and the swell)
        w, h = np.abs(b.wind0), b.heave
        for lo, hi in ((0, 4), (4, 8), (8, 10), (10, 12.01)):
            sel = (w >= lo) & (w < hi)
            print(f"  wind {lo:>2}-{hi:<2.0f} m/s: landed {np.mean(ok[sel]):6.1%}  ({sel.sum()} burns)")
        for lo, hi in ((0, 0.75), (0.75, 1.51)):
            sel = (h >= lo) & (h < hi)
            print(f"  swell {lo:.2f}-{hi:.1f} m: landed {np.mean(ok[sel]):6.1%}")
    return float(np.mean(ok))


if __name__ == "__main__":
    main(*sys.argv[1:])
