"""Start points for the rocket's training ladder (ep18 6j): states along the hand-written controller's landings.

A landing's easy part is its end: lit, low, slow, nearly out of fuel. So the agent first starts a moment before a
touchdown the classic controller made, then further back along the same burn, until it starts where the real
descent does (the backward ladder of Salimans & Chen 2018, "Learning Montezuma's Revenge from a single
demonstration"). The controller only says where to start; the agent never sees what it did from there.
"""

import numpy as np

from ailearns.agents.rocket_classic import act
from ailearns.games.rocket import LANDED, Boosters, RocketConfig, entry

# rocket_jax.State without outcome (a start is always flying); the weather too (9c), so a start in a storm gets
# the wind and swell the controller actually flew through, not a fresh draw with the deck somewhere else
FIELDS = ("x", "y", "vx", "vy", "tilt", "spin", "fuel", "thrust", "gimbal", "time", "lit", "ignitions",
          "wind0", "gust", "gust_phase", "heave", "heave_phase",
          "fin", "heat", "peak_q")  # (11c) from 70 km a start also carries the fins' angle and the heat and air load
# taken so far (all three are 0 in the worlds before, so their starts are unchanged)


def record(n: int = 2000, seed: int = 12345, every: int = 5, config: RocketConfig | None = None) -> dict:
    """Fly the classic controller on n burns (from 70 km, 11c: the hand-written entry, then the classic, a row every
    10 steps); keep the ones that landed.
    -> {"states": (k, L, len(FIELDS)) float64, every `every` steps of each burn, padded with its last state;
        "lens": (k,) how many of the L rows are real}."""
    b = Boosters(n, config=config, seed=seed)
    controller = act
    if entry(b.cfg):
        from ailearns.agents.rocket_entry import act as controller
        every = max(every, 10)
    rows, t = [], 0
    while True:
        flying = b.outcome == 0
        if t % every == 0:
            rows.append((np.stack([getattr(b, f).astype(np.float64) for f in FIELDS], -1), flying.copy()))
        if not flying.any():
            break
        b.step(controller(b))
        t += 1
    states = np.stack([r[0] for r in rows], 1)  # (n, L, len(FIELDS))
    alive = np.stack([r[1] for r in rows], 1)  # (n, L)
    keep = b.outcome == LANDED
    states, alive = states[keep], alive[keep]
    lens = alive.sum(1)
    states = states[:, :lens.max()]  # the rows only failed burns reached
    for i, n_i in enumerate(lens):  # after touchdown: hold the last flying row
        states[i, n_i:] = states[i, n_i - 1]
    return {"states": states, "lens": lens}


if __name__ == "__main__":
    d = record()
    print(f"{len(d['lens'])} landings kept; rows each: {d['lens'].min()}-{d['lens'].max()}; "
          f"{d['states'].nbytes / 1e6:.0f} MB")
