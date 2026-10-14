"""The radio's dictionary (ep11): for each word a pair said on report houses, what was true when it was said and what
the partner did in the next second. `uv run --group jax python scripts/cqb_radio_dict.py runs/cqb-r1-radio:953
[--against runs/cqb-d5-guns:900] [--rounds 64] [--sample-words] [--split]`

The defender's direction is the truth (where he really was, in the speaker's own frame: ahead within 45 degrees,
left, right, behind), whether or not the speaker could see him: that's what a word could be about. --sample-words:
moves and switches best guess, words drawn from the brain's choices (best guess takes the likeliest choice, which
can be silence every time even when the brain talks a lot while sampling)."""

import argparse
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ailearns import record_cqb as R
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb, cqb_scripts as S

NAMES = ["silence", "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel"]


def _ckpt(spec: str):
    run, _, u = spec.partition(":")
    from pathlib import Path
    return Path(run) / "checkpoints" / f"u{int(u):05d}.pt" if u else Path(run) / "latest.pt"


def word_sampler(params):
    """Best-guess moves and switches, the word drawn from the brain's choices."""
    W, n_in = A.n_words(params), params["body.0.weight"].shape[1]

    def play(game, st, who, key):
        mean, logits, _, _ = A.forward(params, game.observe(st, who)[None, :n_in])
        sl, wl = A._split(logits, W)
        sw = jnp.concatenate([(sl > 0).astype(jnp.float32),
                              jax.random.categorical(key, wl)[..., None].astype(jnp.float32)], -1)
        return cqb.full_action(A.to_action(jnp.clip(mean, -1, 1), sw, W)[0])
    return play


def bearing(d, face):
    """Direction of d (world) in a frame facing `face`: 0 ahead, 1 left, 2 right, 3 behind."""
    a = (np.arctan2(d[..., 1], d[..., 0]) - face + np.pi) % (2 * np.pi) - np.pi
    return np.where(np.abs(a) < np.pi / 4, 0, np.where(np.abs(a) > 3 * np.pi / 4, 3, np.where(a > 0, 1, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair")
    ap.add_argument("--against", default="runs/cqb-d5-guns:900")
    ap.add_argument("--rounds", type=int, default=64)
    ap.add_argument("--sample-words", action="store_true")
    ap.add_argument("--split", action="store_true")
    a = ap.parse_args()
    params, ck = A.load(_ckpt(a.pair))
    cfg = R.game_config(a.pair)
    if a.split:
        cfg = replace(cfg, split_entry=True)
    W = A.n_words(params)
    seeds = list(cqb.TEST_SEEDS)[64:64 + a.rounds]
    g = cqb.CQBGame(cqb.make_buildings(seeds, cfg), cfg)
    me = word_sampler(params) if a.sample_words else A.brain(params, 0)
    r = S.play_rounds(g, me, R.player(a.against, g.nA), np.arange(len(seeds)), seed=5000, record=True)
    h = r["history"]
    live = ~np.asarray(h.done)[:, :-1]
    said = np.asarray(h.said)[:, 1:, :2]                      # after each decision
    age = np.asarray(h.said_age)[:, 1:, :2]
    new = (age == 0) & (said > 0) & live[..., None]           # a word started at this decision
    pos, face = np.asarray(h.pos)[:, 1:], np.asarray(h.face)[:, 1:]
    seen = np.asarray(h.seen)[:, 1:]
    alive = np.asarray(h.alive)[:, 1:]
    fired = np.asarray(h.fired)[:, 1:]
    turn = np.asarray(h.turn)[:, 1:]
    lo, hi = np.asarray(g.b.foot_lo)[:, None, None], np.asarray(g.b.foot_hi)[:, None, None]
    inside = ((pos > lo) & (pos < hi)).all(-1)                # (rounds, t, N) inside the house's footprint
    dmid = (np.asarray(g.b.door_lo) + np.asarray(g.b.door_hi)) / 2                       # (rounds, doors, 2)
    dval = np.asarray(g.b.door_valid)
    near = np.where(dval[:, None, None], np.linalg.norm(pos[:, :, :, None] - dmid[:, None, None], axis=-1),
                    np.inf).min(-1) < 1.5                     # (rounds, t, N) within 1.5 m of a doorway
    T = said.shape[1]
    out = np.asarray(r["outcome"])
    talk = new.sum() / max((live[..., None] & alive[:, :, :2]).sum(), 1) / cfg.dt
    print(f"{a.pair} vs {a.against}, {len(seeds)} rounds, words {'sampled' if a.sample_words else 'best guess'}"
          f"{', split entry' if a.split else ''}: won {np.mean(out == 1):.0%}, words per operator-second {talk:.2f}")
    print("| word | said | speaker sees him | partner sees him | he is ahead / left / right / behind the speaker "
          "| ... of the partner | partner in the next 1 s: turns left (+) | fires | gets closer to him (m) "
          "| when (s into the round) | speaker inside | speaker at a doorway |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    k1 = int(1.0 / cfg.dt)
    for w in range(1, W + 1):
        rows = []
        for i, t, s in zip(*np.nonzero(new & (said == w))):
            p = 1 - s                                             # the partner
            d = pos[i, t, 2]
            end = min(t + k1, T - 1)
            rows.append((seen[i, t, s, 2], seen[i, t, p, 2], bearing(d - pos[i, t, s], face[i, t, s]),
                         bearing(d - pos[i, t, p], face[i, t, p]), turn[i, t + 1:end + 1, p].mean() if end > t else 0,
                         fired[i, t + 1:end + 1, p].any(),
                         np.linalg.norm(d - pos[i, t, p]) - np.linalg.norm(pos[i, end, 2] - pos[i, end, p]),
                         alive[i, t, p], t * cfg.dt, inside[i, t, s], near[i, t, s]))
        if not rows:
            print(f"| {NAMES[w]} | 0 | | | | | | | | | | |")
            continue
        x = np.array(rows, dtype=float)
        up = x[:, 7] > 0
        dist = lambda col: " / ".join(f"{np.mean(x[:, col] == k):.0%}" for k in range(4))  # noqa: E731
        print(f"| {NAMES[w]} | {len(x)} ({len(x) / new.sum():.0%}) | {x[:, 0].mean():.0%} | {x[up, 1].mean():.0%} | "
              f"{dist(2)} | {dist(3)} | {x[up, 4].mean():+.2f} | {x[up, 5].mean():.0%} | {x[up, 6].mean():+.2f} | "
              f"{np.median(x[:, 8]):.1f} | {x[:, 9].mean():.0%} | {x[:, 10].mean():.0%} |")
    # the same "what's true" numbers for every decision, as a baseline for the table
    ok = live & alive[:, :, 0] & alive[:, :, 1]
    s_ = np.nonzero(ok)
    print(f"| (any moment) | | {seen[s_[0], s_[1], 0, 2].mean():.0%} | {seen[s_[0], s_[1], 1, 2].mean():.0%} | "
          f"{' / '.join(f'{np.mean(bearing(pos[s_[0], s_[1], 2] - pos[s_[0], s_[1], 0], face[s_[0], s_[1], 0]) == k):.0%}' for k in range(4))} | | | | | {np.median(s_[1]) * cfg.dt:.1f} | {inside[s_[0], s_[1], 0].mean():.0%} | "
          f"{near[s_[0], s_[1], 0].mean():.0%} |")


if __name__ == "__main__":
    main()
