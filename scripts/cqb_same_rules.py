"""Report brains side by side under one set of rules (ep9 3at): is a retrained brain better than the old one playing
the new rules untrained? `uv run --group jax python scripts/cqb_same_rules.py --rules runs/cqb-d6-health:200
runs/cqb-d5-guns:900 runs/cqb-d6-health:200 [--against walker,runs/cqb-a10-guns:350] [--react 0.3] [--no-share]`

Same buildings and seed as a run's report (cqb_eval: test 64-255, seed 5000), best guess, normal starts. The brains'
side comes from their own runs; --against defaults to cqb_eval.STANDARD for that side."""

import argparse
from dataclasses import replace
from pathlib import Path

import jax

from ailearns.agents import cqb_eval as E
from ailearns.agents import ppo_cqb as A
from ailearns.games import cqb


def _ckpt(spec: str) -> Path:
    run, _, u = spec.partition(":")
    return Path(run) / "checkpoints" / f"u{int(u):05d}.pt" if u else Path(run) / "latest.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("brains", nargs="+", help="runs/x:update")
    ap.add_argument("--rules", required=True, help="runs/x:update whose rules everyone plays")
    ap.add_argument("--against", default="")
    ap.add_argument("--react", type=float, default=-1, help="those rules with this reaction time instead (s)")
    ap.add_argument("--no-share", action="store_true", help="those rules without shared sightings (ep11)")
    ap.add_argument("--split", action="store_true", help="those rules with the pair starting at two doors (ep11)")
    ap.add_argument("--relay", type=float, default=-1, help="those rules with fresh callouts this old (s; ep11)")
    ap.add_argument("--intel", type=int, default=-1, help="those rules with this many operators on the drone feed (ep11)")
    ap.add_argument("--route", type=int, default=-1, help="those rules with (1) or without (0) the route hint (ep11)")
    ap.add_argument("--house", default=None, help="those rules in another building preset (cqb.HOUSES; ep11)")
    ap.add_argument("--feed", type=int, default=-1, help="those rules with this many on the drone feed's own inputs")
    ap.add_argument("--pilot", type=int, default=-1, help="those rules with (1) or without (0) a drone pilot (ep11)")
    ap.add_argument("--callout", type=int, default=-1, help="those rules with (1) or without (0) the pilot's protocol")
    ap.add_argument("--radio-test", default="", help="mute or scramble the radio (ep11: are the words doing anything?)")
    ap.add_argument("--sample-words", action="store_true", help="radio brains: words drawn from their choices")
    a = ap.parse_args()
    cfg = E._game_cfg(A.load(_ckpt(a.rules))[1])
    if a.react >= 0:
        cfg = replace(cfg, react=a.react)
    if a.no_share:
        cfg = replace(cfg, share_sight=False)
    if a.split:
        cfg = replace(cfg, split_entry=True)
    if a.relay >= 0:
        cfg = replace(cfg, relay=a.relay)
    if a.intel >= 0:
        cfg = replace(cfg, intel=a.intel)
    if a.route >= 0:
        cfg = replace(cfg, route=bool(a.route))
    if a.house is not None:
        cfg = replace(cfg, **{**{k: getattr(cqb.CQBConfig(), k) for k in cqb.HOUSES["big"]}, **cqb.HOUSES[a.house]})
    if a.feed >= 0:
        cfg = replace(cfg, feed=a.feed)
    if a.pilot >= 0:
        cfg = replace(cfg, pilot=bool(a.pilot))
    if a.callout >= 0:
        cfg = replace(cfg, callout=bool(a.callout))
    if a.radio_test:
        cfg = replace(cfg, radio_test=a.radio_test)
    print(f"rules of {a.rules}: health {cfg.health}, react {cfg.react}, magazine {cfg.magazine}, "
          f"flashbangs {cfg.flashbangs}, teams {cfg.team_sizes}, shared sightings {cfg.share_sight}, "
          f"split entry {cfg.split_entry}, relay {cfg.relay}, intel {cfg.intel}, route {cfg.route}, feed {cfg.feed}, pilot {cfg.pilot}, callout {cfg.callout}, rooms {cfg.rooms}, radio {cfg.radio} {cfg.radio_test}"
          f"{', words sampled' if a.sample_words else ''}")
    lo, hi = E.REPORT
    game = cqb.CQBGame(cqb.make_buildings(list(cqb.TEST_SEEDS)[lo:hi], cfg), cfg)
    print("| brain | vs | won | trades | time-outs |\n|---|---|---|---|---|")
    for spec in a.brains:
        params, ck = A.load(_ckpt(spec))
        side = ck["config"]["side"]
        me = cqb.ASSAULT if side == "assault" else cqb.DEFENSE
        for m in (x.strip() for x in (a.against or E.STANDARD[side]).split(",") if x.strip()):
            r = E._summary(jax.device_get(E._matchup(game, me, params, m, hi - lo, E.SEED_REPORT,
                                                    sample_words=a.sample_words)(params)), me, 0, game)
            print(f"| {spec} | {m} | {r['won']:.0%} | {r['trades']:.0%} | {r['timeouts']:.0%} |", flush=True)


if __name__ == "__main__":
    main()
