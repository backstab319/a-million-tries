"""The games and the AIs that learn them: play, train, evaluate.

    uv run games play                         # Snake, you play
    uv run --group jax games drive            # the race car, you drive
    uv run --group jax games tag              # tag vs the AI, you're the robot (--as blob: you run)
    uv run games train runs/snake-1 --arch cnn-deep --steps 20e6
    uv run --group jax games train-car runs/car-1 --steps 300e6
    uv run games eval runs/snake-1            # best-move games: win rate, mean score
    uv run --group jax games eval-car runs/car-1
    uv run --group jax games train-tag runs/tag-1 --steps 50e6
    uv run --group jax games eval-tag runs/tag-1
"""

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="games", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("play", help="play Snake yourself")
    sp.add_argument("--size", type=int, default=10)
    sp.add_argument("--speed", type=float, default=8.0, help="moves per second")

    sp = sub.add_parser("drive", help="drive the race car yourself (needs --group jax)")
    sp.add_argument("--track", type=int, default=100_000, help="track seed (100000+: tracks the AI never trains on)")

    sp = sub.add_parser("tag", help="play tag yourself against the AI (needs --group jax)")
    sp.add_argument("--as", dest="side", choices=["robot", "blob"], default="robot",
                    help="robot: you chase; blob: you run")
    sp.add_argument("--vs", help="the AI you play (default: the last generation of the other side)")
    sp.add_argument("--arena", type=int, default=100_000, help="first arena (100000+: arenas the AI never trains in)")
    sp.add_argument("--no-fog", action="store_true", help="see the whole arena (the AI doesn't)")
    sp.add_argument("--match", action="store_true", help="the scored ladder: every generation, 10 rounds each")

    sp = sub.add_parser("train", help="train a Snake AI with PPO")
    sp.add_argument("run", help="run directory, e.g. runs/snake-1")
    sp.add_argument("--size", type=int, default=10)
    sp.add_argument("--steps", type=float, default=20e6)
    sp.add_argument("--envs", type=int, default=128)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--arch", choices=["mlp", "cnn", "cnn-deep"], default="mlp")
    sp.add_argument("--view", choices=["ego2", "ego", "board"], default="ego2")
    sp.add_argument("--gamma", type=float, default=0.99, help="how much it values the future (per move)")
    sp.add_argument("--resume", action="store_true", help="continue from <run>/latest.pt")
    sp.add_argument("--max-hours", type=float, help="stop cleanly after this long (resumable)")
    sp.add_argument("--device", default="cpu", help="cpu | cuda (torch backend)")
    sp.add_argument("--backend", choices=["torch", "jax"], default="torch",
                    help="jax: compiled training loop on every device (needs `uv sync --group jax`)")

    sp = sub.add_parser("train-car", help="train a race car AI with PPO (needs --group jax)")
    sp.add_argument("run", help="run directory, e.g. runs/car-1")
    sp.add_argument("--steps", type=float, default=300e6)
    sp.add_argument("--envs", type=int, default=4096)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--hidden", type=int, default=256, help="neurons per layer")
    sp.add_argument("--ray-range", type=float, default=60.0, help="how far the lasers reach (m)")
    sp.add_argument("--messy-starts", action="store_true", help="practice from off-centre, angled, moving starts")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--max-hours", type=float)

    sp = sub.add_parser("train-tag", help="train a tag AI (the chaser, by default) with PPO (needs --group jax)")
    sp.add_argument("run", help="run directory, e.g. runs/tag-1")
    sp.add_argument("--steps", type=float, default=50e6)
    sp.add_argument("--envs", type=int, default=2048)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--hidden", type=int, default=256, help="neurons per layer")
    sp.add_argument("--side", choices=["chaser", "runner"], default="chaser", help="who learns")
    sp.add_argument("--opponent", default="flee",
                    help="who it plays against: flee | smart | still | naive, or frozen brains runs/a:175,runs/b:200")
    sp.add_argument("--closer-reward", type=float, default=0.0, help="reward per metre closer (0: catches only)")
    sp.add_argument("--init-from", default="", help="start from another run's brain, e.g. runs/tag-v2-closer:190")
    sp.add_argument("--remember-heading", action="store_true", help="out of sight: remember which way it was going")
    sp.add_argument("--reset-std", action="store_true", help="with --init-from: experiment again (fresh move noise)")
    sp.add_argument("--resume", action="store_true")
    sp.add_argument("--max-hours", type=float)

    sp = sub.add_parser("eval", help="Snake: best-move games from a checkpoint")
    sp.add_argument("run")
    sp.add_argument("--checkpoint", default="last", help="first | last | <update> | <path>")
    sp.add_argument("--games", type=int, default=256)
    sp.add_argument("--seed", type=int, default=5000)

    sp = sub.add_parser("eval-car", help="race car: one standing-start run on each unseen test track")
    sp.add_argument("run")
    sp.add_argument("--checkpoint", default="last")
    sp.add_argument("--tracks", type=int, default=48)
    sp.add_argument("--first", type=int, default=16)

    sp = sub.add_parser("eval-tag", help="tag: one best-guess round in each unseen test arena")
    sp.add_argument("run")
    sp.add_argument("--checkpoint", default="last")
    sp.add_argument("--rounds", type=int, default=192)
    sp.add_argument("--first", type=int, default=64)
    sp.add_argument("--opponent", help="a script or runs/x:ckpt (default: the one it trained against)")

    sp = sub.add_parser("tag-table", help="tag: tournament table, every chaser vs every runner (catch %%)")
    sp.add_argument("--chasers", nargs="+", required=True)
    sp.add_argument("--runners", nargs="+", required=True)
    sp.add_argument("--rounds", type=int, default=192)
    sp.add_argument("--first", type=int, default=64)
    sp.add_argument("--out", help="also save the table as json (for footage)")

    sp = sub.add_parser("tag-match", help="tag: any chaser vs any runner (scripts or runs/x:ckpt), unseen arenas")
    sp.add_argument("chaser")
    sp.add_argument("runner")
    sp.add_argument("--rounds", type=int, default=192)
    sp.add_argument("--first", type=int, default=64)

    args = p.parse_args()
    if args.cmd == "play":
        from ailearns import play
        play.run(args.size, args.speed)
    elif args.cmd == "drive":
        from ailearns import drive
        drive.run(args.track)
    elif args.cmd == "tag":
        from ailearns import play_tag
        play_tag.run(args.side, args.vs, args.arena, fog=not args.no_fog, match=args.match)
    elif args.cmd == "train":
        from ailearns.agents.ppo import PPOConfig, train
        cfg = PPOConfig(size=args.size, view=args.view, arch=args.arch, total_steps=int(args.steps), n_envs=args.envs,
                        seed=args.seed, gamma=args.gamma)
        if args.backend == "jax":
            from ailearns.agents.ppo_jax import train as jax_train
            jax_train(cfg, args.run, resume=args.resume, max_hours=args.max_hours)
        else:
            train(cfg, args.run, resume=args.resume, max_hours=args.max_hours, device=args.device)
    elif args.cmd == "train-car":
        from ailearns.agents.ppo_car import CarPPOConfig, train
        train(CarPPOConfig(total_steps=int(args.steps), n_envs=args.envs, seed=args.seed, hidden=args.hidden,
                           ray_range=args.ray_range, messy_starts=args.messy_starts),
              args.run, resume=args.resume, max_hours=args.max_hours)
    elif args.cmd == "train-tag":
        from ailearns.agents.ppo_tag import TagPPOConfig, train
        train(TagPPOConfig(total_steps=int(args.steps), n_envs=args.envs, seed=args.seed, hidden=args.hidden,
                           side=args.side, opponent=args.opponent, closer_reward=args.closer_reward,
                           init_from=args.init_from, remember_heading=args.remember_heading,
                           reset_std=args.reset_std),
              args.run, resume=args.resume, max_hours=args.max_hours)
    elif args.cmd == "eval-tag":
        from ailearns.agents.ppo_tag import evaluate
        r = evaluate(args.run, args.checkpoint, args.rounds, args.first, opponent=args.opponent)
        t = r["catch_time"]
        print(f"{args.run} u{r['update']} ({r['moves'] / 1e6:.0f}M moves): caught {r['catch_rate']:.0%} "
              f"of {r['rounds']} rounds" + (f", in {t:.1f} s on average" if t else ""))
    elif args.cmd == "tag-table":
        import json
        from ailearns.agents.ppo_tag import match
        cells = [[match(c, r, args.rounds, args.first)["catch_rate"] for r in args.runners] for c in args.chasers]
        w = max(map(len, args.chasers))
        print(" " * w + "".join(f"{r[-24:]:>26}" for r in args.runners))
        for c, row in zip(args.chasers, cells):
            print(f"{c:<{w}}" + "".join(f"{x:>26.0%}" for x in row))
        if args.out:
            Path(args.out).write_text(json.dumps({"chasers": args.chasers, "runners": args.runners, "rounds": args.rounds,
                                                  "first": args.first, "catch_rate": cells}, indent=2))
    elif args.cmd == "tag-match":
        from ailearns.agents.ppo_tag import match
        r = match(args.chaser, args.runner, args.rounds, args.first)
        t = r["catch_time"]
        print(f"{args.chaser} vs {args.runner}: caught {r['catch_rate']:.0%} of {args.rounds} rounds"
              + (f", in {t:.1f} s on average" if t else ""))
    elif args.cmd == "eval":
        r = eval_snake(args.run, args.checkpoint, args.games, args.seed)
        print(f"{args.run} u{r['update']}: filled the board in {r['wins']}/{r['games']} games "
              f"({r['wins'] / r['games']:.0%}), mean {r['mean_score']:.1f} apples")
    elif args.cmd == "eval-car":
        import numpy as np
        from ailearns.agents.ppo_car import evaluate
        r = evaluate(args.run, args.checkpoint, args.tracks, args.first)
        laps = list(r["flying_laps"].values())
        print(f"{args.run} u{r['update']}: {r['tracks']} tracks, {r['crashes']} crashes, "
              f"{r['completed_lap']} lapped, mean flying lap {np.mean(laps):.2f} s")


def eval_snake(run: str, checkpoint: str = "last", games: int = 256, seed: int = 5000) -> dict:
    """Best-move games (the highest-scoring move every turn) until each one ends."""
    import numpy as np

    from ailearns.agents.ppo import act, load_policy
    from ailearns.checkpoints import pick_checkpoint
    from ailearns.games.snake import SnakeConfig, SnakeGames
    net, ck = load_policy(pick_checkpoint(run, checkpoint))
    cfg = ck["config"]
    g = SnakeGames(games, SnakeConfig(size=cfg["size"]), seed=seed)
    done, won, score = np.zeros(games, bool), np.zeros(games, bool), np.zeros(games, int)
    while not done.all():
        _, d, w = g.step(act(net, g.observe(cfg["view"])))
        new = d & ~done
        won |= new & w
        score[new] = g.score[new]
        done |= d
    return {"update": ck["update"], "games": games, "wins": int(won.sum()), "mean_score": float(score.mean())}


if __name__ == "__main__":
    main()
