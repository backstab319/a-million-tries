"""The games and the AIs that learn them: play, train, evaluate.

    uv run games play                         # Snake, you play
    uv run --group jax games drive            # the race car, you drive
    uv run games train runs/snake-1 --arch cnn-deep --steps 20e6
    uv run --group jax games train-car runs/car-1 --steps 300e6
    uv run games eval runs/snake-1            # best-move games: win rate, mean score
    uv run --group jax games eval-car runs/car-1
"""

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="games", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("play", help="play Snake yourself")
    sp.add_argument("--size", type=int, default=10)
    sp.add_argument("--speed", type=float, default=8.0, help="moves per second")

    sp = sub.add_parser("drive", help="drive the race car yourself (needs --group jax)")
    sp.add_argument("--track", type=int, default=100_000, help="track seed (100000+: tracks the AI never trains on)")

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

    args = p.parse_args()
    if args.cmd == "play":
        from ailearns import play
        play.run(args.size, args.speed)
    elif args.cmd == "drive":
        from ailearns import drive
        drive.run(args.track)
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
