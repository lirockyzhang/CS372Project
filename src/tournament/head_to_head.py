"""Pairwise tournament between a chosen set of agents.

Pass two or more agents via --agents; every unordered pair plays a match.
With exactly two agents this is equivalent to a single head-to-head.

Agents available:

    random                      Uniform random legal move
    ppo                         PPO greedy policy (--ppo-ckpt)
    mcts1k                      Pure MCTS, 1000 sims
    mcts10k                     Pure MCTS, 10000 sims

    AlphaZero-style search (PUCT) on each backbone:
    az-ppo                      PUCT MCTS + PPO weights        (--ppo-ckpt)
    az-selfplay                 PUCT MCTS + AZ-CNN weights     (--az-ckpt)
    az-transformer              PUCT MCTS + Transformer        (--tf-ckpt)

    Gumbel search (root-Gumbel argmax) on the same backbones:
    gumbel-ppo                  Gumbel MCTS + PPO weights      (--ppo-ckpt)
    gumbel-selfplay             Gumbel MCTS + AZ-CNN weights   (--az-ckpt)
    gumbel-transformer          Gumbel MCTS + Transformer      (--tf-ckpt)

All non-random agents play with temperature 0 (greedy / most-visited child).

Usage
-----
    # Head-to-head (two agents)
    python src/tournament/head_to_head.py --agents az-ppo mcts10k

    # Round-robin across several agents
    python src/tournament/head_to_head.py --agents random ppo mcts1k az-ppo \
        --games 50

    # Full zoo
    python src/tournament/head_to_head.py --agents \
        random ppo mcts1k mcts10k az-ppo gumbel-ppo az-selfplay az-transformer

Each pair of games inside a match shares a random opening move; sides are
swapped between the two games of a pair so every opening is played from
both colours.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from agents.alphagumbel.mcts import AlphaGumbelMCTS
from agents.common.network import AlphaZeroNet as AlphaGumbelNet
from agents.alphazero.mcts import AlphaZeroMCTS
from agents.common.network import AlphaZeroNet
from agents.alphazero.transformer import AlphaZeroTransformerNet
from agents.mcts.mcts_agent import MCTSAgent
from agents.ppo.agent import PPOAgent
from env.logic import legal_action_mask, legal_actions, step
from env.observation import observe
from env.state import UTTTState, initial_state


AGENT_NAMES = [
    "random", "ppo",
    "mcts1k", "mcts10k",
    "az-ppo",       "gumbel-ppo",
    "az-selfplay",  "gumbel-selfplay",
    "az-transformer", "gumbel-transformer",
]


# ---------------------------------------------------------------------------
# Unified agent wrappers —all expose .act(state) -> (board, cell)
# ---------------------------------------------------------------------------

class _RandomAgent:
    def act(self, state: UTTTState) -> tuple[int, int]:
        return random.choice(legal_actions(state))


class _PPOAdapter:
    """Wrap PPOAgent so callers can pass a state object like the MCTS agents."""

    def __init__(self, ppo: PPOAgent) -> None:
        self.ppo = ppo

    def act(self, state: UTTTState) -> tuple[int, int]:
        obs = observe(state)
        mask = legal_action_mask(state)
        return self.ppo.act(obs, mask, greedy=True)


# ---------------------------------------------------------------------------
# Network loading helpers
# ---------------------------------------------------------------------------

def _load_state_dict(path: str | Path, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("network_state", "model", "state_dict"):
            if key in ckpt:
                return ckpt[key], ckpt
        return ckpt, ckpt
    return ckpt, {}


def _load_cnn(path: str | Path, device: torch.device, cls=AlphaZeroNet):
    state, raw = _load_state_dict(path, device)
    cfg = raw.get("model_config", {}) if isinstance(raw, dict) else {}
    channels = cfg.get("channels", 64)
    num_blocks = cfg.get("num_blocks", 3)
    net = cls(channels=channels, num_blocks=num_blocks).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def _load_transformer(path: str | Path, device: torch.device) -> AlphaZeroTransformerNet:
    state, raw = _load_state_dict(path, device)
    cfg = raw.get("model_config", {}) if isinstance(raw, dict) else {}
    net = AlphaZeroTransformerNet(
        embed_dim=cfg.get("embed_dim", 128),
        depth=cfg.get("depth", 4),
        num_heads=cfg.get("num_heads", 4),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
    ).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent(
    name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[object, str]:
    """Return (agent, pretty label) for *name*."""
    n = name.lower()

    if n == "random":
        return _RandomAgent(), "Random"

    if n == "ppo":
        agent = PPOAgent(
            checkpoint_path=args.ppo_ckpt,
            device=str(device),
            channels=args.ppo_channels,
            num_blocks=args.ppo_blocks,
        )
        return _PPOAdapter(agent), f"PPO({Path(args.ppo_ckpt).stem})"

    if n == "mcts1k":
        return MCTSAgent(num_simulations=args.mcts_sims_low), f"MCTS({args.mcts_sims_low})"

    if n == "mcts10k":
        return MCTSAgent(num_simulations=args.mcts_sims_high), f"MCTS({args.mcts_sims_high})"

    if n == "az-ppo":
        net = _load_cnn(args.ppo_ckpt, device, cls=AlphaZeroNet)
        agent = AlphaZeroMCTS(
            net, num_simulations=args.az_sims,
            batch_size=args.leaf_batch, device=device,
        )
        return agent, f"AZ-PPO({args.az_sims}s)"

    if n == "gumbel-ppo":
        net = _load_cnn(args.ppo_ckpt, device, cls=AlphaGumbelNet)
        agent = AlphaGumbelMCTS(
            net, num_simulations=args.az_sims, device=device,
        )
        return agent, f"Gumbel-PPO({args.az_sims}s)"

    if n == "az-selfplay":
        if args.az_ckpt is None:
            raise ValueError("--az-ckpt is required for agent 'az-selfplay'")
        net = _load_cnn(args.az_ckpt, device, cls=AlphaZeroNet)
        agent = AlphaZeroMCTS(
            net, num_simulations=args.az_sims,
            batch_size=args.leaf_batch, device=device,
        )
        return agent, f"AZ-SelfPlay({Path(args.az_ckpt).stem})"

    if n == "az-transformer":
        if args.tf_ckpt is None:
            raise ValueError("--tf-ckpt is required for agent 'az-transformer'")
        net = _load_transformer(args.tf_ckpt, device)
        agent = AlphaZeroMCTS(
            net, num_simulations=args.az_sims,
            batch_size=args.leaf_batch, device=device,
        )
        return agent, f"AZ-Transformer({Path(args.tf_ckpt).stem})"

    # ----- Gumbel search wired to AlphaZero-trained backbones --------------

    if n == "gumbel-selfplay":
        if args.az_ckpt is None:
            raise ValueError("--az-ckpt is required for agent 'gumbel-selfplay'")
        net = _load_cnn(args.az_ckpt, device, cls=AlphaZeroNet)
        agent = AlphaGumbelMCTS(net, num_simulations=args.az_sims, device=device)
        return agent, f"Gumbel-SelfPlay({Path(args.az_ckpt).stem})"

    if n == "gumbel-transformer":
        if args.tf_ckpt is None:
            raise ValueError("--tf-ckpt is required for agent 'gumbel-transformer'")
        net = _load_transformer(args.tf_ckpt, device)
        agent = AlphaGumbelMCTS(net, num_simulations=args.az_sims, device=device)
        return agent, f"Gumbel-Transformer({Path(args.tf_ckpt).stem})"

    raise ValueError(f"Unknown agent '{name}'. Choose from: {AGENT_NAMES}")


# ---------------------------------------------------------------------------
# Timed move —records per-move wall time for each agent
# ---------------------------------------------------------------------------

class _MoveTimer:
    __slots__ = ("total_time", "moves")

    def __init__(self) -> None:
        self.total_time = 0.0
        self.moves = 0

    def record(self, dt: float) -> None:
        self.total_time += dt
        self.moves += 1

    @property
    def avg_ms(self) -> float:
        return 1000.0 * self.total_time / self.moves if self.moves else 0.0


def _timed_act(agent, state: UTTTState, timer: _MoveTimer) -> tuple[int, int]:
    t0 = time.perf_counter()
    action = agent.act(state)
    timer.record(time.perf_counter() - t0)
    return action


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def _play_game(
    agent_x, agent_o,
    timer_x: _MoveTimer, timer_o: _MoveTimer,
    opening_move: tuple[int, int] | None,
) -> int:
    """Play one game. Returns winner: -1 draw, 0 X wins, 1 O wins."""
    state = initial_state()
    if opening_move is not None:
        step(state, opening_move)
    while not state.terminated:
        if state.current_player == 0:
            action = _timed_act(agent_x, state, timer_x)
        else:
            action = _timed_act(agent_o, state, timer_o)
        step(state, action)
    return state.winner


# ---------------------------------------------------------------------------
# Match runner
# ---------------------------------------------------------------------------

def run_match(
    agent_a, agent_b,
    label_a: str, label_b: str,
    n_games: int,
) -> dict:
    """Play *n_games* in paired openings. Returns a result dict."""
    timer_a = _MoveTimer()
    timer_b = _MoveTimer()

    wins_a = wins_b = draws = 0
    game_log: list[dict] = []

    col_w = max(len(label_a), len(label_b), 10)
    header = (
        f"  {'Game':>4}  {'Opening':>8}  {'X':<{col_w}}  {'O':<{col_w}}  "
        f"{'Winner':<{col_w}}  {'plies':>5}  "
        f"{label_a + ' ms/mv':>{col_w+8}}  {label_b + ' ms/mv':>{col_w+8}}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    t0 = time.time()
    n_pairs = n_games // 2
    game_idx = 0

    def _log_game(game_no: int, opening, x_label, o_label, winner: int) -> None:
        nonlocal wins_a, wins_b, draws
        if winner == -1:
            draws += 1
            w_label = "Draw"
        else:
            w_is_a = (winner == 0 and x_label == label_a) or (winner == 1 and o_label == label_a)
            if w_is_a:
                wins_a += 1
                w_label = label_a
            else:
                wins_b += 1
                w_label = label_b
        op = f"({opening[0]},{opening[1]})" if opening else "→"
        game_log.append({
            "game": game_no,
            "opening": op,
            "x": x_label,
            "o": o_label,
            "winner": w_label,
            f"{label_a}_wins": wins_a,
            f"{label_b}_wins": wins_b,
            "draws": draws,
            f"{label_a}_ms_per_move": round(timer_a.avg_ms, 2),
            f"{label_b}_ms_per_move": round(timer_b.avg_ms, 2),
        })
        print(
            f"  {game_no:>4}  {op:>8}  {x_label:<{col_w}}  {o_label:<{col_w}}  "
            f"{w_label:<{col_w}}  {'-':>5}  "
            f"{timer_a.avg_ms:>{col_w+5}.1f} ms"
            f"  {timer_b.avg_ms:>{col_w+5}.1f} ms"
        )
        # running W/D/L line
        done = wins_a + wins_b + draws
        print(
            f"        running: {label_a} {wins_a}-{draws}-{wins_b} {label_b}"
            f"   ({done}/{n_games})"
        )

    for _ in range(n_pairs):
        opening = random.choice(legal_actions(initial_state()))

        # Game A: agent_a plays X
        game_idx += 1
        winner = _play_game(
            agent_a, agent_b,
            timer_a, timer_b,
            opening_move=opening,
        )
        _log_game(game_idx, opening, label_a, label_b, winner)

        # Game B: agent_a plays O (same opening, swap colours)
        game_idx += 1
        winner = _play_game(
            agent_b, agent_a,
            timer_b, timer_a,
            opening_move=opening,
        )
        _log_game(game_idx, opening, label_b, label_a, winner)

    # Odd extra game with fresh opening (agent_a as X)
    if n_games % 2 == 1:
        opening = random.choice(legal_actions(initial_state()))
        game_idx += 1
        winner = _play_game(
            agent_a, agent_b,
            timer_a, timer_b,
            opening_move=opening,
        )
        _log_game(game_idx, opening, label_a, label_b, winner)

    elapsed = time.time() - t0

    return {
        "label_a": label_a,
        "label_b": label_b,
        "games": n_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "timer_a": timer_a,
        "timer_b": timer_b,
        "elapsed": elapsed,
        "game_log": game_log,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(r: dict) -> None:
    la, lb = r["label_a"], r["label_b"]
    n = r["games"]
    wa, wb, d = r["wins_a"], r["wins_b"], r["draws"]
    score_a = wa + 0.5 * d
    score_b = wb + 0.5 * d
    ta: _MoveTimer = r["timer_a"]
    tb: _MoveTimer = r["timer_b"]

    width = max(len(la), len(lb), 14) + 2
    sep = "=" * (width * 2 + 50)
    print()
    print(sep)
    print("  MATCH SUMMARY")
    print(sep)
    print(f"  {'Agent':<{width}}  {'W':>4}  {'D':>4}  {'L':>4}  {'Pts':>6}  {'Win%':>6}  {'Moves':>6}  {'Avg ms/mv':>10}")
    print(f"  {'-'*width}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}")
    print(f"  {la:<{width}}  {wa:>4}  {d:>4}  {wb:>4}  {score_a:>6.1f}  {100*score_a/n:>5.1f}%  {ta.moves:>6}  {ta.avg_ms:>10.2f}")
    print(f"  {lb:<{width}}  {wb:>4}  {d:>4}  {wa:>4}  {score_b:>6.1f}  {100*score_b/n:>5.1f}%  {tb.moves:>6}  {tb.avg_ms:>10.2f}")
    print(f"  Games: {n}   Draws: {d}   Wall time: {r['elapsed']/60:.2f} min ({n/r['elapsed']:.2f} g/s)")
    print(sep)


def _write_csv(r: dict, path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not r["game_log"]:
        return
    fieldnames = list(r["game_log"][0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(r["game_log"])
    print(f"  CSV log written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _print_standings(results: list[dict], agent_names: list[str]) -> None:
    """Aggregate per-agent points, W/D/L, moves, and avg ms/move across matches."""
    from collections import defaultdict

    pts = defaultdict(float)
    wins = defaultdict(int)
    losses = defaultdict(int)
    draws_ = defaultdict(int)
    matches_played = defaultdict(int)
    total_time = defaultdict(float)
    total_moves = defaultdict(int)
    labels: dict[str, str] = {}

    for r in results:
        la, lb = r["label_a"], r["label_b"]
        na, nb = r["name_a"], r["name_b"]
        labels[na] = la
        labels[nb] = lb
        wa, wb, d = r["wins_a"], r["wins_b"], r["draws"]
        pts[na] += wa + 0.5 * d
        pts[nb] += wb + 0.5 * d
        wins[na] += wa; losses[na] += wb; draws_[na] += d
        wins[nb] += wb; losses[nb] += wa; draws_[nb] += d
        matches_played[na] += 1; matches_played[nb] += 1
        total_time[na] += r["timer_a"].total_time
        total_time[nb] += r["timer_b"].total_time
        total_moves[na] += r["timer_a"].moves
        total_moves[nb] += r["timer_b"].moves

    ranked = sorted(agent_names, key=lambda n: -pts[n])
    width = max((len(labels[n]) for n in agent_names), default=10) + 2

    sep = "=" * (width + 70)
    print()
    print(sep)
    print("  STANDINGS")
    print(sep)
    print(
        f"  {'#':>2}  {'Agent':<{width}}"
        f"  {'Pts':>6}  {'W':>4}  {'D':>4}  {'L':>4}"
        f"  {'Matches':>7}  {'Moves':>7}  {'Avg ms/mv':>10}"
    )
    print(
        f"  {'--':>2}  {'-'*width}"
        f"  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*4}"
        f"  {'-'*7}  {'-'*7}  {'-'*10}"
    )
    for rank, n in enumerate(ranked, 1):
        avg_ms = 1000 * total_time[n] / total_moves[n] if total_moves[n] else 0.0
        print(
            f"  {rank:>2}  {labels[n]:<{width}}"
            f"  {pts[n]:>6.1f}"
            f"  {wins[n]:>4}  {draws_[n]:>4}  {losses[n]:>4}"
            f"  {matches_played[n]:>7}"
            f"  {total_moves[n]:>7}  {avg_ms:>10.2f}"
        )
    print(sep)


def _print_pairwise_matrix(results: list[dict], agent_names: list[str]) -> None:
    """Print a row-vs-column points matrix (row player's points)."""
    labels = {}
    score: dict[tuple[str, str], float] = {}
    games_map: dict[tuple[str, str], int] = {}
    for r in results:
        na, nb = r["name_a"], r["name_b"]
        labels[na] = r["label_a"]
        labels[nb] = r["label_b"]
        wa, wb, d = r["wins_a"], r["wins_b"], r["draws"]
        score[(na, nb)] = wa + 0.5 * d
        score[(nb, na)] = wb + 0.5 * d
        games_map[(na, nb)] = r["games"]
        games_map[(nb, na)] = r["games"]

    width = max((len(labels.get(n, n)) for n in agent_names), default=8) + 1
    col_w = 8

    sep = "=" * (width + col_w * len(agent_names) + 4)
    print()
    print(sep)
    print("  PAIRWISE POINTS (row vs column, row's score)")
    print(sep)
    header = " " * (width + 2) + "".join(
        f"{labels.get(n, n)[:col_w-1]:>{col_w}}" for n in agent_names
    )
    print(header)
    for row in agent_names:
        row_label = labels.get(row, row)
        cells = []
        for col in agent_names:
            if row == col:
                cells.append(f"{'--':>{col_w}}")
            elif (row, col) in score:
                g = games_map[(row, col)]
                cells.append(f"{score[(row, col)]:>{col_w-2}.1f}/{g}")
            else:
                cells.append(f"{'':>{col_w}}")
        print(f"  {row_label:<{width}}" + "".join(cells))
    print(sep)


def _write_tournament_csv(results: list[dict], path: Path) -> None:
    """Write one row per match: agent pair, score, W/D/L, avg ms/move."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "agent_a": r["label_a"],
            "agent_b": r["label_b"],
            "games": r["games"],
            "wins_a": r["wins_a"],
            "wins_b": r["wins_b"],
            "draws": r["draws"],
            "pts_a": r["wins_a"] + 0.5 * r["draws"],
            "pts_b": r["wins_b"] + 0.5 * r["draws"],
            "a_ms_per_move": round(r["timer_a"].avg_ms, 2),
            "b_ms_per_move": round(r["timer_b"].avg_ms, 2),
            "elapsed_s": round(r["elapsed"], 2),
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Tournament summary CSV -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise tournament between a chosen set of agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agents", type=str, nargs="+", required=True,
                        choices=AGENT_NAMES, metavar="AGENT",
                        help="Two or more agents. Every unordered pair plays a match.")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per match (paired openings)")

    parser.add_argument("--ppo-ckpt", type=str,
                        default="models/ppo_legacy/ppo.pt",
                        help="Checkpoint for PPO / AZ-PPO / Gumbel-PPO")
    parser.add_argument("--ppo-channels", type=int, default=64,
                        help="Channels for the PPO network (must match checkpoint)")
    parser.add_argument("--ppo-blocks", type=int, default=3,
                        help="Residual blocks for the PPO network (must match checkpoint)")
    parser.add_argument("--az-ckpt", type=str,
                        default="models/cnn/iter_050_accepted.pt",
                        help="Checkpoint for AZ-SelfPlay")
    parser.add_argument("--tf-ckpt", type=str,
                        default="models/transformer/iter_020_rejected.pt",
                        help="Checkpoint for AZ-Transformer")

    parser.add_argument("--az-sims", type=int, default=1000,
                        help="MCTS sims for all AlphaZero / Gumbel agents")
    parser.add_argument("--mcts-sims-low", type=int, default=1000,
                        help="Sims for 'mcts1k'")
    parser.add_argument("--mcts-sims-high", type=int, default=10000,
                        help="Sims for 'mcts10k'")
    parser.add_argument("--leaf-batch", type=int, default=8,
                        help="Leaf batch size for AlphaZero MCTS")

    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (auto if omitted)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for CSV output. "
                             "Default: tournament/logs/tourney_<timestamp>/")
    parser.add_argument("--no-log", action="store_true",
                        help="Disable CSV logging entirely")

    args = parser.parse_args()

    if len(args.agents) < 2:
        parser.error("--agents requires at least two names")
    if len(set(args.agents)) != len(args.agents):
        parser.error("--agents must not contain duplicates")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else _detect_device()
    print(f"Device: {device}")

    # ---- Build every agent once; reuse across matches. ----
    print(f"Building {len(args.agents)} agents ...")
    agents: list[tuple[str, object, str]] = []  # (name, agent, label)
    for name in args.agents:
        agent, label = _build_agent(name, args, device)
        agents.append((name, agent, label))
        print(f"  {name:<16} -> {label}")

    from itertools import combinations
    pairs = list(combinations(range(len(agents)), 2))

    # ---- Output directory (default: tournament/logs/tourney_<ts>/) ----
    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.no_log:
        log_dir = None
    elif args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(__file__).resolve().parent / "logs" / f"tourney_{ts}"

    sep = "=" * 70
    print()
    print(sep)
    print(f"  Pairwise tournament: {len(agents)} agents, "
          f"{len(pairs)} matches × {args.games} games")
    print(f"  Temperature 0   paired openings   seed={args.seed}")
    if log_dir is not None:
        print(f"  Log dir: {log_dir}")
    print(sep)

    results: list[dict] = []
    t_start = time.time()
    for m_idx, (i, j) in enumerate(pairs, 1):
        name_a, agent_a, label_a = agents[i]
        name_b, agent_b, label_b = agents[j]

        print()
        print("-" * 70)
        print(f"  Match {m_idx}/{len(pairs)}: {label_a}  vs  {label_b}")
        print("-" * 70)

        result = run_match(agent_a, agent_b, label_a, label_b, args.games)
        result["name_a"] = name_a
        result["name_b"] = name_b
        _print_summary(result)

        if log_dir is not None:
            per_match_csv = log_dir / f"{name_a}_vs_{name_b}.csv"
            _write_csv(result, per_match_csv)

        results.append(result)

    total_elapsed = time.time() - t_start

    # ---- Final reports ----
    _print_standings(results, args.agents)
    if len(agents) > 2:
        _print_pairwise_matrix(results, args.agents)

    print()
    print(f"  Tournament complete in {total_elapsed/60:.2f} min "
          f"({sum(r['games'] for r in results)} games total)")

    if log_dir is not None:
        _write_tournament_csv(results, log_dir / "summary.csv")


if __name__ == "__main__":
    main()
