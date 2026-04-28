"""Evaluation helpers shared by every trainer.

The trainers report **four** strength metrics each iteration:

    1. ``win_rate_random``     — agent vs uniform-random opponent (sanity floor)
    2. ``win_rate_previous``   — agent vs previously-accepted network (gating)
    3. ``win_rate_mcts``       — agent vs pure MCTS at a fixed sim budget (anchored)
    4. (Elo)                   — derived from the gating result by the trainer

All three "win-rate" helpers use the same underlying ``play_eval_game`` /
``evaluate_two_agents`` machinery so the alternation logic and draw scoring
stay consistent across PPO, AlphaZero, AlphaGumbel, and the Transformer
variant.

An "agent" here is any object with ``act(state) -> (board, cell)``.
"""

from __future__ import annotations

import random
from typing import Protocol

from env.logic import legal_actions, step
from env.state import UTTTState, initial_state


class _ActAgent(Protocol):
    def act(self, state: UTTTState) -> tuple[int, int]: ...


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


class RandomAgent:
    """Uniform-random opponent. Useful as a sanity floor in evaluations."""

    def act(self, state: UTTTState) -> tuple[int, int]:
        return random.choice(legal_actions(state))


def play_eval_game(
    new_agent:    _ActAgent,
    old_agent:    _ActAgent,
    new_is_x:     bool,
    random_moves: int = 1,
) -> int:
    """Play one evaluation game. Returns ``state.winner`` (-1 draw, 0 X, 1 O).

    *random_moves* opening moves are sampled uniformly at random so repeated
    calls with deterministic agents do not collapse onto a single trajectory.
    """
    state      = initial_state()
    move_count = 0

    while not state.terminated:
        if move_count < random_moves:
            action = random.choice(legal_actions(state))
        else:
            is_x  = state.current_player == 0
            agent = new_agent if (is_x == new_is_x) else old_agent
            action = agent.act(state)
        step(state, action)
        move_count += 1

    return state.winner


def evaluate_two_agents(
    new_agent:    _ActAgent,
    old_agent:    _ActAgent,
    n_games:      int,
    random_moves: int = 1,
) -> dict[str, float]:
    """Play ``n_games`` between two agents alternating sides.

    Returns a result dict with raw counts plus ``score`` in [0, 1] suitable for
    feeding into an Elo update (1 = win, 0.5 = draw, 0 = loss).
    """
    new_wins = losses = draws = 0
    for i in range(n_games):
        new_is_x = (i % 2 == 0)
        winner   = play_eval_game(new_agent, old_agent, new_is_x, random_moves)

        if winner == -1:
            draws += 1
        elif winner == (0 if new_is_x else 1):
            new_wins += 1
        else:
            losses += 1

    score = (new_wins + 0.5 * draws) / max(n_games, 1)
    return {
        "games":  n_games,
        "wins":   new_wins,
        "losses": losses,
        "draws":  draws,
        "score":  score,
        "win_rate": new_wins / max(n_games, 1),
    }


# ----------------------------------------------------------------------
# Three named evaluations the trainers expose as metrics
# ----------------------------------------------------------------------


def evaluate_vs_random(
    agent:        _ActAgent,
    n_games:      int,
    random_moves: int = 0,
) -> dict[str, float]:
    """Agent vs uniform-random opponent."""
    return evaluate_two_agents(agent, RandomAgent(), n_games, random_moves=random_moves)


def evaluate_vs_previous(
    new_agent:    _ActAgent,
    old_agent:    _ActAgent,
    n_games:      int,
    random_moves: int = 1,
) -> dict[str, float]:
    """New network vs previously-accepted network (the gating eval)."""
    return evaluate_two_agents(new_agent, old_agent, n_games, random_moves=random_moves)


def evaluate_vs_mcts(
    agent:        _ActAgent,
    mcts_sims:    int,
    n_games:      int,
    random_moves: int = 1,
) -> dict[str, float]:
    """Agent vs pure MCTS at a fixed simulation budget — an anchored benchmark."""
    # Imported lazily so importing this module does not pull in MCTSAgent.
    from agents.mcts.mcts_agent import MCTSAgent
    opponent = MCTSAgent(num_simulations=mcts_sims)
    return evaluate_two_agents(agent, opponent, n_games, random_moves=random_moves)
