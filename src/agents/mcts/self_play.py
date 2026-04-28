"""Single-game self-play using MCTSAgent for AlphaZero data generation.

Each call to play_game() runs one full game and returns one (obs, policy, value)
tuple per move —the three arrays that the AlphaZero CNN trains on:

    obs    (6, 9, 9) float32  —board observation from the mover's POV
    policy (9, 9)   float32  —MCTS visit-count distribution (normalized)
    value  float32           —game outcome from the mover's POV (+1/-1/0)

Temperature schedule (faithful to AlphaZero):
    Moves 0 .. temp_threshold-1  : action sampled —visit counts  (explore)
    Moves temp_threshold+        : greedy (most-visited child)     (exploit)

The policy TARGET is always the full visit distribution regardless of which
action is actually played —temperature only governs action selection.
"""

from __future__ import annotations

import random

import numpy as np

from agents.mcts.mcts_agent import MCTSAgent
from env.logic import step
from env.observation import observe
from env.state import initial_state

# Type alias for a single training sample.
Sample = tuple[np.ndarray, np.ndarray, float]


def play_game(
    num_simulations: int = 200,
    temp_threshold: int = 10,
) -> list[Sample]:
    """Play one self-play game and return all training samples.

    Args:
        num_simulations: MCTS simulations per move.
        temp_threshold:  Number of opening moves to play with temperature=1.

    Returns:
        List of (obs, policy, value) tuples, one per move in the game.
    """
    agent = MCTSAgent(num_simulations)
    state = initial_state()

    # Accumulate (obs, policy, player_to_move) before we know the outcome.
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_count = 0

    while not state.terminated:
        obs = observe(state)
        action, policy = agent.act_with_policy(state)

        # Temperature-based action selection.
        # Moves < temp_threshold: sample from the visit distribution so opening
        # play is diverse and the dataset covers a wide range of positions.
        # Later moves: play the most-visited (strongest) child.
        if move_count < temp_threshold:
            # Cast to float64 and renormalize: float32 rounding can push the
            # sum ~1e-6 away from 1.0, which exceeds numpy's 1e-8 tolerance
            # in random.choice and causes a ValueError.
            flat = policy.flatten().astype(np.float64)
            flat /= flat.sum()
            idx = int(np.random.choice(81, p=flat))
            action = (idx // 9, idx % 9)
        # else: use the greedy action already returned by act_with_policy

        history.append((obs, policy, state.current_player))
        step(state, action)
        move_count += 1

    # Assign outcome values from each position's mover perspective.
    # state.winner: -1 = draw, 0 = X won, 1 = O won
    winner = state.winner
    samples: list[Sample] = []
    for obs, policy, player in history:
        if winner == -1:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        samples.append((obs, policy, value))

    return samples
