"""Single-game self-play for AlphaZero data generation.

Each call to play_game() plays one full game using AlphaZeroMCTS and returns
one (obs, policy, value) triple per move —the training targets for the CNN.

Temperature schedule:
    Moves 0 .. temp_threshold-1 : sample action —visit counts  (explore)
    Moves temp_threshold+        : greedy (most-visited child)    (exploit)

Dirichlet noise is always added to the root prior during self-play so the
agent explores rather than collapsing to a single opening.
"""

from __future__ import annotations

import numpy as np

from agents.alphazero.mcts import AlphaZeroMCTS
from env.logic import step
from env.observation import observe
from env.state import initial_state, UTTTState

Sample = tuple[np.ndarray, np.ndarray, float]


def play_game(
    agent:          AlphaZeroMCTS,
    temp_threshold: int = 10,
) -> list[Sample]:
    """Play one self-play game and return all training samples.

    Args:
        agent:          AlphaZeroMCTS instance (network already loaded).
        temp_threshold: Number of opening moves to play with temperature=1.

    Returns:
        List of (obs, policy, value) tuples, one per move in the game.
    """
    state = initial_state()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_count = 0

    while not state.terminated:
        obs = observe(state)
        action, policy = agent.act_with_policy(state, add_noise=True)

        # Temperature-based action selection.
        if move_count < temp_threshold:
            flat = policy.flatten().astype(np.float64)
            flat /= flat.sum()
            idx    = int(np.random.choice(81, p=flat))
            action = (idx // 9, idx % 9)

        history.append((obs, policy, state.current_player))
        step(state, action)
        move_count += 1

    # Assign outcome values from each position's mover perspective.
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
