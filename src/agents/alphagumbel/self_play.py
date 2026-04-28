"""Single-game self-play for AlphaGumbel data generation.

Stores:
- obs: mover-perspective observation
- policy: improved policy target from completed Q-values
- value: final game outcome from the mover's perspective

Action selection:
- early moves: sample from root visit policy for exploration
- later moves: use the deterministic Gumbel-root choice
"""

from __future__ import annotations

import numpy as np

from agents.alphagumbel.mcts import AlphaGumbelMCTS
from env.logic import step
from env.observation import observe
from env.state import initial_state

Sample = tuple[np.ndarray, np.ndarray, float]


def play_game(
    agent: AlphaGumbelMCTS,
    *,
    temp_threshold: int = 10,
) -> list[Sample]:
    state = initial_state()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_count = 0

    while not state.terminated:
        obs = observe(state)
        action, improved_policy, visit_policy = agent.act_with_policy(
            state,
            add_gumbel_noise=True,
        )

        # Early self-play exploration uses visit counts.
        if move_count < temp_threshold:
            flat = visit_policy.reshape(-1).astype(np.float64)
            total = flat.sum()
            if total > 0:
                flat /= total
                idx = int(np.random.choice(81, p=flat))
                action = (idx // 9, idx % 9)

        history.append((obs, improved_policy, state.current_player))
        step(state, action)
        move_count += 1

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
