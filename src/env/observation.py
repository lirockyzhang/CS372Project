"""Observation encoding: UTTTState -> neural network tensor."""

from __future__ import annotations

import numpy as np

from .state import UTTTState


def observe(state: UTTTState) -> np.ndarray:
    """Encode state as a (6, 9, 9) float32 array from the current player's POV.

    Planes:
        0 → current player's marks
        1 → opponent's marks
        2 → legal action mask
        3 → current player's won sub-boards (expanded 3x3 -> 9x9)
        4 → opponent's won sub-boards (expanded)
        5 → drawn sub-boards (expanded)

    Indexing: obs[:, board_idx, cell_idx] → same as action space.
    """
    obs = np.zeros((6, 9, 9), dtype=np.float32)

    if state.current_player == 0:
        my_boards, opp_boards = state.boards_x, state.boards_o
        my_meta, opp_meta = state.meta_x, state.meta_o
    else:
        my_boards, opp_boards = state.boards_o, state.boards_x
        my_meta, opp_meta = state.meta_o, state.meta_x

    # Planes 0-1: per-cell marks.
    for b in range(9):
        my_bits = int(my_boards[b])
        opp_bits = int(opp_boards[b])
        for c in range(9):
            if my_bits & (1 << c):
                obs[0, b, c] = 1.0
            if opp_bits & (1 << c):
                obs[1, b, c] = 1.0

    # Plane 2: legal action mask.
    from .logic import legal_action_mask

    obs[2] = legal_action_mask(state).astype(np.float32)

    # Planes 3-5: meta-board expanded (each won/drawn sub-board fills its row).
    for b in range(9):
        if my_meta & (1 << b):
            obs[3, b, :] = 1.0
        if opp_meta & (1 << b):
            obs[4, b, :] = 1.0
        if state.meta_draw & (1 << b):
            obs[5, b, :] = 1.0

    return obs
