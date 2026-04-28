"""Pure-function game logic for Ultimate Tic-Tac-Toe."""

from __future__ import annotations

import numpy as np

from .state import FULL_BOARD, WIN_MASKS, UTTTState


def check_win(bits: int) -> bool:
    """Check whether a 9-bit board has any winning line."""
    for mask in WIN_MASKS:
        if (bits & mask) == mask:
            return True
    return False


def _is_resolved(state: UTTTState, board_idx: int) -> bool:
    """Return True if sub-board *board_idx* is won or drawn."""
    bit = 1 << board_idx
    return bool((state.meta_x | state.meta_o | state.meta_draw) & bit)


def step(state: UTTTState, action: tuple[int, int]) -> UTTTState:
    """Apply *action* = (board_idx, cell_idx) to *state* IN PLACE and return it.

    Raises ValueError on illegal moves.
    """
    board_idx, cell_idx = action

    # --- validation --------------------------------------------------------
    if state.terminated:
        raise ValueError("Game already terminated")
    if not (0 <= board_idx <= 8 and 0 <= cell_idx <= 8):
        raise ValueError(f"Action out of range: {action}")
    if state.active_board != -1 and board_idx != state.active_board:
        raise ValueError(
            f"Must play in sub-board {state.active_board}, got {board_idx}"
        )
    if _is_resolved(state, board_idx):
        raise ValueError(f"Sub-board {board_idx} is already resolved")

    cell_bit = 1 << cell_idx

    if (state.boards_x[board_idx] | state.boards_o[board_idx]) & cell_bit:
        raise ValueError(f"Cell {cell_idx} in sub-board {board_idx} is occupied")

    # --- place mark --------------------------------------------------------
    if state.current_player == 0:
        state.boards_x[board_idx] |= cell_bit
        player_board = int(state.boards_x[board_idx])
    else:
        state.boards_o[board_idx] |= cell_bit
        player_board = int(state.boards_o[board_idx])

    # --- check sub-board win / draw ----------------------------------------
    board_bit = 1 << board_idx
    if check_win(player_board):
        if state.current_player == 0:
            state.meta_x |= board_bit
        else:
            state.meta_o |= board_bit
    elif (int(state.boards_x[board_idx]) | int(state.boards_o[board_idx])) == FULL_BOARD:
        state.meta_draw |= board_bit

    # --- check game-level termination --------------------------------------
    meta_current = state.meta_x if state.current_player == 0 else state.meta_o
    if check_win(meta_current):
        state.terminated = True
        state.winner = state.current_player
    elif (state.meta_x | state.meta_o | state.meta_draw) == FULL_BOARD:
        state.terminated = True
        state.winner = -1  # draw

    # --- determine next active board ---------------------------------------
    if not state.terminated:
        if _is_resolved(state, cell_idx):
            state.active_board = -1  # free choice
        else:
            state.active_board = cell_idx

    # --- switch player -----------------------------------------------------
    state.current_player ^= 1

    return state


def step_clone(state: UTTTState, action: tuple[int, int]) -> UTTTState:
    """Like step() but operates on a clone, leaving the original unchanged."""
    return step(state.clone(), action)


def legal_action_mask(state: UTTTState) -> np.ndarray:
    """Return a (9, 9) bool mask of legal actions [board_idx, cell_idx]."""
    mask = np.zeros((9, 9), dtype=np.bool_)
    if state.terminated:
        return mask

    resolved = state.meta_x | state.meta_o | state.meta_draw

    if state.active_board == -1:
        # Free choice: any empty cell in any unresolved sub-board.
        for b in range(9):
            if resolved & (1 << b):
                continue
            occupied = int(state.boards_x[b]) | int(state.boards_o[b])
            for c in range(9):
                if not (occupied & (1 << c)):
                    mask[b, c] = True
    else:
        b = state.active_board
        occupied = int(state.boards_x[b]) | int(state.boards_o[b])
        for c in range(9):
            if not (occupied & (1 << c)):
                mask[b, c] = True

    return mask


def legal_actions(state: UTTTState) -> list[tuple[int, int]]:
    """Return list of legal (board_idx, cell_idx) tuples."""
    actions: list[tuple[int, int]] = []
    if state.terminated:
        return actions

    resolved = state.meta_x | state.meta_o | state.meta_draw

    boards = range(9) if state.active_board == -1 else [state.active_board]
    for b in boards:
        if resolved & (1 << b):
            continue
        occupied = int(state.boards_x[b]) | int(state.boards_o[b])
        for c in range(9):
            if not (occupied & (1 << c)):
                actions.append((b, c))

    return actions


def is_terminal(state: UTTTState) -> bool:
    """Check whether the game is over."""
    return state.terminated
