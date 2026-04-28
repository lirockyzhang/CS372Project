"""ASCII rendering for Ultimate Tic-Tac-Toe."""

from __future__ import annotations

from .state import UTTTState


_SYMBOLS = {0: ".", 1: "X", 2: "O"}


def _cell_char(state: UTTTState, board_idx: int, cell_idx: int) -> str:
    bit = 1 << cell_idx
    if int(state.boards_x[board_idx]) & bit:
        return "X"
    if int(state.boards_o[board_idx]) & bit:
        return "O"
    return "."


def _meta_char(state: UTTTState, board_idx: int) -> str:
    bit = 1 << board_idx
    if state.meta_x & bit:
        return "X"
    if state.meta_o & bit:
        return "O"
    if state.meta_draw & bit:
        return "D"
    return "."


def render(state: UTTTState) -> str:
    """Return a human-readable ASCII string of the full board.

    Layout groups sub-boards visually with separators:

        . . . | . . . | . . .
        . . . | . . . | . . .
        . . . | . . . | . . .
        ------+-------+------
        . . . | . . . | . . .
        ...
    """
    lines: list[str] = []

    for meta_row in range(3):  # rows of sub-boards
        for cell_row in range(3):  # rows within a sub-board
            parts: list[str] = []
            for meta_col in range(3):  # columns of sub-boards
                board_idx = meta_row * 3 + meta_col
                cells: list[str] = []
                for cell_col in range(3):
                    cell_idx = cell_row * 3 + cell_col
                    cells.append(_cell_char(state, board_idx, cell_idx))
                parts.append(" ".join(cells))
            lines.append(" | ".join(parts))
        if meta_row < 2:
            lines.append("------+-------+------")

    # Meta-board summary
    lines.append("")
    lines.append("Meta-board:")
    for r in range(3):
        row_chars = [_meta_char(state, r * 3 + c) for c in range(3)]
        lines.append(" ".join(row_chars))

    # Game info
    lines.append("")
    if state.terminated:
        if state.winner == -1:
            lines.append("Result: Draw")
        else:
            lines.append(f"Result: {'XO'[state.winner]} wins")
    else:
        lines.append(f"To move: {'XO'[state.current_player]}")
        if state.active_board == -1:
            lines.append("Active board: any")
        else:
            lines.append(f"Active board: {state.active_board}")

    return "\n".join(lines)
