"""GPU-resident MCTS for Ultimate Tic-Tac-Toe.

Wraps the vendored ``MCTSNC`` engine (Numba CUDA, multi-tree + multi-playout
parallelism) with UTTT-specific game mechanics so all four MCTS phases —
selection, expansion, playout, backup — run as CUDA kernels.

The package keeps its public symbols **lazy** so importing
``agents.mcts_cuda.UTTTState`` (or any other public name) does NOT eagerly
pull in numba / the CUDA driver. Useful on machines without a GPU where
only the host-side state wrapper is needed (e.g. for tests or for the
generate_mcts_data CPU backend).

Public API:

    from agents.mcts_cuda import UTTTState              # always works
    from agents.mcts_cuda import play_games_cuda        # imports numba on first call
    from agents.mcts_cuda import build_engine, play_one_game

Direct submodule imports also work:

    from agents.mcts_cuda.uttt_state   import UTTTState
    from agents.mcts_cuda.self_play    import play_games_cuda
    from agents.mcts_cuda.mctsnc       import MCTSNC

See ATTRIBUTION.md for the CC-BY 4.0 attribution to Klęsk's upstream library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .self_play import build_engine, play_games_cuda, play_one_game
    from .uttt_state import UTTTState

__all__ = ["UTTTState", "build_engine", "play_games_cuda", "play_one_game"]


def __getattr__(name: str) -> Any:
    """Lazily import public names so ``import agents.mcts_cuda.UTTTState`` does
    not eagerly load numba / the CUDA driver."""
    if name == "UTTTState":
        from .uttt_state import UTTTState
        return UTTTState
    if name in ("build_engine", "play_games_cuda", "play_one_game"):
        from . import self_play
        return getattr(self_play, name)
    raise AttributeError(f"module 'agents.mcts_cuda' has no attribute {name!r}")
