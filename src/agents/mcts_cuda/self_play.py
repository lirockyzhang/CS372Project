"""Self-play driver for the CUDA MCTS engine.

Plays full UTTT games end-to-end with one ``MCTSNC`` engine instance per
process. The engine is constructed and ``init_device_side_arrays()``
is called exactly once; only the host driver loop touches the CPU between
moves (one MCTSNC.run call per move, then unpack visit counts → policy
target → step the game).

Output sample shape per yielded game matches ``agents/mcts/self_play.py``
exactly (one ``(obs, policy, value)`` tuple per position) so the chunk-save
/ merge pipeline in ``utils.parallel_selfplay`` consumes both backends
unchanged.
"""

from __future__ import annotations

import random
import warnings
from typing import Iterator

import numpy as np

from env.observation import observe

from .mctsnc import MCTSNC
from .uttt_state import UTTTState


# A position sample as it lives in our parallel_selfplay chunk format:
#   (obs (6,9,9) float32, policy (9,9) float32, value float)
Sample = tuple[np.ndarray, np.ndarray, float]


def build_engine(
    *,
    sims_per_move:  int,
    n_trees:        int   = 8,
    n_playouts:     int   = 128,
    variant:        str   = "acp_prodigal",
    device_memory:  float = 2.0,
    ucb_c:          float = 2.0,
    seed:           int   = 0,
    verbose:        bool  = False,
) -> MCTSNC:
    """Construct a ready-to-run MCTSNC engine for UTTT (CUDA arrays initialised)."""
    ai = MCTSNC(
        UTTTState.get_board_shape(),
        UTTTState.get_extra_info_memory(),
        UTTTState.get_max_actions(),
        search_time_limit  = np.inf,
        search_steps_limit = sims_per_move,
        n_trees            = n_trees,
        n_playouts         = n_playouts,
        variant            = variant,
        device_memory      = device_memory,
        ucb_c              = ucb_c,
        seed               = seed,
        verbose_debug      = False,
        verbose_info       = verbose,
    )
    ai.init_device_side_arrays()
    return ai


def _visits_to_policy(actions_info: dict, max_actions: int) -> np.ndarray:
    """Convert MCTSNC's actions_info dict → flat (max_actions,) probability vector."""
    visits = np.zeros(max_actions, dtype=np.float64)
    for k, entry in actions_info.items():
        if k == "best":
            continue
        n = entry.get("n", 0)
        if n > 0:
            visits[int(k)] = n
    total = visits.sum()
    if total <= 0:
        # Engine returned no visits — fall back to uniform over the legal set so
        # downstream code never sees a NaN policy. Should be vanishingly rare.
        warnings.warn("MCTSNC.run produced zero visits; falling back to uniform policy")
        visits[:] = 1.0
        total = float(max_actions)
    return (visits / total).astype(np.float32)


def _maybe_make_actions_info(ai: MCTSNC) -> dict:
    """Ensure ``ai.actions_info`` is populated after ``ai.run()``.

    The actions_info dict is only built by the engine's verbose path. When
    we're running silently we need to materialise it ourselves; the right
    builder depends on the algorithmic-variant suffix (thrifty / prodigal).
    """
    cached = getattr(ai, "actions_info", None)
    if cached is not None:
        return cached
    if ai.variant.endswith("_thrifty"):
        return ai._make_actions_info_thrifty()
    return ai._make_actions_info_prodigal()


def play_one_game(
    ai:             MCTSNC,
    *,
    temp_threshold: int  = 30,
    rng:            random.Random | None = None,
) -> list[Sample]:
    """Play one self-play game end-to-end. Returns its (obs, policy, value) samples."""
    if rng is None:
        rng = random
    state = UTTTState()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_count = 0
    max_actions = UTTTState.get_max_actions()

    while not state.is_terminal:
        obs = observe(state.env_state)        # (6, 9, 9) float32 — current player's POV
        ai.run(state.get_board(), state.get_extra_info(), state.get_turn())
        actions_info = _maybe_make_actions_info(ai)
        # Reset cached actions_info so the next run() builds fresh.
        ai.actions_info = None
        flat = _visits_to_policy(actions_info, max_actions)
        policy_2d = flat.reshape(9, 9)

        # Action selection: temperature=1 sampling for opening, greedy after.
        if move_count < temp_threshold and flat.sum() > 0:
            action = int(np.random.choice(max_actions, p=flat))
        else:
            action = int(actions_info["best"]["index"])

        history.append((obs, policy_2d, state.current_player_zero_one))
        if not state.take_action_job(action):
            # Should not happen — MCTSNC only ever picks a legal action — but be defensive.
            break
        move_count += 1

    # Stitch outcome → per-position value (mover's POV: +1 win / -1 loss / 0 draw).
    winner = state.env_state.winner   # -1 draw, 0 X won, 1 O won
    samples: list[Sample] = []
    for obs, policy, player_zero_one in history:
        if winner == -1:
            value = 0.0
        elif winner == player_zero_one:
            value = 1.0
        else:
            value = -1.0
        samples.append((obs, policy.astype(np.float32), float(value)))
    return samples


def play_games_cuda(
    *,
    num_games:      int,
    sims_per_move:  int,
    n_trees:        int   = 8,
    n_playouts:     int   = 128,
    variant:        str   = "acp_prodigal",
    device_memory:  float = 2.0,
    ucb_c:          float = 2.0,
    temp_threshold: int   = 30,
    seed:           int   = 42,
    verbose:        bool  = False,
) -> Iterator[list[Sample]]:
    """Yield one finished game's samples at a time. Builds the engine ONCE."""
    ai = build_engine(
        sims_per_move = sims_per_move,
        n_trees       = n_trees,
        n_playouts    = n_playouts,
        variant       = variant,
        device_memory = device_memory,
        ucb_c         = ucb_c,
        seed          = seed,
        verbose       = verbose,
    )
    rng = random.Random(seed)
    for g in range(num_games):
        np.random.seed(seed + g)
        rng.seed(seed + g)
        yield play_one_game(ai, temp_threshold=temp_threshold, rng=rng)
