"""Lightweight Elo bookkeeping for self-play training.

Used by the trainers to track a single rolling rating per network. After each
gating evaluation (new vs previous accepted network), the rating is bumped by
``update_rating(...)`` based on the score over N games.

Anchored to the *previous* accepted network, which itself drifts upward as the
training progresses, so the absolute number is only meaningful relative to its
own history — but the trend captures cumulative strength gain.
"""

from __future__ import annotations

INITIAL_ELO: float = 1500.0
DEFAULT_K:   float = 16.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Logistic Elo expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_rating(
    rating:          float,
    opponent_rating: float,
    score:           float,
    n_games:         int,
    k:               float = DEFAULT_K,
) -> float:
    """Return the player's new Elo after *n_games* against *opponent_rating*.

    *score* is the player's mean score (1 = win, 0.5 = draw, 0 = loss) over
    those games.  Standard Elo update scaled by the number of games.
    """
    if n_games <= 0:
        return rating
    e = expected_score(rating, opponent_rating)
    return rating + k * n_games * (score - e)
