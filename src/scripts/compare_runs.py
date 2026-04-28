"""Cross-run comparison plots for PPO / AlphaZero training studies.

Reads ``train_log.csv`` from each ``--run`` directory, derives a
**comparable Elo** anchored to the shared MCTS@N opponent, and produces:

  1. ``elo_vs_nn_forwards.png`` -- Elo vs cumulative NN forward passes
     (log-x). Pure algorithmic data efficiency.
  2. ``elo_vs_wall_clock.png`` -- Elo vs cumulative training hours
     (linear-x). Practical engineering efficiency.
  3. ``convergence_summary.csv`` -- per-run summary including the first
     NN-forward count and wall-clock hour at which Elo crosses the
     threshold, plus max Elo seen.

Why a *derived* Elo: each trainer also writes an ``elo`` column, but it is
self-referential (anchored to that run's previous accepted snapshot of
itself), so AZ-PUCT-Elo=1450 is not the same skill as PPO-Elo=1450. We
instead convert ``win_rate_mcts`` to Elo against the shared MCTS@N anchor
via the standard log-odds formula, which makes runs directly comparable.

Usage
-----
    uv run python src/scripts/compare_runs.py \
        --run runs/az_cnn_puct \
        --run runs/az_cnn_gumbel \
        --run runs/ppo_warm_matched \
        --convergence-threshold 1600 \
        --out-dir runs/comparison/
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_ANCHOR_ELO = 1500.0
ELO_CLIP = (0.01, 0.99)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-run Elo-vs-(NN forwards | wall clock) plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", action="append", default=[],
                   help="Path to a run directory containing train_log.csv. "
                        "Repeat for multiple runs.")
    p.add_argument("--label", action="append", default=[],
                   help='Optional "name=path" override; otherwise label = dir basename. '
                        "Repeat for multiple runs.")
    p.add_argument("--metric", type=str, default="elo_vs_mcts",
                   choices=["elo_vs_mcts", "win_rate_mcts", "elo"],
                   help="Y-axis metric. 'elo_vs_mcts' (default) is anchored "
                        "to the shared MCTS opponent and is comparable across runs. "
                        "'elo' is the trainer's self-referential rating.")
    p.add_argument("--mcts-anchor-elo", type=float, default=DEFAULT_ANCHOR_ELO,
                   help="Elo assigned to MCTS@N when deriving elo_vs_mcts.")
    p.add_argument("--convergence-threshold", type=float, default=DEFAULT_ANCHOR_ELO + 100,
                   help="Y-value (in --metric units) marking convergence. "
                        "Default = anchor + 100 (one Elo class above MCTS).")
    p.add_argument("--eval-csv", type=str, default=None,
                   help="Optional path to an eval_vs_mcts.py CSV. When given, "
                        "skip --run and plot one line per unique 'run' value in "
                        "the CSV. Useful for post-hoc MCTS@N re-evaluation against "
                        "a different opponent than the one used during training.")
    p.add_argument("--out-dir", type=str, default="runs/comparison",
                   help="Directory for the two PNGs + convergence_summary.csv.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-run loading + derivation
# ---------------------------------------------------------------------------

def _label_map(args: argparse.Namespace) -> dict[str, str]:
    """Build {path: label} from --run + optional --label overrides."""
    mapping: dict[str, str] = {p: Path(p).name for p in args.run}
    for spec in args.label:
        if "=" not in spec:
            raise ValueError(f"--label must be 'name=path', got {spec!r}")
        name, path = spec.split("=", 1)
        mapping[path] = name
    return mapping


def _winrate_to_elo(p: float, anchor_elo: float) -> float:
    """Standard log-odds Elo conversion vs a fixed-rating opponent."""
    p = max(ELO_CLIP[0], min(ELO_CLIP[1], float(p)))
    return anchor_elo + 400.0 * math.log10(p / (1.0 - p))


def load_eval_csv(path: str | Path, anchor_elo: float) -> dict[str, pd.DataFrame]:
    """Load an `eval_vs_mcts.py` CSV and split it by `run` column.

    Each value of the ``run`` column becomes a separate DataFrame keyed by
    that name. The ``warm_start`` rows (one per kind) are folded into each
    real run as the ``forwards=0, time=0`` baseline so the plot includes
    a shared starting point.
    """
    df = pd.read_csv(path)
    required = {"run", "kind", "cumulative_nn_forwards",
                "cumulative_time_s", "win_rate_mcts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: eval CSV missing columns: {missing}")

    df = df.copy()
    df["hours"] = df["cumulative_time_s"] / 3600.0
    df["elo_vs_mcts"] = df["win_rate_mcts"].apply(
        lambda p: _winrate_to_elo(p, anchor_elo) if pd.notna(p) else float("nan")
    )

    # Pull the warm-start rows (one per kind) out as a baseline that gets
    # prepended to every per-kind run.
    warm = df[df["run"] == "warm_start"]
    runs = df[df["run"] != "warm_start"]

    out: dict[str, pd.DataFrame] = {}
    for run_name, run_df in runs.groupby("run", sort=False):
        kind = run_df["kind"].iloc[0]
        warm_for_kind = warm[warm["kind"] == kind]
        merged = pd.concat([warm_for_kind, run_df], ignore_index=True)
        merged = merged.sort_values("cumulative_nn_forwards").reset_index(drop=True)
        out[str(run_name)] = merged
    return out


def load_run(path: str | Path, anchor_elo: float) -> pd.DataFrame:
    """Read train_log.csv from a run dir; return enriched DataFrame.

    Adds:
      - elo_vs_mcts: derived from win_rate_mcts (NaN where win_rate_mcts is NaN)
      - hours: cumulative_time_s / 3600
    Filters require cumulative_nn_forwards + cumulative_time_s columns to
    exist (raises if older logs without instrumentation are passed in).
    """
    csv_path = Path(path) / "train_log.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No train_log.csv in {path}")
    df = pd.read_csv(csv_path)
    for required in ("cumulative_nn_forwards", "cumulative_time_s"):
        if required not in df.columns:
            raise ValueError(
                f"{csv_path}: missing required column '{required}'. "
                "This run was logged before the comparison-study instrumentation; "
                "re-run with the patched trainers to regenerate."
            )
    # ``win_rate_mcts`` may be absent if the run never reached an eval
    # iteration (e.g. only selfplay_only iters before the schema was locked).
    # Treat it as all-NaN so downstream plotting reports "no eval points."
    if "win_rate_mcts" not in df.columns:
        print(f"  warning: {csv_path} has no win_rate_mcts column "
              "(no eval iterations completed); plot will show no points.")
        df["win_rate_mcts"] = float("nan")
    df["hours"] = df["cumulative_time_s"] / 3600.0
    df["elo_vs_mcts"] = df["win_rate_mcts"].apply(
        lambda p: _winrate_to_elo(p, anchor_elo) if pd.notna(p) else float("nan")
    )
    return df


# ---------------------------------------------------------------------------
# Plotting + summary
# ---------------------------------------------------------------------------

def _plot(
    runs:    dict[str, pd.DataFrame],
    metric:  str,
    x_col:   str,
    x_label: str,
    title:   str,
    out_path: Path,
    log_x:   bool,
    anchor_elo: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df in runs.items():
        sub = df.dropna(subset=[metric, x_col])
        if sub.empty:
            print(f"  {label}: no rows with both {metric} and {x_col}")
            continue
        final = sub[metric].iloc[-1]
        ax.plot(
            sub[x_col], sub[metric],
            marker="o", markersize=4, linewidth=1.5,
            label=f"{label} (final {metric}={final:.0f})"
              if metric != "win_rate_mcts"
              else f"{label} (final {metric}={final:.2f})",
        )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if metric.startswith("elo"):
        ax.axhline(anchor_elo, color="grey", linewidth=0.8, linestyle="--",
                   label=f"MCTS anchor ({anchor_elo:.0f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}")


def convergence_table(
    runs:      dict[str, pd.DataFrame],
    metric:    str,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, df in runs.items():
        sub = df.dropna(subset=[metric])
        if sub.empty:
            rows.append({
                "run": label, "first_forwards_to_thr": float("inf"),
                "first_hours_to_thr": float("inf"),
                f"max_{metric}": float("nan"),
                "forwards_at_max": float("nan"),
                "hours_at_max": float("nan"),
            })
            continue
        crossed = sub[sub[metric] >= threshold]
        first_fwd = float(crossed["cumulative_nn_forwards"].iloc[0]) if not crossed.empty else float("inf")
        first_hr  = float(crossed["hours"].iloc[0])                  if not crossed.empty else float("inf")
        max_idx   = sub[metric].idxmax()
        rows.append({
            "run":                   label,
            "first_forwards_to_thr": first_fwd,
            "first_hours_to_thr":    first_hr,
            f"max_{metric}":         float(sub.loc[max_idx, metric]),
            "forwards_at_max":       float(sub.loc[max_idx, "cumulative_nn_forwards"]),
            "hours_at_max":          float(sub.loc[max_idx, "hours"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if not args.run and not args.eval_csv:
        print("error: --run or --eval-csv required", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_csv:
        runs = load_eval_csv(args.eval_csv, args.mcts_anchor_elo)
        print(f"Loaded {sum(len(d) for d in runs.values())} eval rows "
              f"across {len(runs)} run(s) from {args.eval_csv}")
    else:
        label_map = _label_map(args)
        runs = {}
        for path in args.run:
            df = load_run(path, args.mcts_anchor_elo)
            runs[label_map[path]] = df

    metric = args.metric
    print(f"Plotting {metric} (anchor={args.mcts_anchor_elo}) for {len(runs)} run(s)")

    _plot(
        runs, metric=metric,
        x_col="cumulative_nn_forwards",
        x_label="Cumulative NN forward passes (rollout / self-play)",
        title="Algorithmic data efficiency",
        out_path=out_dir / "elo_vs_nn_forwards.png",
        log_x=True,
        anchor_elo=args.mcts_anchor_elo,
    )
    _plot(
        runs, metric=metric,
        x_col="hours",
        x_label="Cumulative wall-clock hours",
        title="Practical engineering efficiency",
        out_path=out_dir / "elo_vs_wall_clock.png",
        log_x=False,
        anchor_elo=args.mcts_anchor_elo,
    )

    summary = convergence_table(runs, metric=metric, threshold=args.convergence_threshold)
    summary_path = out_dir / "convergence_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nConvergence (threshold {metric} >= {args.convergence_threshold}):")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(summary.to_string(index=False))
    print(f"\nSummary CSV: {summary_path}")


if __name__ == "__main__":
    main()
