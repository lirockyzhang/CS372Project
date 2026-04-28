"""Re-evaluate saved AZ / PPO checkpoints against MCTS@N post-hoc.

Used when the during-training panel-eval opponent (MCTS@1000 by default) is
too strong to discriminate at low forward budgets, so the
``elo_vs_mcts`` plot floors to ~700 for everyone. Pointing this script at a
weaker opponent (MCTS@200 say) gives a curve with actual spread.

The output is a CSV that ``compare_runs.py --eval-csv`` can drop in instead
of ``train_log.csv``'s ``win_rate_mcts`` column. We pull each checkpoint's
``cumulative_nn_forwards`` and ``cumulative_time_s`` from the run's
``train_log.csv`` so the re-eval points sit on the same x-axis as the
training trajectory.

Usage
-----
    uv run python src/scripts/eval_vs_mcts.py \
        --run runs/az_cnn_gumbel --kind az-gumbel --az-sims 64 \
        --run runs/ppo_warm_matched --kind ppo \
        --mcts-sims 200 --games 30 \
        --warm-start models/cnn/cnn_c128b3_100k_reg_best.pt \
        --out runs/comparison_mcts200/eval_vs_mcts200.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from agents.alphagumbel.mcts import AlphaGumbelMCTS
from agents.alphazero.mcts import AlphaZeroMCTS
from agents.common.evaluation import evaluate_vs_mcts
from agents.common.network import AlphaZeroNet, masked_log_probs
from env.logic import legal_action_mask
from env.observation import observe
from env.state import UTTTState
from utils.runtime import detect_device


KIND_CHOICES = ["az-puct", "az-gumbel", "ppo"]


# ---------------------------------------------------------------------------
# Adapters so PPO and AZ networks both look like "act(state)" agents
# ---------------------------------------------------------------------------

class _PPONetAdapter:
    """Greedy-policy PPO agent over an already-loaded network.

    Mirrors PPOAgent.act's logic but takes a network directly instead of
    loading from disk -- lets us reuse a pre-loaded net per checkpoint.
    """

    def __init__(self, net: AlphaZeroNet, device: torch.device) -> None:
        self.net = net.eval()
        self.device = device

    @torch.no_grad()
    def act(self, state: UTTTState) -> tuple[int, int]:
        obs = observe(state)
        mask = legal_action_mask(state)
        obs_t  = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        logits, _ = self.net(obs_t)
        log_probs = masked_log_probs(logits, mask_t)  # flat (1, 81)
        idx = int(log_probs.argmax(dim=-1).item())
        return divmod(idx, 9)


def _load_cnn(path: Path, device: torch.device) -> AlphaZeroNet:
    """Load weights from any of: AZ pretrain ({'network_state'}), AZ trainer
    iter checkpoint ({'network_state', 'optimizer_state', 'model_config'}),
    PPO ({'model'}), or raw state_dict.

    PPO checkpoints don't carry ``model_config``; for those we infer
    ``channels`` from the input-conv weight shape and ``num_blocks`` by
    counting ``res_blocks.<i>.conv1.weight`` keys.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        cfg = ckpt.get("model_config", {})
        if "network_state" in ckpt:
            state = ckpt["network_state"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
        cfg = {}

    channels = cfg.get("channels")
    num_blocks = cfg.get("num_blocks")
    if channels is None and "input_conv.weight" in state:
        channels = int(state["input_conv.weight"].shape[0])
    if num_blocks is None:
        block_ids = {
            int(k.split(".")[1])
            for k in state
            if k.startswith("res_blocks.") and k.endswith(".conv1.weight")
        }
        num_blocks = (max(block_ids) + 1) if block_ids else 3
    if channels is None:
        channels = 64

    net = AlphaZeroNet(channels=channels, num_blocks=num_blocks).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def _build_agent(kind: str, net: AlphaZeroNet, az_sims: int, device: torch.device):
    if kind == "az-puct":
        return AlphaZeroMCTS(net, num_simulations=az_sims, batch_size=64, device=device)
    if kind == "az-gumbel":
        return AlphaGumbelMCTS(net, num_simulations=az_sims, device=device)
    if kind == "ppo":
        return _PPONetAdapter(net, device)
    raise ValueError(f"unknown kind: {kind}")


# ---------------------------------------------------------------------------
# Per-run checkpoint discovery + x-axis lookup
# ---------------------------------------------------------------------------

ITER_RE = re.compile(r"iter_(\d+)_(accepted|rejected|selfplay_only)\.pt$")
PPO_RE  = re.compile(r"ppo(?:_(\d+))?\.pt$")


def _iter_for_az_ckpt(path: Path) -> int | None:
    m = ITER_RE.search(path.name)
    return int(m.group(1)) if m else None


def _update_for_ppo_ckpt(path: Path, total_updates: int) -> int | None:
    m = PPO_RE.search(path.name)
    if not m:
        return None
    if m.group(1) is None:           # final ppo.pt
        return total_updates
    return int(m.group(1))


def discover_az_checkpoints(run_dir: Path, log_df: pd.DataFrame) -> list[dict]:
    """Return [{path, label, cum_forwards, cum_time_s}, ...] for AZ run, in order."""
    rows: list[dict] = []
    for p in sorted(run_dir.glob("iter_*.pt")):
        i = _iter_for_az_ckpt(p)
        if i is None:
            continue
        # train_log iter column is 1-based and matches our save iter number.
        match = log_df[log_df["iter"] == i]
        if match.empty:
            print(f"  warning: no log row for iter {i}, skipping {p.name}")
            continue
        rows.append({
            "checkpoint":             str(p),
            "label":                  p.stem,
            "cumulative_nn_forwards": int(match["cumulative_nn_forwards"].iloc[0]),
            "cumulative_time_s":      float(match["cumulative_time_s"].iloc[0]),
        })
    return rows


def discover_ppo_checkpoints(run_dir: Path, log_df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    total_updates = int(log_df["iter"].max())
    for p in sorted(run_dir.glob("ppo*.pt")):
        u = _update_for_ppo_ckpt(p, total_updates)
        if u is None:
            continue
        match = log_df[log_df["iter"] == u]
        if match.empty:
            print(f"  warning: no log row for update {u}, skipping {p.name}")
            continue
        rows.append({
            "checkpoint":             str(p),
            "label":                  p.stem,
            "cumulative_nn_forwards": int(match["cumulative_nn_forwards"].iloc[0]),
            "cumulative_time_s":      float(match["cumulative_time_s"].iloc[0]),
        })
    # de-dupe: ppo.pt and ppo_<final>.pt may both exist
    seen: dict[int, dict] = {}
    for r in rows:
        seen[r["cumulative_nn_forwards"]] = r
    return list(seen.values())


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-evaluate saved checkpoints against MCTS@N for cross-run plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", action="append", default=[],
                   help="Run dir (must contain train_log.csv and *.pt). Repeat.")
    p.add_argument("--kind", action="append", default=[],
                   choices=KIND_CHOICES,
                   help="Agent kind for the matching --run. Repeat in same order.")
    p.add_argument("--label", action="append", default=[],
                   help="Optional 'name=path' override for legend; otherwise dir basename.")
    p.add_argument("--az-sims", type=int, default=64,
                   help="MCTS sims for the AZ-style agent at eval time.")
    p.add_argument("--mcts-sims", type=int, default=200,
                   help="Opponent MCTS sim count.")
    p.add_argument("--games", type=int, default=30,
                   help="Games per checkpoint.")
    p.add_argument("--warm-start", type=str, default=None,
                   help="Optional pretrain checkpoint to evaluate as the 'iter 0' "
                        "starting point on the same axes.")
    p.add_argument("--out", type=str, required=True,
                   help="Output CSV path.")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.run) != len(args.kind):
        raise SystemExit("--run and --kind must be paired (same count, same order)")

    device = detect_device(args.device)
    print(f"Device: {device}")
    print(f"Opponent: MCTS@{args.mcts_sims}  ({args.games} games per checkpoint)")

    label_map: dict[str, str] = {p: Path(p).name for p in args.run}
    for spec in args.label:
        if "=" not in spec:
            raise SystemExit(f"--label must be 'name=path', got {spec!r}")
        name, path = spec.split("=", 1)
        label_map[path] = name

    out_rows: list[dict] = []

    # Optional iter-0 warm-start point shared across all runs.
    if args.warm_start:
        ws_path = Path(args.warm_start)
        net = _load_cnn(ws_path, device)
        kinds_with_az = [k for k in args.kind if k.startswith("az")]
        kinds_with_ppo = [k for k in args.kind if k == "ppo"]
        for k in set(args.kind):
            agent = _build_agent(k, net, args.az_sims, device)
            print(f"  warm-start eval {ws_path.name} as kind={k} ...")
            t0 = time.time()
            res = evaluate_vs_mcts(agent, mcts_sims=args.mcts_sims, n_games=args.games)
            dt = time.time() - t0
            print(f"    win_rate={res['win_rate']:.2%} ({dt:.1f}s)")
            out_rows.append({
                "run":                    "warm_start",
                "kind":                   k,
                "label":                  ws_path.stem,
                "checkpoint":             str(ws_path),
                "cumulative_nn_forwards": 0,
                "cumulative_time_s":      0.0,
                "win_rate_mcts":          res["win_rate"],
                "n_games":                res.get("games", args.games),
                "mcts_sims":              args.mcts_sims,
                "az_sims":                args.az_sims,
                "eval_time_s":            dt,
            })

    # Per-run, per-checkpoint eval.
    for run_dir, kind in zip(args.run, args.kind):
        run_path = Path(run_dir)
        log_df = pd.read_csv(run_path / "train_log.csv")
        rows = (
            discover_az_checkpoints(run_path, log_df) if kind.startswith("az")
            else discover_ppo_checkpoints(run_path, log_df)
        )
        rows.sort(key=lambda r: r["cumulative_nn_forwards"])
        run_label = label_map[run_dir]
        print(f"\n=== {run_label} (kind={kind}, {len(rows)} checkpoints) ===")
        for row in rows:
            net = _load_cnn(Path(row["checkpoint"]), device)
            agent = _build_agent(kind, net, args.az_sims, device)
            t0 = time.time()
            res = evaluate_vs_mcts(agent, mcts_sims=args.mcts_sims, n_games=args.games)
            dt = time.time() - t0
            print(
                f"  {row['label']:<28}  "
                f"forwards={row['cumulative_nn_forwards']:>10,}  "
                f"win_rate={res['win_rate']:.2%}  "
                f"({dt:.1f}s)"
            )
            out_rows.append({
                "run":                    run_label,
                "kind":                   kind,
                "label":                  row["label"],
                "checkpoint":             row["checkpoint"],
                "cumulative_nn_forwards": row["cumulative_nn_forwards"],
                "cumulative_time_s":      row["cumulative_time_s"],
                "win_rate_mcts":          res["win_rate"],
                "n_games":                res.get("games", args.games),
                "mcts_sims":              args.mcts_sims,
                "az_sims":                args.az_sims,
                "eval_time_s":            dt,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else ["run"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
