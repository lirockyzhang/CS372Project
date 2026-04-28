"""AlphaGumbel training loop.

Aligned with AlphaZeroTrainer:
    0. Refresh    — re-inject a random slice of the MCTS bootstrap dataset.
    1. Self-play  — generate games with the current network, add to buffer.
    2. Train      — gradient steps on random minibatches from the buffer.
    3. Evaluate   — new network vs old (gating), vs uniform random,
                    and vs pure MCTS at a fixed sim budget (anchored).
    4. Replace    — accept new network if gating win rate >= --win-threshold.

Per-iteration metrics (training losses, the four strength-evaluation win
rates, and rolling Elo) are emitted to a ``utils.training_logger.TrainingLogger``
that writes CSV + TensorBoard + console summaries in lockstep with the other
trainers.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.alphagumbel.mcts import AlphaGumbelMCTS
from agents.alphagumbel.self_play import play_game
from agents.common.evaluation import (
    evaluate_vs_mcts,
    evaluate_vs_previous,
    evaluate_vs_random,
)
from agents.common.network import AlphaZeroNet
from agents.common.replay_buffer import ReplayBuffer
from utils.elo import DEFAULT_K, INITIAL_ELO, update_rating
from utils.training_logger import TrainingLogger


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    network:    AlphaZeroNet,
    optimizer:  optim.Optimizer,
    buffer:     ReplayBuffer,
    batch_size: int,
    device:     torch.device,
) -> tuple[float, float]:
    """One gradient step. Returns (policy_loss, value_loss)."""
    obs_np, policy_np, value_np = buffer.sample(batch_size)

    obs_t    = torch.from_numpy(obs_np).to(device)
    policy_t = torch.from_numpy(policy_np).to(device)            # (B, 9, 9)
    value_t  = torch.from_numpy(value_np).to(device)             # (B,)

    logits, value_pred = network(obs_t)

    log_probs   = torch.log_softmax(logits.reshape(-1, 81), dim=-1)
    policy_flat = policy_t.reshape(-1, 81)
    policy_loss = -(policy_flat * log_probs).sum(dim=-1).mean()
    value_loss  = nn.functional.mse_loss(value_pred, value_t)

    loss = 0.5 * policy_loss + 2.0 * value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class AlphaGumbelTrainer:
    """Runs the AlphaGumbel self-play / train / evaluate loop."""

    def __init__(
        self,
        run_dir:                Path,
        games_per_iter:         int   = 1_000,
        train_steps:            int   = 500,
        sims:                   int   = 64,
        eval_sims:              int   = 64,
        eval_games:             int   = 40,
        win_threshold:          float = 0.50,
        batch_size:             int   = 256,
        max_root_actions:       int   = 16,
        buffer_size:            int   = 500_000,
        min_buffer:             int   = 10_000,
        temp_threshold:         int   = 10,
        lr_init:                float = 1e-3,
        lr_min:                 float = 1e-4,
        total_iters:            int   = 75,
        channels:               int   = 128,
        num_blocks:             int   = 6,
        device:                 torch.device | str = "cpu",
        mcts_dataset_path:      str | None = None,
        mcts_refresh_positions: int = 5_000,
        # Strength-evaluation knobs
        eval_random_games:      int   = 20,
        eval_mcts_games:        int   = 20,
        eval_mcts_sims:         int   = 1_000,
        elo_k:                  float = DEFAULT_K,
    ) -> None:
        self.run_dir                = Path(run_dir)
        self.games_per_iter         = games_per_iter
        self.train_steps            = train_steps
        self.sims                   = sims
        self.eval_sims              = eval_sims
        self.eval_games             = eval_games
        self.win_threshold          = win_threshold
        self.batch_size             = batch_size
        self.max_root_actions       = max_root_actions
        self.min_buffer             = min_buffer
        self.temp_threshold         = temp_threshold
        self.lr_init                = lr_init
        self.lr_min                 = lr_min
        self.total_iters            = total_iters
        self.channels               = channels
        self.num_blocks             = num_blocks
        self.device                 = torch.device(device)
        self.mcts_refresh_positions = mcts_refresh_positions

        self.eval_random_games      = eval_random_games
        self.eval_mcts_games        = eval_mcts_games
        self.eval_mcts_sims         = eval_mcts_sims
        self.elo_k                  = elo_k

        self._mcts_obs:    np.ndarray | None = None
        self._mcts_policy: np.ndarray | None = None
        self._mcts_value:  np.ndarray | None = None
        if mcts_dataset_path is not None:
            _d = np.load(mcts_dataset_path)
            self._mcts_obs    = _d["obs"].astype(np.float32)
            self._mcts_policy = _d["policy"].astype(np.float32)
            self._mcts_value  = _d["value"].astype(np.float32)
            print(f"MCTS anchor dataset loaded: {len(self._mcts_value):,} positions from {mcts_dataset_path}")

        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.network   = AlphaZeroNet(channels=channels, num_blocks=num_blocks).to(self.device)
        self.buffer    = ReplayBuffer(max_size=buffer_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr_init)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_iters, eta_min=lr_min
        )

        self.elo          = INITIAL_ELO
        self.opponent_elo = INITIAL_ELO

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def load_mcts_dataset(
        self,
        path:           str,
        pretrain_steps: int = 0,
        max_positions:  int | None = None,
    ) -> None:
        """Bootstrap the buffer from an MCTS dataset; optionally pre-train."""
        if self._mcts_obs is not None:
            n_avail = len(self._mcts_value)
            n = min(max_positions, n_avail) if max_positions is not None else n_avail
            self.buffer.add_game(
                self._mcts_obs[:n],
                self._mcts_policy[:n],
                self._mcts_value[:n],
            )
        else:
            n = self.buffer.load_from_npz(path, max_positions=max_positions)
        print(f"Loaded {n:,} positions into buffer  (buffer={self.buffer.size:,})")

        if pretrain_steps > 0 and self.buffer.size >= self.batch_size:
            print(f"Pre-training for {pretrain_steps} steps on MCTS data ...")
            self.network.train()
            policy_losses: list[float] = []
            value_losses:  list[float] = []
            for _ in range(pretrain_steps):
                pl, vl = train_step(
                    self.network, self.optimizer, self.buffer,
                    self.batch_size, self.device,
                )
                policy_losses.append(pl)
                value_losses.append(vl)
            print(
                f"Pre-training done  "
                f"policy_loss={np.mean(policy_losses):.4f}  "
                f"value_loss={np.mean(value_losses):.4f}"
            )

    def run(self, start_iter: int = 0) -> None:
        """Run the training loop from *start_iter*, logging metrics each iteration."""
        print(f"\n{'=' * 68}")
        print("  AlphaGumbel Training")
        print(
            f"  device={self.device}  sims={self.sims}  "
            f"games/iter={self.games_per_iter}  steps/iter={self.train_steps}"
        )
        print(f"  max_root_actions={self.max_root_actions}")
        print(f"  run dir: {self.run_dir}")
        print(f"{'=' * 68}\n")

        with TrainingLogger(
            self.run_dir,
            headline_keys=[
                "iter", "policy_loss", "value_loss",
                "win_rate_previous", "win_rate_random", "win_rate_mcts", "elo",
            ],
            resume=(start_iter > 0),
        ) as logger:
            for iteration in range(start_iter, self.total_iters):
                metrics = self._run_one_iteration(iteration)
                logger.log_iteration(iteration + 1, metrics)

    # ------------------------------------------------------------------
    # Iteration phases
    # ------------------------------------------------------------------

    def _run_one_iteration(self, iteration: int) -> dict[str, Any]:
        iter_start = time.time()
        print(f"--- Iteration {iteration + 1}/{self.total_iters} ---")

        refresh_positions = self._refresh_anchor_data()
        sp_stats          = self._run_self_play()

        if self.buffer.size < self.min_buffer:
            print(f"  buffer too small ({self.buffer.size} < {self.min_buffer}), skipping training")
            self._save_checkpoint(iteration, label="selfplay_only")
            return {
                "phase":             "selfplay_only",
                "buffer_size":       self.buffer.size,
                "refresh_positions": refresh_positions,
                **sp_stats,
                "iter_time_min":     (time.time() - iter_start) / 60,
            }

        old_network = copy.deepcopy(self.network)
        old_network.eval()

        train_stats = self._run_training()
        eval_stats  = self._run_evaluation(old_network)

        accepted = eval_stats["win_rate_previous"] >= self.win_threshold
        if accepted:
            print(f"  new network accepted "
                  f"(win rate {eval_stats['win_rate_previous']:.1%} "
                  f">= {self.win_threshold:.1%})")
            label = "accepted"
        else:
            print(f"  new network rejected "
                  f"(win rate {eval_stats['win_rate_previous']:.1%} "
                  f"< {self.win_threshold:.1%}), reverting")
            self.network.load_state_dict(old_network.state_dict())
            label = "rejected"

        self.elo = update_rating(
            self.elo,
            self.opponent_elo,
            score=eval_stats["score_previous"],
            n_games=self.eval_games,
            k=self.elo_k,
        )
        if accepted:
            self.opponent_elo = self.elo

        self._save_checkpoint(iteration, label=label)

        iter_time = time.time() - iter_start
        print(f"  iteration {iteration + 1} complete in {iter_time/60:.1f} min\n")

        return {
            "phase":             label,
            "buffer_size":       self.buffer.size,
            "refresh_positions": refresh_positions,
            **sp_stats,
            **train_stats,
            **eval_stats,
            "elo":               self.elo,
            "opponent_elo":      self.opponent_elo,
            "iter_time_min":     iter_time / 60,
        }

    def _refresh_anchor_data(self) -> int:
        if self._mcts_obs is None:
            return 0
        n_avail = len(self._mcts_value)
        k = min(self.mcts_refresh_positions, n_avail)
        idx = np.random.randint(0, n_avail, size=k)
        self.buffer.add_game(
            self._mcts_obs[idx],
            self._mcts_policy[idx],
            self._mcts_value[idx],
        )
        print(f"  re-injected {k:,} MCTS positions (random sample)  buffer={self.buffer.size:,}")
        return k

    def _run_self_play(self) -> dict[str, Any]:
        self.network.eval()
        agent = AlphaGumbelMCTS(
            self.network,
            num_simulations=self.sims,
            max_root_actions=self.max_root_actions,
            device=self.device,
        )

        sp_start = time.time()
        positions_added = 0
        for g in range(self.games_per_iter):
            samples = play_game(agent, temp_threshold=self.temp_threshold)
            obs    = np.stack([s[0] for s in samples])
            policy = np.stack([s[1] for s in samples])
            value  = np.array([s[2] for s in samples], dtype=np.float32)
            self.buffer.add_game(obs, policy, value)
            positions_added += len(samples)

            if (g + 1) % 100 == 0:
                elapsed = time.time() - sp_start
                print(
                    f"  self-play {g+1}/{self.games_per_iter}"
                    f"  positions={positions_added:,}"
                    f"  buffer={self.buffer.size:,}"
                    f"  {(g+1)/elapsed:.2f} games/s"
                )

        sp_time = time.time() - sp_start
        avg_game_len = positions_added / max(self.games_per_iter, 1)
        print(f"  self-play done in {sp_time/60:.1f} min  "
              f"+{positions_added:,} positions  buffer={self.buffer.size:,}")

        return {
            "selfplay_games":         self.games_per_iter,
            "selfplay_positions":     positions_added,
            "selfplay_time_s":        sp_time,
            "selfplay_games_per_sec": self.games_per_iter / max(sp_time, 1e-9),
            "avg_game_length":        avg_game_len,
        }

    def _run_training(self) -> dict[str, Any]:
        one_epoch    = self.buffer.size // self.batch_size
        actual_steps = min(self.train_steps, one_epoch)

        self.network.train()
        train_start = time.time()
        policy_losses: list[float] = []
        value_losses:  list[float] = []
        for _ in range(actual_steps):
            pl, vl = train_step(
                self.network, self.optimizer, self.buffer,
                self.batch_size, self.device,
            )
            policy_losses.append(pl)
            value_losses.append(vl)

        self.scheduler.step()
        lr_now = self.scheduler.get_last_lr()[0]
        train_time = time.time() - train_start

        policy_loss_mean = float(np.mean(policy_losses)) if policy_losses else 0.0
        value_loss_mean  = float(np.mean(value_losses))  if value_losses  else 0.0
        print(
            f"  training done in {train_time:.1f}s"
            f"  steps={actual_steps}/{self.train_steps}"
            f"  policy_loss={policy_loss_mean:.4f}"
            f"  value_loss={value_loss_mean:.4f}"
            f"  lr={lr_now:.2e}"
        )

        return {
            "train_steps_actual": actual_steps,
            "train_time_s":       train_time,
            "policy_loss":        policy_loss_mean,
            "value_loss":         value_loss_mean,
            "lr":                 lr_now,
        }

    def _run_evaluation(self, old_network: AlphaZeroNet) -> dict[str, Any]:
        self.network.eval()
        eval_start = time.time()

        new_agent = AlphaGumbelMCTS(
            self.network,
            num_simulations=self.eval_sims,
            max_root_actions=self.max_root_actions,
            device=self.device,
        )
        old_agent = AlphaGumbelMCTS(
            old_network,
            num_simulations=self.eval_sims,
            max_root_actions=self.max_root_actions,
            device=self.device,
        )

        prev = evaluate_vs_previous(new_agent, old_agent, n_games=self.eval_games)
        rand = evaluate_vs_random(new_agent,            n_games=self.eval_random_games)
        mcts = evaluate_vs_mcts(   new_agent,
                                   mcts_sims=self.eval_mcts_sims,
                                   n_games=self.eval_mcts_games)

        eval_time = time.time() - eval_start
        print(
            f"  evaluation done in {eval_time:.1f}s  "
            f"vs_prev={prev['win_rate']:.1%}  "
            f"vs_random={rand['win_rate']:.1%}  "
            f"vs_mcts({self.eval_mcts_sims})={mcts['win_rate']:.1%}"
        )

        return {
            "eval_time_s":       eval_time,
            "win_rate_previous": prev["win_rate"],
            "score_previous":    prev["score"],
            "draws_previous":    prev["draws"],
            "win_rate_random":   rand["win_rate"],
            "win_rate_mcts":     mcts["win_rate"],
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, iteration: int, label: str = "") -> None:
        name = f"iter_{iteration + 1:03d}"
        if label:
            name += f"_{label}"
        path = self.run_dir / f"{name}.pt"
        torch.save(
            {
                "iteration":       iteration + 1,
                "network_state":   self.network.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "buffer_size":     self.buffer.size,
                "elo":             self.elo,
                "opponent_elo":    self.opponent_elo,
                "model_config": {
                    "channels":   self.channels,
                    "num_blocks": self.num_blocks,
                },
            },
            path,
        )
        print(f"  checkpoint saved -> {path}")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and return the iteration to resume from."""
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.elo          = float(ckpt.get("elo",          INITIAL_ELO))
        self.opponent_elo = float(ckpt.get("opponent_elo", INITIAL_ELO))
        start_iter = ckpt["iteration"]
        print(f"Resumed from {path}  (iteration {start_iter}, elo {self.elo:.1f})")
        return start_iter
