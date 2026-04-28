# PPO vs AlphaZero-Gumbel — Head-to-head from a shared supervised backbone

This doc records what we learned pitting our **PPO** baseline against the **AlphaZero-Gumbel** pipeline. Both runs share the same supervised pretrain, the same network architecture, and the same evaluation protocol, so any divergence in playing strength is attributable to the post-pretraining stage — pure policy-gradient (PPO) vs Gumbel-search self-play (AlphaZero-Gumbel) — not to the network or the training data.

| Component | Value |
|---|---|
| Backbone (both runs) | `models/cnn/cnn_c128b3_100k_reg_best.pt` (regularized supervised pretrain on the 100k MCTS bootstrap dataset) |
| Channels / blocks | 128 / 3 |
| PPO run dir | `runs/ppo_warm_matched/` |
| AZ-Gumbel run dir | `runs/az_cnn_gumbel/` |
| PPO recipe | Clipped PPO, 150 update steps, 512 envs, 128 steps/rollout, 4 epochs, 4 minibatches, lr 3e-4 |
| AZ-Gumbel recipe | Gumbel root + sequential halving, 64 sims/move, win-threshold gate 0.45, `--max-root-actions 16` |
| Move policy at eval | Greedy (PPO argmax, Gumbel most-visited child, temperature 0) |
| Hardware | NVIDIA RTX 4060 Laptop, CUDA |

## Phase 1 — Common-opponent comparison (vs MCTS@200)

Both agents are evaluated against the **same** opponent — a pure-MCTS player at 200 simulations per move — using `src/scripts/compare_runs.py` against the bootstrap MCTS baseline. This is the best apples-to-apples strength comparison without a direct head-to-head match.

Eval settings: 30 games per checkpoint, agents at 64 sims/move, MCTS at 200 sims/move. Source rows are in `runs/comparison_mcts200/eval_vs_mcts200.csv` and `runs/comparison_mcts200/convergence_summary.csv`.

| Run                 | Best checkpoint               | Win rate vs MCTS@200 | Cum. NN forwards | Cum. wall-clock | Peak Elo vs MCTS |
|---------------------|-------------------------------|----------------------|------------------|-----------------|------------------|
| Warm start (no RL)  | `cnn_c128b3_100k_reg_best`    | 6.7 %                | 0                | 0 h             | — (baseline)     |
| **PPO**             | `ppo.pt` (final, iter 150)    | **16.7 %**           | 5.01 M           | 0.85 h          | **1220.4**       |
| **AlphaZero-Gumbel**| `iter_002_accepted.pt`        | **16.7 %**           | 2.04 M           | 1.10 h          | **1220.4**       |

### Findings

1. **Equivalent peak strength.** Both agents top out at the same win rate (5/30 games) vs the shared MCTS@200 opponent, giving an identical peak Elo of 1220.4. From the same supervised starting point, neither RL recipe significantly outperforms the other on this benchmark in the budget we ran.
2. **AlphaZero-Gumbel is more sample-efficient.** It reaches its peak in ~2.0 M cumulative NN forwards vs PPO's ~5.0 M — roughly **2.5× fewer forward passes per unit of playing strength**. This matches the AlphaZero intuition that search improves the training targets, so each gradient step carries more information.
3. **PPO is wall-clock cheaper.** PPO completes 150 iterations in ~51 min vs AlphaZero-Gumbel's ~66 min for 5 iterations, because PPO has no search overhead at training time (Gumbel runs 64 sims/move during self-play). On a per-NN-forward basis Gumbel is slower; on a per-iteration basis it does much more work.
4. **Pretrain dominates.** Both warm-start and final checkpoints land in a tight band on this benchmark (6.7 % → 16.7 % win rate vs MCTS@200). Most of the playing strength comes from the supervised pretrain on the 100k MCTS dataset; the RL stage adds a relatively small delta in the budget we explored.

## Phase 2 — Acceptance gating and self-play dynamics

The two pipelines have very different "did it get better?" loops:

| Property | PPO | AlphaZero-Gumbel |
|---|---|---|
| Update gate | None — every iteration is kept | 0.45-win-rate gate vs previous best (`--win-threshold .45`) |
| Iterations run | 150 | 5 |
| Iterations accepted | 150/150 | **1/5** (`iter_002_accepted.pt`) |
| Self-play opponent | Past *self* (running average) | Previous *best* (last accepted iteration) |
| Self-play win rate at end | ~88 % vs running average (`runs/ppo_warm_matched/train_log.csv`) | 0.43–0.63 vs previous best (`runs/az_cnn_gumbel/train_log.csv`) |

### Findings

1. **Self-play Elo can climb past a rejected iteration.** AlphaZero-Gumbel's internal self-play Elo column rises from 1484 (iter 1) to 1614 (iter 4), but iter 4 was rejected by the validation match against `iter_002_accepted` — that is, the network felt stronger in self-play but did not transfer to the gating opponent. This is consistent with self-play drift / overfitting to recent trajectories, and is exactly what the gate is there to filter.
2. **PPO has no equivalent guard.** Without a gate, PPO accepts every update, so the 150-iter checkpoint is whatever the most recent gradient step produced. The fact that the *final* PPO checkpoint matches the *best* AZ-Gumbel checkpoint vs MCTS@200 suggests PPO did not catastrophically degrade, but a gated variant would be a fairer comparison and is a natural follow-up.

## Phase 3 — Direct head-to-head

50 games, paired openings (each opening played from both sides), temperature-0 greedy moves on both sides, seed 42, 64 sims/move for the Gumbel agent, single RTX 4060 Laptop. Tool: `head_to_head.py --agents ppo gumbel-selfplay`.

Source CSVs: `runs/ppo_vs_gumbel/ppo_vs_gumbel-selfplay.csv` (per-game) + `runs/ppo_vs_gumbel/summary.csv` (match summary).

| Agent                              | W  | D | L  | Pts  | Win % | Avg ms/move |
|------------------------------------|---:|--:|---:|-----:|------:|------------:|
| **PPO** (`models/ppo/ppo.pt`)      | 27 | 1 | 22 | 27.5 | **55.0 %** |  **1.63** |
| Gumbel-SelfPlay (`iter_002_accepted.pt`, 64 sims) | 22 | 1 | 27 | 22.5 | 45.0 % | 93.50 |

Wall time: **1.82 min** (50 games, ~0.46 g/s).

### Findings

1. **PPO wins the direct head-to-head 55–45.** With only 50 games the binomial 95 % CI on PPO's score (~27.5/50) is roughly ±14 pp, so this does **not** establish a significant strength gap — but the direction is the opposite of what you'd guess from the AlphaZero literature, where search-augmented training is expected to dominate a pure policy-gradient baseline.
2. **Common-opponent vs head-to-head can disagree.** Phase 1 said both agents reach the same peak Elo vs MCTS@200; Phase 3 shows PPO is modestly better when they actually play each other. This is a known limitation of common-opponent benchmarks — the shared opponent may not discriminate styles that exploit each other differently.
3. **PPO is ~57× cheaper per move at inference.** PPO ran at 1.63 ms/move on cuda; Gumbel-SelfPlay at 64 sims/move ran at 93.5 ms/move. Even if PPO loses some ground at higher Gumbel sim budgets, the wall-clock gap would still favor PPO substantially in interactive use.
4. **Caveat: only iter_002 of AZ-Gumbel was tested.** This is the gated "accepted" checkpoint and the one that peaked vs MCTS@200, but the later self-play-Elo-leading iterations (rejected by the gate) were not evaluated in this match. A fairer test would sweep multiple AZ-Gumbel iterations or compare to a longer-trained Gumbel run.

### Reproduction

```bash
python src/tournament/head_to_head.py --agents ppo gumbel-selfplay \
  --ppo-ckpt models/ppo/ppo.pt --ppo-channels 128 \
  --az-ckpt models/cnn/iter_002_accepted.pt \
  --games 50 --az-sims 64 --seed 42 \
  --log-dir runs/ppo_vs_gumbel
```

`--ppo-channels 128` is required because the bundled PPO checkpoint was trained with 128 channels (head_to_head.py defaults to 64). Output is written to `runs/ppo_vs_gumbel/{ppo_vs_gumbel-selfplay.csv, summary.csv}`.