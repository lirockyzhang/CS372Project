# Setup

This document explains how to install AlphaToe and run the training, evaluation, and tournament scripts. The project is pure-Python (PyTorch + NumPy) and designed to be run locally with consumer-grade hardware. It uses no external APIs nor services.

## 1. Prerequisites

This project uses [`uv`](https://docs.astral.sh/uv/) as its package and project manager. Please refer to the `uv` official documentation for specific installing instruction.  

A CUDA-compatable GPU is **recommended** for training but **not required**. Every script accepts `--device cpu` and the test scripts in the README run on CPU in a few minutes.

If you are using GPU, please change the CUDA version to match the CUDA installed on your machine in following lines from `pyproject.toml`.
```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

## 2. Install

From the repository root:

```bash
uv sync
```

`uv sync` reads `pyproject.toml` + `uv.lock` and creates a `.venv/` with the
exact pinned versions of `torch`, `numpy`, `tqdm`, `tensorboard`, `numba`, and
`numba-cuda`.

## 3. Verify the install

```bash
uv run python -c "import sys; sys.path.insert(0, 'src'); import env.state, agents.mcts.mcts_agent; print('alphatoe OK')"
```

The inline `sys.path.insert` mirrors what every entry-point script under
`src/scripts/` and `src/tournament/` does at import time (see Troubleshooting
below), so this one-liner works on Windows / macOS / Linux without setting
`PYTHONPATH`.

## 4. External services / APIs

**None.** The project trains and evaluates entirely from local self-play data.
Graders do **not** need any API keys, accounts, or network access beyond
`uv` package downloads on first install.

## 5. Troubleshooting

**`ModuleNotFoundError: agents` / `env`** — run scripts via `uv run python …` from the repo root so the `sys.path` injection at the top of each entry point can resolve `src/`. If you import these modules from your own code, set `PYTHONPATH=src`.

The project source under `src/` is a flat collection of top-level packages
(`agents/`, `env/`, `tournament/`, `utils/`, plus the `scripts/` entry points).
Every entry-point script under `src/scripts/` and `src/tournament/` injects
`src/` onto `sys.path` at the top of the file:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

so they can be invoked directly via `uv run python src/scripts/...` without
an editable install step.
