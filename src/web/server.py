"""FastAPI app: human vs trained agent for Ultimate Tic-Tac-Toe.

Run from the repo root:

    uv run uvicorn web.server:app --reload --app-dir src

Then open http://localhost:8000/.

Models are auto-discovered from three subdirectories under ``models/``:

    models/cnn/*.pt          AlphaZero CNN checkpoints       -> AlphaZeroMCTS
    models/transformer/*.pt  AlphaZero Transformer ckpts     -> AlphaZeroMCTS
    models/ppo/*.pt          PPO checkpoints (UTTTNet)       -> raw policy net
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Make `agents`, `env` importable when run via uvicorn --app-dir src.
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agents.alphagumbel.mcts import AlphaGumbelMCTS  # noqa: E402
from agents.alphazero.mcts import AlphaZeroMCTS  # noqa: E402
from agents.alphazero.transformer import AlphaZeroTransformerNet  # noqa: E402
from agents.common.network import AlphaZeroNet, masked_softmax  # noqa: E402
from env.logic import legal_action_mask  # noqa: E402
from env.observation import observe  # noqa: E402
from env.state import FULL_BOARD, UTTTState  # noqa: E402

REPO_ROOT = _SRC.parent
MODELS_ROOT = REPO_ROOT / "models"
STATIC_DIR = Path(__file__).parent / "static"


def _detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class StatePayload(BaseModel):
    boards_x: list[int] = Field(min_length=9, max_length=9)
    boards_o: list[int] = Field(min_length=9, max_length=9)
    meta_x: int
    meta_o: int
    meta_draw: int
    current_player: int
    active_board: int

    @field_validator("boards_x", "boards_o")
    @classmethod
    def _validate_sub_boards(cls, v: list[int]) -> list[int]:
        for b in v:
            if not 0 <= b <= FULL_BOARD:
                raise ValueError(f"sub-board out of range: {b}")
        return v

    @field_validator("current_player")
    @classmethod
    def _validate_player(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("current_player must be 0 or 1")
        return v

    @field_validator("active_board")
    @classmethod
    def _validate_active(cls, v: int) -> int:
        if not -1 <= v <= 8:
            raise ValueError("active_board must be in [-1, 8]")
        return v


class MoveRequest(BaseModel):
    state: StatePayload
    model_id: str
    num_simulations: int = Field(default=200, ge=10, le=2000)


class TopMove(BaseModel):
    board: int
    cell: int
    visits: float
    q: float


class MoveResponse(BaseModel):
    action: tuple[int, int]
    policy: list[list[float]]
    value: float
    top_moves: list[TopMove]
    elapsed_ms: int
    kind: str  # "az_cnn" | "az_transformer" | "az_gumbel" | "ppo"


class ModelInfo(BaseModel):
    id: str
    label: str
    kind: str
    uses_simulations: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# State conversion + validation
# ---------------------------------------------------------------------------


def payload_to_state(p: StatePayload) -> UTTTState:
    boards_x = np.asarray(p.boards_x, dtype=np.uint16)
    boards_o = np.asarray(p.boards_o, dtype=np.uint16)

    if np.any(boards_x & boards_o):
        raise HTTPException(400, "boards_x and boards_o overlap")

    if (p.meta_x & p.meta_o) or (p.meta_x & p.meta_draw) or (p.meta_o & p.meta_draw):
        raise HTTPException(400, "meta bits overlap")

    resolved = p.meta_x | p.meta_o | p.meta_draw
    if resolved & ~FULL_BOARD:
        raise HTTPException(400, "meta bits out of range")

    if p.active_board != -1 and (resolved >> p.active_board) & 1:
        raise HTTPException(400, "active_board points to a resolved sub-board")

    state = UTTTState(
        boards_x=boards_x,
        boards_o=boards_o,
        meta_x=p.meta_x,
        meta_o=p.meta_o,
        meta_draw=p.meta_draw,
        current_player=p.current_player,
        active_board=p.active_board,
        terminated=False,
        winner=-1,
    )
    if not legal_action_mask(state).any():
        raise HTTPException(400, "no legal actions available (terminated state?)")
    return state


# ---------------------------------------------------------------------------
# Adapters — uniform interface across model kinds
# ---------------------------------------------------------------------------


class _AdapterResult:
    __slots__ = ("action", "policy", "value", "top_moves")

    def __init__(
        self,
        action: tuple[int, int],
        policy: np.ndarray,
        value: float,
        top_moves: list[TopMove],
    ) -> None:
        self.action = action
        self.policy = policy
        self.value = value
        self.top_moves = top_moves


class Adapter(Protocol):
    kind: str
    label: str
    uses_simulations: bool

    def play(self, state: UTTTState, num_simulations: int) -> _AdapterResult: ...


class AlphaZeroAdapter:
    """Wraps an AlphaZeroMCTS so policy = visit-share, top_moves carry Q."""

    kind: str  # set per-instance ("az_cnn" or "az_transformer")
    uses_simulations = True

    def __init__(self, mcts: AlphaZeroMCTS, label: str, kind: str) -> None:
        self.mcts = mcts
        self.label = label
        self.kind = kind

    def play(self, state: UTTTState, num_simulations: int) -> _AdapterResult:
        prev = self.mcts.num_simulations
        self.mcts.num_simulations = num_simulations
        try:
            root = self.mcts._build_tree(state, add_noise=False)
        finally:
            self.mcts.num_simulations = prev

        if not root.children:
            raise HTTPException(400, "agent produced no children for this state")

        total_visits = sum(c.visits for c in root.children) or 1
        policy = np.zeros((9, 9), dtype=np.float32)
        children_sorted = sorted(root.children, key=lambda c: c.visits, reverse=True)
        for child in children_sorted:
            b, c = child.action  # type: ignore[misc]
            policy[b, c] = child.visits / total_visits

        best_action = children_sorted[0].action
        assert best_action is not None

        # Each child's q_value is from the root-mover's POV by AlphaZero's
        # alternating-sign backprop convention.
        top_moves = [
            TopMove(
                board=int(child.action[0]),  # type: ignore[index]
                cell=int(child.action[1]),  # type: ignore[index]
                visits=float(child.visits / total_visits),
                q=float(child.q_value),
            )
            for child in children_sorted[:3]
        ]

        # Root has no parent: q_value is averaged from the opponent's POV.
        # Negate once to report from root mover's POV.
        value = -float(root.q_value) if root.visits else 0.0

        return _AdapterResult(
            action=(int(best_action[0]), int(best_action[1])),
            policy=policy,
            value=value,
            top_moves=top_moves,
        )


class AlphaGumbelAdapter:
    """Wraps an AlphaGumbelMCTS so the visit_policy is exposed as the move
    heatmap and top_moves carry per-move Q from the search root.
    """

    kind = "az_gumbel"
    uses_simulations = True

    def __init__(self, mcts: AlphaGumbelMCTS, label: str) -> None:
        self.mcts = mcts
        self.label = label

    def play(self, state: UTTTState, num_simulations: int) -> _AdapterResult:
        prev = self.mcts.num_simulations
        self.mcts.num_simulations = num_simulations
        try:
            out = self.mcts.search(state, add_gumbel_noise=False)
        finally:
            self.mcts.num_simulations = prev

        root = self.mcts._last_root
        if root is None or not root.children:
            raise HTTPException(400, "agent produced no children for this state")

        policy = out.visit_policy.astype(np.float32, copy=True)

        # Q values come from the searched root children. Some Gumbel root
        # candidates may be unvisited (sequential halving eliminates losers
        # early); skip those and order by visit_policy mass.
        children_by_action = {
            child.action: child
            for child in root.children
            if child.action is not None
        }
        flat = policy.reshape(-1)
        order = np.argsort(-flat)

        top_moves: list[TopMove] = []
        for idx in order:
            if flat[idx] <= 0 or len(top_moves) >= 3:
                break
            b, c = divmod(int(idx), 9)
            child = children_by_action.get((b, c))
            q = float(child.q_value) if (child is not None and child.visits > 0) else 0.0
            top_moves.append(TopMove(board=b, cell=c, visits=float(flat[idx]), q=q))

        # Root has no parent: q_value is averaged from the opponent's POV.
        # Negate once to report from the root mover's POV (matches AlphaZero).
        value = -float(root.q_value) if root.visits else 0.0

        return _AdapterResult(
            action=(int(out.action[0]), int(out.action[1])),
            policy=policy,
            value=value,
            top_moves=top_moves,
        )


class PPOAdapter:
    """Raw policy/value network — no tree search.

    The "policy" returned to the client is the network's softmax over
    legal moves (a stand-in for visit-share). top_moves' ``q`` carries
    the position-level value (the same number is repeated for every
    candidate move) — PPO has no per-move Q without rollouts.
    """

    kind = "ppo"
    uses_simulations = False

    def __init__(self, network: AlphaZeroNet, label: str, device: torch.device) -> None:
        self.network = network
        self.label = label
        self.device = device

    @torch.no_grad()
    def play(self, state: UTTTState, num_simulations: int) -> _AdapterResult:
        del num_simulations  # ignored: PPO uses raw policy

        obs = observe(state)
        mask = legal_action_mask(state)

        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        logits, value_t = self.network(obs_t)
        probs = masked_softmax(logits, mask_t).squeeze(0).cpu().numpy()
        value = float(value_t.squeeze(0).item())

        flat = probs.reshape(-1)
        order = np.argsort(-flat)  # descending
        top_moves: list[TopMove] = []
        for idx in order[:3]:
            if flat[idx] <= 0:
                break
            b, c = divmod(int(idx), 9)
            top_moves.append(TopMove(board=b, cell=c, visits=float(flat[idx]), q=value))

        best_idx = int(order[0])
        return _AdapterResult(
            action=divmod(best_idx, 9),
            policy=probs,
            value=value,
            top_moves=top_moves,
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_state_dict(
    ckpt: dict[str, Any] | Any,
) -> tuple[dict, dict[str, Any]]:
    """Return (state_dict, full_ckpt_meta). Tolerates a few key conventions."""
    if isinstance(ckpt, dict):
        for key in ("network_state", "model", "state_dict"):
            if key in ckpt:
                return ckpt[key], ckpt
        return ckpt, ckpt
    return ckpt, {}


def _max_block_index(state: dict, prefix: str) -> int:
    """Return max N + 1 for keys like ``{prefix}.N.*``, or 0 if none."""
    seen: set[int] = set()
    for key in state:
        if key.startswith(prefix + "."):
            try:
                seen.add(int(key.split(".", 2)[1]))
            except ValueError:
                continue
    return max(seen) + 1 if seen else 0


def _infer_cnn_dims(state: dict) -> tuple[int, int]:
    """(channels, num_blocks) from an AlphaZeroNet state_dict."""
    channels = int(state["input_conv.weight"].shape[0])
    num_blocks = _max_block_index(state, "res_blocks")
    return channels, num_blocks


def _infer_transformer_dims(state: dict) -> tuple[int, int, int]:
    """(embed_dim, depth, num_heads) from an AlphaZeroTransformerNet state_dict.

    embed_dim and depth are deterministic from shapes. num_heads is not
    recoverable from PyTorch's fused in-projection weight; we pick the
    largest common choice that divides embed_dim, matching the project's
    naming convention (e.g. e192 -> heads=6, e128 -> heads=4).
    """
    embed_dim = int(state["input_proj.weight"].shape[0])
    depth = _max_block_index(state, "blocks")
    for h in (8, 6, 4, 2, 1):
        if embed_dim % h == 0:
            num_heads = h
            break
    else:  # pragma: no cover — embed_dim always divisible by 1
        num_heads = 1
    return embed_dim, depth, num_heads


def _load_az_cnn(path: Path, device: torch.device) -> AlphaZeroAdapter:
    raw = torch.load(path, map_location=device, weights_only=False)
    state, meta = _load_state_dict(raw)
    cfg = meta.get("model_config", {}) if isinstance(meta, dict) else {}

    if "channels" in cfg and "num_blocks" in cfg:
        channels, num_blocks = cfg["channels"], cfg["num_blocks"]
    else:
        channels, num_blocks = _infer_cnn_dims(state)

    net = AlphaZeroNet(channels=channels, num_blocks=num_blocks).to(device)
    net.load_state_dict(state)
    net.eval()

    mcts = AlphaZeroMCTS(net, num_simulations=200, batch_size=8, device=device)
    label = f"AlphaZero CNN — {path.stem} (ch={channels}, blocks={num_blocks})"
    return AlphaZeroAdapter(mcts, label=label, kind="az_cnn")


def _patch_legacy_transformer_state(state: dict) -> dict:
    """Older transformer checkpoints used a 4-layer value_head (no Identity
    placeholder for dropout). The current architecture has the final Linear
    at index 4. Rename ``value_head.3.*`` → ``value_head.4.*`` if needed.
    """
    if "value_head.3.weight" in state and "value_head.4.weight" not in state:
        state = dict(state)
        state["value_head.4.weight"] = state.pop("value_head.3.weight")
        if "value_head.3.bias" in state:
            state["value_head.4.bias"] = state.pop("value_head.3.bias")
    return state


def _load_az_transformer(path: Path, device: torch.device) -> AlphaZeroAdapter:
    raw = torch.load(path, map_location=device, weights_only=False)
    state, meta = _load_state_dict(raw)
    cfg = meta.get("model_config", {}) if isinstance(meta, dict) else {}

    if {"embed_dim", "depth", "num_heads"} <= cfg.keys():
        embed_dim = cfg["embed_dim"]
        depth = cfg["depth"]
        num_heads = cfg["num_heads"]
    else:
        embed_dim, depth, num_heads = _infer_transformer_dims(state)

    net = AlphaZeroTransformerNet(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
    ).to(device)
    net.load_state_dict(_patch_legacy_transformer_state(state))
    net.eval()

    mcts = AlphaZeroMCTS(net, num_simulations=200, batch_size=8, device=device)
    label = (
        f"AlphaZero Transformer — {path.stem} "
        f"(e{embed_dim}d{depth}h{num_heads})"
    )
    return AlphaZeroAdapter(mcts, label=label, kind="az_transformer")


def _load_az_gumbel(path: Path, device: torch.device) -> AlphaGumbelAdapter:
    """Load a CNN backbone and wrap it with AlphaGumbelMCTS.

    The Gumbel deployment uses the regularized CNN pretrain (or any
    AlphaZero-CNN checkpoint with the same arch) — only the search algorithm
    differs from ``_load_az_cnn``.
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    state, meta = _load_state_dict(raw)
    cfg = meta.get("model_config", {}) if isinstance(meta, dict) else {}

    if "channels" in cfg and "num_blocks" in cfg:
        channels, num_blocks = cfg["channels"], cfg["num_blocks"]
    else:
        channels, num_blocks = _infer_cnn_dims(state)

    net = AlphaZeroNet(channels=channels, num_blocks=num_blocks).to(device)
    net.load_state_dict(state)
    net.eval()

    mcts = AlphaGumbelMCTS(net, num_simulations=64, device=device)
    label = f"AlphaGumbel — {path.stem} (ch={channels}, blocks={num_blocks})"
    return AlphaGumbelAdapter(mcts, label=label)


def _load_ppo(path: Path, device: torch.device) -> PPOAdapter:
    raw = torch.load(path, map_location=device, weights_only=False)
    state, meta = _load_state_dict(raw)
    cfg = meta.get("model_config", {}) if isinstance(meta, dict) else {}

    if "channels" in cfg and "num_blocks" in cfg:
        channels, num_blocks = cfg["channels"], cfg["num_blocks"]
    else:
        channels, num_blocks = _infer_cnn_dims(state)

    net = AlphaZeroNet(channels=channels, num_blocks=num_blocks).to(device)
    net.load_state_dict(state)
    net.eval()

    label = f"PPO — {path.stem} (ch={channels}, blocks={num_blocks})"
    return PPOAdapter(network=net, label=label, device=device)


# ---------------------------------------------------------------------------
# Registry build: scan models/{cnn,transformer,ppo}/*.pt
# ---------------------------------------------------------------------------


_KIND_LOADERS = {
    "cnn":         (_load_az_cnn,         "cnn"),
    "transformer": (_load_az_transformer, "tx"),
    "ppo":         (_load_ppo,            "ppo"),
}


_DEFAULT_GUMBEL_CHECKPOINT = "models/cnn/cnn_c128b3_100k_reg_best.pt"


def _build_gumbel_only_registry(device: torch.device) -> dict[str, Adapter]:
    """Single-model deployment mode (e.g. Coolify): expose only AlphaGumbel.

    Checkpoint path is taken from ``WEB_GUMBEL_CHECKPOINT`` (relative to repo
    root or absolute), defaulting to the regularized CNN pretrain used in the
    Gumbel-vs-PUCT benchmark. The registered model id is ``gumbel`` so the
    frontend doesn't need to know the underlying filename.
    """
    rel = os.environ.get("WEB_GUMBEL_CHECKPOINT", _DEFAULT_GUMBEL_CHECKPOINT)
    path = Path(rel)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        print(f"[web] WEB_MODE=alphagumbel but checkpoint not found: {path}", file=sys.stderr)
        return {}
    return {"gumbel": _load_az_gumbel(path, device)}


def build_registry() -> dict[str, Adapter]:
    device = _detect_device()

    if os.environ.get("WEB_MODE", "").lower() == "alphagumbel":
        return _build_gumbel_only_registry(device)

    registry: dict[str, Adapter] = {}
    if not MODELS_ROOT.is_dir():
        return registry

    for subdir, (loader, prefix) in _KIND_LOADERS.items():
        kind_dir = MODELS_ROOT / subdir
        if not kind_dir.is_dir():
            continue
        for path in sorted(kind_dir.glob("*.pt")):
            mid = f"{prefix}__{path.stem}"
            try:
                registry[mid] = loader(path, device)
            except Exception as exc:  # noqa: BLE001
                print(f"[web] failed to load {path}: {exc}", file=sys.stderr)
    return registry


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AlphaToe Web Play", version="0.2.0")
_REGISTRY: dict[str, Adapter] = {}


@app.on_event("startup")
def _startup() -> None:
    global _REGISTRY
    _REGISTRY = build_registry()
    if _REGISTRY:
        print(f"[web] loaded {len(_REGISTRY)} model(s):")
        for mid, ad in _REGISTRY.items():
            print(f"  - {mid}  ({ad.kind})")
    else:
        print(f"[web] no models found under {MODELS_ROOT}", file=sys.stderr)


@app.get("/api/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(
        models=[
            ModelInfo(
                id=k,
                label=v.label,
                kind=v.kind,
                uses_simulations=v.uses_simulations,
            )
            for k, v in _REGISTRY.items()
        ]
    )


@app.post("/api/move", response_model=MoveResponse)
def post_move(req: MoveRequest) -> MoveResponse:
    adapter = _REGISTRY.get(req.model_id)
    if adapter is None:
        raise HTTPException(404, f"unknown model_id: {req.model_id}")

    state = payload_to_state(req.state)

    t0 = time.perf_counter()
    result = adapter.play(state, req.num_simulations)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return MoveResponse(
        action=result.action,
        policy=result.policy.tolist(),
        value=result.value,
        top_moves=result.top_moves,
        elapsed_ms=elapsed_ms,
        kind=adapter.kind,
    )


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
