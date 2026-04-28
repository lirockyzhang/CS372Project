// AlphaToe web client — UTTT vs trained agents.
// Mirrors enough of env/logic.py to validate moves and step the board.

const FULL_BOARD = 0b111_111_111;
const WIN_MASKS = [
  0b000_000_111, 0b000_111_000, 0b111_000_000,
  0b001_001_001, 0b010_010_010, 0b100_100_100,
  0b100_010_001, 0b001_010_100,
];

function checkWin(bits) {
  for (const m of WIN_MASKS) if ((bits & m) === m) return true;
  return false;
}

function initialState() {
  return {
    boards_x: new Array(9).fill(0),
    boards_o: new Array(9).fill(0),
    meta_x: 0, meta_o: 0, meta_draw: 0,
    current_player: 0,         // 0 = X, 1 = O
    active_board: -1,
    terminated: false,
    winner: -1,
    last_move: null,
  };
}

function isResolved(s, b) {
  return ((s.meta_x | s.meta_o | s.meta_draw) >> b) & 1;
}

function legalMask(s) {
  const mask = Array.from({length: 9}, () => new Array(9).fill(false));
  if (s.terminated) return mask;
  const boards = s.active_board === -1 ? [0,1,2,3,4,5,6,7,8] : [s.active_board];
  for (const b of boards) {
    if (isResolved(s, b)) continue;
    const occ = s.boards_x[b] | s.boards_o[b];
    for (let c = 0; c < 9; c++) {
      if (!((occ >> c) & 1)) mask[b][c] = true;
    }
  }
  return mask;
}

function step(s, [b, c]) {
  // Apply move in-place. Caller is responsible for legality.
  const bit = 1 << c;
  const boardBit = 1 << b;

  if (s.current_player === 0) s.boards_x[b] |= bit;
  else                        s.boards_o[b] |= bit;

  const playerBoard = s.current_player === 0 ? s.boards_x[b] : s.boards_o[b];

  if (checkWin(playerBoard)) {
    if (s.current_player === 0) s.meta_x |= boardBit;
    else                        s.meta_o |= boardBit;
  } else if ((s.boards_x[b] | s.boards_o[b]) === FULL_BOARD) {
    s.meta_draw |= boardBit;
  }

  const metaCurrent = s.current_player === 0 ? s.meta_x : s.meta_o;
  if (checkWin(metaCurrent)) {
    s.terminated = true;
    s.winner = s.current_player;
  } else if ((s.meta_x | s.meta_o | s.meta_draw) === FULL_BOARD) {
    s.terminated = true;
    s.winner = -1;
  }

  if (!s.terminated) {
    s.active_board = isResolved(s, c) ? -1 : c;
  }
  s.current_player ^= 1;
  s.last_move = [b, c];
  return s;
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

const els = {
  board:          document.getElementById("board"),
  status:         document.getElementById("status"),
  model:          document.getElementById("model"),
  playAs:         document.getElementById("play-as"),
  difficulty:     document.getElementById("difficulty"),
  showHeatmap:    document.getElementById("show-heatmap"),
  newGame:        document.getElementById("new-game"),
  topMoves:       document.getElementById("top-moves"),
  valueNum:       document.getElementById("value-num"),
  valueFill:      document.getElementById("value-fill"),
  elapsed:        document.getElementById("elapsed"),
};

// AlphaGumbel simulation budgets, tuned for CPU-only deployment (e.g.
// Coolify arm64). Each sim is ~one CNN forward pass on a c128b3 backbone.
const GUMBEL_SIMS = { easy: 32, medium: 64, hard: 128 };

let state = initialState();
let humanSide = 0;            // 0 = X, 1 = O
let difficulty = "medium";    // "easy" | "medium" | "hard"
let lastPolicy = null;        // 9x9 visit-share from the most recent AI move
let busy = false;             // disable input while AI is thinking

function currentSims() {
  return GUMBEL_SIMS[difficulty];
}

const CELL_LAYOUT = [
  // Within one sub-board, cell index 0..8 maps to (row, col) row-major.
  [0,0],[0,1],[0,2],
  [1,0],[1,1],[1,2],
  [2,0],[2,1],[2,2],
];

function buildBoard() {
  els.board.innerHTML = "";
  for (let b = 0; b < 9; b++) {
    const sub = document.createElement("div");
    sub.className = "subboard";
    sub.dataset.board = b;

    const overlay = document.createElement("div");
    overlay.className = "overlay";
    sub.appendChild(overlay);

    for (let c = 0; c < 9; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.board = b;
      cell.dataset.cell = c;
      cell.addEventListener("click", onCellClick);
      sub.appendChild(cell);
    }
    els.board.appendChild(sub);
  }
}

function render() {
  const mask = legalMask(state);
  const heatOn = els.showHeatmap.checked && lastPolicy !== null && state.current_player === humanSide;

  for (let b = 0; b < 9; b++) {
    const sub = els.board.children[b];
    sub.classList.toggle("active", !state.terminated && (state.active_board === -1 || state.active_board === b) && !isResolved(state, b));
    sub.classList.toggle("won-x", !!((state.meta_x >> b) & 1));
    sub.classList.toggle("won-o", !!((state.meta_o >> b) & 1));
    sub.classList.toggle("drawn", !!((state.meta_draw >> b) & 1) && !((state.meta_x >> b) & 1) && !((state.meta_o >> b) & 1));

    const overlay = sub.querySelector(".overlay");
    overlay.textContent = ((state.meta_x >> b) & 1) ? "X" : ((state.meta_o >> b) & 1) ? "O" : "";

    // children index 0 is the overlay, cells start at 1.
    for (let c = 0; c < 9; c++) {
      const cell = sub.children[c + 1];
      const xMark = (state.boards_x[b] >> c) & 1;
      const oMark = (state.boards_o[b] >> c) & 1;
      cell.classList.toggle("x", !!xMark);
      cell.classList.toggle("o", !!oMark);
      cell.textContent = xMark ? "X" : oMark ? "O" : "";

      const legal = mask[b][c] && !state.terminated;
      cell.classList.toggle("disabled", !legal || state.current_player !== humanSide || busy);

      const isLast = state.last_move && state.last_move[0] === b && state.last_move[1] === c;
      cell.classList.toggle("last-move", !!isLast);

      if (heatOn && legal && lastPolicy[b][c] > 0) {
        const a = Math.min(0.6, 0.15 + lastPolicy[b][c] * 0.85);
        cell.classList.add("heat");
        cell.style.setProperty("--heat-alpha", a.toFixed(3));
      } else {
        cell.classList.remove("heat");
        cell.style.removeProperty("--heat-alpha");
      }
    }
  }

  updateStatus();
}

function updateStatus() {
  els.status.classList.remove("thinking", "error", "win", "loss", "draw");
  if (state.terminated) {
    if (state.winner === -1) {
      els.status.textContent = "Draw.";
      els.status.classList.add("draw");
    } else if (state.winner === humanSide) {
      els.status.textContent = "You won.";
      els.status.classList.add("win");
    } else {
      els.status.textContent = "AI won.";
      els.status.classList.add("loss");
    }
    return;
  }
  if (busy) {
    els.status.textContent = "AI thinking…";
    els.status.classList.add("thinking");
  } else if (state.current_player === humanSide) {
    els.status.textContent = "Your turn.";
  } else {
    els.status.textContent = "AI's turn.";
  }
}

function showError(msg) {
  els.status.textContent = msg;
  els.status.classList.add("error");
}

// ---------------------------------------------------------------------------
// Click handling + AI move
// ---------------------------------------------------------------------------

async function onCellClick(ev) {
  if (busy || state.terminated) return;
  if (state.current_player !== humanSide) return;

  const b = +ev.currentTarget.dataset.board;
  const c = +ev.currentTarget.dataset.cell;
  const mask = legalMask(state);
  if (!mask[b][c]) return;

  step(state, [b, c]);
  lastPolicy = null;
  render();
  if (state.terminated) return;
  await aiMove();
}

async function aiMove() {
  busy = true;
  render();

  const payload = {
    state: {
      boards_x: state.boards_x.slice(),
      boards_o: state.boards_o.slice(),
      meta_x: state.meta_x,
      meta_o: state.meta_o,
      meta_draw: state.meta_draw,
      current_player: state.current_player,
      active_board: state.active_board,
    },
    model_id: els.model.value,
    num_simulations: currentSims(),
  };

  try {
    const r = await fetch("/api/move", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: r.statusText}));
      throw new Error(err.detail || "request failed");
    }
    const data = await r.json();
    step(state, data.action);
    lastPolicy = data.policy;
    renderIntrospection(data);
  } catch (e) {
    showError("Error: " + e.message);
  } finally {
    busy = false;
    render();
  }
}

function renderIntrospection(data) {
  // Server returns value/Q from the AI's POV (the side that just moved).
  // The value bar labels read "O wins ←→ X wins" so we always project to
  // X's POV — and Q follows the same convention so signs stay aligned no
  // matter which side the AI is playing.
  const aiSide = state.current_player ^ 1; // step() already toggled current_player
  const toXPov = (v) => (aiSide === 0 ? v : -v);

  const valueXPov = toXPov(data.value);
  els.valueNum.textContent = (valueXPov >= 0 ? "+" : "") + valueXPov.toFixed(2);
  const half = Math.abs(valueXPov) * 50;
  if (valueXPov >= 0) {
    els.valueFill.style.left = "50%";
    els.valueFill.style.width = half + "%";
  } else {
    els.valueFill.style.left = (50 - half) + "%";
    els.valueFill.style.width = half + "%";
  }

  els.topMoves.innerHTML = "";
  for (const m of data.top_moves) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.className = "move-label";
    label.textContent = `B${m.board}·C${m.cell}`;
    const visits = document.createElement("span");
    visits.textContent = (m.visits * 100).toFixed(0) + "%";
    const qXPov = toXPov(m.q);
    const q = document.createElement("span");
    q.textContent = "Q " + (qXPov >= 0 ? "+" : "") + qXPov.toFixed(2);
    li.append(label, visits, q);
    els.topMoves.appendChild(li);
  }

  els.elapsed.textContent = `${data.elapsed_ms} ms`;
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

let modelsById = {};

async function loadModels() {
  const r = await fetch("/api/models");
  const data = await r.json();
  els.model.innerHTML = "";
  modelsById = {};
  if (data.models.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "(no models loaded)";
    opt.disabled = true;
    els.model.appendChild(opt);
    els.newGame.disabled = true;
    showError("AlphaGumbel checkpoint not loaded — check the server log.");
    return;
  }

  // Single-kind deployment — flat list, no optgroups needed.
  for (const m of data.models) {
    modelsById[m.id] = m;
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label.replace(/^AlphaGumbel — /, "");
    els.model.appendChild(opt);
  }

  // Single-model deployment: hide the picker entirely so the UI doesn't
  // expose an interaction with no choices.
  if (data.models.length <= 1) {
    const row = els.model.closest(".panel-row");
    if (row) row.style.display = "none";
  }
}

function setDifficulty(level) {
  if (!(level in GUMBEL_SIMS)) return;
  difficulty = level;
  for (const b of els.difficulty.querySelectorAll("button")) {
    b.classList.toggle("active", b.dataset.level === level);
  }
}

function setSide(side) {
  humanSide = side === "X" ? 0 : 1;
  for (const b of els.playAs.querySelectorAll("button")) {
    b.classList.toggle("active", b.dataset.side === side);
  }
}

async function newGame() {
  state = initialState();
  lastPolicy = null;
  els.topMoves.innerHTML = "";
  els.valueNum.textContent = "—";
  els.valueFill.style.width = "0";
  els.valueFill.style.left = "50%";
  els.elapsed.textContent = "";
  render();
  // If human plays O, AI moves first.
  if (humanSide === 1) await aiMove();
}

function init() {
  buildBoard();
  els.showHeatmap.addEventListener("change", render);
  els.newGame.addEventListener("click", newGame);
  els.playAs.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    setSide(b.dataset.side);
  });
  els.difficulty.addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    setDifficulty(b.dataset.level);
  });
  loadModels().then(() => { newGame(); });
}

init();
