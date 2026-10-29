const ROWS = 6, COLS = 7, EMPTY = 0, AI = 2, HUMAN = 1;
const SPEED_STEPS = [3000, 2000, 1000, 500, 250];
const SNAPSHOT_FRAME_MS = 30;
const TRACE_FRAMES = 60;
const TRACE_FRAME_MS = 25;
const DECISIVE_SCORE = 100000;

const state = {
  mode: null,
  board: null,
  currentToken: AI,
  playerToken: null,
  botAId: null,
  botBId: null,
  moveCount: 0,
  gameOver: false,
  botThinking: false,
  simulationHandle: null,
  simulationSpeed: 1000,
  record: { a: 0, draws: 0, b: 0 },
  bots: [],
  lastScores: null,
  lastKind: null,
  statusMeta: null,
  winningCells: [],
  // Every column played this game, human or bot, in order.
  moves: [],
  // Bumped on reset/back so late async responses from an old game are ignored.
  gameId: 0,
  overlayOn: false,
  cellHeat: null,
  analysis: null,
  scrubPly: 0,
  comparePending: false,
  // Bumped per minimax replay so an older animation stops writing into the panel.
  traceSeq: 0,
};

const els = {
  screens: {
    mode: document.getElementById("mode-screen"),
    setup: document.getElementById("setup-screen"),
    game: document.getElementById("game-screen"),
  },
  setupTitle: document.getElementById("setup-title"),
  setupContent: document.getElementById("setup-content"),
  board: document.getElementById("board"),
  boardIndex: document.getElementById("board-index"),
  heatmap: document.getElementById("heatmap"),
  heatmapIndex: document.getElementById("heatmap-index"),
  heatmapPanel: document.getElementById("heatmap-panel"),
  statusText: document.getElementById("status-text"),
  winnerBadge: document.getElementById("winner-badge"),
  moveCounter: document.getElementById("move-counter"),
  controlsPanel: document.getElementById("controls-panel"),
  recordText: document.getElementById("record-text"),
  backButton: document.getElementById("back-button"),
  resetButton: document.getElementById("reset-button"),
  nextButton: document.getElementById("next-button"),
  simulateButton: document.getElementById("simulate-button"),
  pauseButton: document.getElementById("pause-button"),
  speedSlider: document.getElementById("speed-slider"),
  speedLabel: document.getElementById("speed-label"),
  overlayToggle: document.getElementById("overlay-toggle"),
  overlayCaption: document.getElementById("overlay-caption"),
  analysisPanel: document.getElementById("analysis-panel"),
  searchPanel: document.getElementById("search-panel"),
  searchCaption: document.getElementById("search-caption"),
  searchList: document.getElementById("search-list"),
  compareButton: document.getElementById("compare-button"),
  comparePanel: document.getElementById("compare-panel"),
  compareCaption: document.getElementById("compare-caption"),
  compareBody: document.getElementById("compare-body"),
};

function initBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(EMPTY));
}

function lowestRow(board, col) {
  for (let row = ROWS - 1; row >= 0; row -= 1) {
    if (board[row][col] === EMPTY) return row;
  }
  return -1;
}

function placePiece(board, col, token) {
  const row = lowestRow(board, col);
  if (row === -1) return -1;
  board[row][col] = token;
  return row;
}

function checkWin(board, token) {
  return findWinningLine(board, token).length > 0;
}

function findWinningLine(board, token) {
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col <= COLS - 4; col += 1) {
      if ([0, 1, 2, 3].every((offset) => board[row][col + offset] === token)) {
        return [0, 1, 2, 3].map((offset) => [row, col + offset]);
      }
    }
  }
  for (let col = 0; col < COLS; col += 1) {
    for (let row = 0; row <= ROWS - 4; row += 1) {
      if ([0, 1, 2, 3].every((offset) => board[row + offset][col] === token)) {
        return [0, 1, 2, 3].map((offset) => [row + offset, col]);
      }
    }
  }
  for (let row = 0; row <= ROWS - 4; row += 1) {
    for (let col = 0; col <= COLS - 4; col += 1) {
      if ([0, 1, 2, 3].every((offset) => board[row + offset][col + offset] === token)) {
        return [0, 1, 2, 3].map((offset) => [row + offset, col + offset]);
      }
    }
  }
  for (let row = 3; row < ROWS; row += 1) {
    for (let col = 0; col <= COLS - 4; col += 1) {
      if ([0, 1, 2, 3].every((offset) => board[row - offset][col + offset] === token)) {
        return [0, 1, 2, 3].map((offset) => [row - offset, col + offset]);
      }
    }
  }
  return [];
}

function isFull(board) {
  return board[0].every((cell) => cell !== EMPTY);
}

function tokenLabel(token) {
  return token === AI ? "Red" : "Yellow";
}

function otherToken(token) {
  return token === AI ? HUMAN : AI;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `${url} failed (${response.status})`);
  return data;
}

function boardAfter(moves, count) {
  const board = initBoard();
  let token = AI;
  moves.slice(0, count).forEach((col) => {
    placePiece(board, col, token);
    token = otherToken(token);
  });
  return board;
}

function botLabel(botId) {
  return state.bots.find((bot) => bot.id === botId)?.label || botId;
}

function playerLabel(token) {
  if (state.mode === "pvp") return "Human";
  if (state.mode === "pvbot") return token === state.playerToken ? "Human" : botLabel(state.botAId);
  return botLabel(token === AI ? state.botAId : state.botBId);
}

function evaluatorLabel(evaluator) {
  return evaluator === "neural" ? "neural value net" : "minimax heuristic (no model loaded)";
}

function formatValue(value) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

// Red-perspective tint: red cells help Red, yellow cells help Yellow.
function heatTint(value, scale) {
  const alpha = 0.12 + 0.6 * Math.min(1, Math.abs(value) / scale);
  return value >= 0 ? `rgba(255, 68, 68, ${alpha.toFixed(3)})` : `rgba(255, 216, 77, ${alpha.toFixed(3)})`;
}

function heatScale(grid) {
  const values = grid.flat().filter((value) => value !== null).map(Math.abs);
  return Math.max(1e-6, ...values);
}

function showScreen(name) {
  Object.entries(els.screens).forEach(([key, node]) => {
    node.classList.toggle("active", key === name);
  });
}

function updateStatus(extra = "") {
  if (state.gameOver) return;
  els.statusText.parentElement.classList.remove("win-red", "win-yellow", "draw");
  els.winnerBadge.classList.add("hidden");
  const turn = `${tokenLabel(state.currentToken)}'s turn`;
  els.statusText.textContent = extra ? `${turn} / ${extra}` : turn;
  updateCompareButton();
}

function renderIndexes() {
  const labels = Array.from({ length: COLS }, (_, idx) => `<div>${idx + 1}</div>`).join("");
  els.boardIndex.innerHTML = labels;
  els.heatmapIndex.innerHTML = labels;
}

function renderBoard() {
  els.board.innerHTML = "";
  const heat = state.overlayOn ? state.cellHeat : null;
  const scale = heat ? heatScale(heat) : 1;
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      const button = document.createElement("button");
      button.className = "cell";
      const piece = document.createElement("div");
      piece.className = "piece";
      if (state.board[row][col] === AI) piece.classList.add("red");
      if (state.board[row][col] === HUMAN) piece.classList.add("yellow");
      button.appendChild(piece);
      if (state.winningCells.some(([winRow, winCol]) => winRow === row && winCol === col)) {
        button.classList.add("winning");
      }
      if (heat && heat[row][col] !== null) {
        button.style.background = heatTint(heat[row][col], scale);
        button.title = `${formatValue(heat[row][col])} to Red's value`;
      }

      const interactive = !state.gameOver
        && !state.botThinking
        && (state.mode === "pvp" || (state.mode === "pvbot" && state.currentToken === state.playerToken));
      if (!interactive || lowestRow(state.board, col) === -1) {
        button.classList.add("disabled");
        button.disabled = true;
      } else {
        button.addEventListener("click", () => handleColumnClick(col));
      }
      els.board.appendChild(button);
    }
  }
  els.moveCounter.textContent = `Move: ${state.moveCount}`;
}

function formatScore(score, kind) {
  if (kind === "mcts") return String(Math.round(score));
  if (kind === "neural") return score.toFixed(2);
  if (Math.abs(score) >= DECISIVE_SCORE) return score > 0 ? "win" : "loss";
  return String(Math.round(score));
}

function renderHeatmap(scores = null, kind = null, maxOverride = null) {
  els.heatmap.innerHTML = "";
  const usableScores = scores || Array(COLS).fill(null);
  const captionEl = document.getElementById('score-caption');
  if (!scores) {
    captionEl.textContent = 'Scores unavailable for this agent';
  } else if (kind === 'mcts') {
    captionEl.textContent = 'MCTS simulations through each column';
  } else if (kind === 'neural') {
    captionEl.textContent = 'Neural value after each move';
  } else {
    captionEl.textContent = 'One-move estimates, not win probabilities';
  }
  const numeric = usableScores.filter((value) => value !== null);
  const maxScore = maxOverride ?? (numeric.length ? Math.max(...numeric) : 1);
  const minScore = maxOverride !== null ? 0 : (numeric.length ? Math.min(...numeric) : 0);
  const uniform = maxOverride === null && numeric.length && numeric.every((value) => value === numeric[0]);

  usableScores.forEach((score) => {
    const cell = document.createElement("div");
    cell.className = "heat-cell";
    if (state.mode === "pvbot" && state.currentToken === state.playerToken) {
      cell.classList.add("dimmed");
    }
    if (score === null) {
      cell.style.background = "#424242";
    } else {
      const normalized = maxScore > minScore ? (score - minScore) / (maxScore - minScore) : 0;
      const intensity = uniform ? 0.2 : Math.max(0.08, normalized);
      const r = Math.round(20  + 220 * intensity);
      const g = Math.round(40  + 152 * intensity);
      const b = Math.round(80  - 16  * intensity);
      cell.style.background = `rgb(${r}, ${g}, ${b})`;
      cell.textContent = formatScore(score, kind);
    }
    els.heatmap.appendChild(cell);
  });
}

// Replays the search's visit counts so the bars grow the way the tree did.
async function animateSnapshots(snapshots) {
  const gameId = state.gameId;
  const finalMax = Math.max(1, ...snapshots[snapshots.length - 1]);
  for (const snapshot of snapshots) {
    if (gameId !== state.gameId) return;
    renderHeatmap(snapshot.map((visits) => (visits > 0 ? visits : null)), "mcts", finalMax);
    await sleep(SNAPSHOT_FRAME_MS);
  }
}

function setThinking(on) {
  state.botThinking = on;
  document.getElementById('thinking-indicator').classList.toggle('active', on);
}

async function requestMove(botId) {
  const gameId = state.gameId;
  setThinking(true);
  renderBoard();
  updateStatus("Bot thinking...");
  try {
    return await postJSON("/api/move", { board: state.board, token: state.currentToken, bot_id: botId });
  } finally {
    if (gameId === state.gameId) {
      setThinking(false);
      renderBoard();
    }
  }
}

function dropPiece(col, token) {
  const row = placePiece(state.board, col, token);
  if (row === -1) return -1;
  state.moves.push(col);
  state.cellHeat = null;
  state.moveCount += 1;
  renderBoard();
  updateStatus();
  return row;
}

function checkGameOver(token) {
  const winningLine = findWinningLine(state.board, token);
  if (winningLine.length) {
    state.gameOver = true;
    state.winningCells = winningLine;
    els.statusText.textContent = `${tokenLabel(token)} wins with 4 in a row`;
    els.winnerBadge.textContent = `${tokenLabel(token)} Win`;
    els.winnerBadge.classList.remove("hidden");
    els.statusText.parentElement.classList.add(token === AI ? "win-red" : "win-yellow");
    if (state.mode === "bvb") {
      if (token === AI) state.record.a += 1;
      else state.record.b += 1;
      updateRecord();
      stopSimulation();
    }
    renderBoard();
    onGameOver(token);
    return true;
  }
  if (isFull(state.board)) {
    state.gameOver = true;
    state.winningCells = [];
    els.statusText.textContent = "Draw. Board is full";
    els.winnerBadge.textContent = "Draw";
    els.winnerBadge.classList.remove("hidden");
    els.statusText.parentElement.classList.add("draw");
    if (state.mode === "bvb") {
      state.record.draws += 1;
      updateRecord();
      stopSimulation();
    }
    renderBoard();
    onGameOver(token);
    return true;
  }
  return false;
}

function onGameOver(lastMover) {
  updateCompareButton();
  refreshCellHeat(otherToken(lastMover));
  runAnalysis();
}

function updateRecord() {
  els.recordText.innerHTML =
    `<span style="color:var(--red)">${state.record.a}</span> · ` +
    `<span style="color:var(--muted-hi)">${state.record.draws} draws</span> · ` +
    `<span style="color:var(--yellow)">${state.record.b}</span>`;
}

function hideInsightPanels() {
  els.analysisPanel.classList.add("hidden");
  els.analysisPanel.innerHTML = "";
  els.searchPanel.classList.add("hidden");
  els.comparePanel.classList.add("hidden");
}

function resetBoardState() {
  state.gameId += 1;
  state.board = initBoard();
  state.currentToken = AI;
  state.moveCount = 0;
  state.gameOver = false;
  setThinking(false);
  state.lastScores = null;
  state.lastKind = null;
  state.winningCells = [];
  state.moves = [];
  state.cellHeat = null;
  state.analysis = null;
  els.statusText.parentElement.classList.remove("win-red", "win-yellow", "draw");
  els.winnerBadge.classList.add("hidden");
  hideInsightPanels();
  renderBoard();
  renderHeatmap();
  updateStatus();
  refreshCellHeat(state.currentToken);
}

// ── Value overlay over the board ──

async function refreshCellHeat(toMove) {
  if (!state.overlayOn || !state.mode) return;
  const gameId = state.gameId;
  const moveCount = state.moves.length;
  try {
    const body = await postJSON("/api/heatmap", { board: state.board, token: toMove });
    if (!state.overlayOn || gameId !== state.gameId || moveCount !== state.moves.length) return;
    state.cellHeat = body.cells;
    els.overlayCaption.textContent = `${evaluatorLabel(body.evaluator)} · red helps Red, yellow helps Yellow`;
    renderBoard();
  } catch (error) {
    if (state.overlayOn) els.overlayCaption.textContent = `Overlay unavailable: ${error.message}`;
  }
}

function setOverlay(on) {
  state.overlayOn = on;
  state.cellHeat = null;
  els.overlayCaption.textContent = on ? "Evaluating…" : "Off";
  renderBoard();
  if (on) refreshCellHeat(state.gameOver ? otherToken(state.currentToken) : state.currentToken);
}

// ── Minimax search replay ──

async function showSearchTrace(board, token, moveNumber) {
  const gameId = state.gameId;
  state.traceSeq += 1;
  const seq = state.traceSeq;
  const current = () => gameId === state.gameId && seq === state.traceSeq;
  let trace;
  try {
    trace = await postJSON("/api/minimax-trace", { board, token });
  } catch (error) {
    return;
  }
  if (!current()) return;
  els.searchPanel.classList.remove("hidden");
  els.searchCaption.textContent =
    `Move ${moveNumber} (${tokenLabel(token)}) · depth-${trace.depth} replay; the bot searches depth ${trace.play_depth}`
    + (trace.truncated ? ` · first ${trace.events.length} events` : "");

  const rows = {};
  trace.root.forEach((entry) => {
    rows[entry.col] = { leaves: 0, pruned: 0, score: null, entry };
  });
  const maxNodes = Math.max(1, ...trace.root.map((entry) => entry.nodes));
  const render = (activeCol, finished) => {
    els.searchList.innerHTML = Object.keys(rows).map(Number).sort((a, b) => a - b).map((col) => {
      const row = rows[col];
      const nodes = finished ? row.entry.nodes : row.leaves;
      const pruned = finished ? row.entry.pruned : row.pruned;
      const score = finished ? row.entry.score : row.score;
      const classes = ["search-row"];
      if (col === activeCol && !finished) classes.push("active");
      if (finished && col === trace.best_col) classes.push("best");
      return `<div class="${classes.join(" ")}">
        <span class="search-col">${col + 1}</span>
        <span class="search-bar"><span style="width:${(100 * nodes / maxNodes).toFixed(1)}%"></span></span>
        <span class="search-stat" title="${finished ? "nodes searched" : "positions evaluated so far"}">${nodes}</span>
        <span class="search-stat pruned" title="moves skipped by alpha-beta cut-offs">✂ ${pruned}</span>
        <span class="search-score">${score === null ? "…" : formatScore(score, "minimax")}</span>
      </div>`;
    }).join("");
  };

  const batch = Math.max(1, Math.ceil(trace.events.length / TRACE_FRAMES));
  for (let start = 0; start < trace.events.length; start += batch) {
    if (!current()) return;
    let activeCol = null;
    trace.events.slice(start, start + batch).forEach((event) => {
      const row = rows[event.path[0]];
      activeCol = event.path[0];
      if (event.kind === "leaf") row.leaves += 1;
      if (event.kind === "prune") row.pruned += event.score;
      if (event.kind === "done") row.score = event.score;
    });
    render(activeCol, false);
    await sleep(TRACE_FRAME_MS);
  }
  if (current()) render(null, true);
}

// ── Ask every agent ──

function updateCompareButton() {
  els.compareButton.disabled = !state.mode || state.gameOver || state.botThinking || state.comparePending;
}

async function compareAgents() {
  const gameId = state.gameId;
  const moveCount = state.moves.length;
  const token = state.currentToken;
  state.comparePending = true;
  els.compareButton.disabled = true;
  els.compareButton.textContent = "Asking…";
  els.comparePanel.classList.remove("hidden");
  els.compareCaption.textContent = "Every available agent is choosing a move…";
  els.compareBody.innerHTML = "";
  try {
    const rows = await postJSON("/api/compare-position", { board: state.board, token });
    if (gameId !== state.gameId) return;
    if (moveCount !== state.moves.length) {
      els.compareCaption.textContent = "The position changed before every agent answered; ask again.";
      return;
    }
    const votes = {};
    rows.forEach((row) => { votes[row.col] = (votes[row.col] || 0) + 1; });
    const top = Math.max(...Object.values(votes));
    els.compareCaption.textContent = `After move ${moveCount}, ${tokenLabel(token)} to play`;
    els.compareBody.innerHTML = rows.map((row) => `
      <tr class="${votes[row.col] === top && top > 1 ? "consensus" : ""}">
        <td>${escapeHtml(row.label)}</td>
        <td>${row.col + 1}</td>
        <td>${row.elapsed_ms < 10 ? row.elapsed_ms.toFixed(1) : Math.round(row.elapsed_ms)}</td>
      </tr>`).join("");
  } catch (error) {
    els.compareCaption.textContent = `Comparison failed: ${error.message}`;
  } finally {
    state.comparePending = false;
    els.compareButton.textContent = "Ask every agent";
    updateCompareButton();
  }
}

// ── Post-game analysis ──

async function runAnalysis() {
  const gameId = state.gameId;
  const moves = state.moves.slice();
  els.analysisPanel.classList.remove("hidden");
  els.analysisPanel.innerHTML = `<div class="panel-header"><h3>Game analysis</h3><span class="panel-caption">Evaluating every position…</span></div>`;
  try {
    const analysis = await postJSON("/api/analyze", { moves });
    if (gameId !== state.gameId) return;
    state.analysis = { ...analysis, moves };
    state.scrubPly = analysis.pivot === null ? moves.length : analysis.pivot + 1;
    renderAnalysis();
  } catch (error) {
    if (gameId !== state.gameId) return;
    state.analysis = null;
    els.analysisPanel.innerHTML = `<div class="panel-header"><h3>Game analysis</h3>
      <button class="secondary-button" data-action="export">Export game</button></div>
      <p class="analysis-note">Analysis unavailable: ${escapeHtml(error.message)}</p>`;
  }
}

function miniBoard(board, heat, highlight) {
  const scale = heat ? heatScale(heat) : 1;
  let html = '<div class="mini-board">';
  for (let row = 0; row < ROWS; row += 1) {
    for (let col = 0; col < COLS; col += 1) {
      const cell = board[row][col];
      const value = heat ? heat[row][col] : null;
      const style = value !== null ? ` style="background:${heatTint(value, scale)}"` : "";
      const title = value !== null ? ` title="${formatValue(value)}"` : "";
      const mark = highlight && highlight[0] === row && highlight[1] === col ? " last" : "";
      const piece = cell === AI ? "red" : cell === HUMAN ? "yellow" : "";
      html += `<div class="mini-cell${mark}"${style}${title}><span class="mini-piece ${piece}"></span></div>`;
    }
  }
  return `${html}</div>`;
}

function landingCell(moves, ply) {
  if (ply === 0) return null;
  const before = boardAfter(moves, ply - 1);
  return [lowestRow(before, moves[ply - 1]), moves[ply - 1]];
}

// Positions with Red to move and with Yellow to move are drawn as separate lines:
// the value net favours the side to move, so only same-colour points compare cleanly.
function valueChart(values, pivotIndex) {
  const width = 560, height = 150, pad = 12;
  const step = (width - 2 * pad) / Math.max(1, values.length - 1);
  const x = (i) => pad + i * step;
  const y = (v) => height / 2 - v * (height / 2 - pad);
  const series = (parity) => values
    .map((v, i) => [i, v])
    .filter(([i]) => i % 2 === parity)
    .map(([i, v]) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`)
    .join(" ");
  const dots = values.map((v, i) => `<circle class="chart-dot ${i % 2 === 0 ? "red-to-move" : "yellow-to-move"}${i === pivotIndex ? " pivot" : ""}" data-ply="${i}"
      cx="${x(i).toFixed(1)}" cy="${y(v).toFixed(1)}" r="${i === pivotIndex ? 6 : 3.5}"><title>After move ${i}: ${formatValue(v)} (${tokenLabel(i % 2 === 0 ? AI : HUMAN)} to move)</title></circle>`).join("");
  return `<svg class="value-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img"
      aria-label="Red's value after each move, split by side to move">
    <line class="chart-zero" x1="${pad}" x2="${width - pad}" y1="${y(0)}" y2="${y(0)}"></line>
    <text class="chart-label" x="${pad}" y="${pad + 4}">Red winning</text>
    <text class="chart-label" x="${pad}" y="${height - 4}">Yellow winning</text>
    <line class="chart-cursor" id="chart-cursor" y1="${pad}" y2="${height - pad}"></line>
    <polyline class="chart-line red-to-move" points="${series(0)}"></polyline>
    <polyline class="chart-line yellow-to-move" points="${series(1)}"></polyline>
    ${dots}
  </svg>
  <div class="chart-legend"><span class="legend red-to-move">Red to move</span><span class="legend yellow-to-move">Yellow to move</span></div>`;
}

function renderAnalysis() {
  const { plies, values, pivot, heatmaps, evaluator, moves } = state.analysis;
  const pivotPly = pivot === null ? null : plies[pivot];
  // Compare with the position two moves earlier: same side to move, so the net's turn bias cancels.
  const before = pivotPly ? (pivotPly.swing === null ? pivotPly.ply - 1 : pivotPly.ply - 2) : 0;
  const movesInSwing = pivotPly ? moves.slice(before, pivotPly.ply).map((col) => col + 1).join(" then ") : "";
  const pivotSummary = pivotPly
    ? `Biggest swing ends at move ${pivotPly.ply} (columns ${movesInSwing}): with ${tokenLabel(pivotPly.ply % 2 === 0 ? AI : HUMAN)} to move both times, Red's value went ${formatValue(values[before])} → ${formatValue(values[pivotPly.ply])}.`
    : "No moves to analyse.";
  const positionName = (index) => (index === 0 ? "Start" : `After move ${index}`);
  const pivotBoards = pivotPly ? `
    <div class="pivot-boards">
      <figure>${miniBoard(boardAfter(moves, before), heatmaps[before], landingCell(moves, before))}
        <figcaption>${positionName(before)} · ${formatValue(values[before])}</figcaption></figure>
      <figure>${miniBoard(boardAfter(moves, pivotPly.ply), heatmaps[pivotPly.ply], landingCell(moves, pivotPly.ply))}
        <figcaption>${positionName(pivotPly.ply)} · ${formatValue(values[pivotPly.ply])}</figcaption></figure>
    </div>` : "";
  const chips = plies.map((ply, index) => `
    <button class="timeline-move${index === pivot ? " pivot" : ""}" data-ply="${ply.ply}" title="Red value ${formatValue(ply.value)}; swing vs. two moves earlier">
      <span class="move-num">${ply.ply}</span>
      <span class="move-dot ${ply.token === AI ? "red" : "yellow"}"></span>
      <span class="move-col">${ply.col + 1}</span>
      <span class="move-score">${ply.swing === null ? "—" : formatValue(ply.swing)}</span>
    </button>`).join("");

  els.analysisPanel.innerHTML = `
    <div class="panel-header">
      <h3>Game analysis</h3>
      <span class="panel-caption">${escapeHtml(evaluatorLabel(evaluator))}</span>
      <button class="secondary-button" data-action="export">Export game</button>
    </div>
    <p class="analysis-note">${pivotSummary} Tinted cells show how much each piece (or each next move) shifts Red's value.</p>
    ${valueChart(values, pivotPly ? pivotPly.ply : -1)}
    ${pivotBoards}
    <div class="scrubber">
      <div class="scrub-board" id="scrub-board"></div>
      <div class="scrub-controls">
        <div class="control-row">
          <button class="secondary-button" data-action="prev">◀</button>
          <input id="scrub-range" type="range" min="0" max="${moves.length}" step="1" value="${state.scrubPly}">
          <button class="secondary-button" data-action="next">▶</button>
        </div>
        <div class="scrub-label" id="scrub-label"></div>
      </div>
    </div>
    <div class="move-timeline">${chips}</div>`;
  updateScrub();
}

function updateScrub() {
  const { values, heatmaps, moves, plies } = state.analysis;
  const ply = state.scrubPly;
  document.getElementById("scrub-board").innerHTML = miniBoard(boardAfter(moves, ply), heatmaps[ply], landingCell(moves, ply));
  document.getElementById("scrub-range").value = String(ply);
  const mover = ply === 0 ? "Start position" : `Move ${ply}: ${tokenLabel(plies[ply - 1].token)} played column ${moves[ply - 1] + 1}`;
  document.getElementById("scrub-label").textContent = `${mover} · Red's value ${formatValue(values[ply])}`;
  const dot = els.analysisPanel.querySelector(`.chart-dot[data-ply="${ply}"]`);
  const cursor = document.getElementById("chart-cursor");
  cursor.setAttribute("x1", dot.getAttribute("cx"));
  cursor.setAttribute("x2", dot.getAttribute("cx"));
  els.analysisPanel.querySelectorAll(".timeline-move").forEach((chip) => {
    chip.classList.toggle("active", Number(chip.dataset.ply) === ply);
  });
}

function setScrub(ply) {
  if (!state.analysis) return;
  state.scrubPly = Math.max(0, Math.min(state.analysis.moves.length, ply));
  updateScrub();
}

function exportGame() {
  const winnerToken = state.winningCells.length ? state.board[state.winningCells[0][0]][state.winningCells[0][1]] : null;
  const analysis = state.analysis;
  const record = {
    schema: "connect4-game/1",
    exported_at: new Date().toISOString(),
    mode: state.mode,
    red: playerLabel(AI),
    yellow: playerLabel(HUMAN),
    moves: state.moves.slice(),
    columns: "0-based, left to right; Red moves first",
    winner: winnerToken === AI ? "red" : winnerToken === HUMAN ? "yellow" : null,
    analysis: analysis ? {
      evaluator: analysis.evaluator,
      red_values: analysis.values,
      pivot_move: analysis.pivot === null ? null : analysis.plies[analysis.pivot].ply,
    } : null,
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `connect4-game-${record.exported_at.replace(/[:.]/g, "-")}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function resetStateForMode(mode) {
  if (state.simulationHandle) stopSimulation();
  state.mode = mode;
  state.playerToken = null;
  state.botAId = null;
  state.botBId = null;
  resetBoardState();
  els.controlsPanel.classList.toggle("hidden", mode !== "bvb");
  els.heatmapPanel.classList.toggle("hidden", mode === "pvp");
}

function buildBotOptions() {
  return state.bots.map((bot) => {
    const suffix = bot.available ? "" : ` (unavailable: ${bot.reason})`;
    return `<option value="${bot.id}" ${bot.available ? "" : "disabled"}>${bot.label}${suffix}</option>`;
  }).join("");
}

function selectedAvailableBot(select) {
  const option = Array.from(select.options).find((item) => !item.disabled);
  if (option && !select.value) select.value = option.value;
}

function renderSetup(mode) {
  showScreen("setup");
  els.setupTitle.textContent = mode === "pvbot" ? "Player vs Bot" : "Bot vs Bot";
  if (mode === "pvbot") {
    els.setupContent.innerHTML = `
      <div class="setup-form">
        <div class="setup-row">
          <label>Choose your colour</label>
          <div class="choice-row">
            <button class="choice-button active" data-token="${AI}">Red</button>
            <button class="choice-button" data-token="${HUMAN}">Yellow</button>
          </div>
        </div>
        <div class="setup-row">
          <label for="bot-select">Choose bot</label>
          <select id="bot-select">${buildBotOptions()}</select>
        </div>
        <button id="start-game" class="primary-button">Start Game</button>
      </div>
    `;
    let chosenToken = AI;
    els.setupContent.querySelectorAll(".choice-button").forEach((button) => {
      button.addEventListener("click", () => {
        chosenToken = Number(button.dataset.token);
        els.setupContent.querySelectorAll(".choice-button").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
      });
    });
    const botSelect = document.getElementById("bot-select");
    selectedAvailableBot(botSelect);
    document.getElementById("start-game").addEventListener("click", () => {
      resetStateForMode("pvbot");
      state.playerToken = chosenToken;
      state.botAId = botSelect.value;
      startGame();
    });
  } else {
    els.setupContent.innerHTML = `
      <div class="setup-form">
        <div class="setup-row">
          <label for="bot-a-select">Bot A (Red)</label>
          <select id="bot-a-select">${buildBotOptions()}</select>
        </div>
        <div class="setup-row">
          <label for="bot-b-select">Bot B (Yellow)</label>
          <select id="bot-b-select">${buildBotOptions()}</select>
        </div>
        <button id="start-game" class="primary-button">Start Game</button>
      </div>
    `;
    const botASelect = document.getElementById("bot-a-select");
    const botBSelect = document.getElementById("bot-b-select");
    selectedAvailableBot(botASelect);
    selectedAvailableBot(botBSelect);
    document.getElementById("start-game").addEventListener("click", () => {
      resetStateForMode("bvb");
      state.botAId = botASelect.value;
      state.botBId = botBSelect.value;
      startGame();
    });
  }
}

function finishTurn() {
  state.currentToken = otherToken(state.currentToken);
  renderBoard();
  updateStatus();
  refreshCellHeat(state.currentToken);
}

async function handleColumnClick(col) {
  if (dropPiece(col, state.currentToken) === -1) return;
  if (checkGameOver(state.currentToken)) return;
  finishTurn();
  if (state.mode === "pvbot") {
    await takeBotTurn(state.botAId);
  }
}

// One bot move: ask the server, replay its thinking, then play the column.
async function playBotMove(botId) {
  const gameId = state.gameId;
  const boardBefore = state.board.map((row) => row.slice());
  const token = state.currentToken;
  const movesBefore = state.moves.length;
  const moveNumber = state.moveCount + 1;
  // Anything that changed the game while we waited makes this answer stale.
  const stale = () => gameId !== state.gameId || state.moves.length !== movesBefore || state.currentToken !== token;
  const { col, scores, kind, snapshots } = await requestMove(botId);
  if (stale()) return false;
  state.lastScores = scores;
  state.lastKind = kind || null;
  if (kind === "minimax") showSearchTrace(boardBefore, token, moveNumber);
  if (kind === "mcts" && snapshots && snapshots.length) {
    setThinking(true);
    await animateSnapshots(snapshots);
    if (gameId !== state.gameId) return false;
    setThinking(false);
    if (stale()) return false;
  }
  renderHeatmap(scores, kind);
  dropPiece(col, token);
  if (checkGameOver(token)) return false;
  finishTurn();
  return true;
}

async function playBotMoveSafely(botId) {
  const gameId = state.gameId;
  try {
    await playBotMove(botId);
  } catch (error) {
    if (gameId !== state.gameId) return;
    stopSimulation();
    setThinking(false);
    renderBoard();
    els.statusText.textContent = `Bot move failed: ${error.message}`;
    updateCompareButton();
  }
}

async function takeBotTurn(botId) {
  if (state.gameOver) return;
  await playBotMoveSafely(botId);
}

async function stepBotVsBot() {
  if (state.gameOver || state.botThinking) return;
  await playBotMoveSafely(state.currentToken === AI ? state.botAId : state.botBId);
}

function stopSimulation() {
  if (state.simulationHandle) {
    clearInterval(state.simulationHandle);
    state.simulationHandle = null;
  }
}

function startSimulation() {
  stopSimulation();
  state.simulationHandle = setInterval(() => {
    stepBotVsBot();
  }, state.simulationSpeed);
}

function goBack() {
  stopSimulation();
  state.gameId += 1;
  state.mode = null;
  state.board = initBoard();
  state.currentToken = AI;
  state.playerToken = null;
  state.botAId = null;
  state.botBId = null;
  state.moveCount = 0;
  state.gameOver = false;
  setThinking(false);
  state.lastScores = null;
  state.winningCells = [];
  state.moves = [];
  state.cellHeat = null;
  state.analysis = null;
  hideInsightPanels();
  els.controlsPanel.classList.add("hidden");
  els.heatmapPanel.classList.remove("hidden");
  showScreen("mode");
}

function startGame() {
  showScreen("game");
  renderBoard();
  renderHeatmap();
  updateStatus();
  refreshCellHeat(state.currentToken);
  if (state.mode === "pvbot" && state.playerToken !== state.currentToken) {
    takeBotTurn(state.botAId);
  }
}

async function loadInitialData() {
  const [botsResponse, statusResponse] = await Promise.all([
    fetch("/api/bots"),
    fetch("/api/status"),
  ]);
  state.bots = await botsResponse.json();
  state.statusMeta = await statusResponse.json();
}

function bindEvents() {
  document.querySelectorAll(".mode-card").forEach((button) => {
    button.addEventListener("click", () => {
      const mode = button.dataset.mode;
      if (mode === "pvp") {
        resetStateForMode("pvp");
        startGame();
      } else {
        renderSetup(mode);
      }
    });
  });
  els.backButton.addEventListener("click", goBack);
  els.resetButton.addEventListener("click", () => {
    if (state.mode !== "bvb") return;
    resetBoardState();
  });
  els.nextButton.addEventListener("click", () => {
    if (state.mode !== "bvb") return;
    stepBotVsBot();
  });
  els.simulateButton.addEventListener("click", () => {
    if (state.mode !== "bvb" || state.gameOver) return;
    startSimulation();
  });
  els.pauseButton.addEventListener("click", stopSimulation);
  els.speedSlider.addEventListener("input", () => {
    const index = Number(els.speedSlider.value);
    state.simulationSpeed = SPEED_STEPS[index];
    els.speedLabel.textContent = `${(state.simulationSpeed / 1000).toFixed(state.simulationSpeed < 1000 ? 2 : 1)}s`;
    if (state.simulationHandle) startSimulation();
  });
  els.overlayToggle.addEventListener("change", () => setOverlay(els.overlayToggle.checked));
  els.compareButton.addEventListener("click", compareAgents);
  els.analysisPanel.addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "export") exportGame();
    if (action === "prev") setScrub(state.scrubPly - 1);
    if (action === "next") setScrub(state.scrubPly + 1);
    const plyTarget = event.target.closest("[data-ply]");
    if (plyTarget) setScrub(Number(plyTarget.dataset.ply));
  });
  els.analysisPanel.addEventListener("input", (event) => {
    if (event.target.id === "scrub-range") setScrub(Number(event.target.value));
  });
}

async function boot() {
  renderIndexes();
  resetBoardState();
  updateRecord();
  bindEvents();
  await loadInitialData();
}

boot().catch((error) => {
  els.statusText.textContent = `Failed to load viewer: ${error.message}`;
});

window.render_game_to_text = () => JSON.stringify({
  mode: state.mode,
  board: state.board,
  coordinates: 'row 0 top; columns 0..6 left to right; 0 empty, 2 red, 1 yellow',
  currentToken: state.currentToken,
  moveCount: state.moveCount,
  gameOver: state.gameOver,
  thinking: state.botThinking,
  botA: state.botAId,
  botB: state.botBId,
  moves: state.moves,
});
// Turn-based game: server responses and clicks advance play, not elapsed time.
window.advanceTime = async () => {};
