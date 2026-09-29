const ROWS = 6, COLS = 7, EMPTY = 0, AI = 2, HUMAN = 1;
const SPEED_STEPS = [3000, 2000, 1000, 500, 250];

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
  statusMeta: null,
  winningCells: [],
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
}

function renderIndexes() {
  const labels = Array.from({ length: COLS }, (_, idx) => `<div>${idx + 1}</div>`).join("");
  els.boardIndex.innerHTML = labels;
  els.heatmapIndex.innerHTML = labels;
}

function renderBoard() {
  els.board.innerHTML = "";
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

function renderHeatmap(scores = null) {
  els.heatmap.innerHTML = "";
  const usableScores = scores || Array(COLS).fill(null);
  document.getElementById('score-caption').textContent = scores
    ? 'One-move estimates, not win probabilities'
    : 'Scores unavailable for this agent';
  const numeric = usableScores.filter((value) => value !== null);
  const maxScore = numeric.length ? Math.max(...numeric) : 1;
  const minScore = numeric.length ? Math.min(...numeric) : 0;
  const uniform = numeric.length && numeric.every((value) => value === numeric[0]);

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
    }
    els.heatmap.appendChild(cell);
  });
}

async function requestMove(botId) {
  state.botThinking = true;
  document.getElementById('thinking-indicator').classList.add('active');
  renderBoard();
  updateStatus("Bot thinking...");
  try {
    const response = await fetch("/api/move", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ board: state.board, token: state.currentToken, bot_id: botId }),
    });
    if (!response.ok) throw new Error("Move request failed");
    return await response.json();
  } finally {
    state.botThinking = false;
    document.getElementById('thinking-indicator').classList.remove('active');
    renderBoard();
  }
}

function dropPiece(col, token) {
  const row = placePiece(state.board, col, token);
  if (row === -1) return -1;
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
    return true;
  }
  return false;
}

function updateRecord() {
  els.recordText.innerHTML =
    `<span style="color:var(--red)">${state.record.a}</span> · ` +
    `<span style="color:var(--muted-hi)">${state.record.draws} draws</span> · ` +
    `<span style="color:var(--yellow)">${state.record.b}</span>`;
}

function resetBoardState() {
  state.board = initBoard();
  state.currentToken = AI;
  state.moveCount = 0;
  state.gameOver = false;
  state.botThinking = false;
  state.lastScores = null;
  state.winningCells = [];
  els.statusText.parentElement.classList.remove("win-red", "win-yellow", "draw");
  els.winnerBadge.classList.add("hidden");
  renderBoard();
  renderHeatmap();
  updateStatus();
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

async function handleColumnClick(col) {
  if (dropPiece(col, state.currentToken) === -1) return;
  if (checkGameOver(state.currentToken)) return;
  state.currentToken = state.currentToken === AI ? HUMAN : AI;
  updateStatus();
  if (state.mode === "pvbot") {
    await takeBotTurn(state.botAId);
  }
}

async function takeBotTurn(botId) {
  if (state.gameOver) return;
  const { col, scores } = await requestMove(botId);
  state.lastScores = scores;
  renderHeatmap(scores);
  dropPiece(col, state.currentToken);
  if (checkGameOver(state.currentToken)) return;
  state.currentToken = state.currentToken === AI ? HUMAN : AI;
  renderBoard();
  renderHeatmap(state.lastScores);
  updateStatus();
}

async function stepBotVsBot() {
  if (state.gameOver || state.botThinking) return;
  const currentBotId = state.currentToken === AI ? state.botAId : state.botBId;
  const { col, scores } = await requestMove(currentBotId);
  state.lastScores = scores;
  renderHeatmap(scores);
  dropPiece(col, state.currentToken);
  if (checkGameOver(state.currentToken)) return;
  state.currentToken = state.currentToken === AI ? HUMAN : AI;
  renderBoard();
  updateStatus();
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
  state.mode = null;
  state.board = initBoard();
  state.currentToken = AI;
  state.playerToken = null;
  state.botAId = null;
  state.botBId = null;
  state.moveCount = 0;
  state.gameOver = false;
  state.botThinking = false;
  state.lastScores = null;
  state.winningCells = [];
  els.controlsPanel.classList.add("hidden");
  els.heatmapPanel.classList.remove("hidden");
  showScreen("mode");
}

function startGame() {
  showScreen("game");
  renderBoard();
  renderHeatmap();
  updateStatus();
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
});
// Turn-based game: server responses and clicks advance play, not elapsed time.
window.advanceTime = async () => {};
