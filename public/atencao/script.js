/* ============================================================
   Self-Attention — laboratório educacional
   JavaScript puro (ES6+). Toda a matemática roda no navegador.

   Organização:
     1) Utilidades de álgebra linear (transpose, matrixMultiply, dotProduct…)
     2) Pipeline de atenção (positionalEncoding, calculateQKV, calculateScores, scaleScores,
        applyCausalMask, softmax, calculateAttention)
     3) Renderização (renderMatrix, renderHeatmap, …)
     4) Estado + recálculo reativo
     5) Navegação de módulos, animação (animateStep) e exercícios
   ============================================================ */

/* ---------- 1. ÁLGEBRA LINEAR ---------- */

/** Produto escalar entre dois vetores de mesmo tamanho. */
function dotProduct(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Transposta de uma matriz (linhas viram colunas). */
function transpose(M) {
  return M[0].map((_, j) => M.map((row) => row[j]));
}

/** Multiplicação de matrizes A(n×m) · B(m×p) = C(n×p). */
function matrixMultiply(A, B) {
  const Bt = transpose(B);
  return A.map((row) => Bt.map((col) => dotProduct(row, col)));
}

/** Soma ponderada de vetores: Σ w[j]·V[j]. */
function weightedSum(weights, V) {
  const out = new Array(V[0].length).fill(0);
  weights.forEach((w, j) => V[j].forEach((v, c) => (out[c] += w * v)));
  return out;
}

/** Gerador pseudoaleatório determinístico a partir de uma string. */
function seededRandom(seed) {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return () => {
    h += 0x6d2b79f5;
    let t = h;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- 2. PIPELINE DE ATENÇÃO ---------- */

/** Tokenização didática: minúsculas, sem pontuação, no máximo 6 tokens. */
function tokenize(sentence) {
  const t = sentence
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
  return t.length ? t.slice(0, 6) : ["a", "anta", "comeu", "banana"];
}

/** Embeddings didáticos de 3 dimensões, estáveis para a mesma palavra. */
const DEFAULT_SENTENCE = "A Anta comeu banana";
const DEFAULT_SENTENCE_KEY = DEFAULT_SENTENCE.toLowerCase();
const DEFAULT_X = [
  [0.2, 0.7, 0.1],
  [0.9, 0.1, 0.3],
  [0.4, 0.8, 0.6],
  [0.1, 0.3, 0.9],
];

function createEmbeddings(tokens, sentence) {
  if (sentence && sentence.trim().toLowerCase() === DEFAULT_SENTENCE_KEY) {
    return DEFAULT_X.map((r) => r.slice());
  }
  return tokens.map((tk) => {
    const rnd = seededRandom(tk);
    return [0, 1, 2].map(() => Math.round(rnd() * 100) / 100);
  });
}

/** Positional encoding senoidal: PE(pos,2i)=sen(...), PE(pos,2i+1)=cos(...). */
function createPositionalEncoding(rows, cols) {
  return Array.from({ length: rows }, (_, pos) =>
    Array.from({ length: cols }, (_, dim) => {
      const exponent = (2 * Math.floor(dim / 2)) / cols;
      const angle = pos / Math.pow(10000, exponent);
      return dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    })
  );
}

function addMatrices(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

/** Projeções lineares Q = X·W_Q, K = X·W_K, V = X·W_V. */
function calculateQKV(X, WQ, WK, WV) {
  return {
    Q: matrixMultiply(X, WQ),
    K: matrixMultiply(X, WK),
    V: matrixMultiply(X, WV),
  };
}

/** Scores brutos de compatibilidade: Q·Kᵀ. */
function calculateScores(Q, K) {
  return matrixMultiply(Q, transpose(K));
}

/** Divide os scores por √d_k. */
function scaleScores(S, dk) {
  const f = Math.sqrt(dk);
  return S.map((r) => r.map((v) => v / f));
}

/** Máscara causal: posições futuras (j > i) recebem -Infinity. */
function applyCausalMask(S) {
  return S.map((row, i) => row.map((v, j) => (j > i ? -Infinity : v)));
}

/** Softmax numericamente estável de um vetor. */
function softmax(vec) {
  const max = Math.max(...vec.filter((v) => Number.isFinite(v)));
  const exps = vec.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

/** Pipeline completo: devolve todas as etapas intermediárias. */
function calculateAttention(X, WQ, WK, WV, dk, causal) {
  const PE = createPositionalEncoding(X.length, X[0].length);
  const Xpos = addMatrices(X, PE);
  const { Q, K, V } = calculateQKV(Xpos, WQ, WK, WV);
  const scores = calculateScores(Q, K);
  let scaled = scaleScores(scores, dk);
  const masked = causal ? applyCausalMask(scaled) : scaled;
  const weights = masked.map(softmax);
  const output = matrixMultiply(weights, V);
  return { PE, Xpos, Q, K, V, scores, scaled, masked, weights, output };
}

/* ---------- 3. ESTADO ---------- */

const state = {
  sentence: DEFAULT_SENTENCE,
  tokens: tokenize(DEFAULT_SENTENCE),
  X: DEFAULT_X.map((r) => r.slice()),
  WQ: [
    [2.4, -0.6, 0.3],
    [-0.4, 2.2, 0.8],
    [0.5, -0.3, 2.5],
  ],
  WK: [
    [2.3, 0.4, -0.7],
    [0.6, 2.4, 0.3],
    [-0.5, 0.7, 2.2],
  ],
  WV: [
    [0.5, 0.4, 0.1],
    [0.3, 0.6, 0.2],
    [0.2, 0.3, 0.7],
  ],
  dk: 3,
  causal: false,
  module: 1,
  playing: false,
  speed: 1,
  selectedRow: 0,
  demoSoftmax: false,
};

let R = {}; // resultados do pipeline

/* ---------- 4. HELPERS DE RENDER ---------- */

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const fmt = (v) => (Number.isFinite(v) ? v.toFixed(2) : "−∞");

/** Desenha uma matriz numérica, opcionalmente com inputs editáveis. */
function renderMatrix(el, M, opts = {}) {
  if (!el) return;
  const { rowLabels, colLabels, editable, onEdit } = opts;
  const t = document.createElement("table");

  if (colLabels) {
    const tr = document.createElement("tr");
    if (rowLabels) tr.appendChild(document.createElement("th"));
    colLabels.forEach((c) => {
      const th = document.createElement("th");
      th.textContent = c;
      tr.appendChild(th);
    });
    t.appendChild(tr);
  }

  M.forEach((row, i) => {
    const tr = document.createElement("tr");
    if (rowLabels) {
      const th = document.createElement("th");
      th.textContent = rowLabels[i];
      tr.appendChild(th);
    }
    row.forEach((v, j) => {
      const td = document.createElement("td");
      if (editable) {
        const inp = document.createElement("input");
        inp.type = "number";
        inp.step = "0.1";
        inp.value = v;
        inp.setAttribute("aria-label", `linha ${i + 1} coluna ${j + 1}`);
        inp.addEventListener("input", () => onEdit(i, j, parseFloat(inp.value) || 0));
        td.appendChild(inp);
      } else {
        td.textContent = fmt(v);
      }
      tr.appendChild(td);
    });
    t.appendChild(tr);
  });

  el.innerHTML = "";
  el.appendChild(t);
}

/** Cor do heatmap: ciano (baixo) → azul → violeta (alto). */
function heatColor(t) {
  const hue = 186 + t * 84; // 186 (ciano) → 270 (violeta)
  const light = 16 + t * 34;
  return `hsl(${hue} 85% ${light}%)`;
}

/** Desenha um heatmap clicável com rótulos de tokens. */
function renderHeatmap(el, M, opts = {}) {
  if (!el) return;
  const labels = state.tokens;
  const finite = M.flat().filter(Number.isFinite);
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  const span = max - min || 1;

  const t = document.createElement("table");
  const head = document.createElement("tr");
  head.appendChild(document.createElement("th"));
  labels.forEach((l) => {
    const th = document.createElement("th");
    th.textContent = l;
    head.appendChild(th);
  });
  t.appendChild(head);

  M.forEach((row, i) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.className = "rowh";
    th.textContent = labels[i];
    th.tabIndex = 0;
    if (opts.onRow) {
      const pick = () => opts.onRow(i);
      th.addEventListener("click", pick);
      th.addEventListener("keydown", (e) => e.key === "Enter" && pick());
      if (opts.selectedRow === i) th.classList.add("sel");
    }
    tr.appendChild(th);

    row.forEach((v, j) => {
      const td = document.createElement("td");
      const norm = Number.isFinite(v) ? (v - min) / span : 0;
      td.style.background = Number.isFinite(v) ? heatColor(norm) : "rgba(255,107,127,.15)";
      td.textContent = fmt(v);
      if (opts.selectedRow != null && opts.selectedRow !== i) td.classList.add("dim");
      td.addEventListener("mouseenter", () => opts.onHover && opts.onHover(i, j, v));
      td.addEventListener("click", () => opts.onCell && opts.onCell(i, j, v));
      tr.appendChild(td);
    });
    t.appendChild(tr);
  });

  el.innerHTML = "";
  el.appendChild(t);
}

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), 2200);
}

/* ---------- 5. RENDER DOS MÓDULOS ---------- */

function renderModule1() {
  $("#fraseM1").textContent = state.sentence;
  const tr = $("#tokensM1");
  const ar = $("#tokenArrows");
  const ep = $("#embPreview");
  tr.innerHTML = ar.innerHTML = ep.innerHTML = "";
  state.tokens.forEach((tk, i) => {
    const d = document.createElement("div");
    d.className = "token";
    d.style.animationDelay = i * 0.12 + "s";
    d.textContent = tk;
    tr.appendChild(d);

    const a = document.createElement("span");
    a.textContent = "↓";
    a.style.animationDelay = 0.3 + i * 0.12 + "s";
    ar.appendChild(a);

    const c = document.createElement("div");
    c.className = "embchip";
    c.style.animationDelay = 0.5 + i * 0.12 + "s";
    c.textContent = "[" + state.X[i].map(fmt).join(", ") + "]";
    ep.appendChild(c);
  });
}

function renderModule2() {
  $("#dimX").textContent = `(${state.tokens.length}×3)`;
  renderMatrix($("#matX"), state.X, {
    rowLabels: state.tokens,
    colLabels: ["d₁", "d₂", "d₃"],
    editable: true,
    onEdit: (i, j, v) => {
      state.X[i][j] = v;
      recompute({ skip: "matX" });
    },
  });
}

function renderModulePositional() {
  const rl = { rowLabels: state.tokens, colLabels: ["d₁", "d₂", "d₃"] };
  renderMatrix($("#matPosX"), state.X, rl);
  renderMatrix($("#matPE"), R.PE, rl);
  renderMatrix($("#matXPos"), R.Xpos, rl);
}

function renderModule3() {
  const wOpts = (name) => ({
    editable: true,
    onEdit: (i, j, v) => {
      state[name][i][j] = v;
      recompute({ skip: "w" + name });
    },
  });
  renderMatrix($("#matWQ"), state.WQ, wOpts("WQ"));
  renderMatrix($("#matWK"), state.WK, wOpts("WK"));
  renderMatrix($("#matWV"), state.WV, wOpts("WV"));
  const rl = { rowLabels: state.tokens };
  renderMatrix($("#matQ"), R.Q, rl);
  renderMatrix($("#matK"), R.K, rl);
  renderMatrix($("#matV"), R.V, rl);
}

function renderModule4() {
  renderHeatmap($("#heatScores"), R.scores, {
    onHover: (i, j, v) =>
      ($("#tipScores").textContent =
        `Query: "${state.tokens[i]}"\nKey: "${state.tokens[j]}"\nScore = Q${i + 1}·K${j + 1} = ${fmt(v)}`),
    onCell: (i, j, v) => checkExercise2(v),
  });
}

function renderModule5() {
  $("#dkVal").textContent = state.dk;
  $("#sqrtVal").textContent = Math.sqrt(state.dk).toFixed(2);
  $("#dk").value = state.dk;
  const o = { rowLabels: state.tokens, colLabels: state.tokens };
  renderMatrix($("#matScoresRaw"), R.scores, o);
  renderMatrix($("#matScoresScaled"), R.scaled, o);
}

function renderModule6() {
  const sel = $("#softmaxRow");
  if (sel.options.length !== state.tokens.length) {
    sel.innerHTML = "";
    state.tokens.forEach((t, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = `linha de "${t}"`;
      sel.appendChild(o);
    });
  }
  sel.value = state.selectedRow;

  const labels = state.demoSoftmax ? ["a", "b", "c", "d"] : state.tokens;
  const xs = state.demoSoftmax ? [2, 4, 1, 3] : R.masked[state.selectedRow];
  const w = softmax(xs);
  const max = Math.max(...xs.filter(Number.isFinite));
  const exps = xs.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));

  const body = $("#softBody");
  body.innerHTML = "";
  xs.forEach((x, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${labels[i]}</td><td>${fmt(x)}</td><td>${exps[i].toFixed(3)}</td>` +
      `<td>${w[i].toFixed(3)}</td><td><div class="bar" style="width:0"></div></td>`;
    body.appendChild(tr);
    requestAnimationFrame(() => {
      tr.querySelector(".bar").style.width = Math.max(2, w[i] * 100) + "%";
    });
  });

  $("#sumLine").textContent =
    w.map((v) => v.toFixed(2)).join(" + ") +
    " = " +
    w.reduce((a, b) => a + b, 0).toFixed(2);
}

function renderModule7() {
  renderHeatmap($("#heatWeights"), R.weights, {
    selectedRow: state.selectedRow,
    onRow: (i) => {
      state.selectedRow = i;
      renderModule7();
      renderModule6();
      renderModule8();
    },
    onHover: (i, j, v) =>
      ($("#tipWeights").textContent =
        `"${state.tokens[i]}" olha para "${state.tokens[j]}" com peso ${fmt(v)}`),
    onCell: (i, j, v) => {
      state.selectedRow = i;
      checkExercise5(i, j);
      renderModule7();
    },
  });
  drawLinks();
}

/** Desenha em SVG as conexões token→token com espessura proporcional ao peso. */
function drawLinks() {
  const svg = $("#linksSvg");
  const n = state.tokens.length;
  const row = R.weights[state.selectedRow];
  const H = 260;
  const gap = H / (n + 1);
  let out = "";

  state.tokens.forEach((tk, j) => {
    const y = gap * (j + 1);
    out += `<line class="lnk" x1="96" y1="${gap * (state.selectedRow + 1)}" x2="224" y2="${y}"
      stroke-width="${0.6 + row[j] * 9}" stroke-opacity="${0.15 + row[j] * 0.85}" />`;
  });
  state.tokens.forEach((tk, j) => {
    const y = gap * (j + 1);
    out += `<rect x="8" y="${y - 14}" width="88" height="28" rx="8"/><text x="52" y="${y + 4}" text-anchor="middle">${tk}</text>`;
    out += `<rect x="224" y="${y - 14}" width="88" height="28" rx="8"/><text x="268" y="${y + 4}" text-anchor="middle">${tk}</text>`;
  });
  svg.innerHTML = out;
}

function renderModule8() {
  const sel = $("#sumToken");
  if (sel.options.length !== state.tokens.length) {
    sel.innerHTML = "";
    state.tokens.forEach((t, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = t;
      sel.appendChild(o);
    });
  }
  sel.value = state.selectedRow;

  const w = R.weights[state.selectedRow];
  const box = $("#sumSteps");
  box.innerHTML = "";
  w.forEach((wi, j) => {
    const d = document.createElement("div");
    d.style.animationDelay = j * 0.1 + "s";
    d.textContent = `${wi.toFixed(2)} × V(${state.tokens[j]}) = [${R.V[j].map((v) => (v * wi).toFixed(2)).join(", ")}]`;
    box.appendChild(d);
  });
  const res = weightedSum(w, R.V);
  $("#sumResult").textContent = `${state.tokens[state.selectedRow]} → [${res.map(fmt).join(", ")}]`;
}

function renderModule10() {
  const n = state.tokens.length;
  const mask = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (j > i ? 0 : 1))
  );
  const el = $("#maskMat");
  renderMatrix(el, mask, { rowLabels: state.tokens, colLabels: state.tokens });
  el.querySelectorAll("td").forEach((td) => {
    td.classList.add(td.textContent === "1.00" ? "allow" : "block");
    td.textContent = td.textContent === "1.00" ? "1" : "0";
  });

  const causalScaled = applyCausalMask(R.scaled);
  renderMatrix($("#maskedScores"), causalScaled, {
    rowLabels: state.tokens,
    colLabels: state.tokens,
  });
  renderHeatmap($("#maskedWeights"), causalScaled.map(softmax));
  $("#causal").checked = state.causal;
  $("#causal2").checked = state.causal;
}

const HEAD_DEFS = [
  { name: "Head 1", role: "relação sintática", shift: 0.15 },
  { name: "Head 2", role: "relação semântica", shift: -0.2 },
  { name: "Head 3", role: "relação de posição", shift: 0.4 },
  { name: "Head 4", role: "outra relação contextual", shift: -0.35 },
];

function renderModule11() {
  const box = $("#heads");
  box.innerHTML = "";
  HEAD_DEFS.forEach((h, idx) => {
    // Cada cabeça usa uma variação determinística dos pesos: projeções diferentes.
    const twist = (M) => M.map((r, i) => r.map((v, j) => v + h.shift * ((i + j + idx) % 3 === 0 ? 1 : -0.4)));
    const res = calculateAttention(state.X, twist(state.WQ), twist(state.WK), state.WV, state.dk, state.causal);
    const div = document.createElement("div");
    div.className = "head";
    div.innerHTML = `<h5>${h.name}</h5><p>${h.role}</p>`;
    const hm = document.createElement("div");
    hm.className = "heatmap tiny";
    div.appendChild(hm);
    box.appendChild(div);
    renderHeatmap(hm, res.weights);
  });
}

function renderModule12() {
  $("#frase").value = state.sentence;
  const tr = $("#labTokens");
  tr.innerHTML = "";
  state.tokens.forEach((t, i) => {
    const d = document.createElement("div");
    d.className = "token";
    d.style.animationDelay = i * 0.08 + "s";
    d.textContent = t;
    tr.appendChild(d);
  });
  const rl = { rowLabels: state.tokens };
  const sq = { rowLabels: state.tokens, colLabels: state.tokens };
  renderMatrix($("#labX"), state.X, rl);
  renderMatrix($("#labPE"), R.PE, rl);
  renderMatrix($("#labXPos"), R.Xpos, rl);
  renderMatrix($("#labQ"), R.Q, rl);
  renderMatrix($("#labK"), R.K, rl);
  renderMatrix($("#labV"), R.V, rl);
  renderMatrix($("#labS"), R.scores, sq);
  renderMatrix($("#labSc"), R.masked, sq);
  renderHeatmap($("#labA"), R.weights);
  renderMatrix($("#labO"), R.output, rl);
}

/* ---------- 6. RECÁLCULO REATIVO ---------- */

/** Recalcula o pipeline inteiro e redesenha tudo. */
function recompute(opts = {}) {
  if (state.selectedRow >= state.tokens.length) state.selectedRow = 0;
  R = calculateAttention(state.X, state.WQ, state.WK, state.WV, state.dk, state.causal);

  renderModule1();
  if (opts.skip !== "matX") renderModule2();
  renderModulePositional();
  if (!String(opts.skip || "").startsWith("w")) renderModule3();
  else {
    const rl = { rowLabels: state.tokens };
    renderMatrix($("#matQ"), R.Q, rl);
    renderMatrix($("#matK"), R.K, rl);
    renderMatrix($("#matV"), R.V, rl);
  }
  renderModule4();
  renderModule5();
  renderModule6();
  renderModule7();
  renderModule8();
  renderModule10();
  renderModule11();
  renderModule12();
}

/* ---------- 7. NAVEGAÇÃO E ANIMAÇÃO ---------- */

const MODULE_TITLES = [
  "O problema",
  "Embeddings",
  "Positional Encoding",
  "Query, Key e Value",
  "Produto QKᵀ",
  "Scaling",
  "Softmax",
  "Attention Weights",
  "Weighted Sum",
  "A fórmula completa",
  "Atenção causal",
  "Multi-Head Attention",
  "Simulação interativa",
];
const TOTAL = MODULE_TITLES.length;

function buildNav() {
  const nav = $("#nav");
  MODULE_TITLES.forEach((t, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<b>${i + 1}</b>${t}`;
    li.tabIndex = 0;
    li.addEventListener("click", () => goTo(i + 1));
    li.addEventListener("keydown", (e) => e.key === "Enter" && goTo(i + 1));
    nav.appendChild(li);
  });
}

/** Mostra um módulo, atualiza nav, barra de progresso e destaque do fluxo. */
function animateStep(n) {
  state.module = Math.min(TOTAL, Math.max(1, n));
  $$(".module").forEach((s, i) => s.classList.toggle("active", i === state.module - 1));
  $$("#nav li").forEach((li, i) => li.classList.toggle("active", i === state.module - 1));
  $("#progressBar").style.width = (state.module / TOTAL) * 100 + "%";
  $("#progressLabel").textContent = `Etapa ${state.module} de ${TOTAL} — ${MODULE_TITLES[state.module - 1]}`;

  const stage = $$(".module")[state.module - 1].dataset.stage;
  $$("#pipeline li").forEach((li) => li.classList.toggle("on", li.dataset.stage === stage));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function goTo(n) {
  animateStep(n);
  if (state.module === TOTAL) stopPlay();
}

function play() {
  if (state.playing) return;
  state.playing = true;
  const tick = () => {
    if (!state.playing) return;
    if (state.module >= TOTAL) return stopPlay();
    animateStep(state.module + 1);
    state._timer = setTimeout(tick, 3800 / state.speed);
  };
  state._timer = setTimeout(tick, 2600 / state.speed);
  toast("Executando a explicação passo a passo…");
}

function stopPlay() {
  state.playing = false;
  clearTimeout(state._timer);
}

/** Volta tudo ao estado inicial. */
function resetSimulation() {
  stopPlay();
  state.sentence = DEFAULT_SENTENCE;
  state.tokens = tokenize(DEFAULT_SENTENCE);
  state.X = DEFAULT_X.map((r) => r.slice());
  state.dk = 3;
  state.causal = false;
  state.selectedRow = 0;
  state.demoSoftmax = false;
  $("#dk").value = 3;
  $$(".exercise").forEach((ex) => {
    ex.querySelector(".fb").textContent = "";
    ex.querySelectorAll(".opt").forEach((o) => o.classList.remove("right", "wrong"));
  });
  recompute();
  animateStep(1);
  toast("Simulação reiniciada.");
}

/* ---------- 8. EXERCÍCIOS ---------- */

function feedback(ex, ok, msg) {
  const fb = ex.querySelector(".fb");
  fb.textContent = ok ? "✓ Correto! " + (msg || "") : "✗ Tente novamente. " + (msg || "");
  fb.className = "fb " + (ok ? "ok" : "no");
}

/** Exercícios 1, 3 e 4: múltipla escolha. */
function wireChoiceExercises() {
  $$(".exercise .opt").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ex = btn.closest(".exercise");
      const ok = btn.dataset.correct === "1";
      ex.querySelectorAll(".opt").forEach((o) => o.classList.remove("right", "wrong"));
      btn.classList.add(ok ? "right" : "wrong");
      feedback(ex, ok);
    });
  });
}

/** Exercício 2: clicar na célula de maior score. */
function checkExercise2(v) {
  const ex = document.querySelector('[data-ex="2"]');
  const max = Math.max(...R.scores.flat());
  feedback(ex, Math.abs(v - max) < 1e-9, `Maior score = ${fmt(max)}.`);
}

/** Exercício 5: clicar no maior peso da última linha. */
function checkExercise5(i, j) {
  const ex = document.querySelector('[data-ex="5"]');
  const last = state.tokens.length - 1;
  const row = R.weights[last];
  const best = row.indexOf(Math.max(...row));
  feedback(
    ex,
    i === last && j === best,
    `"${state.tokens[last]}" olha mais para "${state.tokens[best]}" (${fmt(row[best])}).`
  );
}

/* ---------- 9. LIGAÇÃO DOS CONTROLES ---------- */

function wireControls() {
  $("#btnPrev").addEventListener("click", () => { stopPlay(); goTo(state.module - 1); });
  $("#btnNext").addEventListener("click", () => { stopPlay(); goTo(state.module + 1); });
  $("#btnPlay").addEventListener("click", play);
  $("#btnPause").addEventListener("click", () => { stopPlay(); toast("Pausado."); });
  $("#btnReset").addEventListener("click", resetSimulation);

  $("#speed").addEventListener("input", (e) => {
    state.speed = parseFloat(e.target.value);
    $("#speedVal").textContent = state.speed + "×";
  });

  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.key === "ArrowRight") { stopPlay(); goTo(state.module + 1); }
    if (e.key === "ArrowLeft") { stopPlay(); goTo(state.module - 1); }
    if (e.key === " ") { e.preventDefault(); state.playing ? stopPlay() : play(); }
  });

  $("#btnResetX").addEventListener("click", () => {
    state.X = createEmbeddings(state.tokens, state.sentence);
    recompute();
  });
  $("#btnRandX").addEventListener("click", () => {
    state.X = state.X.map((r) => r.map(() => Math.round(Math.random() * 100) / 100));
    recompute();
  });

  $("#dk").addEventListener("input", (e) => {
    state.dk = parseInt(e.target.value, 10);
    recompute();
  });

  $("#softmaxRow").addEventListener("change", (e) => {
    state.selectedRow = parseInt(e.target.value, 10);
    state.demoSoftmax = false;
    renderModule6();
    renderModule7();
    renderModule8();
  });
  $("#btnDemoRow").addEventListener("click", () => {
    state.demoSoftmax = !state.demoSoftmax;
    renderModule6();
  });

  $("#sumToken").addEventListener("change", (e) => {
    state.selectedRow = parseInt(e.target.value, 10);
    renderModule8();
    renderModule7();
    renderModule6();
  });

  const setCausal = (v) => { state.causal = v; recompute(); };
  $("#causal").addEventListener("change", (e) => setCausal(e.target.checked));
  $("#causal2").addEventListener("change", (e) => setCausal(e.target.checked));

  $$(".buildup .step").forEach((s) =>
    s.addEventListener("click", () => goTo(parseInt(s.dataset.goto, 10)))
  );

  const run = () => {
    state.sentence = $("#frase").value.trim() || DEFAULT_SENTENCE;
    state.tokens = tokenize(state.sentence);
    state.X = createEmbeddings(state.tokens, state.sentence);
    state.selectedRow = 0;
    recompute();
    toast(`Recalculado para ${state.tokens.length} tokens.`);
  };
  $("#btnRun").addEventListener("click", run);
  $("#frase").addEventListener("keydown", (e) => e.key === "Enter" && run());
}

/* ---------- 10. TEMA CLARO/ESCURO ---------- */

function initTheme() {
  const root = document.documentElement;
  const saved = localStorage.getItem("attention-theme");
  const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  const theme = saved || (prefersLight ? "light" : "dark");
  root.setAttribute("data-theme", theme);
  updateThemeIcon(theme);

  $("#btnTheme").addEventListener("click", () => {
    const current = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", current);
    localStorage.setItem("attention-theme", current);
    updateThemeIcon(current);
  });
}

function updateThemeIcon(theme) {
  $("#themeIcon").textContent = theme === "light" ? "☀️" : "🌙";
  $("#btnTheme").setAttribute("aria-label", theme === "light" ? "Mudar para tema escuro" : "Mudar para tema claro");
}

/* ---------- 11. BOOT ---------- */

buildNav();
wireControls();
wireChoiceExercises();
initTheme();
recompute();
animateStep(1);
