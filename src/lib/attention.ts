/*
  Self-Attention — matemática pura, sem DOM.
  Tudo aqui é determinístico e testável (ver attention.test.ts).
*/

export type Vector = number[];
export type Matrix = number[][];

/** Dimensão didática dos embeddings. Modelos reais usam centenas ou milhares. */
export const DIM = 3;
/** Rótulos das colunas de X, PE e X + PE. */
export const DIM_LABELS = ["d₁", "d₂", "d₃"];
/** Limite de tokens para as matrizes continuarem legíveis na tela. */
export const MAX_TOKENS = 6;

/* ---------- álgebra linear ---------- */

export function dotProduct(a: Vector, b: Vector): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] ?? 0) * (b[i] ?? 0);
  return s;
}

export function transpose(M: Matrix): Matrix {
  const first = M[0];
  if (!first) return [];
  return first.map((_, j) => M.map((row) => row[j] ?? 0));
}

/** A(n×m) · B(m×p) = C(n×p). */
export function matrixMultiply(A: Matrix, B: Matrix): Matrix {
  const Bt = transpose(B);
  return A.map((row) => Bt.map((col) => dotProduct(row, col)));
}

/** Σ_j w[j]·V[j] — combinação convexa das linhas de V quando w vem do softmax. */
export function weightedSum(weights: Vector, V: Matrix): Vector {
  const width = V[0]?.length ?? 0;
  const out: Vector = new Array<number>(width).fill(0);
  weights.forEach((w, j) => {
    const row = V[j];
    if (!row) return;
    row.forEach((v, c) => {
      out[c] = (out[c] ?? 0) + w * v;
    });
  });
  return out;
}

export function addMatrices(A: Matrix, B: Matrix): Matrix {
  return A.map((row, i) => row.map((v, j) => v + (B[i]?.[j] ?? 0)));
}

export function cloneMatrix(M: Matrix): Matrix {
  return M.map((r) => r.slice());
}

/** Índice do maior valor de um vetor (primeiro em caso de empate). */
export function argmax(v: Vector): number {
  let best = 0;
  for (let i = 1; i < v.length; i++) if ((v[i] ?? -Infinity) > (v[best] ?? -Infinity)) best = i;
  return best;
}

/** Posição [i, j] do maior valor de uma matriz. */
export function argmaxMatrix(M: Matrix): [number, number] {
  let best: [number, number] = [0, 0];
  let max = -Infinity;
  M.forEach((row, i) =>
    row.forEach((v, j) => {
      if (v > max) {
        max = v;
        best = [i, j];
      }
    }),
  );
  return best;
}

/** Gerador pseudoaleatório determinístico (FNV-1a + mulberry32) a partir de uma string. */
export function seededRandom(seed: string): () => number {
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

/* ---------- valores padrão ---------- */

export const DEFAULT_SENTENCE = "A Anta comeu banana";
const DEFAULT_SENTENCE_KEY = DEFAULT_SENTENCE.toLowerCase();

export const DEFAULT_X: Matrix = [
  [0.2, 0.7, 0.1],
  [0.9, 0.1, 0.3],
  [0.4, 0.8, 0.6],
  [0.1, 0.3, 0.9],
];

/*
  Magnitude escolhida para que, com o positional encoding somado, os pesos de
  atenção fiquem distribuídos (máximo por linha entre ~0,35 e ~0,60). Com pesos
  maiores o softmax satura e a "média ponderada" vira uma cópia de um token só.
*/
export const DEFAULT_WQ: Matrix = [
  [1.2, -0.3, 0.2],
  [-0.2, 1.1, 0.4],
  [0.3, -0.1, 1.3],
];
export const DEFAULT_WK: Matrix = [
  [1.2, 0.2, -0.3],
  [0.3, 1.2, 0.2],
  [-0.2, 0.4, 1.1],
];
export const DEFAULT_WV: Matrix = [
  [0.5, 0.4, 0.1],
  [0.3, 0.6, 0.2],
  [0.2, 0.3, 0.7],
];

/* ---------- pipeline ---------- */

export interface TokenizeResult {
  tokens: string[];
  /** true quando a frase tinha mais palavras do que MAX_TOKENS. */
  truncated: boolean;
}

/** Tokenização didática: minúsculas, sem pontuação, no máximo MAX_TOKENS tokens. */
export function tokenize(sentence: string): TokenizeResult {
  const all = sentence
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
  if (all.length === 0) return { tokens: tokenize(DEFAULT_SENTENCE).tokens, truncated: false };
  return { tokens: all.slice(0, MAX_TOKENS), truncated: all.length > MAX_TOKENS };
}

/** Embeddings didáticos: estáveis para a mesma palavra; a frase padrão usa DEFAULT_X. */
export function createEmbeddings(tokens: string[], sentence: string): Matrix {
  if (sentence.trim().toLowerCase() === DEFAULT_SENTENCE_KEY) return cloneMatrix(DEFAULT_X);
  return tokens.map((tk) => {
    const rnd = seededRandom(tk);
    return Array.from({ length: DIM }, () => Math.round(rnd() * 100) / 100);
  });
}

/**
 * Positional encoding senoidal (Vaswani et al., 2017):
 *   PE(pos, 2i)   = sen(pos / 10000^(2i/d))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
 */
export function createPositionalEncoding(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, (_, pos) =>
    Array.from({ length: cols }, (_, dim) => {
      const exponent = (2 * Math.floor(dim / 2)) / cols;
      const angle = pos / Math.pow(10000, exponent);
      return dim % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }),
  );
}

export function calculateQKV(X: Matrix, WQ: Matrix, WK: Matrix, WV: Matrix) {
  return {
    Q: matrixMultiply(X, WQ),
    K: matrixMultiply(X, WK),
    V: matrixMultiply(X, WV),
  };
}

/** Scores brutos S[i][j] = Q_i · K_j (equivale a Q·Kᵀ sem transpor de fato). */
export function calculateScores(Q: Matrix, K: Matrix): Matrix {
  return Q.map((q) => K.map((k) => dotProduct(q, k)));
}

export function scaleScores(S: Matrix, dk: number): Matrix {
  const f = Math.sqrt(dk);
  return S.map((r) => r.map((v) => v / f));
}

/** Máscara causal: posições futuras (j > i) recebem −∞. */
export function applyCausalMask(S: Matrix): Matrix {
  return S.map((row, i) => row.map((v, j) => (j > i ? -Infinity : v)));
}

/** Matriz 0/1 da máscara causal, para exibição. */
export function causalMaskMatrix(n: number): Matrix {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (j > i ? 0 : 1)));
}

/** Softmax numericamente estável; entradas −∞ viram peso 0. */
export function softmax(vec: Vector): Vector {
  const finite = vec.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return vec.map(() => 0);
  const max = Math.max(...finite);
  const exps = vec.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

/** Exponenciais deslocadas usadas pelo softmax (para mostrar a conta na tela). */
export function softmaxExps(vec: Vector): Vector {
  const finite = vec.filter((v) => Number.isFinite(v));
  const max = finite.length ? Math.max(...finite) : 0;
  return vec.map((v) => (Number.isFinite(v) ? Math.exp(v - max) : 0));
}

export interface AttentionResult {
  PE: Matrix;
  Xpos: Matrix;
  Q: Matrix;
  K: Matrix;
  V: Matrix;
  scores: Matrix;
  scaled: Matrix;
  masked: Matrix;
  weights: Matrix;
  output: Matrix;
}

/** Pipeline completo, devolvendo todas as etapas intermediárias. */
export function calculateAttention(
  X: Matrix,
  WQ: Matrix,
  WK: Matrix,
  WV: Matrix,
  dk: number,
  causal: boolean,
): AttentionResult {
  const PE = createPositionalEncoding(X.length, X[0]?.length ?? DIM);
  const Xpos = addMatrices(X, PE);
  const { Q, K, V } = calculateQKV(Xpos, WQ, WK, WV);
  const scores = calculateScores(Q, K);
  const scaled = scaleScores(scores, dk);
  const masked = causal ? applyCausalMask(scaled) : scaled;
  const weights = masked.map(softmax);
  const output = matrixMultiply(weights, V);
  return { PE, Xpos, Q, K, V, scores, scaled, masked, weights, output };
}

/** Formata um número com 2 casas; −∞ vira o símbolo. */
export function fmt(v: number): string {
  return Number.isFinite(v) ? v.toFixed(2) : "−∞";
}
