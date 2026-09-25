import { describe, expect, it } from "vitest";

import {
  DEFAULT_SENTENCE,
  DEFAULT_WK,
  DEFAULT_WQ,
  DEFAULT_WV,
  DEFAULT_X,
  MAX_TOKENS,
  applyCausalMask,
  argmax,
  argmaxMatrix,
  calculateAttention,
  calculateScores,
  causalMaskMatrix,
  createEmbeddings,
  createPositionalEncoding,
  matrixMultiply,
  scaleScores,
  softmax,
  tokenize,
  transpose,
  weightedSum,
} from "./attention";

const closeTo = (a: number[], b: number[]) =>
  a.forEach((v, i) => expect(v).toBeCloseTo(b[i] ?? NaN, 6));

describe("álgebra linear", () => {
  it("transpõe", () => {
    expect(
      transpose([
        [1, 2, 3],
        [4, 5, 6],
      ]),
    ).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
    expect(transpose([])).toEqual([]);
  });

  it("multiplica matrizes", () => {
    expect(
      matrixMultiply(
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ),
    ).toEqual([
      [19, 22],
      [43, 50],
    ]);
  });

  it("Q·Kᵀ sem transpor bate com a multiplicação explícita", () => {
    const Q = [
      [1, 2, 3],
      [0, 1, 0],
    ];
    const K = [
      [1, 0, 1],
      [2, 2, 2],
      [3, 1, 0],
    ];
    expect(calculateScores(Q, K)).toEqual(matrixMultiply(Q, transpose(K)));
  });

  it("soma ponderada", () => {
    expect(
      weightedSum(
        [0.5, 0.5],
        [
          [2, 4],
          [4, 8],
        ],
      ),
    ).toEqual([3, 6]);
  });

  it("argmax", () => {
    expect(argmax([1, 5, 3])).toBe(1);
    expect(
      argmaxMatrix([
        [1, 2],
        [9, 0],
      ]),
    ).toEqual([1, 0]);
  });
});

describe("softmax", () => {
  it("soma 1 e preserva a ordem", () => {
    const w = softmax([2, 4, 1, 3]);
    expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(argmax(w)).toBe(1);
  });

  it("é invariante a somar uma constante", () => {
    closeTo(softmax([1, 2, 3]), softmax([101, 102, 103]));
  });

  it("zera entradas −∞", () => {
    const w = softmax([1, -Infinity, 1]);
    expect(w[1]).toBe(0);
    expect(w[0]).toBeCloseTo(0.5, 10);
  });

  it("não explode com vetor só de −∞", () => {
    expect(softmax([-Infinity, -Infinity])).toEqual([0, 0]);
  });
});

describe("máscara e scaling", () => {
  it("mascara posições futuras", () => {
    expect(
      applyCausalMask([
        [1, 2],
        [3, 4],
      ]),
    ).toEqual([
      [1, -Infinity],
      [3, 4],
    ]);
    expect(causalMaskMatrix(3)).toEqual([
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
    ]);
  });

  it("divide por √dk", () => {
    expect(scaleScores([[4, 8]], 4)).toEqual([[2, 4]]);
  });
});

describe("positional encoding", () => {
  it("segue a fórmula seno/cosseno", () => {
    const PE = createPositionalEncoding(3, 4);
    expect(PE[0]).toEqual([0, 1, 0, 1]);
    closeTo(PE[1] ?? [], [Math.sin(1), Math.cos(1), Math.sin(1 / 100), Math.cos(1 / 100)]);
    closeTo(PE[2] ?? [], [Math.sin(2), Math.cos(2), Math.sin(2 / 100), Math.cos(2 / 100)]);
  });
});

describe("tokenização e embeddings", () => {
  it("normaliza e remove pontuação", () => {
    expect(tokenize("O cão, viu o GATO!").tokens).toEqual(["o", "cão", "viu", "o", "gato"]);
  });

  it("limita a MAX_TOKENS e avisa", () => {
    const r = tokenize("um dois três quatro cinco seis sete oito");
    expect(r.tokens).toHaveLength(MAX_TOKENS);
    expect(r.truncated).toBe(true);
    expect(tokenize("a b").truncated).toBe(false);
  });

  it("frase vazia cai no padrão", () => {
    expect(tokenize("   ").tokens).toEqual(["a", "anta", "comeu", "banana"]);
  });

  it("remove qualquer caractere de marcação", () => {
    expect(tokenize("<svg onload=alert(1)>").tokens.join(" ")).not.toMatch(/[<>=()]/);
  });

  it("embeddings são estáveis por palavra e usam DEFAULT_X na frase padrão", () => {
    expect(createEmbeddings(tokenize(DEFAULT_SENTENCE).tokens)).toEqual(DEFAULT_X);
    expect(createEmbeddings(tokenize("a anta comeu banana.").tokens)).toEqual(DEFAULT_X);
    const a = createEmbeddings(["gato"]);
    const b = createEmbeddings(["gato"]);
    expect(a).toEqual(b);
    expect(a[0]).toHaveLength(3);
  });
});

describe("pipeline completo", () => {
  const R = calculateAttention(DEFAULT_X, DEFAULT_WQ, DEFAULT_WK, DEFAULT_WV, 3, false);

  it("tem os formatos certos", () => {
    expect(R.Q).toHaveLength(4);
    expect(R.scores.map((r) => r.length)).toEqual([4, 4, 4, 4]);
    expect(R.output[0]).toHaveLength(3);
  });

  it("cada linha de pesos soma 1", () => {
    R.weights.forEach((row) => expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10));
  });

  it("os pesos padrão não saturam o softmax", () => {
    R.weights.forEach((row) => expect(Math.max(...row)).toBeLessThan(0.7));
  });

  it("com máscara causal, pesos futuros são 0", () => {
    const C = calculateAttention(DEFAULT_X, DEFAULT_WQ, DEFAULT_WK, DEFAULT_WV, 3, true);
    expect(C.weights[0]?.slice(1)).toEqual([0, 0, 0]);
    expect(C.weights[0]?.[0]).toBeCloseTo(1, 10);
  });

  it("output = A·V linha a linha", () => {
    R.output.forEach((row, i) => closeTo(row, weightedSum(R.weights[i] ?? [], R.V)));
  });
});
