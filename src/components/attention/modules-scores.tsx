import { useState } from "react";

import { DIM, argmaxMatrix, causalMaskMatrix, fmt, softmax, softmaxExps } from "@/lib/attention";

import { Heatmap } from "./Heatmap";
import { Matrix } from "./Matrix";
import { CellExercise, ChoiceExercise, Formulas, Fx, M, Why } from "./shared";
import type { ModuleProps } from "./types";

/* ═══════════ 5. Produto QKᵀ ═══════════ */
export function ScoresModule({ state, R, actions, exercises }: ModuleProps) {
  const [tip, setTip] = useState("Passe o mouse, toque ou navegue com Tab sobre uma célula.");
  const tokens = state.tokens;

  return (
    <>
      <Formulas>
        <Fx>Scores = Q·Kᵀ</Fx>
      </Formulas>
      <p className="lead">
        Cada célula mede a <strong>compatibilidade</strong> entre a Query de um token (linha) e a
        Key de um token (coluna) — inclusive dele mesmo, na diagonal.
      </p>

      <div className="card">
        <div className="matlabel">Heatmap dos scores brutos</div>
        <Heatmap
          data={R.scores}
          labels={tokens}
          label="Scores brutos Q·Kᵀ"
          onHover={(i, j, v) =>
            setTip(
              `Query: "${tokens[i]}"\nKey: "${tokens[j]}"\nScore = Q${i + 1}·K${j + 1} = ${fmt(v)}`,
            )
          }
          onCell={(i, j, v) => {
            if (exercises.armed !== 2) return;
            const [bi, bj] = argmaxMatrix(R.scores);
            const max = R.scores[bi]?.[bj] ?? 0;
            actions.setExerciseResult(2, {
              // Compara pelo valor para aceitar qualquer célula empatada com o máximo.
              ok: v === max,
              msg: `Maior score = ${fmt(max)}, em "${tokens[bi]}" × "${tokens[bj]}". Você clicou em ${fmt(v)}.`,
            });
          }}
        />
        <div className="tooltipbox" aria-live="polite">
          {tip}
        </div>
      </div>

      <Why
        why="Precisamos de um número que diga “quanto o token A deve olhar para o token B”. O produto escalar faz isso: vetores longos apontando em direções parecidas dão valores altos; direções opostas dão valores negativos."
        math={
          <>
            <M>
              S[i][j] = Q<sub>i</sub> · K<sub>j</sub> = Σ<sub>c</sub> Q[i][c]·K[j][c]
            </M>
            . O resultado é uma matriz quadrada n×n, que não precisa ser simétrica: Q e K vêm de
            matrizes de pesos diferentes, então S[i][j] ≠ S[j][i] em geral.
          </>
        }
      />

      <CellExercise
        id={2}
        question={
          <>
            Clique na célula do heatmap acima com o <strong>maior score</strong>.
          </>
        }
        exercises={exercises}
        actions={actions}
      />
    </>
  );
}

/* ═══════════ 6. Scaling ═══════════ */
export function ScalingModule({ state, R, actions, exercises }: ModuleProps) {
  const o = { rowLabels: state.tokens, colLabels: state.tokens };
  return (
    <>
      <Formulas>
        <Fx big>
          Scores escalados = Q·Kᵀ ⁄ √d<sub>k</sub>
        </Fx>
      </Formulas>
      <p className="lead">
        Dividimos os scores por{" "}
        <M>
          √d<sub>k</sub>
        </M>{" "}
        para evitar valores grandes demais, que deixariam o softmax quase “tudo ou nada” (um peso
        perto de 1 e os outros perto de 0) e travariam o aprendizado.
      </p>

      <div className="card">
        <label className="slider">
          d<sub>k</sub> = <span>{state.dk}</span> &nbsp;→&nbsp; √d<sub>k</sub> ={" "}
          <span>{Math.sqrt(state.dk).toFixed(2)}</span>
          <input
            type="range"
            min={1}
            max={64}
            step={1}
            value={state.dk}
            onChange={(e) => actions.setDk(parseInt(e.target.value, 10))}
            aria-label="Simular outro valor de d_k"
          />
        </label>
        <p className="hint">
          <strong>Simulação “e se”:</strong> aqui K tem 3 colunas, então o d<sub>k</sub> real é 3. O
          slider só troca o divisor, mantendo os scores brutos fixos, para você ver o efeito da
          divisão isoladamente. Num modelo real, um d<sub>k</sub> maior também faria os próprios
          scores brutos crescerem — e é justamente isso que a divisão compensa.
        </p>
        {state.dk !== DIM && (
          <p className="hint warn">
            Atenção: o valor escolhido aqui vale para todas as etapas seguintes (softmax, pesos,
            saída). Com d<sub>k</sub> ≠ {DIM}, elas deixam de mostrar o cálculo correto.{" "}
            <button type="button" className="btn small" onClick={() => actions.setDk(DIM)}>
              Voltar para d<sub>k</sub> = {DIM}
            </button>
          </p>
        )}
        <div className="sidebyside">
          <div>
            <div className="matlabel">Q·Kᵀ (bruto)</div>
            <Matrix data={R.scores} {...o} label="Scores brutos" />
          </div>
          <div className="divarrow" aria-hidden="true">
            ÷ √d<sub>k</sub> →
          </div>
          <div>
            <div className="matlabel">
              Q·Kᵀ / √d<sub>k</sub>
            </div>
            <Matrix data={R.scaled} {...o} className="accent" label="Scores escalados" />
          </div>
        </div>
      </div>

      <Why
        why={
          <>
            Com d<sub>k</sub> grande, o produto escalar soma muitos termos e tende a ficar grande em
            valor absoluto. Scores enormes fazem o softmax devolver quase 1 e 0, e os gradientes
            ficam minúsculos.
          </>
        }
        math={
          <>
            Se as componentes de q e k são independentes, com média 0 e variância 1, o produto
            escalar q·k (soma de d<sub>k</sub> termos) tem variância d<sub>k</sub>. Dividir por √d
            <sub>k</sub> devolve variância 1.
          </>
        }
      />

      <ChoiceExercise
        id={3}
        question={
          <>
            Nesta simulação, mantendo os scores brutos fixos e aumentando d<sub>k</sub> no slider,
            os scores escalados ficam:
          </>
        }
        options={[
          { label: "maiores em valor absoluto", correct: false },
          { label: "mais próximos de zero e mais próximos entre si", correct: true },
          { label: "inalterados", correct: false },
        ]}
        exercises={exercises}
        actions={actions}
      />
    </>
  );
}

/* ═══════════ 7. Máscara causal ═══════════ */
export function MaskModule({ state, R, actions }: ModuleProps) {
  const n = state.tokens.length;
  const sq = { rowLabels: state.tokens, colLabels: state.tokens };
  return (
    <>
      <p className="lead">
        Em modelos que geram texto, um token <strong>não pode enxergar o futuro</strong>. Posições
        futuras recebem −∞ antes do softmax, o que zera o peso.
      </p>

      <div className="card">
        <label className="switch">
          <input
            type="checkbox"
            checked={state.causal}
            onChange={(e) => actions.setCausal(e.target.checked)}
          />
          <span>Ativar máscara causal</span>
        </label>
        <p className="hint">
          {state.causal
            ? "Máscara ativa: as duas matrizes da direita já refletem o bloqueio, assim como as etapas seguintes (softmax, pesos e saída). Os scores brutos e escalados não mudam, porque a máscara entra depois deles."
            : "Máscara desativada: a matriz da esquerda mostra o que seria bloqueado; as da direita ainda estão sem máscara."}
        </p>
        <div className="sidebyside">
          <div>
            <div className="matlabel">Máscara (1 = permitido)</div>
            <Matrix
              data={causalMaskMatrix(n)}
              {...sq}
              format={(v) => String(v)}
              cellClass={(v) => (v === 1 ? "allow" : "block")}
              className="mask"
              label="Máscara causal"
            />
          </div>
          <div>
            <div className="matlabel">
              {state.causal ? "Scores após máscara" : "Scores escalados (sem máscara)"}
            </div>
            <Matrix data={R.masked} {...sq} label="Scores após máscara" />
          </div>
          <div>
            <div className="matlabel">Pesos após softmax</div>
            <Heatmap data={R.weights} labels={state.tokens} label="Pesos após softmax" tiny />
          </div>
        </div>
        <p className="flowline">Scores → Máscara → Softmax → Attention</p>
      </div>

      <Why
        why="No treino, o modelo aprende a prever o próximo token em cada posição, mas recebe a frase inteira de uma vez. Sem máscara ele “colaria” a resposta olhando o próximo token. Na geração, os tokens futuros nem existem ainda. (Modelos que só codificam texto, sem gerá-lo, como o BERT, não usam essa máscara: cada token vê a frase inteira.)"
        math={
          <>
            Somamos −∞ (na prática, um número muito negativo) onde <M>j &gt; i</M>. Como{" "}
            <M>
              e<sup>−∞</sup> = 0
            </M>{" "}
            (no limite), o peso vira 0. A diagonal nunca é mascarada: cada token sempre pode olhar
            para si mesmo, então o primeiro token dá peso 1 a si próprio.
          </>
        }
      />
    </>
  );
}

/* ═══════════ 8. Softmax ═══════════ */
const DEMO_XS = [2, 4, 1, 3];
const DEMO_LABELS = ["a", "b", "c", "d"];

export function SoftmaxModule({ state, R, actions, exercises }: ModuleProps) {
  const demo = state.demoSoftmax;
  const labels = demo ? DEMO_LABELS : state.tokens;
  const xs = demo ? DEMO_XS : (R.masked[state.selectedRow] ?? []);
  const w = softmax(xs);
  const exps = softmaxExps(xs);

  return (
    <>
      <Formulas>
        <Fx>
          softmax(x<sub>i</sub>) = e
          <sup>
            x<sub>i</sub>
          </sup>{" "}
          ⁄ Σ<sub>j</sub> e
          <sup>
            x<sub>j</sub>
          </sup>
        </Fx>
      </Formulas>
      <p className="lead">
        O softmax transforma uma linha de scores em <strong>pesos entre 0 e 1 que somam 1</strong>.
        Sem máscara, todos os pesos são positivos; posições mascaradas (−∞) recebem peso exatamente
        0.
      </p>

      <div className="card">
        <div className="row">
          <label>
            Linha analisada:{" "}
            <select
              value={state.selectedRow}
              onChange={(e) => actions.selectRow(parseInt(e.target.value, 10))}
              disabled={demo}
            >
              {state.tokens.map((t, i) => (
                <option key={`${i}-${t}`} value={i}>
                  linha de “{t}”
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="btn small"
            aria-pressed={demo}
            onClick={actions.toggleDemoSoftmax}
          >
            {demo ? "Voltar aos scores da frase" : "Usar exemplo [2, 4, 1, 3]"}
          </button>
        </div>
        <table className="softtable">
          <thead>
            <tr>
              <th scope="col">token</th>
              <th scope="col">
                x<sub>i</sub> {demo ? "" : "(score escalado, após máscara)"}
              </th>
              <th scope="col">
                e
                <sup>
                  x<sub>i</sub> − max
                </sup>
              </th>
              <th scope="col">peso</th>
              <th scope="col">barra</th>
            </tr>
          </thead>
          <tbody>
            {xs.map((x, i) => (
              <tr key={i}>
                <td>{labels[i]}</td>
                <td>{fmt(x)}</td>
                <td>{(exps[i] ?? 0).toFixed(3)}</td>
                <td>{(w[i] ?? 0).toFixed(3)}</td>
                <td>
                  <div
                    className="bar"
                    style={{ width: `${Math.max(2, (w[i] ?? 0) * 100).toFixed(1)}%` }}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="sumline">
          {w.map((v) => v.toFixed(2)).join(" + ")} = {w.reduce((a, b) => a + b, 0).toFixed(2)}
        </p>
      </div>

      <Why
        why="Queremos uma média ponderada. Para isso os pesos precisam ser não-negativos e somar 1 — exatamente o que o softmax garante."
        math="A exponencial transforma qualquer número em positivo e amplia as diferenças; a divisão pela soma normaliza. Somar uma constante a todos os x não muda o resultado — por isso subtraímos o máximo antes de exponenciar, só para evitar números gigantes. Os pesos exibidos estão arredondados para 2 casas, então somá-los de cabeça pode dar 0.99 ou 1.01; a soma exata é sempre 1."
      />

      <ChoiceExercise
        id={4}
        question="Quanto sempre soma uma linha de pesos após o softmax?"
        options={[
          { label: "0", correct: false },
          { label: "1", correct: true },
          {
            label: (
              <>
                depende de d<sub>k</sub>
              </>
            ),
            correct: false,
          },
        ]}
        exercises={exercises}
        actions={actions}
      />
    </>
  );
}
