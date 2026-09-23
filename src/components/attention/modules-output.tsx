import { useEffect, useMemo, useState } from "react";

import { MAX_TOKENS, argmax, calculateAttention, fmt, weightedSum } from "@/lib/attention";
import type { Matrix as MatrixData } from "@/lib/attention";

import { Heatmap } from "./Heatmap";
import { Matrix } from "./Matrix";
import { CellExercise, Formulas, Fx, M, TokenChips, Why } from "./shared";
import type { ModuleProps, Stage } from "./types";

/* ═══════════ 9. Attention weights ═══════════ */
export function WeightsModule({ state, R, actions, exercises }: ModuleProps) {
  const [tip, setTip] = useState("Clique num token à esquerda para ver as conexões dele.");
  const tokens = state.tokens;

  return (
    <>
      <p className="lead">
        Cada linha responde: <strong>“para onde este token está olhando?”</strong> Clique num token
        à esquerda para ver as conexões.
      </p>

      <div className="card twocol">
        <div>
          <div className="matlabel">Matriz de pesos (softmax aplicado por linha)</div>
          <Heatmap
            data={R.weights}
            labels={tokens}
            label="Pesos de atenção"
            selectedRow={state.selectedRow}
            onSelectRow={actions.selectRow}
            onHover={(i, j, v) =>
              setTip(`"${tokens[i]}" olha para "${tokens[j]}" com peso ${fmt(v)}`)
            }
            onCell={(i, j) => {
              actions.selectRow(i);
              if (exercises.armed !== 5) return;
              const last = tokens.length - 1;
              const row = R.weights[last] ?? [];
              const best = argmax(row);
              actions.setExerciseResult(5, {
                ok: i === last && j === best,
                msg: `"${tokens[last]}" olha mais para "${tokens[best]}" (${fmt(row[best] ?? 0)}).`,
              });
            }}
          />
          <div className="tooltipbox" aria-live="polite">
            {tip}
          </div>
        </div>
        <div>
          <div className="matlabel">Conexões do token selecionado</div>
          <Links tokens={tokens} weights={R.weights} selectedRow={state.selectedRow} />
        </div>
      </div>

      <Why
        why="É aqui que “contexto” vira número: a linha do token “comeu” mostra quanta informação ele puxa de cada outro token."
        math={
          <>
            <M>
              A = softmax<sub>linha</sub>(Q·Kᵀ/√d<sub>k</sub>)
            </M>
            , com cada linha somando 1.
          </>
        }
      />

      <CellExercise
        id={5}
        question={
          <>
            Clique na célula que mostra{" "}
            <strong>para qual token o último token mais presta atenção</strong>.
          </>
        }
        exercises={exercises}
        actions={actions}
      />
    </>
  );
}

/** Conexões token→token em SVG, com espessura proporcional ao peso. */
function Links({
  tokens,
  weights,
  selectedRow,
}: {
  tokens: string[];
  weights: MatrixData;
  selectedRow: number;
}) {
  const n = tokens.length;
  const H = 260;
  const gap = H / (n + 1);
  const row = weights[selectedRow] ?? [];
  const short = (t: string) => (t.length > 11 ? t.slice(0, 10) + "…" : t);
  return (
    <svg
      className="links"
      viewBox="0 0 320 260"
      role="img"
      aria-label="Conexões de atenção do token selecionado"
    >
      {tokens.map((_, j) => {
        const w = row[j] ?? 0;
        return (
          <line
            key={`l${j}`}
            className="lnk"
            x1={96}
            y1={gap * (selectedRow + 1)}
            x2={224}
            y2={gap * (j + 1)}
            // Arredondado: servidor e navegador diferem no último dígito do float e o React
            // reclamaria de hidratação divergente.
            strokeWidth={(0.6 + w * 9).toFixed(2)}
            strokeOpacity={(0.15 + w * 0.85).toFixed(3)}
          />
        );
      })}
      {tokens.map((tk, j) => {
        const y = gap * (j + 1);
        return (
          <g key={`t${j}`}>
            <rect x={8} y={y - 14} width={88} height={28} rx={8} />
            <text x={52} y={y + 4} textAnchor="middle">
              {short(tk)}
            </text>
            <rect x={224} y={y - 14} width={88} height={28} rx={8} />
            <text x={268} y={y + 4} textAnchor="middle">
              {short(tk)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ═══════════ 10. Weighted sum ═══════════ */
export function WeightedSumModule({ state, R, actions }: ModuleProps) {
  const w = R.weights[state.selectedRow] ?? [];
  const res = weightedSum(w, R.V);
  return (
    <>
      <Formulas>
        <Fx>Output = A · V</Fx>
      </Formulas>
      <p className="lead">
        A saída de cada token é a soma dos vetores V de todos os tokens, cada um multiplicado pelo
        seu peso.
      </p>

      <div className="card">
        <label className="row">
          Token de saída:{" "}
          <select
            value={state.selectedRow}
            onChange={(e) => actions.selectRow(parseInt(e.target.value, 10))}
          >
            {state.tokens.map((t, i) => (
              <option key={`${i}-${t}`} value={i}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <div className="sumsteps">
          {w.map((wi, j) => (
            <div key={j} style={{ animationDelay: `${j * 0.1}s` }}>
              {wi.toFixed(2)} × V({state.tokens[j]}) = [
              {(R.V[j] ?? []).map((v) => (v * wi).toFixed(2)).join(", ")}]
            </div>
          ))}
        </div>
        <div className="matlabel">Representação contextualizada</div>
        <div className="vector accent">
          {state.tokens[state.selectedRow]} → [{res.map(fmt).join(", ")}]
        </div>
      </div>

      <Why
        why="O vetor final já não é só “comeu”: ele carrega pedaços de “anta” e “banana” na proporção dos pesos."
        math={
          <>
            <M>
              Out<sub>i</sub> = Σ<sub>j</sub> A[i][j]·V<sub>j</sub>
            </M>{" "}
            — uma combinação convexa das linhas de V.
          </>
        }
      />
    </>
  );
}

/* ═══════════ 11. A fórmula completa ═══════════ */
const BUILDUP: { label: React.ReactNode; stage: Stage | "lab"; final?: boolean }[] = [
  { label: "Q·Kᵀ", stage: "scores" },
  {
    label: (
      <>
        Q·Kᵀ ⁄ √d<sub>k</sub>
      </>
    ),
    stage: "scale",
  },
  {
    label: (
      <>
        softmax( Q·Kᵀ ⁄ √d<sub>k</sub> )
      </>
    ),
    stage: "softmax",
  },
  {
    label: (
      <>
        softmax( Q·Kᵀ ⁄ √d<sub>k</sub> )·V
      </>
    ),
    stage: "sum",
  },
  { label: "Attention(Q, K, V)", stage: "lab", final: true },
];

export function FormulaModule({ actions }: ModuleProps) {
  return (
    <>
      <Formulas>
        <Fx big>
          Attention(Q,K,V) = softmax( Q·Kᵀ ⁄ √d<sub>k</sub> ) · V
        </Fx>
      </Formulas>
      <p className="lead">Clique em cada parte para ir ao módulo correspondente.</p>
      <div className="buildup">
        {BUILDUP.map((b, i) => (
          <div key={i} style={{ display: "contents" }}>
            {i > 0 && (
              <div className="down" aria-hidden="true">
                ↓
              </div>
            )}
            <button
              type="button"
              className={`step${b.final ? " final" : ""}`}
              onClick={() =>
                b.stage === "lab"
                  ? actions.goTo(Number.POSITIVE_INFINITY)
                  : actions.goToStage(b.stage)
              }
            >
              {b.label}
            </button>
          </div>
        ))}
      </div>
    </>
  );
}

/* ═══════════ 12. Multi-Head Attention ═══════════ */
const HEAD_DEFS = [
  { name: "Head 1", role: "relação sintática", shift: 0.15 },
  { name: "Head 2", role: "relação semântica", shift: -0.2 },
  { name: "Head 3", role: "relação de posição", shift: 0.4 },
  { name: "Head 4", role: "outra relação contextual", shift: -0.35 },
];

export function HeadsModule({ state }: ModuleProps) {
  const { X, WQ, WK, WV, dk, causal } = state;
  const heads = useMemo(
    () =>
      HEAD_DEFS.map((h, idx) => {
        // Cada cabeça usa uma variação determinística dos pesos: projeções diferentes.
        const twist = (Mx: MatrixData) =>
          Mx.map((r, i) => r.map((v, j) => v + h.shift * ((i + j + idx) % 3 === 0 ? 1 : -0.4)));
        return {
          ...h,
          weights: calculateAttention(X, twist(WQ), twist(WK), WV, dk, causal).weights,
        };
      }),
    [X, WQ, WK, WV, dk, causal],
  );

  return (
    <>
      <p className="lead">
        Transformers rodam várias atenções em paralelo, cada uma com suas próprias projeções, e
        concatenam os resultados.
      </p>

      <div className="headflow">
        <div className="hnode">Input</div>
        <div className="down" aria-hidden="true">
          ↓
        </div>
        <div className="heads">
          {heads.map((h) => (
            <div className="head" key={h.name}>
              <h5>{h.name}</h5>
              <p>{h.role}</p>
              <Heatmap data={h.weights} labels={state.tokens} label={`Pesos da ${h.name}`} tiny />
            </div>
          ))}
        </div>
        <div className="down" aria-hidden="true">
          ↓
        </div>
        <div className="hnode">Concatenação</div>
        <div className="down" aria-hidden="true">
          ↓
        </div>
        <div className="hnode">
          W<sub>O</sub>
        </div>
        <div className="down" aria-hidden="true">
          ↓
        </div>
        <div className="hnode final">Output</div>
      </div>
      <p className="disclaimer">
        Os rótulos “sintática”, “semântica” e “posicional” são simplificações didáticas. Cabeças
        reais não se especializam necessariamente assim.
      </p>
    </>
  );
}

/* ═══════════ 13. Simulação interativa ═══════════ */
export function LabModule({ state, R, actions }: ModuleProps) {
  const [text, setText] = useState(state.sentence);
  useEffect(() => setText(state.sentence), [state.sentence]);
  const run = () => actions.runSentence(text);
  const rl = { rowLabels: state.tokens };
  const sq = { rowLabels: state.tokens, colLabels: state.tokens };

  return (
    <>
      <p className="lead">
        Escreva uma frase. Tudo é recalculado no navegador, do token ao vetor final.
      </p>

      <div className="card">
        <div className="row">
          <input
            className="textinput"
            type="text"
            value={text}
            maxLength={60}
            aria-label="Frase"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
          />
          <button type="button" className="btn primary small" onClick={run}>
            Recalcular
          </button>
          <label className="switch">
            <input
              type="checkbox"
              checked={state.causal}
              onChange={(e) => actions.setCausal(e.target.checked)}
            />{" "}
            <span>máscara causal</span>
          </label>
        </div>
        <p className={`hint${state.truncated ? " warn" : ""}`}>
          {state.truncated
            ? `A frase tinha mais de ${MAX_TOKENS} palavras: para as matrizes continuarem legíveis, só as ${MAX_TOKENS} primeiras foram usadas.`
            : `Limite didático: ${MAX_TOKENS} tokens, para as matrizes caberem na tela.`}
        </p>
        <div className="labgrid">
          <div>
            <div className="matlabel">Tokens</div>
            <TokenChips tokens={state.tokens} delay={0.08} />
          </div>
          <div>
            <div className="matlabel">X (embeddings)</div>
            <Matrix data={state.X} {...rl} label="X" />
          </div>
          <div>
            <div className="matlabel">PE (seno/cosseno)</div>
            <Matrix data={R.PE} {...rl} label="PE" />
          </div>
          <div>
            <div className="matlabel">X + PE</div>
            <Matrix data={R.Xpos} {...rl} className="accent" label="X + PE" />
          </div>
          <div>
            <div className="matlabel">Q</div>
            <Matrix data={R.Q} {...rl} label="Q" />
          </div>
          <div>
            <div className="matlabel">K</div>
            <Matrix data={R.K} {...rl} label="K" />
          </div>
          <div>
            <div className="matlabel">V</div>
            <Matrix data={R.V} {...rl} label="V" />
          </div>
          <div>
            <div className="matlabel">Q·Kᵀ</div>
            <Matrix data={R.scores} {...sq} label="Scores" />
          </div>
          <div>
            <div className="matlabel">
              ÷ √d<sub>k</sub> {state.causal ? "(+ máscara)" : ""}
            </div>
            <Matrix data={R.masked} {...sq} label="Scores escalados" />
          </div>
          <div>
            <div className="matlabel">Attention weights</div>
            <Heatmap data={R.weights} labels={state.tokens} label="Pesos de atenção" tiny />
          </div>
          <div>
            <div className="matlabel">Output = A·V</div>
            <Matrix data={R.output} {...rl} className="accent" label="Output" />
          </div>
        </div>
      </div>

      <div className="panel note">
        <h4>Precisão conceitual</h4>
        <ul>
          <li>
            Q, K e V são <strong>projeções lineares</strong> de X + PE.
          </li>
          <li>
            O positional encoding soma seno e cosseno aos embeddings para informar a ordem dos
            tokens.
          </li>
          <li>Q·Kᵀ produz os scores de compatibilidade.</li>
          <li>
            √d<sub>k</sub> é o fator de scaling.
          </li>
          <li>Softmax converte scores em pesos que somam 1.</li>
          <li>Os pesos combinam as linhas de V.</li>
          <li>Multi-Head executa várias projeções/atenções em paralelo.</li>
          <li>Na atenção causal, posições futuras são mascaradas.</li>
          <li>
            Esta é uma <strong>visualização didática</strong>; as dimensões minúsculas usadas aqui
            não são as de um LLM real.
          </li>
        </ul>
      </div>
    </>
  );
}
