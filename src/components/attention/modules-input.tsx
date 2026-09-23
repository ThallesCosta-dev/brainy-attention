import { DIM, DIM_LABELS, fmt } from "@/lib/attention";

import { Matrix } from "./Matrix";
import { ChoiceExercise, Formulas, Fx, M, TokenChips, Why } from "./shared";
import type { ModuleProps } from "./types";

/* ═══════════ 1. O problema ═══════════ */
export function ProblemModule({ state }: ModuleProps) {
  return (
    <>
      <p className="lead">
        Um modelo de linguagem precisa representar cada palavra{" "}
        <em>levando em conta as outras palavras</em> da frase. A palavra “comeu” só faz sentido
        pleno perto de “anta” e “banana”.
      </p>

      <div className="card">
        <p className="frase">“{state.sentence}”</p>
        <TokenChips tokens={state.tokens} />
        <p className="hint">
          Cada token recebe uma representação numérica chamada <strong>embedding</strong> — um vetor
          de números.
        </p>
        <div className="tokenrow arrows" aria-hidden="true">
          {state.tokens.map((tk, i) => (
            <span key={`${i}-${tk}`} style={{ animationDelay: `${0.3 + i * 0.12}s` }}>
              ↓
            </span>
          ))}
        </div>
        <div className="tokenrow">
          {state.tokens.map((tk, i) => (
            <div
              key={`${i}-${tk}`}
              className="embchip"
              style={{ animationDelay: `${0.5 + i * 0.12}s` }}
            >
              [{(state.X[i] ?? []).map(fmt).join(", ")}]
            </div>
          ))}
        </div>
      </div>

      <Why
        why={
          <>
            Sem contexto, “banco” em “sentei no banco” e “fui ao banco” teriam o mesmo vetor. A
            atenção deixa cada token misturar informação dos outros.
          </>
        }
        math={
          <>
            Transformamos uma matriz <M>X</M> de <M>n</M> tokens × <M>d</M> dimensões em outra
            matriz do mesmo formato, mas onde cada linha é uma <em>média ponderada</em> das linhas
            originais.
          </>
        }
      />
    </>
  );
}

/* ═══════════ 2. Embeddings ═══════════ */
export function EmbeddingsModule({ state, actions }: ModuleProps) {
  return (
    <>
      <p className="lead">
        Cada linha de <M>X</M> é um token. Aqui usamos apenas <strong>{DIM} dimensões</strong> para
        conseguirmos ver os números; modelos reais usam centenas ou milhares.
      </p>

      <div className="card">
        <div className="matlabel">
          X{" "}
          <span className="dim">
            ({state.tokens.length}×{DIM})
          </span>{" "}
          — clique nos números e edite
        </div>
        <Matrix
          key={state.matrixVersion}
          data={state.X}
          rowLabels={state.tokens}
          colLabels={DIM_LABELS}
          onEdit={actions.editX}
          label="Matriz X de embeddings, editável"
          className="editable"
        />
        <div className="row">
          <button type="button" className="btn small" onClick={actions.resetX}>
            Restaurar valores padrão
          </button>
          <button type="button" className="btn small" onClick={actions.randomizeX}>
            Sortear embeddings
          </button>
        </div>
      </div>

      <Why
        why="Computadores não operam sobre letras. O embedding transforma cada token em um ponto no espaço, onde “perto” significa “parecido”."
        math={
          <>
            <M>
              X ∈ ℝ<sup>n×d</sup>
            </M>
            , com <M>n</M> = número de tokens e <M>d</M> = dimensão do modelo. Aqui <M>d = {DIM}</M>
            , valor didático.
          </>
        }
      />
    </>
  );
}

/* ═══════════ 3. Positional Encoding ═══════════ */
export function PositionalModule({ state, R }: ModuleProps) {
  const rl = { rowLabels: state.tokens, colLabels: DIM_LABELS };
  return (
    <>
      <p className="lead">
        Depois do embedding, o modelo soma uma informação de posição. Ela começa na saída de X e
        termina na matriz X + PE, que segue para Query, Key e Value.
      </p>

      <Formulas>
        <Fx>
          PE(pos, 2i) = sen(pos / 10000<sup>2i/d</sup>)
        </Fx>
        <Fx>
          PE(pos, 2i+1) = cos(pos / 10000<sup>2i/d</sup>)
        </Fx>
      </Formulas>

      <div className="card">
        <div className="posbadge start">Começa aqui: embeddings X</div>
        <div className="posflow">
          <div>
            <div className="matlabel">X</div>
            <Matrix data={state.X} {...rl} label="X" />
          </div>
          <div className="eqsym" aria-hidden="true">
            +
          </div>
          <div>
            <div className="matlabel">PE — seno e cosseno por posição</div>
            <Matrix data={R.PE} {...rl} label="PE" />
          </div>
          <div className="eqsym" aria-hidden="true">
            =
          </div>
          <div>
            <div className="matlabel">X + PE</div>
            <Matrix data={R.Xpos} {...rl} className="accent" label="X + PE" />
          </div>
        </div>
        <div className="posbadge end">Termina aqui: entrada das projeções Q/K/V</div>
        <p className="hint">
          Por que a coluna <M>d₃</M> fica quase zero? Cada par de dimensões usa uma frequência menor
          que o anterior. Com <M>d = 3</M>, a terceira dimensão tem frequência 1/10000<sup>2/3</sup>{" "}
          ≈ 1/464: para posições pequenas, sen(pos/464) é praticamente 0. Ela só varia de forma
          visível em sequências longas, com centenas de tokens.
        </p>
      </div>

      <Why
        why="A atenção compara tokens, mas os embeddings sozinhos não dizem se “anta” veio antes de “banana”. O positional encoding injeta a ordem na sequência."
        math={
          <>
            Para cada posição <M>pos</M> e dimensão <M>i</M>, calculamos ondas de seno e cosseno em
            frequências diferentes e somamos esse vetor ao embedding do token. Dimensões pares usam
            seno, ímpares usam cosseno.
          </>
        }
      />
    </>
  );
}

/* ═══════════ 4. Query, Key e Value ═══════════ */
export function QKVModule({ state, R, actions, exercises }: ModuleProps) {
  const cols = [
    {
      name: "WQ" as const,
      letter: "Q",
      out: R.Q,
      intuition: "“o que este token está procurando?”",
    },
    {
      name: "WK" as const,
      letter: "K",
      out: R.K,
      intuition: "“que tipo de informação eu ofereço?”",
    },
    { name: "WV" as const, letter: "V", out: R.V, intuition: "“qual conteúdo eu transmito?”" },
  ];
  const roles: Record<string, string> = { Q: "Query", K: "Key", V: "Value" };
  return (
    <>
      <p className="lead">
        Três <strong>projeções lineares</strong> da matriz X + PE. São só multiplicações de matrizes
        com pesos aprendidos.
      </p>

      <Formulas>
        {cols.map((c) => (
          <Fx key={c.letter}>
            {c.letter} = (X + PE)·W<sub>{c.letter}</sub>
          </Fx>
        ))}
      </Formulas>

      <div className="qkvflow">
        {cols.map((c) => (
          <div className="qkvcol" key={c.letter}>
            <div className="matlabel">
              W<sub>{c.letter}</sub> (editável)
            </div>
            <Matrix
              key={state.matrixVersion}
              data={state[c.name]}
              onEdit={(i, j, v) => actions.editW(c.name, i, j, v)}
              className="editable small"
              label={`Matriz W${c.letter}, editável`}
            />
            <div className="flowarrow" aria-hidden="true">
              ↓ (X + PE) · W<sub>{c.letter}</sub>
            </div>
            <div className="matlabel">{c.letter}</div>
            <Matrix data={c.out} rowLabels={state.tokens} label={c.letter} />
            <p className="intuition">
              <strong>{roles[c.letter]}:</strong> {c.intuition}
            </p>
          </div>
        ))}
      </div>

      <Why
        why="Um mesmo token precisa de papéis diferentes: quem pergunta, quem se anuncia e quem entrega conteúdo. Três matrizes de pesos dão esses três papéis."
        math={
          <>
            Multiplicação de matrizes: o elemento <M>Q[i][j]</M> é o produto escalar da linha{" "}
            <M>i</M> de X + PE com a coluna <M>j</M> de W<sub>Q</sub>. Nada de não-linearidade aqui
            — são projeções lineares.
          </>
        }
      />

      <ChoiceExercise
        id={1}
        question={
          <>
            Se X tem formato n×{DIM} e W<sub>Q</sub> tem {DIM}×{DIM}, qual é o formato de Q?
          </>
        }
        options={[
          { label: `${DIM}×${DIM}`, correct: false },
          { label: `n×${DIM}`, correct: true },
          { label: "n×n", correct: false },
        ]}
        exercises={exercises}
        actions={actions}
      />
    </>
  );
}
