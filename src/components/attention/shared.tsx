import type { ReactNode } from "react";

import type { ExerciseState, LabActions } from "./types";

/** Par de painéis "Por que" / "O que está acontecendo matematicamente". */
export function Why({ why, math }: { why: ReactNode; math: ReactNode }) {
  return (
    <div className="why">
      <div className="panel">
        <h4>Por que isso é necessário?</h4>
        <p>{why}</p>
      </div>
      <div className="panel">
        <h4>O que está acontecendo matematicamente?</h4>
        <p>{math}</p>
      </div>
    </div>
  );
}

export function Formulas({ children }: { children: ReactNode }) {
  return <div className="formulas">{children}</div>;
}

export function Fx({ children, big }: { children: ReactNode; big?: boolean | undefined }) {
  return <div className={`fx${big ? " big" : ""}`}>{children}</div>;
}

/** Símbolo matemático inline. */
export function M({ children }: { children: ReactNode }) {
  return <span className="m">{children}</span>;
}

interface ChoiceExerciseProps {
  id: number;
  question: ReactNode;
  options: { label: ReactNode; correct: boolean }[];
  exercises: ExerciseState;
  actions: LabActions;
}

export function ChoiceExercise({ id, question, options, exercises, actions }: ChoiceExerciseProps) {
  const result = exercises.results[id];
  return (
    <div className="exercise">
      <h4>Exercício {id}</h4>
      <p>{question}</p>
      <div className="opts" role="group" aria-label={`Alternativas do exercício ${id}`}>
        {options.map((o, idx) => {
          const picked = result?.picked === idx;
          const cls = picked ? (result?.ok ? " right" : " wrong") : "";
          return (
            <button
              key={idx}
              type="button"
              className={`opt${cls}`}
              aria-pressed={picked}
              onClick={() =>
                actions.setExerciseResult(id, {
                  ok: o.correct,
                  msg: "",
                  picked: idx,
                })
              }
            >
              {o.label}
            </button>
          );
        })}
      </div>
      <Feedback id={id} exercises={exercises} />
    </div>
  );
}

interface CellExerciseProps {
  id: number;
  question: ReactNode;
  exercises: ExerciseState;
  actions: LabActions;
}

/**
 * Exercício de clique numa célula. Só avalia depois de "Responder", para o
 * aluno poder explorar o heatmap sem receber ✓/✗ a cada clique.
 */
export function CellExercise({ id, question, exercises, actions }: CellExerciseProps) {
  const armed = exercises.armed === id;
  return (
    <div className="exercise">
      <h4>Exercício {id}</h4>
      <p>{question}</p>
      <div className="opts">
        <button
          type="button"
          className="opt"
          aria-pressed={armed}
          onClick={() => actions.armExercise(id)}
        >
          {armed ? "Aguardando o clique na célula…" : "Responder"}
        </button>
      </div>
      <Feedback id={id} exercises={exercises} />
    </div>
  );
}

function Feedback({ id, exercises }: { id: number; exercises: ExerciseState }) {
  const result = exercises.results[id];
  const armed = exercises.armed === id;
  let cls = "fb";
  let text = "";
  if (armed) {
    cls += " armed";
    text = "Clique (ou use Tab + Enter) na célula que você acha correta.";
  } else if (result) {
    cls += result.ok ? " ok" : " no";
    text = (result.ok ? "✓ Correto! " : "✗ Tente novamente. ") + result.msg;
  }
  return (
    <p className={cls} aria-live="polite">
      {text}
    </p>
  );
}

export function TokenChips({
  tokens,
  delay = 0.12,
}: {
  tokens: string[];
  delay?: number | undefined;
}) {
  return (
    <div className="tokenrow">
      {tokens.map((tk, i) => (
        <div key={`${i}-${tk}`} className="token" style={{ animationDelay: `${i * delay}s` }}>
          {tk}
        </div>
      ))}
    </div>
  );
}
