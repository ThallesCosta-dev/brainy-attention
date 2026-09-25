import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  DEFAULT_SENTENCE,
  DEFAULT_WK,
  DEFAULT_WQ,
  DEFAULT_WV,
  DEFAULT_X,
  DIM,
  MAX_TOKENS,
  calculateAttention,
  cloneMatrix,
  createEmbeddings,
  tokenize,
} from "@/lib/attention";
import type { Matrix } from "@/lib/attention";

import { MODULES, PIPELINE, TOTAL_MODULES, moduleForStage } from "./modules";
import type { ExerciseResult, ExerciseState, LabActions, LabState, WeightName } from "./types";

export const THEME_STORAGE_KEY = "attention-theme";
type Theme = "light" | "dark";

const AUTOPLAY_MS = 3800;

function initialState(): LabState {
  const { tokens, truncated } = tokenize(DEFAULT_SENTENCE);
  return {
    sentence: DEFAULT_SENTENCE,
    tokens,
    truncated,
    X: cloneMatrix(DEFAULT_X),
    WQ: cloneMatrix(DEFAULT_WQ),
    WK: cloneMatrix(DEFAULT_WK),
    WV: cloneMatrix(DEFAULT_WV),
    dk: DIM,
    causal: false,
    selectedRow: 0,
    demoSoftmax: false,
    matrixVersion: 0,
  };
}

function setCell(Mx: Matrix, i: number, j: number, value: number): Matrix {
  return Mx.map((row, ri) => (ri === i ? row.map((v, ci) => (ci === j ? value : v)) : row));
}

const clampModule = (n: number) => Math.min(TOTAL_MODULES, Math.max(1, Math.round(n)));

interface AttentionLabProps {
  /** Módulo ativo (1-based). Vem da URL para permitir deep-link e botão voltar. */
  module: number;
  onModuleChange: (module: number) => void;
}

export function AttentionLab({ module, onModuleChange }: AttentionLabProps) {
  const current = clampModule(module);
  const def = MODULES[current - 1] ?? MODULES[0]!;

  const [state, setState] = useState<LabState>(initialState);
  const [exercises, setExercises] = useState<ExerciseState>({ results: {}, armed: null });
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [theme, setTheme] = useState<Theme | null>(null);
  const [toast, setToast] = useState<{ text: string; id: number } | null>(null);

  /* Recalcula o pipeline só quando algo que entra nele muda. Cada módulo lê de R;
     nada é redesenhado fora do módulo ativo. */
  const R = useMemo(
    () => calculateAttention(state.X, state.WQ, state.WK, state.WV, state.dk, state.causal),
    [state.X, state.WQ, state.WK, state.WV, state.dk, state.causal],
  );

  const showToast = useCallback((text: string) => setToast({ text, id: Date.now() }), []);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 2600);
    return () => clearTimeout(t);
  }, [toast]);

  const goTo = useCallback(
    (n: number) => {
      setPlaying(false);
      onModuleChange(clampModule(n));
    },
    [onModuleChange],
  );

  /* ---------- apresentação automática ---------- */
  useEffect(() => {
    if (!playing) return;
    if (current >= TOTAL_MODULES) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => onModuleChange(current + 1), AUTOPLAY_MS / speed);
    return () => clearTimeout(t);
  }, [playing, current, speed, onModuleChange]);

  /* ---------- rolar para o topo ao trocar de módulo ---------- */
  const firstRender = useRef(true);
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    window.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" });
  }, [current]);

  /* ---------- atalhos de teclado ---------- */
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement as HTMLElement | null;
      const tag = el?.tagName ?? "";
      if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) || el?.isContentEditable) return;
      if (e.key === "ArrowRight") goTo(current + 1);
      else if (e.key === "ArrowLeft") goTo(current - 1);
      else if (e.key === " " && tag !== "BUTTON") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [current, goTo]);

  /* ---------- tema (o <script> inline no <head> já aplicou o atributo antes da hidratação) ---------- */
  useEffect(() => {
    setTheme(document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark");
  }, []);
  const toggleTheme = () => {
    const next: Theme = theme === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    setTheme(next);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
    } catch {
      /* modo privado / armazenamento bloqueado: o tema vale só nesta visita */
    }
  };

  /* ---------- ações ---------- */
  const actions = useMemo<LabActions>(
    () => ({
      editX: (i, j, v) => setState((s) => ({ ...s, X: setCell(s.X, i, j, v) })),
      editW: (name: WeightName, i, j, v) =>
        setState((s) => ({ ...s, [name]: setCell(s[name], i, j, v) })),
      resetX: () =>
        setState((s) => ({
          ...s,
          X: createEmbeddings(s.tokens),
          matrixVersion: s.matrixVersion + 1,
        })),
      randomizeX: () =>
        setState((s) => ({
          ...s,
          X: s.X.map((r) => r.map(() => Math.round(Math.random() * 100) / 100)),
          matrixVersion: s.matrixVersion + 1,
        })),
      setDk: (dk) => setState((s) => ({ ...s, dk })),
      setCausal: (causal) => setState((s) => ({ ...s, causal })),
      selectRow: (i) => setState((s) => ({ ...s, selectedRow: i, demoSoftmax: false })),
      toggleDemoSoftmax: () => setState((s) => ({ ...s, demoSoftmax: !s.demoSoftmax })),
      runSentence: (raw) => {
        const sentence = raw.trim() || DEFAULT_SENTENCE;
        const { tokens, truncated } = tokenize(sentence);
        setState((s) => ({
          ...s,
          sentence,
          tokens,
          truncated,
          X: createEmbeddings(tokens),
          selectedRow: 0,
          demoSoftmax: false,
          matrixVersion: s.matrixVersion + 1,
        }));
        setExercises({ results: {}, armed: null });
        showToast(
          truncated
            ? `Frase cortada em ${MAX_TOKENS} tokens (limite didático).`
            : `Recalculado para ${tokens.length} token${tokens.length === 1 ? "" : "s"}.`,
        );
      },
      goTo,
      goToStage: (stage) => goTo(moduleForStage(stage)),
      setExerciseResult: (id: number, result: ExerciseResult) =>
        setExercises((e) => ({
          results: { ...e.results, [id]: result },
          armed: e.armed === id ? null : e.armed,
        })),
      armExercise: (id: number) =>
        setExercises((e) => ({
          results: { ...e.results, [id]: undefined },
          armed: e.armed === id ? null : id,
        })),
    }),
    [goTo, showToast],
  );

  const reset = () => {
    setPlaying(false);
    setState(initialState());
    setExercises({ results: {}, armed: null });
    goTo(1);
    showToast("Simulação reiniciada.");
  };

  const Component = def.Component;
  const progress = (current / TOTAL_MODULES) * 100;

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true" />
          <div>
            <h1>Como funciona o mecanismo de atenção</h1>
            <p className="sub">Self-Attention explicado passo a passo — laboratório interativo</p>
          </div>
        </div>

        <div className="controls" role="group" aria-label="Navegação e apresentação">
          <button
            type="button"
            className="btn"
            title="Módulo anterior (←)"
            disabled={current === 1}
            onClick={() => goTo(current - 1)}
          >
            ← Anterior
          </button>
          <button
            type="button"
            className="btn primary"
            title="Avança os módulos sozinho (espaço)"
            aria-pressed={playing}
            disabled={!playing && current === TOTAL_MODULES}
            onClick={() => setPlaying((p) => !p)}
          >
            {playing ? "⏸ Pausar" : "▶ Apresentação automática"}
          </button>
          <button
            type="button"
            className="btn"
            title="Próximo módulo (→)"
            disabled={current === TOTAL_MODULES}
            onClick={() => goTo(current + 1)}
          >
            Próximo →
          </button>
          <button
            type="button"
            className="btn ghost"
            title="Volta tudo ao estado inicial"
            onClick={reset}
          >
            ⟲ Reiniciar
          </button>
          <label className="speed" title="Ritmo da apresentação automática">
            Ritmo
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.5}
              value={speed}
              onChange={(e) => setSpeed(parseFloat(e.target.value))}
              aria-label="Ritmo da apresentação automática"
            />
            <span>{speed}×</span>
          </label>
          <button
            type="button"
            className="btn ghost theme"
            onClick={toggleTheme}
            aria-label={theme === "light" ? "Mudar para tema escuro" : "Mudar para tema claro"}
            title="Alternar tema claro/escuro"
          >
            <span aria-hidden="true">{theme === "light" ? "☀️" : "🌙"}</span>
          </button>
        </div>
      </header>

      <div className="progresswrap">
        <div
          className="progress"
          role="progressbar"
          aria-valuemin={1}
          aria-valuemax={TOTAL_MODULES}
          aria-valuenow={current}
        >
          <div style={{ width: `${progress}%` }} />
        </div>
        <div className="progresslabel">
          Etapa {current} de {TOTAL_MODULES} — {def.title}
        </div>
      </div>

      <div className="layout">
        <nav className="sidebar" aria-label="Módulos">
          <ol>
            {MODULES.map((m, i) => (
              <li key={m.title}>
                <button
                  type="button"
                  className="navbtn"
                  aria-current={i + 1 === current ? "step" : undefined}
                  onClick={() => goTo(i + 1)}
                >
                  <b>{i + 1}</b>
                  {m.title}
                </button>
              </li>
            ))}
          </ol>
          <div className="pipeline">
            <h3>Fluxo dos dados</h3>
            <ul>
              {PIPELINE.map((p) => (
                <li
                  key={p.stage}
                  className={p.stage === def.stage ? "on" : ""}
                  aria-current={p.stage === def.stage ? "true" : undefined}
                >
                  {p.label}
                </li>
              ))}
            </ul>
          </div>
        </nav>

        <main id="main">
          <section className="module" key={current} aria-labelledby="module-title">
            <h2 id="module-title">
              <span className="num" aria-hidden="true">
                {current}
              </span>
              {def.title}
            </h2>
            <Component state={state} R={R} actions={actions} exercises={exercises} />
          </section>
        </main>
      </div>

      <div className={`toast${toast ? " show" : ""}`} role="status" aria-live="polite">
        {toast?.text}
      </div>
    </>
  );
}
