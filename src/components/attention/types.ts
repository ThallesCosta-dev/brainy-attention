import type { ComponentType } from "react";

import type { AttentionResult, Matrix } from "@/lib/attention";

/** Etapas do fluxo de dados mostrado na barra lateral. */
export type Stage =
  | "tokens"
  | "emb"
  | "pos"
  | "qkv"
  | "scores"
  | "scale"
  | "mask"
  | "softmax"
  | "weights"
  | "sum"
  | "out"
  | "heads";

export type WeightName = "WQ" | "WK" | "WV";

export interface LabState {
  sentence: string;
  tokens: string[];
  /** A frase digitada tinha mais palavras do que o limite. */
  truncated: boolean;
  X: Matrix;
  WQ: Matrix;
  WK: Matrix;
  WV: Matrix;
  dk: number;
  causal: boolean;
  selectedRow: number;
  demoSoftmax: boolean;
  /** Incrementa quando X ou W são substituídos de fora, para descartar o que estava digitado nos inputs. */
  matrixVersion: number;
}

export interface ExerciseResult {
  ok: boolean;
  msg: string;
  /** Índice da alternativa escolhida (exercícios de múltipla escolha). */
  picked?: number;
}

export interface ExerciseState {
  results: Partial<Record<number, ExerciseResult>>;
  /** Exercício de clique que está aguardando a resposta, se houver. */
  armed: number | null;
}

export interface LabActions {
  editX: (i: number, j: number, value: number) => void;
  editW: (name: WeightName, i: number, j: number, value: number) => void;
  resetX: () => void;
  randomizeX: () => void;
  setDk: (dk: number) => void;
  setCausal: (causal: boolean) => void;
  selectRow: (i: number) => void;
  toggleDemoSoftmax: () => void;
  runSentence: (sentence: string) => void;
  goTo: (module: number) => void;
  goToStage: (stage: Stage) => void;
  setExerciseResult: (exercise: number, result: ExerciseResult) => void;
  armExercise: (exercise: number) => void;
}

export interface ModuleProps {
  state: LabState;
  R: AttentionResult;
  actions: LabActions;
  exercises: ExerciseState;
}

export interface ModuleDef {
  title: string;
  stage: Stage;
  Component: ComponentType<ModuleProps>;
}
