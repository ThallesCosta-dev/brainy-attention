import { EmbeddingsModule, PositionalModule, ProblemModule, QKVModule } from "./modules-input";
import {
  FormulaModule,
  HeadsModule,
  LabModule,
  WeightedSumModule,
  WeightsModule,
} from "./modules-output";
import { MaskModule, ScalingModule, ScoresModule, SoftmaxModule } from "./modules-scores";
import type { ModuleDef, Stage } from "./types";

/**
 * Única fonte de verdade da ordem, dos títulos e do estágio de cada módulo.
 * A ordem segue o fluxo real dos dados: a máscara entra entre o scaling e o softmax.
 */
export const MODULES: readonly ModuleDef[] = [
  { title: "O problema", stage: "tokens", Component: ProblemModule },
  { title: "Embeddings", stage: "emb", Component: EmbeddingsModule },
  { title: "Positional Encoding", stage: "pos", Component: PositionalModule },
  { title: "Query, Key e Value", stage: "qkv", Component: QKVModule },
  { title: "Produto QKᵀ", stage: "scores", Component: ScoresModule },
  { title: "Scaling", stage: "scale", Component: ScalingModule },
  { title: "Atenção causal", stage: "mask", Component: MaskModule },
  { title: "Softmax", stage: "softmax", Component: SoftmaxModule },
  { title: "Attention Weights", stage: "weights", Component: WeightsModule },
  { title: "Weighted Sum", stage: "sum", Component: WeightedSumModule },
  { title: "A fórmula completa", stage: "out", Component: FormulaModule },
  { title: "Multi-Head Attention", stage: "heads", Component: HeadsModule },
  { title: "Simulação interativa", stage: "out", Component: LabModule },
];

export const TOTAL_MODULES = MODULES.length;

/** Primeiro módulo (1-based) que cobre um estágio. */
export function moduleForStage(stage: Stage): number {
  const idx = MODULES.findIndex((m) => m.stage === stage);
  return idx === -1 ? 1 : idx + 1;
}

export const PIPELINE: readonly { stage: Stage; label: string }[] = [
  { stage: "tokens", label: "Tokens" },
  { stage: "emb", label: "Embeddings" },
  { stage: "pos", label: "Positional encoding" },
  { stage: "qkv", label: "Q / K / V" },
  { stage: "scores", label: "QKᵀ" },
  { stage: "scale", label: "Scaling ÷ √dk" },
  { stage: "mask", label: "Máscara causal" },
  { stage: "softmax", label: "Softmax" },
  { stage: "weights", label: "Attention weights" },
  { stage: "sum", label: "Weighted sum" },
  { stage: "out", label: "Representação contextual" },
  { stage: "heads", label: "Multi-head: concat + W_O" },
];
