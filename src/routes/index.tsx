import { createFileRoute } from "@tanstack/react-router";
import { useCallback } from "react";

import attentionCss from "../attention.css?url";
import { AttentionLab } from "../components/attention/AttentionLab";
import { TOTAL_MODULES } from "../components/attention/modules";

const TITLE = "Mecanismo de Atenção — laboratório interativo";
const DESC =
  "Aprenda Self-Attention passo a passo: tokens, embeddings, positional encoding, Q/K/V, QKᵀ, scaling, softmax, pesos de atenção e soma ponderada, com matrizes editáveis e animações.";

type Search = { modulo?: number };

export const Route = createFileRoute("/")({
  // ?modulo=N dá deep-link para cada etapa e faz o botão voltar do navegador funcionar.
  validateSearch: (search: Record<string, unknown>): Search => {
    const n = Number(search["modulo"]);
    return Number.isInteger(n) && n >= 1 && n <= TOTAL_MODULES ? { modulo: n } : {};
  },
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESC },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESC },
    ],
    links: [{ rel: "stylesheet", href: attentionCss }],
  }),
  component: Index,
});

function Index() {
  const { modulo = 1 } = Route.useSearch();
  const navigate = Route.useNavigate();
  const setModule = useCallback(
    (n: number) => {
      void navigate({ search: n === 1 ? {} : { modulo: n }, resetScroll: false });
    },
    [navigate],
  );
  return <AttentionLab module={modulo} onModuleChange={setModule} />;
}
