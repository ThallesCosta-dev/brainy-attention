import { createFileRoute } from "@tanstack/react-router";

const TITLE = "Mecanismo de Atenção — laboratório interativo";
const DESC =
  "Aprenda Self-Attention passo a passo: tokens, embeddings, Q/K/V, QKᵀ, scaling, softmax, pesos de atenção e soma ponderada, com matrizes editáveis e animações.";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESC },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESC },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <iframe
      src="/atencao/index.html"
      title={TITLE}
      className="h-screen w-screen border-0"
    />
  );
}
