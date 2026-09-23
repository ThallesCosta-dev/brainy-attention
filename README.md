# Como funciona o mecanismo de atenção

Laboratório interativo, em português, para aprender **Self-Attention** passo a passo: tokens → embeddings →
positional encoding → Q/K/V → Q·Kᵀ → scaling → máscara causal → softmax → attention weights → weighted sum,
com matrizes editáveis, heatmaps, exercícios e uma simulação livre. Toda a matemática roda no navegador.

**App publicado**: https://brainy-attention.lovable.app

## Estrutura

| Caminho                     | O que é                                                                                               |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `src/lib/attention.ts`      | Matemática pura (tokenização, PE senoidal, projeções, softmax, máscara). Sem DOM, coberta por testes. |
| `src/components/attention/` | Os 13 módulos didáticos, o heatmap/matriz acessíveis e o estado do laboratório.                       |
| `src/routes/index.tsx`      | Rota principal. `?modulo=N` abre uma etapa específica.                                                |
| `src/attention.css`         | Estilos do laboratório (temas claro e escuro via `data-theme`).                                       |
| `src/routes/__root.tsx`     | Casca TanStack Start: metadados, tema inicial, 404 e página de erro.                                  |

## Desenvolvimento

O projeto usa [Bun](https://bun.sh) (o lockfile é `bun.lock`).

```sh
bun install
bun run dev        # servidor de desenvolvimento
bun run test       # testes da matemática (vitest)
bun run typecheck  # tsc --noEmit
bun run lint
bun run build
```

Sem o Bun instalado, `npx bun@latest install` funciona sem instalação global.

## Lovable

Este projeto está conectado ao [Lovable](https://lovable.dev/projects/6356981f-547f-49ac-b9ff-2889a5799db7).
Commits enviados para `main` aparecem no editor, e mudanças feitas lá voltam para este repositório. Não
reescreva histórico já publicado (force push, rebase, squash) para não perder o histórico do lado do Lovable.
