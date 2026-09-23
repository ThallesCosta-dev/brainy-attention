import type { Matrix } from "@/lib/attention";
import { fmt } from "@/lib/attention";

/** Cor do heatmap: ciano (baixo) → azul → violeta (alto). Luminosidade 16%–50%. */
function heatColor(t: number): string {
  const hue = 186 + t * 84;
  const light = 16 + t * 34;
  return `hsl(${hue.toFixed(0)} 85% ${light.toFixed(0)}%)`;
}

interface HeatmapProps {
  data: Matrix;
  labels: string[];
  label: string;
  selectedRow?: number | undefined;
  onSelectRow?: ((i: number) => void) | undefined;
  /** Chamado ao clicar ou ativar uma célula pelo teclado. */
  onCell?: ((i: number, j: number, value: number) => void) | undefined;
  /** Chamado ao passar o mouse ou focar uma célula (também no toque, via clique). */
  onHover?: ((i: number, j: number, value: number) => void) | undefined;
  tiny?: boolean | undefined;
}

/**
 * Heatmap acessível: cabeçalhos de linha e células são <button>, então tudo
 * que dá para fazer com o mouse dá para fazer com Tab + Enter.
 */
export function Heatmap({
  data,
  labels,
  label,
  selectedRow,
  onSelectRow,
  onCell,
  onHover,
  tiny,
}: HeatmapProps) {
  const finite = data.flat().filter(Number.isFinite);
  const min = finite.length ? Math.min(...finite) : 0;
  const max = finite.length ? Math.max(...finite) : 1;
  const span = max - min || 1;
  const interactive = Boolean(onCell || onHover);

  return (
    <div className={`heatmap${tiny ? " tiny" : ""}${interactive ? "" : " static"}`}>
      <table aria-label={label}>
        <thead>
          <tr>
            <th scope="col" />
            {labels.map((l, j) => (
              <th key={j} scope="col">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              <th scope="row">
                {onSelectRow ? (
                  <button
                    type="button"
                    className="rowbtn"
                    aria-pressed={selectedRow === i}
                    onClick={() => onSelectRow(i)}
                  >
                    {labels[i]}
                  </button>
                ) : (
                  labels[i]
                )}
              </th>
              {row.map((v, j) => {
                const masked = !Number.isFinite(v);
                const norm = masked ? 0 : (v - min) / span;
                const dimmed = selectedRow != null && selectedRow !== i;
                const cls = `cell${dimmed ? " dim" : ""}${masked ? " masked" : ""}`;
                const style = masked ? undefined : { background: heatColor(norm) };
                const title = `${labels[i]} → ${labels[j]}: ${fmt(v)}`;
                return (
                  <td key={j}>
                    {interactive ? (
                      <button
                        type="button"
                        className={cls}
                        style={style}
                        title={title}
                        aria-label={title}
                        onMouseEnter={() => onHover?.(i, j, v)}
                        onFocus={() => onHover?.(i, j, v)}
                        onClick={() => {
                          onHover?.(i, j, v);
                          onCell?.(i, j, v);
                        }}
                      >
                        {fmt(v)}
                      </button>
                    ) : (
                      <span className={cls} style={style} title={title}>
                        {fmt(v)}
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
