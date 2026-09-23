import type { Matrix as MatrixData } from "@/lib/attention";
import { fmt } from "@/lib/attention";

interface MatrixProps {
  data: MatrixData;
  rowLabels?: string[] | undefined;
  colLabels?: string[] | undefined;
  /** Quando definido, cada célula vira um <input type="number">. */
  onEdit?: ((i: number, j: number, value: number) => void) | undefined;
  /** Classe extra por célula (ex.: "allow"/"block" na máscara). */
  cellClass?: ((value: number, i: number, j: number) => string | undefined) | undefined;
  /** Formatação alternativa (ex.: "1"/"0" na máscara). */
  format?: ((value: number) => string) | undefined;
  className?: string | undefined;
  label?: string | undefined;
}

/**
 * Tabela numérica. Os inputs são não controlados (defaultValue) para permitir
 * digitar "-0." sem o React reescrever o campo; o pai troca a `key` do
 * componente quando quer descartar o que foi digitado (restaurar, sortear…).
 */
export function Matrix({
  data,
  rowLabels,
  colLabels,
  onEdit,
  cellClass,
  format = fmt,
  className = "",
  label,
}: MatrixProps) {
  return (
    <div className={`matrix ${className}`.trim()}>
      <table aria-label={label}>
        {colLabels && (
          <thead>
            <tr>
              {rowLabels && <th scope="col" />}
              {colLabels.map((c, j) => (
                <th key={j} scope="col">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {data.map((row, i) => (
            <tr key={i}>
              {rowLabels && <th scope="row">{rowLabels[i]}</th>}
              {row.map((v, j) => (
                <td key={j} className={cellClass?.(v, i, j)}>
                  {onEdit ? (
                    <input
                      type="number"
                      step="0.1"
                      defaultValue={v}
                      aria-label={`linha ${rowLabels?.[i] ?? i + 1}, coluna ${colLabels?.[j] ?? j + 1}`}
                      onChange={(e) => {
                        const n = parseFloat(e.target.value);
                        onEdit(i, j, Number.isFinite(n) ? n : 0);
                      }}
                    />
                  ) : (
                    format(v)
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
