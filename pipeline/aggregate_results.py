#!/usr/bin/env python3
"""
=============================================================================
aggregate_results.py — Agrega metrics.json de todas as classes avaliadas
=============================================================================

Varre results/<experimento>/*/metrics.json (gerados por evaluate.py através
do run_pipeline.sh) e monta uma tabela consolidada de accuracy / IoU / F1
por classe, com uma linha adicional de totais/médias do set inteiro no final.

USO:
  python aggregate_results.py --results results/icoseg_full
  python aggregate_results.py --results results/icoseg_full --sort iou
  python aggregate_results.py --results results/icoseg_full --csv resumo_agregado.csv
  python aggregate_results.py --results results/icoseg_full --json resumo_agregado.json
  python aggregate_results.py --results results/icoseg_full --csv resumo_agregado.csv --json resumo_agregado.json

Se nenhum metrics.json for encontrado, avisa e sugere rodar o pipeline com
--export (necessário para a etapa EVAL gerar máscaras avaliáveis).

NOTA: este é o mesmo comando que o run_pipeline.sh já executa automaticamente
no final de cada execução, salvando em <results_base>/resumo_agregado.csv e
<results_base>/resumo_agregado.json. Rodar manualmente com esses mesmos nomes
sobrescreve os arquivos já gerados (sem duplicar conteúdo com outro nome).
=============================================================================
"""

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

OVERALL_LABEL = "TOTAL / MÉDIA GERAL"


def load_rows(results_root: Path) -> list[dict]:
    rows = []
    for f in sorted(results_root.glob("*/metrics.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Aviso] Falha lendo {f}: {e}", file=sys.stderr)
            continue

        agg = d.get("aggregate", {})
        rows.append({
            "classe":     d.get("collection", f.parent.name),
            "n_images":   agg.get("n_images", 0),
            "n_flipped":  agg.get("n_flipped", 0),
            "accuracy":   agg.get("accuracy",  {}).get("mean", 0.0),
            "precision":  agg.get("precision", {}).get("mean", 0.0),
            "recall":     agg.get("recall",    {}).get("mean", 0.0),
            "f1":         agg.get("f1",        {}).get("mean", 0.0),
            "iou":        agg.get("iou",       {}).get("mean", 0.0),
            "iou_std":    agg.get("iou",       {}).get("std", 0.0),
            "metrics_path": str(f),
        })
    return rows


def build_overall_row(rows: list[dict]) -> dict | None:
    """
    Calcula a linha de totais/médias do set inteiro (todas as classes).

    accuracy/precision/recall/f1/iou: média simples entre as classes
    (cada classe pesa igual, independente do número de imagens — mesmo
    critério já usado na tabela impressa no terminal).
    n_images/n_flipped: soma total do set.
    iou_std: desvio padrão do IoU *entre classes* (não confundir com o
    iou_std de cada classe, que é o desvio padrão entre imagens daquela
    classe).
    """
    n = len(rows)
    if n == 0:
        return None

    return {
        "classe":       OVERALL_LABEL,
        "n_images":     sum(r["n_images"] for r in rows),
        "n_flipped":    sum(r["n_flipped"] for r in rows),
        "accuracy":     st.mean(r["accuracy"]  for r in rows),
        "precision":    st.mean(r["precision"] for r in rows),
        "recall":       st.mean(r["recall"]    for r in rows),
        "f1":           st.mean(r["f1"]        for r in rows),
        "iou":          st.mean(r["iou"]       for r in rows),
        "iou_std":      st.pstdev(r["iou"] for r in rows) if n > 1 else 0.0,
        "metrics_path": "",
    }


def print_table(rows: list[dict], sort_key: str) -> None:
    if not rows:
        print("[Aviso] Nenhum metrics.json encontrado. "
              "Rode o pipeline com --export e --set <nome> primeiro.")
        return

    rows_sorted = sorted(rows, key=lambda r: r[sort_key])

    name_w = max(len(r["classe"]) for r in rows + [{"classe": OVERALL_LABEL}])
    name_w = min(name_w + 2, 60)

    header = (f"  {'CLASSE':<{name_w}} {'N':>4} {'ACC':>8} "
              f"{'PREC':>8} {'REC':>8} {'F1':>8} {'IoU':>8}")
    sep = "  " + "─" * (len(header) - 2)

    print()
    print(f"  Agregado de {len(rows)} classe(s)  (ordenado por '{sort_key}')")
    print(sep)
    print(header)
    print(sep)
    for r in rows_sorted:
        name = r["classe"][:name_w]
        print(f"  {name:<{name_w}} {r['n_images']:>4} "
              f"{r['accuracy']:>8.4f} {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['iou']:>8.4f}")
    print(sep)

    overall = build_overall_row(rows)
    print(f"  {overall['classe']:<{name_w}} {overall['n_images']:>4} "
          f"{overall['accuracy']:>8.4f} "
          f"{overall['precision']:>8.4f} "
          f"{overall['recall']:>8.4f} "
          f"{overall['f1']:>8.4f} "
          f"{overall['iou']:>8.4f}")
    print(sep)

    n_flipped_total = overall["n_flipped"]
    if n_flipped_total:
        print(f"\n  [!] {n_flipped_total} imagem(ns) tiveram FG/BG invertido em alguma classe.")
    print()


def save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["classe", "n_images", "n_flipped", "accuracy",
                  "precision", "recall", "f1", "iou", "iou_std", "metrics_path"]
    overall = build_overall_row(rows)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        if overall:
            writer.writerow(overall)
    print(f"[Salvo] {path}")


def save_json(rows: list[dict]) -> dict:
    overall = build_overall_row(rows)
    n = len(rows)
    summary = {
        "n_classes": n,
        "classes": rows,
        # "overall" tem o mesmo shape de uma linha de "classes", pronto para
        # ser tratado como "mais uma linha" por quem consome o JSON.
        "overall": overall,
        # Mantido por compatibilidade com consumidores antigos do JSON.
        "mean": {
            "accuracy":  overall["accuracy"]  if overall else None,
            "precision": overall["precision"] if overall else None,
            "recall":    overall["recall"]    if overall else None,
            "f1":        overall["f1"]        if overall else None,
            "iou":       overall["iou"]       if overall else None,
        },
        "total_images":  overall["n_images"]  if overall else 0,
        "total_flipped": overall["n_flipped"] if overall else 0,
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agrega metrics.json de todas as classes avaliadas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results", type=Path, default=Path("results/icoseg_full"),
                   help="Pasta results/<experimento> a varrer.")
    p.add_argument("--sort", choices=["classe", "n_images", "accuracy",
                                       "precision", "recall", "f1", "iou"],
                   default="iou", help="Coluna usada para ordenar a tabela.")
    p.add_argument("--csv", type=Path, default=None,
                   help="Se fornecido, salva a tabela agregada em CSV neste caminho.")
    p.add_argument("--json", type=Path, default=None,
                   help="Se fornecido, salva o resumo agregado em JSON neste caminho.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.results.exists():
        print(f"[ERRO] Pasta não encontrada: {args.results}")
        sys.exit(1)

    rows = load_rows(args.results)
    print_table(rows, sort_key=args.sort)

    if rows and args.csv:
        save_csv(rows, args.csv)

    if rows and args.json:
        summary = save_json(rows)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"[Salvo] {args.json}")

    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()