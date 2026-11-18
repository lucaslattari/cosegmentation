#!/usr/bin/env python3
"""
Compara os resultados de avaliação da Etapa 4 (SAM, uma passada) vs Etapa 5
(refinamento iterativo), gerados pela avaliação comparativa suplementar do
run_pipeline.sh (results/<experimento>/<Classe>/e4/metrics.json e .../e5/metrics.json).

Uso:
  python3 compare_e4_e5_final.py --results results/icoseg_full --csv comparacao_e4_e5.csv
"""
import argparse
import csv
import json
from pathlib import Path


def load_metrics(path: Path):
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # 1) formatos com agregado direto na raiz ou aninhado — só aceita se
    #    "iou" for mesmo um número; se for um dict/lista aninhado (formato
    #    diferente do esperado), ignora e cai no cálculo via per_image.
    for candidate in (data, data.get("metrics", {}), data.get("summary", {}), data.get("aggregate", {})):
        if isinstance(candidate, dict) and isinstance(candidate.get("iou"), (int, float)):
            return candidate

    # 2) formato real observado: {"collection":..., "per_image": [ {..., "iou":...}, ... ]}
    #    calcula media e desvio padrao direto da lista por imagem — mesmo
    #    calculo que o resumo_agregado.csv já usa (colunas iou/iou_std).
    per_image = data.get("per_image")
    if isinstance(per_image, list) and per_image:
        ious = [img["iou"] for img in per_image
               if isinstance(img, dict) and isinstance(img.get("iou"), (int, float))]
        if ious:
            mean = sum(ious) / len(ious)
            var = sum((x - mean) ** 2 for x in ious) / len(ious)
            return {
                "iou": mean,
                "iou_std": var ** 0.5,
                "n_images": len(ious),
            }

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path,
                    help="Pasta results/<experimento> (ex: results/icoseg_full)")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    class_dirs = sorted(d for d in args.results.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"[ERRO] Nenhuma subpasta de classe encontrada em {args.results}")
        return

    rows = []
    for cdir in class_dirs:
        m4 = load_metrics(cdir / "e4" / "metrics.json")
        m5 = load_metrics(cdir / "e5" / "metrics.json")
        if m4 is None and m5 is None:
            continue
        rows.append({
            "classe": cdir.name,
            "iou_e4": m4.get("iou") if m4 else None,
            "iou_e5": m5.get("iou") if m5 else None,
            "n_images": (m5 or m4 or {}).get("n_images"),
        })

    if not rows:
        print(f"[ERRO] Nenhum e4/metrics.json ou e5/metrics.json encontrado em {args.results}/*/")
        print("       Rode o run_pipeline.sh atualizado (avaliação comparativa E4/E5) primeiro.")
        return

    print(f"{'classe':45s} {'iou_e4':>8s} {'iou_e5':>8s} {'delta':>8s}")
    print("-" * 75)
    deltas = []
    for r in sorted(rows, key=lambda r: r["classe"]):
        e4 = r["iou_e4"]; e5 = r["iou_e5"]
        e4s = f"{e4:.4f}" if e4 is not None else "n/a"
        e5s = f"{e5:.4f}" if e5 is not None else "n/a"
        if e4 is not None and e5 is not None:
            d = e5 - e4
            deltas.append(d)
            ds = f"{d:+.4f}"
        else:
            ds = "n/a"
        print(f"{r['classe'][:45]:45s} {e4s:>8s} {e5s:>8s} {ds:>8s}")

    if deltas:
        print("-" * 75)
        media = sum(deltas) / len(deltas)
        n_melhor = sum(1 for d in deltas if d > 0.01)
        n_pior = sum(1 for d in deltas if d < -0.01)
        n_neutro = len(deltas) - n_melhor - n_pior
        print(f"{'MÉDIA DO DELTA (E5 - E4)':45s} {'':>8s} {'':>8s} {media:>+8.4f}")
        print(f"\nE5 melhorou {n_melhor}, piorou {n_pior}, neutro em {n_neutro} (de {len(deltas)} classes com os dois resultados)")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["classe", "n_images", "iou_e4", "iou_e5", "delta"])
            w.writeheader()
            for r in sorted(rows, key=lambda r: r["classe"]):
                d = (r["iou_e5"] - r["iou_e4"]) if (r["iou_e4"] is not None and r["iou_e5"] is not None) else None
                w.writerow({**r, "delta": d})
        print(f"\n[Salvo] {args.csv}")


if __name__ == "__main__":
    main()
