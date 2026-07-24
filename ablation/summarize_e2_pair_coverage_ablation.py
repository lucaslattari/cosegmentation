#!/usr/bin/env python3
"""
Resume baseline vs. E2 per-image kNN.

Lê metrics.json por classe e produz:
  - tabela no terminal;
  - CSV em ablation_results/e2_pair_coverage_summary.csv

Uso:
  python ablation/summarize_e2_pair_coverage_ablation.py \
      --baseline-suffix bg_recurrence_l0p50 \
      --classes person boat bottle bus \
      --k-values 2 4 8
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def metric_mean(data: dict, metric: str):
    # Estrutura atual esperada: aggregate -> metric -> mean
    aggregate = data.get("aggregate", {})
    value = aggregate.get(metric)

    if isinstance(value, dict):
        v = value.get("mean")
        if isinstance(v, (int, float)):
            return float(v)

    # Fallbacks para versões antigas.
    for key in (
        f"{metric}_mean",
        metric,
    ):
        v = aggregate.get(key)
        if isinstance(v, (int, float)):
            return float(v)
        v = data.get(key)
        if isinstance(v, (int, float)):
            return float(v)

    return None


def n_images(data: dict):
    for key in ("n_images", "num_images", "n_evaluated"):
        v = data.get(key)
        if isinstance(v, int):
            return v
    per_image = data.get("per_image")
    if isinstance(per_image, list):
        return len(per_image)
    return None


def load_metrics(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline-suffix",
        default="bg_recurrence_l0p50",
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=["person", "boat", "bottle", "bus"],
    )
    p.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[2, 4, 8],
    )
    p.add_argument(
        "--output",
        default="ablation_results/e2_pair_coverage_summary.csv",
    )
    args = p.parse_args()

    variants = [("baseline", None)] + [
        (f"kpi{k}", k) for k in args.k_values
    ]

    rows = []

    for class_name in args.classes:
        for variant, k in variants:
            if k is None:
                exp = f"pascal_coseg_{args.baseline_suffix}"
            else:
                exp = f"pascal_coseg_{args.baseline_suffix}_kpi{k}"

            path = (
                PROJECT_ROOT
                / "results"
                / exp
                / class_name
                / "metrics.json"
            )
            data = load_metrics(path)

            rows.append({
                "class": class_name,
                "variant": variant,
                "k_per_image": "" if k is None else k,
                "n_images": n_images(data) if data else "",
                "accuracy": metric_mean(data, "accuracy") if data else "",
                "f1": metric_mean(data, "f1") if data else "",
                "iou": metric_mean(data, "iou") if data else "",
                "metrics_path": str(path),
                "exists": bool(data),
            })

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 88)
    print("E2 PAIR COVERAGE — RESUMO")
    print("=" * 88)

    for class_name in args.classes:
        print(f"\n{class_name}")
        print(f"{'variante':<12} {'acc':>10} {'iou':>10} {'ΔIoU':>10}")
        print("-" * 46)

        class_rows = [r for r in rows if r["class"] == class_name]
        baseline = next(
            (r for r in class_rows if r["variant"] == "baseline"),
            None,
        )
        baseline_iou = (
            baseline["iou"]
            if baseline and isinstance(baseline["iou"], float)
            else None
        )

        for r in class_rows:
            acc = r["accuracy"]
            iou = r["iou"]
            delta = (
                iou - baseline_iou
                if isinstance(iou, float) and baseline_iou is not None
                else None
            )
            acc_s = f"{acc:.4f}" if isinstance(acc, float) else "-"
            iou_s = f"{iou:.4f}" if isinstance(iou, float) else "-"
            d_s = f"{delta:+.4f}" if delta is not None else "-"
            print(f"{r['variant']:<12} {acc_s:>10} {iou_s:>10} {d_s:>10}")

    print("\n" + "=" * 88)
    print("MÉDIA NAS CLASSES TESTADAS")
    print("=" * 88)

    for variant, _ in variants:
        vals = [
            r["iou"]
            for r in rows
            if r["variant"] == variant and isinstance(r["iou"], float)
        ]
        if vals:
            print(f"{variant:<12}: IoU médio = {mean(vals):.4f}  (n={len(vals)})")
        else:
            print(f"{variant:<12}: sem resultados")

    print(f"\n[Salvo] {out}")


if __name__ == "__main__":
    main()
