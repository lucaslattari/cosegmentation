#!/usr/bin/env python3
"""
compare_e3_adaptive_bypass_results.py
=====================================

Compara metrics.json por classe entre baseline e adaptive bypass.

Uso PASCAL:
  python ablation/compare_e3_adaptive_bypass_results.py \
      --baseline-root results/pascal_coseg_bg_recurrence_l0p50 \
      --candidate-root results/pascal_coseg_bg_recurrence_l0p50_e3bypass_top20 \
      --output ablation_results/compare_pascal_e3bypass.csv

Uso iCoseg controles:
  python ablation/compare_e3_adaptive_bypass_results.py \
      --baseline-root results/icoseg_full_bg_recurrence_l0p50 \
      --candidate-root results/icoseg_full_bg_recurrence_l0p50_e3bypass_top20 \
      --output ablation_results/compare_icoseg_controls_e3bypass.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def extract_metric(data: dict, name: str):
    # Formato atual esperado.
    aggregate = data.get("aggregate", {})
    value = aggregate.get(name)
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, (int, float)):
        return float(value)

    # Fallbacks de versões anteriores.
    for key in (name, f"{name}_mean"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        value = aggregate.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    return None


def load_metrics(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_classes(root: Path):
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "metrics.json").exists()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-root", required=True, type=Path)
    p.add_argument("--candidate-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--classes", nargs="*")
    args = p.parse_args()

    baseline_root = args.baseline_root.resolve()
    candidate_root = args.candidate_root.resolve()

    if args.classes:
        classes = args.classes
    else:
        classes = sorted(
            set(discover_classes(baseline_root))
            | set(discover_classes(candidate_root))
        )

    rows = []
    for cls in classes:
        base_path = baseline_root / cls / "metrics.json"
        cand_path = candidate_root / cls / "metrics.json"

        base = load_metrics(base_path)
        cand = load_metrics(cand_path)

        base_iou = extract_metric(base, "iou") if base else None
        cand_iou = extract_metric(cand, "iou") if cand else None
        base_acc = extract_metric(base, "accuracy") if base else None
        cand_acc = extract_metric(cand, "accuracy") if cand else None

        rows.append({
            "class": cls,
            "baseline_iou": base_iou,
            "candidate_iou": cand_iou,
            "delta_iou": (
                cand_iou - base_iou
                if base_iou is not None and cand_iou is not None
                else None
            ),
            "baseline_accuracy": base_acc,
            "candidate_accuracy": cand_acc,
            "delta_accuracy": (
                cand_acc - base_acc
                if base_acc is not None and cand_acc is not None
                else None
            ),
            "baseline_exists": base is not None,
            "candidate_exists": cand is not None,
        })

    valid = [
        r for r in rows
        if r["baseline_iou"] is not None and r["candidate_iou"] is not None
    ]

    print("=" * 88)
    print("BASELINE vs E3 ADAPTIVE BYPASS")
    print("=" * 88)
    print(
        f"{'Classe':<28}"
        f"{'Baseline':>11}"
        f"{'Candidate':>11}"
        f"{'Δ IoU':>11}"
    )
    print("-" * 88)

    for r in sorted(
        valid,
        key=lambda x: x["delta_iou"],
        reverse=True,
    ):
        print(
            f"{r['class']:<28}"
            f"{r['baseline_iou']:>11.4f}"
            f"{r['candidate_iou']:>11.4f}"
            f"{r['delta_iou']:>+11.4f}"
        )

    if valid:
        macro_base = mean(r["baseline_iou"] for r in valid)
        macro_cand = mean(r["candidate_iou"] for r in valid)
        n_better = sum(r["delta_iou"] > 1e-9 for r in valid)
        n_worse = sum(r["delta_iou"] < -1e-9 for r in valid)
        n_same = len(valid) - n_better - n_worse

        print("-" * 88)
        print(
            f"{'MACRO':<28}"
            f"{macro_base:>11.4f}"
            f"{macro_cand:>11.4f}"
            f"{macro_cand-macro_base:>+11.4f}"
        )
        print()
        print(
            f"Melhoraram: {n_better} | "
            f"Iguais: {n_same} | "
            f"Pioraram: {n_worse} | "
            f"Comparadas: {len(valid)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Salvo] {args.output.resolve()}")


if __name__ == "__main__":
    main()
