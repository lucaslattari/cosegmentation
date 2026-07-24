#!/usr/bin/env python3
"""
aggregate_step2_orientation.py

Agrega múltiplos arquivos step2_quality.json produzidos por
analyze_step2_correspondence.py em:

  - step2_orientation_summary.csv
  - step2_orientation_summary.json

Uso:
  python ablation/aggregate_step2_orientation.py \
      --input ablation_results/step2_orientation \
      --output ablation_results/step2_orientation

O script procura recursivamente por arquivos chamados step2_quality.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# Colunas principais a extrair de metrics.
METRIC_COLUMNS = [
    "n_images",
    "n_patches",
    "n_pairs",
    "polarization_fraction_decided",
    "polarization_bimodality_score",
    "fg_fraction_mean",
    "pair_agreement_mean",
    "stability_loo_mae_mean",
    "cls_correlation_mean",
    "gt_best_iou",
    "gt_best_threshold",
    "gt_iou_at_threshold_05",
    "gt_auc_roc",
    "gt_best_iou_raw",
    "gt_best_iou_inverted",
    "gt_best_iou_oracle_polarity",
    "gt_auc_roc_raw",
    "gt_auc_roc_inverted",
    "gt_auc_roc_oracle_polarity",
    "orientation_accuracy",
    "orientation_accuracy_size_based",
    "orientation_accuracy_spatial_based",
    "orientation_accuracy_combined",
    "orientation_best_heuristic",
    "orientation_best_accuracy",
    "orientation_improvement_over_baseline",
    "pair_orientation_selected_iou_mean",
    "pair_orientation_inverted_iou_mean",
    "pair_orientation_oracle_iou_mean",
    "pair_orientation_loss_mean",
    "oracle_orientation_imagewise_aggregated_best_iou",
    "oracle_orientation_pairwise_aggregated_best_iou",
    "pair_coverage_min",
    "pair_coverage_max",
    "pair_coverage_mean",
    "pair_coverage_n_images_without_pairs",
]


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def first_existing(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def classify_failure(metrics: Dict[str, Any]) -> str:
    """
    Classificação heurística útil para leitura rápida da tabela consolidada.

    - good: resultado raw já é forte.
    - orientation_failure: raw ruim, mas oracle de polaridade/orientação melhora muito.
    - structural_failure: nem oracle de polaridade/orientação salva.
    - intermediate: demais casos.
    """
    raw_iou = safe_float(first_existing(
        metrics,
        "gt_best_iou_raw",
        "gt_best_iou",
    ))
    oracle_iou = safe_float(first_existing(
        metrics,
        "oracle_orientation_pairwise_aggregated_best_iou",
        "gt_best_iou_oracle_polarity",
    ))
    raw_auc = safe_float(first_existing(
        metrics,
        "gt_auc_roc_raw",
        "gt_auc_roc",
    ))
    oracle_auc = safe_float(metrics.get("gt_auc_roc_oracle_polarity"))

    if raw_iou is not None and raw_iou >= 0.55:
        return "good"

    if raw_iou is not None and oracle_iou is not None:
        gain = oracle_iou - raw_iou
        if raw_iou < 0.35 and oracle_iou >= 0.45 and gain >= 0.20:
            return "orientation_failure"
        if raw_iou < 0.35 and oracle_iou < 0.35:
            return "structural_failure"

    if raw_auc is not None and oracle_auc is not None:
        if raw_auc < 0.35 and oracle_auc >= 0.80:
            return "orientation_failure"

    return "intermediate"


def read_quality_file(path: Path, input_root: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("'metrics' não é um objeto JSON.")

    try:
        relative_parent = path.parent.relative_to(input_root)
        class_name = relative_parent.as_posix()
    except ValueError:
        class_name = path.parent.name

    # Se o JSON estiver diretamente na raiz, tenta inferir pela pasta
    # do correspondence_path.
    if class_name in ("", "."):
        corr_path = metrics.get("correspondence_path")
        if corr_path:
            try:
                class_name = Path(str(corr_path)).parent.name
            except Exception:
                class_name = path.parent.name
        else:
            class_name = path.parent.name

    row: Dict[str, Any] = {
        "class": class_name,
        "source_json": str(path),
        "diagnosis": payload.get("diagnosis"),
        "failure_profile": classify_failure(metrics),
    }

    for key in METRIC_COLUMNS:
        row[key] = metrics.get(key)

    # Compatibilidade com versões em que algumas métricas tinham outros nomes.
    aliases = {
        "gt_best_iou_raw": ("gt_best_iou_raw", "gt_best_iou"),
        "gt_auc_roc_raw": ("gt_auc_roc_raw", "gt_auc_roc"),
        "pair_coverage_mean": ("pair_coverage_mean", "pair_coverage_pairs_per_image_mean"),
        "pair_coverage_min": ("pair_coverage_min", "pair_coverage_pairs_per_image_min"),
        "pair_coverage_max": ("pair_coverage_max", "pair_coverage_pairs_per_image_max"),
    }
    for target, keys in aliases.items():
        if row.get(target) is None:
            row[target] = first_existing(metrics, *keys)

    return row


def discover_quality_files(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.rglob("step2_quality.json")
        if p.is_file()
    )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "class",
        "failure_profile",
        *METRIC_COLUMNS,
        "diagnosis",
        "source_json",
    ]

    # Evita colunas duplicadas caso um nome apareça duas vezes.
    fieldnames = list(dict.fromkeys(fieldnames))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: List[Dict[str, Any]], path: Path) -> None:
    summary = {
        "n_classes": len(rows),
        "classes": rows,
    }
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 78)
    print("RESUMO CONSOLIDADO — ETAPA 2 / ORIENTAÇÃO")
    print("=" * 78)

    counts: Dict[str, int] = {}
    for row in rows:
        profile = str(row.get("failure_profile", "unknown"))
        counts[profile] = counts.get(profile, 0) + 1

    print(f"Classes agregadas: {len(rows)}")
    for profile in ["good", "orientation_failure", "structural_failure", "intermediate"]:
        if profile in counts:
            print(f"  {profile:>20}: {counts[profile]}")

    print("\nPrincipais métricas:")
    print(
        f"{'classe':40} {'perfil':20} "
        f"{'IoU raw':>8} {'IoU oracle':>10} {'AUC raw':>8} {'AUC oracle':>10} {'orient':>8}"
    )
    print("-" * 112)

    for row in rows:
        raw_iou = safe_float(first_existing(row, "gt_best_iou_raw", "gt_best_iou"))
        oracle_iou = safe_float(first_existing(
            row,
            "oracle_orientation_pairwise_aggregated_best_iou",
            "gt_best_iou_oracle_polarity",
        ))
        raw_auc = safe_float(first_existing(row, "gt_auc_roc_raw", "gt_auc_roc"))
        oracle_auc = safe_float(row.get("gt_auc_roc_oracle_polarity"))
        orient = safe_float(row.get("orientation_accuracy"))

        def fmt(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:.4f}"

        class_name = str(row["class"])
        if len(class_name) > 40:
            class_name = class_name[:37] + "..."

        print(
            f"{class_name:40} {str(row['failure_profile']):20} "
            f"{fmt(raw_iou):>8} {fmt(oracle_iou):>10} "
            f"{fmt(raw_auc):>8} {fmt(oracle_auc):>10} {fmt(orient):>8}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agrega step2_quality.json de múltiplas classes em CSV e JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Diretório raiz contendo as subpastas com step2_quality.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Diretório de saída. Default: mesmo diretório de --input.",
    )
    parser.add_argument(
        "--csv_name",
        default="step2_orientation_summary.csv",
        help="Nome do CSV consolidado.",
    )
    parser.add_argument(
        "--json_name",
        default="step2_orientation_summary.json",
        help="Nome do JSON consolidado.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input
    output_dir = args.output or input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_quality_files(input_dir)
    if not files:
        raise FileNotFoundError(
            f"Nenhum step2_quality.json encontrado recursivamente em: {input_dir}"
        )

    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for path in files:
        try:
            rows.append(read_quality_file(path, input_dir))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    if not rows:
        raise RuntimeError(
            "Nenhum JSON pôde ser agregado.\n" + "\n".join(errors)
        )

    # Ordenação útil: perfil e nome da classe.
    profile_order = {
        "orientation_failure": 0,
        "structural_failure": 1,
        "intermediate": 2,
        "good": 3,
    }
    rows.sort(
        key=lambda r: (
            profile_order.get(str(r.get("failure_profile")), 99),
            str(r.get("class", "")).lower(),
        )
    )

    csv_path = output_dir / args.csv_name
    json_path = output_dir / args.json_name

    write_csv(rows, csv_path)
    write_json(rows, json_path)

    print_summary(rows)

    print(f"\n[Salvo] CSV : {csv_path}")
    print(f"[Salvo] JSON: {json_path}")

    if errors:
        print("\n[Avisos] Alguns arquivos não puderam ser processados:")
        for err in errors:
            print(f"  - {err}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERRO] {exc}")
        sys.exit(1)
