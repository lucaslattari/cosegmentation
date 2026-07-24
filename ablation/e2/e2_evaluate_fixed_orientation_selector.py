#!/usr/bin/env python3
"""
evaluate_fixed_orientation_selector.py

Avalia o seletor fixo CLS vs Spatial usando três fontes:

1) fixed_orientation_predictions.csv
   - predição do seletor
   - coluna is_training_class

2) baseline JSON
   - resultados do pipeline com orientação CLS/baseline
   - formato esperado:
       {
         "Classe": {
           "iou": ...,
           ...
         }
       }

3) spatial CSV
   - resultados do pipeline com orientação Spatial
   - precisa conter colunas de classe e IoU

O script calcula, separadamente para:
  - validação independente (is_training_class=0)
  - conjunto de treino (is_training_class=1)
  - todas as classes

Métricas:
  - accuracy do seletor
  - balanced accuracy
  - matriz de confusão
  - IoU média usando sempre CLS
  - IoU média usando sempre Spatial
  - IoU média usando o seletor
  - IoU média Oracle = max(CLS, Spatial) por classe
  - gap seletor -> Oracle
  - ganho do seletor sobre sempre CLS
  - ganho do seletor sobre sempre Spatial

Também gera CSV detalhado por classe.

Exemplo:

python ablation/evaluate_fixed_orientation_selector.py \
    --predictions ablation_results/fixed_orientation_selector/fixed_orientation_predictions.csv \
    --baseline_json results/icoseg_full_e5_v2/all_classes.json \
    --spatial_csv results/icoseg_full_e2_spatial/resumo_agregado.csv \
    --output ablation_results/fixed_orientation_selector_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
except ImportError as exc:
    raise SystemExit(
        "scikit-learn é necessário. Instale com: pip install scikit-learn"
    ) from exc


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def safe_float(value: Any) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return float("nan")


def normalize_class_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def detect_column(fieldnames, candidates):
    lower_map = {str(x).lower(): str(x) for x in fieldnames}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


# ---------------------------------------------------------------------------
# Leitura dos resultados
# ---------------------------------------------------------------------------

def load_baseline_json(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))

    result = {}

    if not isinstance(data, dict):
        raise ValueError("O baseline JSON deve ser um objeto/dicionário.")

    for cls, payload in data.items():
        if not isinstance(payload, dict):
            continue

        if "iou" not in payload:
            continue

        iou = safe_float(payload["iou"])

        if math.isfinite(iou):
            result[normalize_class_name(cls)] = iou

    return result


def load_spatial_csv(path: Path) -> Dict[str, float]:
    rows = read_csv(path)

    if not rows:
        raise ValueError("Spatial CSV vazio.")

    fields = rows[0].keys()

    class_col = detect_column(
        fields,
        ["class", "classe", "collection", "colecao", "coleção"],
    )

    iou_col = detect_column(
        fields,
        ["iou", "IoU", "iou_mean", "mean_iou"],
    )

    if class_col is None or iou_col is None:
        raise ValueError(
            f"Não consegui identificar colunas de classe/IoU em {path}. "
            f"Colunas encontradas: {list(fields)}"
        )

    result = {}

    for row in rows:
        cls = normalize_class_name(row[class_col])
        iou = safe_float(row[iou_col])

        if math.isfinite(iou):
            result[cls] = iou

    return result


# ---------------------------------------------------------------------------
# Junção das fontes
# ---------------------------------------------------------------------------

def build_evaluation_rows(
    predictions_path: Path,
    baseline_iou: Dict[str, float],
    spatial_iou: Dict[str, float],
    tie_delta: float,
) -> List[Dict[str, Any]]:
    pred_rows = read_csv(predictions_path)

    output = []

    for row in pred_rows:
        cls_display = row["class"]
        cls_norm = normalize_class_name(cls_display)

        if cls_norm not in baseline_iou:
            print(f"[AVISO] Sem baseline CLS para: {cls_display}")
            continue

        if cls_norm not in spatial_iou:
            print(f"[AVISO] Sem resultado Spatial para: {cls_display}")
            continue

        pred = row["predicted_orientation"].strip()

        if pred not in {"cls", "spatial"}:
            print(
                f"[AVISO] Predição inválida para {cls_display}: {pred}"
            )
            continue

        iou_cls = baseline_iou[cls_norm]
        iou_spatial = spatial_iou[cls_norm]

        delta = iou_spatial - iou_cls

        if abs(delta) < tie_delta:
            true_winner = "tie"
        else:
            true_winner = "spatial" if delta > 0 else "cls"

        selected_iou = (
            iou_cls
            if pred == "cls"
            else iou_spatial
        )

        oracle_iou = max(
            iou_cls,
            iou_spatial,
        )

        output.append({
            "class": cls_display,
            "is_training_class": int(
                float(row.get("is_training_class", 0))
            ),
            "predicted_orientation": pred,
            "true_winner": true_winner,
            "correct": (
                ""
                if true_winner == "tie"
                else int(pred == true_winner)
            ),
            "iou_cls": iou_cls,
            "iou_spatial": iou_spatial,
            "iou_delta_spatial_minus_cls": delta,
            "iou_selector": selected_iou,
            "iou_oracle": oracle_iou,
            "selector_gap_to_oracle": (
                oracle_iou - selected_iou
            ),
            "prob_cls": safe_float(
                row.get("prob_cls")
            ),
            "prob_spatial": safe_float(
                row.get("prob_spatial")
            ),
            "confidence": safe_float(
                row.get("confidence")
            ),
        })

    return output


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def evaluate_subset(
    rows: List[Dict[str, Any]],
    subset_name: str,
) -> Dict[str, Any]:
    if not rows:
        return {
            "subset": subset_name,
            "n_classes": 0,
        }

    labeled = [
        r for r in rows
        if r["true_winner"] in {"cls", "spatial"}
    ]

    y_true = [
        r["true_winner"]
        for r in labeled
    ]

    y_pred = [
        r["predicted_orientation"]
        for r in labeled
    ]

    if labeled:
        accuracy = float(
            np.mean([
                int(a == b)
                for a, b in zip(y_true, y_pred)
            ])
        )

        balanced = float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=["cls", "spatial"],
        ).tolist()
    else:
        accuracy = float("nan")
        balanced = float("nan")
        cm = [[0, 0], [0, 0]]

    mean_cls = float(
        np.mean([
            r["iou_cls"]
            for r in rows
        ])
    )

    mean_spatial = float(
        np.mean([
            r["iou_spatial"]
            for r in rows
        ])
    )

    mean_selector = float(
        np.mean([
            r["iou_selector"]
            for r in rows
        ])
    )

    mean_oracle = float(
        np.mean([
            r["iou_oracle"]
            for r in rows
        ])
    )

    return {
        "subset": subset_name,
        "n_classes": len(rows),
        "n_labeled_for_accuracy": len(labeled),
        "n_ties": len(rows) - len(labeled),

        "accuracy": accuracy,
        "balanced_accuracy": balanced,

        "confusion_matrix_labels": [
            "cls",
            "spatial",
        ],
        "confusion_matrix": cm,

        "mean_iou_always_cls": mean_cls,
        "mean_iou_always_spatial": mean_spatial,
        "mean_iou_selector": mean_selector,
        "mean_iou_oracle": mean_oracle,

        "selector_gain_vs_cls": (
            mean_selector - mean_cls
        ),
        "selector_gain_vs_spatial": (
            mean_selector - mean_spatial
        ),
        "selector_gap_to_oracle": (
            mean_oracle - mean_selector
        ),
    }


def print_report(
    report: Dict[str, Any],
) -> None:
    print("\n" + "=" * 110)
    print("AVALIAÇÃO DO SELETOR FIXO CLS vs SPATIAL")
    print("=" * 110)

    for subset in report["subsets"]:
        print()
        print("-" * 110)
        print(
            f"{subset['subset']} "
            f"(n={subset['n_classes']})"
        )
        print("-" * 110)

        if subset["n_classes"] == 0:
            print("Nenhuma classe disponível.")
            continue

        acc = subset["accuracy"]
        bal = subset["balanced_accuracy"]

        if math.isfinite(acc):
            print(
                f"Accuracy               : "
                f"{acc:.4f}"
            )
            print(
                f"Balanced accuracy      : "
                f"{bal:.4f}"
            )

        print(
            f"IoU média — sempre CLS : "
            f"{subset['mean_iou_always_cls']:.4f}"
        )
        print(
            f"IoU média — sempre SPA : "
            f"{subset['mean_iou_always_spatial']:.4f}"
        )
        print(
            f"IoU média — seletor    : "
            f"{subset['mean_iou_selector']:.4f}"
        )
        print(
            f"IoU média — oracle     : "
            f"{subset['mean_iou_oracle']:.4f}"
        )

        print(
            f"Ganho seletor vs CLS   : "
            f"{subset['selector_gain_vs_cls']:+.4f}"
        )
        print(
            f"Ganho seletor vs SPA   : "
            f"{subset['selector_gain_vs_spatial']:+.4f}"
        )
        print(
            f"Gap seletor → oracle   : "
            f"{subset['selector_gap_to_oracle']:.4f}"
        )

        print(
            f"Empates (< tie_delta)  : "
            f"{subset['n_ties']}"
        )

    print("\n" + "=" * 110)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Avalia o seletor fixo CLS vs Spatial contra os resultados "
            "reais dos dois pipelines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="fixed_orientation_predictions.csv",
    )

    p.add_argument(
        "--baseline_json",
        required=True,
        type=Path,
        help="JSON com resultados do pipeline CLS/baseline.",
    )

    p.add_argument(
        "--spatial_csv",
        required=True,
        type=Path,
        help="CSV com resultados do pipeline Spatial.",
    )

    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Diretório de saída.",
    )

    p.add_argument(
        "--tie_delta",
        type=float,
        default=0.01,
        help=(
            "Diferença absoluta de IoU abaixo da qual CLS e Spatial "
            "são tratados como empate para a métrica de classificação."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        raise FileNotFoundError(
            f"Predictions não encontrado: {args.predictions}"
        )

    if not args.baseline_json.exists():
        raise FileNotFoundError(
            f"Baseline JSON não encontrado: {args.baseline_json}"
        )

    if not args.spatial_csv.exists():
        raise FileNotFoundError(
            f"Spatial CSV não encontrado: {args.spatial_csv}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline_iou = load_baseline_json(
        args.baseline_json
    )

    spatial_iou = load_spatial_csv(
        args.spatial_csv
    )

    rows = build_evaluation_rows(
        predictions_path=args.predictions,
        baseline_iou=baseline_iou,
        spatial_iou=spatial_iou,
        tie_delta=args.tie_delta,
    )

    if not rows:
        raise RuntimeError(
            "Nenhuma classe pôde ser avaliada."
        )

    independent = [
        r for r in rows
        if r["is_training_class"] == 0
    ]

    training = [
        r for r in rows
        if r["is_training_class"] == 1
    ]

    report = {
        "tie_delta": args.tie_delta,
        "n_total_classes": len(rows),
        "subsets": [
            evaluate_subset(
                independent,
                "VALIDAÇÃO INDEPENDENTE",
            ),
            evaluate_subset(
                training,
                "CLASSES DE TREINO",
            ),
            evaluate_subset(
                rows,
                "TODAS AS CLASSES",
            ),
        ],
    }

    detail_path = (
        args.output
        / "fixed_orientation_evaluation_by_class.csv"
    )

    report_path = (
        args.output
        / "fixed_orientation_evaluation_report.json"
    )

    write_csv(
        detail_path,
        rows,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print_report(
        report
    )

    print("Arquivos salvos:")
    print(
        f"  {detail_path}"
    )
    print(
        f"  {report_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrompido pelo usuário."
        )
        sys.exit(130)
    except Exception as exc:
        print(
            f"\n[ERRO] {exc}"
        )
        sys.exit(1)
