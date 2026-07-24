#!/usr/bin/env python3
"""
run_fixed_orientation_selector.py

Treina um seletor FIXO de orientação por coleção usando exatamente:

  Modelo:
    Logistic Regression

  Features:
    1. semantic_advantage_spatial_mean
    2. border_advantage_spatial

O objetivo é congelar a regra encontrada na etapa de seleção de features
e aplicá-la a novas coleções sem usar GT como entrada.

Entradas:
  --train_labels
      collection_orientation_labels.csv contendo as classes de desenvolvimento
      e a coluna "winner" (cls/spatial). O GT é usado apenas para treinar o
      seletor offline.

  --selector_root
      Pasta contendo, para cada coleção:
        orientation_selector_summary.json
        orientation_selector_pairs.csv

      Pode conter tanto as classes de treino quanto novas classes.

Saídas:
  fixed_orientation_predictions.csv
  fixed_orientation_model.json

Exemplo:

  python ablation/run_fixed_orientation_selector.py \
      --train_labels ablation_results/collection_orientation_selector/collection_orientation_labels.csv \
      --selector_root ablation_results/orientation_selector \
      --output ablation_results/fixed_orientation_selector

IMPORTANTE:
  As duas features usadas pelo modelo são extraídas SEM GT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "scikit-learn é necessário. Instale com: pip install scikit-learn"
    ) from exc


FEATURES = [
    "semantic_advantage_spatial_mean",
    "border_advantage_spatial",
]


def safe_float(value: Any) -> float:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return float("nan")


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


def normalize_class_name(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def choose_side_value(row: Dict[str, Any], prefix: str, choice: int) -> float:
    return safe_float(row.get(f"{prefix}_side{choice}"))


def extract_two_features(selector_root: Path) -> List[Dict[str, Any]]:
    summaries = sorted(
        selector_root.rglob("orientation_selector_summary.json")
    )

    if not summaries:
        raise FileNotFoundError(
            f"Nenhum orientation_selector_summary.json encontrado em {selector_root}"
        )

    output = []

    for summary_path in summaries:
        pair_path = summary_path.parent / "orientation_selector_pairs.csv"

        if not pair_path.exists():
            print(
                f"[AVISO] Ignorando {summary_path.parent.name}: "
                "orientation_selector_pairs.csv ausente"
            )
            continue

        pairs = read_csv(pair_path)

        if not pairs:
            continue

        semantic_advantages = []
        border_advantages = []

        for row in pairs:
            cls_choice = int(float(row["cls_choice"]))
            spatial_choice = int(float(row["spatial_choice"]))

            cls_sem = choose_side_value(
                row,
                "semantic_margin",
                cls_choice,
            )
            spatial_sem = choose_side_value(
                row,
                "semantic_margin",
                spatial_choice,
            )

            cls_border = choose_side_value(
                row,
                "border_contact",
                cls_choice,
            )
            spatial_border = choose_side_value(
                row,
                "border_contact",
                spatial_choice,
            )

            semantic_advantages.append(
                spatial_sem - cls_sem
            )

            # Positivo = Spatial toca menos borda que CLS.
            border_advantages.append(
                cls_border - spatial_border
            )

        output.append({
            "class": summary_path.parent.name,
            "semantic_advantage_spatial_mean": mean_or_nan(
                semantic_advantages
            ),
            "border_advantage_spatial": mean_or_nan(
                border_advantages
            ),
        })

    return output


def load_training_rows(
    labels_path: Path,
) -> List[Dict[str, Any]]:
    rows = read_csv(labels_path)

    result = []

    for row in rows:
        winner = row.get("winner", "").strip()

        if winner not in {"cls", "spatial"}:
            continue

        item = {
            "class": row["class"],
            "winner": winner,
        }

        for feature in FEATURES:
            value = safe_float(row.get(feature))

            if not math.isfinite(value):
                raise ValueError(
                    f"Feature '{feature}' ausente ou inválida "
                    f"na classe '{row['class']}'."
                )

            item[feature] = value

        result.append(item)

    if len(result) < 4:
        raise RuntimeError(
            f"Poucas classes de treino válidas: {len(result)}"
        )

    if len({r["winner"] for r in result}) < 2:
        raise RuntimeError(
            "É preciso haver exemplos CLS e Spatial no treino."
        )

    return result


def build_model() -> Pipeline:
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def matrix(
    rows: List[Dict[str, Any]],
) -> np.ndarray:
    return np.array(
        [
            [
                safe_float(row[feature])
                for feature in FEATURES
            ]
            for row in rows
        ],
        dtype=float,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Treina o seletor fixo Logistic Regression com duas features "
            "e aplica às coleções disponíveis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--train_labels",
        required=True,
        type=Path,
        help=(
            "collection_orientation_labels.csv usado para treinar "
            "o seletor fixo."
        ),
    )

    parser.add_argument(
        "--selector_root",
        required=True,
        type=Path,
        help=(
            "Raiz contendo os resultados do analyze_orientation_selector.py."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Diretório de saída.",
    )

    args = parser.parse_args()

    if not args.train_labels.exists():
        raise FileNotFoundError(
            f"Arquivo de treino não encontrado: {args.train_labels}"
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_rows = load_training_rows(
        args.train_labels
    )

    target_rows = extract_two_features(
        args.selector_root
    )

    if not target_rows:
        raise RuntimeError(
            "Nenhuma coleção disponível para predição."
        )

    model = build_model()

    X_train = matrix(train_rows)
    y_train = np.array(
        [r["winner"] for r in train_rows],
        dtype=object,
    )

    model.fit(
        X_train,
        y_train,
    )

    X_target = matrix(target_rows)

    predictions = model.predict(
        X_target
    )

    probabilities = model.predict_proba(
        X_target
    )

    classes = list(
        model.classes_
    )

    train_names = {
        normalize_class_name(r["class"])
        for r in train_rows
    }

    output_rows = []

    for idx, row in enumerate(target_rows):
        pred = str(predictions[idx])

        prob_map = {
            cls: float(
                probabilities[idx, class_idx]
            )
            for class_idx, cls in enumerate(classes)
        }

        prob_cls = prob_map.get(
            "cls",
            float("nan"),
        )
        prob_spatial = prob_map.get(
            "spatial",
            float("nan"),
        )

        confidence = max(
            prob_cls,
            prob_spatial,
        )

        is_train_class = (
            normalize_class_name(row["class"])
            in train_names
        )

        output_rows.append({
            "class": row["class"],
            "predicted_orientation": pred,
            "prob_cls": prob_cls,
            "prob_spatial": prob_spatial,
            "confidence": confidence,
            "is_training_class": int(
                is_train_class
            ),
            "semantic_advantage_spatial_mean": row[
                "semantic_advantage_spatial_mean"
            ],
            "border_advantage_spatial": row[
                "border_advantage_spatial"
            ],
        })

    predictions_path = (
        args.output
        / "fixed_orientation_predictions.csv"
    )

    write_csv(
        predictions_path,
        output_rows,
    )

    # Salva parâmetros do modelo para reprodutibilidade.
    scaler = model.named_steps["scaler"]
    logistic = model.named_steps["model"]

    model_info = {
        "model": "LogisticRegression",
        "features": FEATURES,
        "n_training_classes": len(
            train_rows
        ),
        "training_classes": [
            r["class"]
            for r in train_rows
        ],
        "class_order": classes,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": logistic.coef_.tolist(),
        "intercept": logistic.intercept_.tolist(),
        "note": (
            "Modelo congelado usando exatamente duas features não supervisionadas. "
            "As classes marcadas is_training_class=1 participaram do treino; "
            "use is_training_class=0 para validação independente."
        ),
    }

    model_path = (
        args.output
        / "fixed_orientation_model.json"
    )

    model_path.write_text(
        json.dumps(
            model_info,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print(
        "SELETOR FIXO DE ORIENTAÇÃO "
        "— LOGISTIC REGRESSION"
    )
    print("=" * 100)
    print(
        f"Classes de treino : "
        f"{len(train_rows)}"
    )
    print(
        f"Coleções preditas : "
        f"{len(output_rows)}"
    )
    print(
        "Features:"
    )
    for feature in FEATURES:
        print(
            f"  - {feature}"
        )

    print("\n" + "-" * 100)
    print(
        f"{'CLASSE':55} "
        f"{'PRED':8} "
        f"{'P(CLS)':>8} "
        f"{'P(SPA)':>8} "
        f"{'CONF':>8} "
        f"{'TREINO':>7}"
    )
    print("-" * 100)

    for row in output_rows:
        print(
            f"{row['class'][:55]:55} "
            f"{row['predicted_orientation']:8} "
            f"{row['prob_cls']:8.3f} "
            f"{row['prob_spatial']:8.3f} "
            f"{row['confidence']:8.3f} "
            f"{row['is_training_class']:7d}"
        )

    print("\nArquivos salvos:")
    print(
        f"  {predictions_path}"
    )
    print(
        f"  {model_path}"
    )

    print(
        "\nPara validação independente, considere apenas "
        "as linhas com is_training_class=0."
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
