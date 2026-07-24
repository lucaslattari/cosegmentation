#!/usr/bin/env python3
"""
select_orientation_features.py

Busca subconjuntos pequenos de features para o seletor CLS vs Spatial
por coleção, usando Leave-One-Out Cross-Validation (LOOCV).

Objetivo:
  Encontrar uma regra/modelo compacto, interpretável e menos sujeito a
  overfitting do que usar dezenas de features com poucas coleções.

Entrada esperada:
  collection_orientation_labels.csv

Esse CSV deve conter:
  - class
  - winner              ("cls", "spatial" ou "tie")
  - iou_cls
  - iou_spatial
  - iou_delta_spatial_minus_cls
  - features numéricas não supervisionadas

O script:
  1) remove colunas de rótulo/leakage;
  2) remove features constantes/quase inválidas;
  3) testa combinações de 1 até N features;
  4) avalia Logistic Regression e/ou árvores rasas com LOOCV;
  5) ranqueia por balanced accuracy, accuracy e tamanho do subconjunto;
  6) salva os melhores subconjuntos e as predições por classe.

IMPORTANTE:
  O winner é usado SOMENTE para avaliação offline.
  Nenhuma feature derivada de IoU/GT entra no modelo.

Exemplo:
  python ablation/select_orientation_features.py \
      --labels ablation_results/collection_orientation_selector/collection_orientation_labels.csv \
      --output ablation_results/orientation_feature_selection \
      --max_features 5 \
      --top_n 30
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text
except ImportError as exc:
    raise SystemExit(
        "scikit-learn é necessário. Instale com: pip install scikit-learn"
    ) from exc


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

LEAKAGE_COLUMNS = {
    "class",
    "winner",
    "iou_cls",
    "iou_spatial",
    "iou_delta_spatial_minus_cls",
}

# Features redundantes óbvias podem ser excluídas por padrão.
# Ex.: agreement e disagreement carregam a mesma informação.
DEFAULT_EXCLUDE = {
    "disagreement",
    "cls_confidence_mean_summary",
    "spatial_confidence_mean_summary",
}

DEFAULT_PREFERRED = [
    "agreement",
    "cls_confidence_mean",
    "spatial_confidence_mean",
    "confidence_gap_cls_minus_spatial",
    "semantic_advantage_spatial_mean",
    "semantic_advantage_spatial_std",
    "border_advantage_spatial",
    "centrality_advantage_spatial",
    "compactness_advantage_spatial",
    "spatial_score_advantage_mean",
    "disagreement_fraction",
    "dis_semantic_advantage_spatial",
    "dis_centrality_advantage_spatial",
    "dis_compactness_advantage_spatial",
    "dis_border_advantage_spatial",
    "semantic_prefers_cls_fraction_disagreement",
    "semantic_prefers_spatial_fraction_disagreement",
]


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


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: List[str] = []
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


def valid_feature_columns(
    rows: List[Dict[str, str]],
    exclude: set[str],
    preferred_only: bool,
) -> List[str]:
    if not rows:
        return []

    candidates = [
        c for c in rows[0].keys()
        if c not in LEAKAGE_COLUMNS and c not in exclude
    ]

    if preferred_only:
        candidates = [c for c in DEFAULT_PREFERRED if c in candidates]

    valid = []

    for col in candidates:
        vals = np.array([safe_float(r.get(col)) for r in rows], dtype=float)
        finite = vals[np.isfinite(vals)]

        # Exige ao menos metade dos valores válidos e alguma variância.
        if finite.size < max(3, len(rows) // 2):
            continue

        if np.nanstd(vals) < 1e-12:
            continue

        valid.append(col)

    return valid


def matrix_from_rows(
    rows: List[Dict[str, str]],
    features: Tuple[str, ...] | List[str],
) -> np.ndarray:
    return np.array(
        [[safe_float(r.get(f)) for f in features] for r in rows],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

def build_model(name: str, random_state: int = 42) -> Pipeline:
    if name == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])

    if name == "tree2":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=2,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])

    if name == "tree3":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])

    raise ValueError(f"Modelo desconhecido: {name}")


def evaluate_subset(
    rows: List[Dict[str, str]],
    features: Tuple[str, ...],
    model_name: str,
) -> Dict[str, Any]:
    X = matrix_from_rows(rows, features)
    y = np.array([r["winner"] for r in rows], dtype=object)

    loo = LeaveOneOut()
    model = build_model(model_name)

    preds = cross_val_predict(
        model,
        X,
        y,
        cv=loo,
        method="predict",
    )

    acc = float(accuracy_score(y, preds))
    bal = float(balanced_accuracy_score(y, preds))
    cm = confusion_matrix(y, preds, labels=["cls", "spatial"]).tolist()

    prediction_rows = []
    for row, pred in zip(rows, preds):
        prediction_rows.append({
            "class": row["class"],
            "truth": row["winner"],
            "predicted": str(pred),
            "correct": bool(pred == row["winner"]),
        })

    return {
        "model": model_name,
        "n_features": len(features),
        "features": list(features),
        "loocv_accuracy": acc,
        "loocv_balanced_accuracy": bal,
        "confusion_matrix_labels": ["cls", "spatial"],
        "confusion_matrix": cm,
        "predictions": prediction_rows,
    }


# ---------------------------------------------------------------------------
# Busca exaustiva
# ---------------------------------------------------------------------------

def search_feature_subsets(
    rows: List[Dict[str, str]],
    candidate_features: List[str],
    max_features: int,
    models: List[str],
) -> List[Dict[str, Any]]:
    results = []

    total = 0
    for k in range(1, min(max_features, len(candidate_features)) + 1):
        total += math.comb(len(candidate_features), k) * len(models)

    print(f"\nCombinações/modelos a avaliar: {total}")

    done = 0

    for k in range(1, min(max_features, len(candidate_features)) + 1):
        for subset in itertools.combinations(candidate_features, k):
            for model_name in models:
                result = evaluate_subset(rows, subset, model_name)
                results.append(result)

                done += 1
                if done % 100 == 0 or done == total:
                    print(f"  progresso: {done}/{total}")

    # Ordenação:
    # 1) maior balanced accuracy
    # 2) maior accuracy
    # 3) menos features
    # 4) nome do modelo
    results.sort(
        key=lambda r: (
            -r["loocv_balanced_accuracy"],
            -r["loocv_accuracy"],
            r["n_features"],
            r["model"],
        )
    )

    return results


# ---------------------------------------------------------------------------
# Estabilidade das features
# ---------------------------------------------------------------------------

def summarize_feature_frequency(
    results: List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    top = results[:top_n]
    counts: Dict[str, int] = {}

    for result in top:
        for feat in result["features"]:
            counts[feat] = counts.get(feat, 0) + 1

    rows = [
        {
            "feature": feat,
            "count_in_top_n": count,
            "fraction_in_top_n": count / len(top) if top else 0.0,
        }
        for feat, count in counts.items()
    ]

    rows.sort(
        key=lambda r: (
            -r["count_in_top_n"],
            r["feature"],
        )
    )

    return rows


# ---------------------------------------------------------------------------
# Interpretação da melhor árvore
# ---------------------------------------------------------------------------

def fit_best_tree_text(
    rows: List[Dict[str, str]],
    best_result: Dict[str, Any],
) -> str | None:
    if best_result["model"] not in {"tree2", "tree3"}:
        return None

    features = best_result["features"]
    X = matrix_from_rows(rows, features)
    y = np.array([r["winner"] for r in rows], dtype=object)

    model = build_model(best_result["model"])
    model.fit(X, y)

    imputer = model.named_steps["imputer"]
    tree = model.named_steps["model"]

    X_imp = imputer.transform(X)
    tree.fit(X_imp, y)

    return export_text(tree, feature_names=list(features))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Seleciona subconjuntos pequenos de features para prever CLS vs Spatial "
            "com LOOCV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="collection_orientation_labels.csv",
    )

    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Diretório de saída.",
    )

    p.add_argument(
        "--max_features",
        type=int,
        default=5,
        help="Número máximo de features por subconjunto.",
    )

    p.add_argument(
        "--top_n",
        type=int,
        default=30,
        help="Quantidade de melhores resultados salvos em detalhe.",
    )

    p.add_argument(
        "--models",
        nargs="+",
        default=["logistic", "tree2", "tree3"],
        choices=["logistic", "tree2", "tree3"],
        help="Modelos a testar.",
    )

    p.add_argument(
        "--all_features",
        action="store_true",
        help=(
            "Testa todas as features numéricas válidas. "
            "Sem essa flag, usa apenas um conjunto pré-selecionado e interpretável."
        ),
    )

    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Features adicionais a excluir.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.labels.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.labels}")

    args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.labels)

    # Remove empates.
    rows = [
        r for r in rows
        if r.get("winner") in {"cls", "spatial"}
    ]

    if len(rows) < 4:
        raise RuntimeError(
            f"Poucas classes rotuladas: {len(rows)}"
        )

    labels = {r["winner"] for r in rows}
    if len(labels) < 2:
        raise RuntimeError(
            "É preciso haver exemplos de CLS e Spatial."
        )

    exclude = set(DEFAULT_EXCLUDE) | set(args.exclude)

    features = valid_feature_columns(
        rows,
        exclude=exclude,
        preferred_only=not args.all_features,
    )

    print("\n" + "=" * 100)
    print("SELEÇÃO DE FEATURES — ORIENTATION SELECTOR")
    print("=" * 100)
    print(f"Classes rotuladas : {len(rows)}")
    print(f"Features candidatas: {len(features)}")
    print(f"Máximo por subset : {args.max_features}")
    print(f"Modelos           : {', '.join(args.models)}")
    print()

    for f in features:
        print(f"  - {f}")

    results = search_feature_subsets(
        rows=rows,
        candidate_features=features,
        max_features=args.max_features,
        models=args.models,
    )

    top_n = min(args.top_n, len(results))
    top_results = results[:top_n]

    # Resumo tabular.
    summary_rows = []
    for rank, r in enumerate(top_results, start=1):
        summary_rows.append({
            "rank": rank,
            "model": r["model"],
            "n_features": r["n_features"],
            "loocv_accuracy": r["loocv_accuracy"],
            "loocv_balanced_accuracy": r["loocv_balanced_accuracy"],
            "features": " | ".join(r["features"]),
        })

    write_csv(
        args.output / "feature_selection_top.csv",
        summary_rows,
    )

    frequency = summarize_feature_frequency(
        results,
        top_n=top_n,
    )

    write_csv(
        args.output / "feature_frequency_top.csv",
        frequency,
    )

    best = results[0]

    report = {
        "n_labeled_classes": len(rows),
        "candidate_features": features,
        "max_features": args.max_features,
        "models": args.models,
        "best_result": best,
        "top_results": top_results,
        "feature_frequency_top_n": frequency,
    }

    tree_text = fit_best_tree_text(rows, best)
    if tree_text is not None:
        report["best_tree_rule"] = tree_text

    (args.output / "feature_selection_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Predições LOOCV do melhor resultado.
    write_csv(
        args.output / "best_loocv_predictions.csv",
        best["predictions"],
    )

    print("\n" + "=" * 100)
    print("MELHORES SUBCONJUNTOS")
    print("=" * 100)

    for rank, r in enumerate(top_results[:10], start=1):
        print(
            f"{rank:2d}. "
            f"{r['model']:8} "
            f"n={r['n_features']} "
            f"acc={r['loocv_accuracy']:.3f} "
            f"bal_acc={r['loocv_balanced_accuracy']:.3f} "
            f"features={', '.join(r['features'])}"
        )

    print("\n" + "=" * 100)
    print("MELHOR RESULTADO")
    print("=" * 100)
    print(f"Modelo      : {best['model']}")
    print(f"N features  : {best['n_features']}")
    print(f"Accuracy    : {best['loocv_accuracy']:.3f}")
    print(f"Balanced acc: {best['loocv_balanced_accuracy']:.3f}")
    print("Features:")
    for f in best["features"]:
        print(f"  - {f}")

    print("\nPredições LOOCV:")
    for p in best["predictions"]:
        status = "OK" if p["correct"] else "ERRO"
        print(
            f"  {status:4} "
            f"{p['class'][:50]:50} "
            f"real={p['truth']:7} pred={p['predicted']}"
        )

    if tree_text is not None:
        print("\nRegra da árvore treinada em todas as classes:")
        print(tree_text)

    print("\nArquivos salvos:")
    print(f"  {args.output / 'feature_selection_top.csv'}")
    print(f"  {args.output / 'feature_frequency_top.csv'}")
    print(f"  {args.output / 'best_loocv_predictions.csv'}")
    print(f"  {args.output / 'feature_selection_report.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERRO] {exc}")
        sys.exit(1)
