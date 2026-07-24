#!/usr/bin/env python3
"""
analyze_collection_orientation_selector.py

Análise por COLEÇÃO para tentar decidir entre orientação CLS e Spatial
sem usar GT como entrada do seletor.

O script tem duas funções:

1) Extrair descritores NÃO SUPERVISIONADOS por coleção a partir dos arquivos:
     orientation_selector_summary.json
     orientation_selector_pairs.csv

2) Opcionalmente, em modo de PESQUISA/OFFLINE, comparar esses descritores com
   o vencedor real CLS vs Spatial obtido em avaliações anteriores.
   O GT entra SOMENTE para criar o rótulo de treinamento/avaliação offline.
   As features usadas pelo modelo continuam sendo 100% não supervisionadas.

Arquivos de saída:
  collection_orientation_features.csv
  collection_orientation_labels.csv               (se houver rótulos)
  collection_orientation_predictions.csv           (se houver rótulos)
  collection_orientation_model_report.json          (se houver rótulos)

Uso típico:

  python ablation/analyze_collection_orientation_selector.py \
      --selector_root ablation_results/orientation_selector \
      --baseline_json all_classes.json \
      --spatial_csv results/icoseg_full_e2_spatial/resumo_agregado.csv \
      --output ablation_results/collection_orientation_selector

O script aceita CSV com nomes de coluna flexíveis:
  classe / class / collection
  iou / IoU

O JSON baseline esperado pode ser:
  {
    "ClasseA": {"iou": 0.50, ...},
    "ClasseB": {"iou": 0.70, ...}
  }

Observação metodológica:
  Com poucas coleções rotuladas, qualquer classificador pode overfitar.
  Por isso o script avalia modelos simples com Leave-One-Out Cross-Validation
  e reporta explicitamente a acurácia fora da amostra.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:
    raise SystemExit(
        "scikit-learn é necessário para o modo de modelagem. "
        "Instale com: pip install scikit-learn"
    ) from exc


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isfinite(x):
            return x
        return default
    except (TypeError, ValueError):
        return default


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def std_or_nan(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.std(vals))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def choose_side_value(row: Dict[str, Any], prefix: str, choice: int) -> float:
    key = f"{prefix}_side{choice}"
    return safe_float(row.get(key))


def normalize_class_name(name: str) -> str:
    """
    Normaliza nomes de classe vindos de fontes diferentes.

    Exemplos que passam a casar:
      "Agra Taj Mahal"
      "Agra_Taj_Mahal"
      "Agra-Taj-Mahal"

    Também remove pontuação residual e colapsa separadores.
    """
    s = str(name).strip().lower()
    s = re.sub(r"[\s_\-]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Extração de features por coleção
# ---------------------------------------------------------------------------

def extract_collection_features(selector_root: Path) -> List[Dict[str, Any]]:
    summaries = sorted(selector_root.rglob("orientation_selector_summary.json"))
    if not summaries:
        raise FileNotFoundError(
            f"Nenhum orientation_selector_summary.json encontrado em {selector_root}"
        )

    all_rows: List[Dict[str, Any]] = []

    for summary_path in summaries:
        pair_path = summary_path.parent / "orientation_selector_pairs.csv"
        if not pair_path.exists():
            print(f"[AVISO] Ignorando {summary_path.parent.name}: pairs.csv ausente")
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        pairs = read_csv_rows(pair_path)

        if not pairs:
            continue

        cls_name = summary_path.parent.name

        # Listas globais.
        cls_conf = []
        spatial_conf = []
        cls_sim = []

        cls_semantic = []
        spatial_semantic = []
        spatial_minus_cls_semantic = []

        cls_centrality = []
        spatial_centrality = []
        cls_compactness = []
        spatial_compactness = []
        cls_border = []
        spatial_border = []

        spatial_score_advantage = []

        # Somente nos pares em que CLS e Spatial discordam.
        dis_cls_semantic = []
        dis_spatial_semantic = []
        dis_semantic_advantage = []

        dis_cls_centrality = []
        dis_spatial_centrality = []
        dis_cls_compactness = []
        dis_spatial_compactness = []
        dis_cls_border = []
        dis_spatial_border = []

        semantic_prefers_cls = 0
        semantic_prefers_spatial = 0
        semantic_ties = 0
        n_disagree = 0

        for r in pairs:
            c_choice = int(float(r["cls_choice"]))
            s_choice = int(float(r["spatial_choice"]))
            agree = c_choice == s_choice

            c_conf = safe_float(r.get("cls_confidence"))
            s_conf = safe_float(r.get("spatial_confidence"))
            pair_sim = safe_float(r.get("cls_pair_similarity"))

            cls_conf.append(c_conf)
            spatial_conf.append(s_conf)
            cls_sim.append(pair_sim)

            c_sem = choose_side_value(r, "semantic_margin", c_choice)
            s_sem = choose_side_value(r, "semantic_margin", s_choice)

            c_cent = choose_side_value(r, "centrality", c_choice)
            s_cent = choose_side_value(r, "centrality", s_choice)

            c_comp = choose_side_value(r, "compactness", c_choice)
            s_comp = choose_side_value(r, "compactness", s_choice)

            c_border = choose_side_value(r, "border_contact", c_choice)
            s_border = choose_side_value(r, "border_contact", s_choice)

            c_sp_score = choose_side_value(r, "spatial_score", c_choice)
            s_sp_score = choose_side_value(r, "spatial_score", s_choice)

            cls_semantic.append(c_sem)
            spatial_semantic.append(s_sem)
            spatial_minus_cls_semantic.append(s_sem - c_sem)

            cls_centrality.append(c_cent)
            spatial_centrality.append(s_cent)
            cls_compactness.append(c_comp)
            spatial_compactness.append(s_comp)
            cls_border.append(c_border)
            spatial_border.append(s_border)
            spatial_score_advantage.append(s_sp_score - c_sp_score)

            if not agree:
                n_disagree += 1

                dis_cls_semantic.append(c_sem)
                dis_spatial_semantic.append(s_sem)
                dis_semantic_advantage.append(s_sem - c_sem)

                dis_cls_centrality.append(c_cent)
                dis_spatial_centrality.append(s_cent)
                dis_cls_compactness.append(c_comp)
                dis_spatial_compactness.append(s_comp)
                dis_cls_border.append(c_border)
                dis_spatial_border.append(s_border)

                if s_sem > c_sem + 1e-12:
                    semantic_prefers_spatial += 1
                elif c_sem > s_sem + 1e-12:
                    semantic_prefers_cls += 1
                else:
                    semantic_ties += 1

        n_pairs = len(pairs)
        disagreement_fraction = n_disagree / n_pairs if n_pairs else 0.0

        row = {
            "class": cls_name,
            "n_pairs": n_pairs,

            # Métricas já consolidadas.
            "agreement": safe_float(summary.get("cls_spatial_agreement")),
            "disagreement": safe_float(summary.get("cls_spatial_disagreement")),
            "cls_confidence_mean_summary": safe_float(summary.get("cls_confidence_mean")),
            "spatial_confidence_mean_summary": safe_float(summary.get("spatial_confidence_mean")),

            # Confiança.
            "cls_confidence_mean": mean_or_nan(cls_conf),
            "cls_confidence_std": std_or_nan(cls_conf),
            "spatial_confidence_mean": mean_or_nan(spatial_conf),
            "spatial_confidence_std": std_or_nan(spatial_conf),
            "confidence_gap_cls_minus_spatial": (
                mean_or_nan(cls_conf) - mean_or_nan(spatial_conf)
            ),

            # Similaridade global dos pares.
            "pair_cls_similarity_mean": mean_or_nan(cls_sim),
            "pair_cls_similarity_std": std_or_nan(cls_sim),

            # Margem semântica das hipóteses escolhidas por cada heurística.
            "cls_semantic_margin_mean": mean_or_nan(cls_semantic),
            "spatial_semantic_margin_mean": mean_or_nan(spatial_semantic),
            "semantic_advantage_spatial_mean": mean_or_nan(spatial_minus_cls_semantic),
            "semantic_advantage_spatial_std": std_or_nan(spatial_minus_cls_semantic),

            # Geometria.
            "cls_centrality_mean": mean_or_nan(cls_centrality),
            "spatial_centrality_mean": mean_or_nan(spatial_centrality),
            "centrality_advantage_spatial": (
                mean_or_nan(spatial_centrality) - mean_or_nan(cls_centrality)
            ),

            "cls_compactness_mean": mean_or_nan(cls_compactness),
            "spatial_compactness_mean": mean_or_nan(spatial_compactness),
            "compactness_advantage_spatial": (
                mean_or_nan(spatial_compactness) - mean_or_nan(cls_compactness)
            ),

            "cls_border_contact_mean": mean_or_nan(cls_border),
            "spatial_border_contact_mean": mean_or_nan(spatial_border),
            "border_advantage_spatial": (
                mean_or_nan(cls_border) - mean_or_nan(spatial_border)
            ),

            "spatial_score_advantage_mean": mean_or_nan(spatial_score_advantage),

            # Foco somente nos pares em desacordo.
            "n_disagreement_pairs": n_disagree,
            "disagreement_fraction": disagreement_fraction,

            "dis_cls_semantic_margin_mean": mean_or_nan(dis_cls_semantic),
            "dis_spatial_semantic_margin_mean": mean_or_nan(dis_spatial_semantic),
            "dis_semantic_advantage_spatial": mean_or_nan(dis_semantic_advantage),

            "dis_cls_centrality_mean": mean_or_nan(dis_cls_centrality),
            "dis_spatial_centrality_mean": mean_or_nan(dis_spatial_centrality),
            "dis_centrality_advantage_spatial": (
                mean_or_nan(dis_spatial_centrality) - mean_or_nan(dis_cls_centrality)
            ),

            "dis_cls_compactness_mean": mean_or_nan(dis_cls_compactness),
            "dis_spatial_compactness_mean": mean_or_nan(dis_spatial_compactness),
            "dis_compactness_advantage_spatial": (
                mean_or_nan(dis_spatial_compactness) - mean_or_nan(dis_cls_compactness)
            ),

            "dis_cls_border_contact_mean": mean_or_nan(dis_cls_border),
            "dis_spatial_border_contact_mean": mean_or_nan(dis_spatial_border),
            "dis_border_advantage_spatial": (
                mean_or_nan(dis_cls_border) - mean_or_nan(dis_spatial_border)
            ),

            "semantic_prefers_cls_fraction_disagreement": (
                semantic_prefers_cls / n_disagree if n_disagree else 0.0
            ),
            "semantic_prefers_spatial_fraction_disagreement": (
                semantic_prefers_spatial / n_disagree if n_disagree else 0.0
            ),
            "semantic_tie_fraction_disagreement": (
                semantic_ties / n_disagree if n_disagree else 0.0
            ),
        }

        all_rows.append(row)

    return all_rows


# ---------------------------------------------------------------------------
# Leitura dos resultados reais CLS / Spatial para gerar rótulos offline
# ---------------------------------------------------------------------------

def load_baseline_json(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result: Dict[str, float] = {}

    if isinstance(data, dict):
        # Formato all_classes.json: {classe: {iou: ...}}
        for cls, payload in data.items():
            if isinstance(payload, dict) and "iou" in payload:
                result[normalize_class_name(cls)] = safe_float(payload["iou"])

    return result


def detect_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_map = {str(x).lower(): str(x) for x in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def load_result_csv(path: Path) -> Dict[str, float]:
    rows = read_csv_rows(path)
    if not rows:
        return {}

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
            f"Colunas: {list(fields)}"
        )

    result: Dict[str, float] = {}

    for row in rows:
        cls = normalize_class_name(row[class_col])
        result[cls] = safe_float(row[iou_col])

    return result


def attach_labels(
    features: List[Dict[str, Any]],
    baseline_iou: Dict[str, float],
    spatial_iou: Dict[str, float],
    min_delta: float,
) -> List[Dict[str, Any]]:
    labeled = []

    for row in features:
        cls_display = row["class"]
        cls = normalize_class_name(cls_display)

        if cls not in baseline_iou or cls not in spatial_iou:
            continue

        iou_cls = baseline_iou[cls]
        iou_spatial = spatial_iou[cls]

        if not (math.isfinite(iou_cls) and math.isfinite(iou_spatial)):
            continue

        delta = iou_spatial - iou_cls

        # Pequenas diferenças podem ser tratadas como empate.
        if abs(delta) < min_delta:
            winner = "tie"
        else:
            winner = "spatial" if delta > 0 else "cls"

        item = dict(row)
        item.update({
            "iou_cls": iou_cls,
            "iou_spatial": iou_spatial,
            "iou_delta_spatial_minus_cls": delta,
            "winner": winner,
        })
        labeled.append(item)

    return labeled



def print_matching_diagnostics(
    features: List[Dict[str, Any]],
    baseline_iou: Dict[str, float],
    spatial_iou: Dict[str, float],
) -> None:
    feature_names = {
        normalize_class_name(r["class"]): r["class"]
        for r in features
    }

    matched = []
    missing_baseline = []
    missing_spatial = []

    for norm, display in feature_names.items():
        has_base = norm in baseline_iou
        has_spatial = norm in spatial_iou

        if has_base and has_spatial:
            matched.append(display)
        else:
            if not has_base:
                missing_baseline.append(display)
            if not has_spatial:
                missing_spatial.append(display)

    print("\n" + "=" * 100)
    print("DIAGNÓSTICO DE CORRESPONDÊNCIA DE NOMES")
    print("=" * 100)
    print(f"Coleções com features : {len(feature_names)}")
    print(f"Casaram nas 2 fontes  : {len(matched)}")
    print(f"Ausentes no baseline  : {len(missing_baseline)}")
    print(f"Ausentes no spatial   : {len(missing_spatial)}")

    if missing_baseline:
        print("\nNão encontradas no baseline:")
        for name in missing_baseline:
            print(f"  - {name}")

    if missing_spatial:
        print("\nNão encontradas no spatial CSV:")
        for name in missing_spatial:
            print(f"  - {name}")


# ---------------------------------------------------------------------------
# Modelagem simples com LOOCV
# ---------------------------------------------------------------------------

NON_FEATURE_COLUMNS = {
    "class",
    "winner",
    "iou_cls",
    "iou_spatial",
    "iou_delta_spatial_minus_cls",
}


def numeric_feature_columns(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []

    cols = []

    for key in rows[0].keys():
        if key in NON_FEATURE_COLUMNS:
            continue

        vals = [safe_float(r.get(key)) for r in rows]
        finite_count = sum(math.isfinite(v) for v in vals)

        if finite_count >= max(2, len(rows) // 2):
            cols.append(key)

    return cols


def rows_to_matrix(
    rows: List[Dict[str, Any]],
    feature_cols: List[str],
) -> np.ndarray:
    return np.array(
        [
            [safe_float(r.get(col)) for col in feature_cols]
            for r in rows
        ],
        dtype=np.float64,
    )


def build_models(random_state: int = 42) -> Dict[str, Pipeline]:
    return {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "tree_depth2": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=2,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
        "tree_depth3": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }


def evaluate_models(
    labeled_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str, List[str]]:
    # Exclui empates do treino binário.
    train_rows = [r for r in labeled_rows if r["winner"] in {"cls", "spatial"}]

    if len(train_rows) < 4:
        raise RuntimeError(
            "Poucas classes rotuladas para avaliar modelos. "
            "São necessárias pelo menos 4."
        )

    classes_present = {r["winner"] for r in train_rows}
    if len(classes_present) < 2:
        raise RuntimeError(
            "Os rótulos disponíveis possuem apenas uma classe. "
            "É preciso ter exemplos CLS e Spatial."
        )

    feature_cols = numeric_feature_columns(train_rows)
    X = rows_to_matrix(train_rows, feature_cols)
    y = np.array([r["winner"] for r in train_rows], dtype=object)

    loo = LeaveOneOut()
    models = build_models()

    report: Dict[str, Any] = {
        "n_labeled_classes": len(train_rows),
        "feature_columns": feature_cols,
        "models": {},
    }

    best_name = None
    best_key = (-1.0, -1.0)

    for name, model in models.items():
        preds = cross_val_predict(
            model,
            X,
            y,
            cv=loo,
            method="predict",
        )

        acc = float(accuracy_score(y, preds))
        bal = float(balanced_accuracy_score(y, preds))
        cm = confusion_matrix(
            y,
            preds,
            labels=["cls", "spatial"],
        ).tolist()

        per_class = []
        for row, pred in zip(train_rows, preds):
            per_class.append({
                "class": row["class"],
                "truth": row["winner"],
                "predicted": str(pred),
                "correct": bool(pred == row["winner"]),
            })

        report["models"][name] = {
            "loocv_accuracy": acc,
            "loocv_balanced_accuracy": bal,
            "confusion_matrix_labels": ["cls", "spatial"],
            "confusion_matrix": cm,
            "predictions": per_class,
        }

        key = (bal, acc)
        if key > best_key:
            best_key = key
            best_name = name

    assert best_name is not None
    report["best_model"] = best_name

    return report, best_name, feature_cols


def fit_and_predict_all(
    best_model_name: str,
    feature_cols: List[str],
    labeled_rows: List[Dict[str, Any]],
    all_feature_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    train_rows = [r for r in labeled_rows if r["winner"] in {"cls", "spatial"}]

    X_train = rows_to_matrix(train_rows, feature_cols)
    y_train = np.array([r["winner"] for r in train_rows], dtype=object)

    model = build_models()[best_model_name]
    model.fit(X_train, y_train)

    X_all = rows_to_matrix(all_feature_rows, feature_cols)
    preds = model.predict(X_all)

    # Probabilidade quando disponível.
    proba = None
    classes = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_all)
            classes = list(model.classes_)
        except Exception:
            pass

    output = []

    for idx, (row, pred) in enumerate(zip(all_feature_rows, preds)):
        item = {
            "class": row["class"],
            "predicted_orientation": str(pred),
        }

        if proba is not None and classes is not None:
            for c_idx, cls in enumerate(classes):
                item[f"prob_{cls}"] = float(proba[idx, c_idx])

        output.append(item)

    return output


# ---------------------------------------------------------------------------
# Escrita
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def print_feature_summary(features: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 100)
    print("FEATURES NÃO SUPERVISIONADAS POR COLEÇÃO")
    print("=" * 100)
    print(
        f"{'CLASSE':45} "
        f"{'ACORDO':>8} "
        f"{'CLS CONF':>9} "
        f"{'SPA CONF':>9} "
        f"{'SEM Δ SPA':>10} "
        f"{'BORDER Δ':>9}"
    )
    print("-" * 100)

    for r in features:
        print(
            f"{r['class'][:45]:45} "
            f"{r['agreement']:8.3f} "
            f"{r['cls_confidence_mean']:9.3f} "
            f"{r['spatial_confidence_mean']:9.3f} "
            f"{r['semantic_advantage_spatial_mean']:10.3f} "
            f"{r['border_advantage_spatial']:9.3f}"
        )


def print_model_summary(
    report: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> None:
    print("\n" + "=" * 100)
    print("AVALIAÇÃO OFFLINE DO SELETOR POR COLEÇÃO")
    print("=" * 100)

    for name, data in report["models"].items():
        print(
            f"{name:16} "
            f"LOOCV acc={data['loocv_accuracy']:.3f}  "
            f"balanced_acc={data['loocv_balanced_accuracy']:.3f}"
        )

    print(f"\nMelhor modelo: {report['best_model']}")

    print("\nPredições finais treinadas em todas as classes rotuladas:")
    print(f"{'CLASSE':55} {'PREDIÇÃO':10}")
    print("-" * 70)

    for row in predictions:
        print(
            f"{row['class'][:55]:55} "
            f"{row['predicted_orientation']:10}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extrai métricas não supervisionadas por coleção e, opcionalmente, "
            "avalia um seletor CLS vs Spatial usando rótulos apenas offline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--selector_root",
        required=True,
        type=Path,
        help=(
            "Raiz contendo subpastas com orientation_selector_summary.json "
            "e orientation_selector_pairs.csv."
        ),
    )

    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Diretório de saída.",
    )

    p.add_argument(
        "--baseline_json",
        type=Path,
        default=None,
        help=(
            "JSON com resultados do pipeline CLS/baseline. "
            "Usado apenas para gerar rótulos offline."
        ),
    )

    p.add_argument(
        "--spatial_csv",
        type=Path,
        default=None,
        help=(
            "CSV com resultados do pipeline Spatial. "
            "Usado apenas para gerar rótulos offline."
        ),
    )

    p.add_argument(
        "--min_delta",
        type=float,
        default=0.01,
        help=(
            "Diferença mínima absoluta de IoU para declarar CLS ou Spatial vencedor. "
            "Diferenças menores são marcadas como tie e excluídas do treino binário."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    features = extract_collection_features(args.selector_root)
    if not features:
        raise RuntimeError("Nenhuma coleção pôde ser processada.")

    features_path = args.output / "collection_orientation_features.csv"
    write_csv(features_path, features)

    print_feature_summary(features)
    print(f"\n[Salvo] {features_path}")

    # Sem resultados reais: apenas extrai features.
    if args.baseline_json is None or args.spatial_csv is None:
        print(
            "\n[INFO] baseline_json/spatial_csv não fornecidos. "
            "Features extraídas sem modelagem supervisionada offline."
        )
        return

    baseline_iou = load_baseline_json(args.baseline_json)
    spatial_iou = load_result_csv(args.spatial_csv)

    print_matching_diagnostics(
        features,
        baseline_iou,
        spatial_iou,
    )

    labeled = attach_labels(
        features,
        baseline_iou,
        spatial_iou,
        args.min_delta,
    )

    labels_path = args.output / "collection_orientation_labels.csv"
    write_csv(labels_path, labeled)
    print(f"[Salvo] {labels_path}")

    if len(labeled) < 4:
        print(
            f"[AVISO] Apenas {len(labeled)} coleções casaram entre features, baseline e spatial. "
            "Não foi possível avaliar modelos."
        )
        return

    report, best_name, feature_cols = evaluate_models(labeled)

    predictions = fit_and_predict_all(
        best_name,
        feature_cols,
        labeled,
        features,
    )

    predictions_path = args.output / "collection_orientation_predictions.csv"
    write_csv(predictions_path, predictions)

    report_path = args.output / "collection_orientation_model_report.json"
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[Salvo] {predictions_path}")
    print(f"[Salvo] {report_path}")

    print_model_summary(report, predictions)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERRO] {exc}")
        sys.exit(1)
