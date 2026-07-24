#!/usr/bin/env python3
"""
analyze_orientation_selector.py

Diagnóstico NÃO SUPERVISIONADO para decidir entre orientação CLS e Spatial
na Etapa 2 do pipeline de co-segmentação.

O script NÃO usa ground truth.

Ele reaproveita:
  - features.npz da Etapa 1
  - correspondence.npz da Etapa 2

Importante:
  Não é necessário recalcular o NCut nem salvar o vetor de Fiedler.
  Cada voto salvo em pair_votes representa um dos dois lados da bipartição.
  O outro lado é simplesmente o complemento dentro da região de conteúdo.
  Assim podemos avaliar, para a MESMA bipartição:
      - qual lado o critério CLS escolheria;
      - qual lado o critério Spatial escolheria;
      - qual lado possui melhor separabilidade semântica cross-image.

Saídas:
  orientation_selector_summary.json
  orientation_selector_pairs.csv

Exemplo:
  python ablation/analyze_orientation_selector.py \
      --features features/icoseg_full_e2_spatial/Kendo-EKC_2008_Helsinki-SanSuiSo/features.npz \
      --correspondence features/icoseg_full_e2_spatial/Kendo-EKC_2008_Helsinki-SanSuiSo/correspondence.npz \
      --output ablation_results/orientation_selector/Kendo-EKC_2008_Helsinki-SanSuiSo
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import normalize as sk_normalize


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return np.load(path, allow_pickle=True)


def compute_content_mask(
    resize_info: "np.ndarray | None",
    patch_grid: tuple[int, int],
) -> "np.ndarray | None":
    """
    Replica o mascaramento de padding da Etapa 2.
    Retorna (N, P) booleano ou None quando não há resize_info.
    """
    if resize_info is None:
        return None

    H_p, W_p = patch_grid
    N = resize_info.shape[0]

    canvas_h = int(resize_info[0, 4])
    canvas_w = int(resize_info[0, 5])

    patch_h = canvas_h / H_p
    patch_w = canvas_w / W_p

    row_starts = np.arange(H_p) * patch_h
    col_starts = np.arange(W_p) * patch_w
    row_ends = row_starts + patch_h
    col_ends = col_starts + patch_w

    masks = np.zeros((N, H_p, W_p), dtype=bool)

    for i in range(N):
        resized_h, resized_w, pad_top, pad_left, _, _ = [
            int(x) for x in resize_info[i]
        ]

        content_row = (
            (row_ends > pad_top)
            & (row_starts < pad_top + resized_h)
        )
        content_col = (
            (col_ends > pad_left)
            & (col_starts < pad_left + resized_w)
        )

        masks[i] = content_row[:, None] & content_col[None, :]

    return masks.reshape(N, -1)


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Estatísticas espaciais
# ---------------------------------------------------------------------------

def spatial_components(
    side_mask: np.ndarray,
    patch_indices: np.ndarray,
    patch_grid: tuple[int, int],
) -> dict[str, float]:
    """
    Calcula:
      - centrality  : 1 = centro, 0 = bordas/cantos
      - compactness : 1 = concentrado, 0 = espalhado
      - border_contact: fração dos patches selecionados que tocam a borda
      - score       : sqrt(centrality * compactness)

    A definição de score é compatível com a heurística spatial usada
    na ablação anterior.
    """
    if not side_mask.any():
        return {
            "centrality": 0.0,
            "compactness": 0.0,
            "border_contact": 1.0,
            "score": 0.0,
        }

    H_p, W_p = patch_grid

    rows_int = patch_indices // W_p
    cols_int = patch_indices % W_p

    rows = rows_int.astype(np.float32) / max(H_p - 1, 1)
    cols = cols_int.astype(np.float32) / max(W_p - 1, 1)

    rows_sel = rows[side_mask]
    cols_sel = cols[side_mask]

    max_dist = math.sqrt(0.5 ** 2 + 0.5 ** 2)
    dist_center = (
        np.sqrt((rows_sel - 0.5) ** 2 + (cols_sel - 0.5) ** 2)
        / max_dist
    )

    centrality = 1.0 - float(dist_center.mean())

    std_r = float(rows_sel.std())
    std_c = float(cols_sel.std())
    compactness = 1.0 - min(((std_r + std_c) / 2.0) / 0.289, 1.0)

    rows_sel_int = rows_int[side_mask]
    cols_sel_int = cols_int[side_mask]

    border = (
        (rows_sel_int == 0)
        | (rows_sel_int == H_p - 1)
        | (cols_sel_int == 0)
        | (cols_sel_int == W_p - 1)
    )
    border_contact = float(border.mean()) if border.size else 1.0

    score = float(
        math.sqrt(
            max(centrality, 0.0)
            * max(compactness, 0.0)
        )
    )

    return {
        "centrality": centrality,
        "compactness": compactness,
        "border_contact": border_contact,
        "score": score,
    }


def pair_spatial_score(
    side_i: np.ndarray,
    side_j: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    patch_grid: tuple[int, int],
) -> tuple[float, dict[str, float]]:
    comp_i = spatial_components(side_i, idx_i, patch_grid)
    comp_j = spatial_components(side_j, idx_j, patch_grid)

    merged = {
        key: 0.5 * (comp_i[key] + comp_j[key])
        for key in comp_i
    }

    return float(merged["score"]), merged


# ---------------------------------------------------------------------------
# Coerência semântica cross-image
# ---------------------------------------------------------------------------

def topk_cross_similarity(
    sim_matrix: np.ndarray,
    rows_mask: np.ndarray,
    cols_mask: np.ndarray,
    k: int,
) -> float:
    """
    Similaridade cross-image simétrica usando top-k correspondências.

    Para o bloco selecionado:
      - para cada patch da imagem i, média dos top-k matches em j;
      - para cada patch da imagem j, média dos top-k matches em i;
      - retorna a média das duas direções.

    Isso reduz o efeito de pares irrelevantes quando uma região contém
    muitos patches.
    """
    row_idx = np.flatnonzero(rows_mask)
    col_idx = np.flatnonzero(cols_mask)

    if row_idx.size == 0 or col_idx.size == 0:
        return 0.0

    block = sim_matrix[np.ix_(row_idx, col_idx)]

    k_row = min(k, block.shape[1])
    k_col = min(k, block.shape[0])

    # Top-k por linha.
    row_top = np.partition(
        block,
        kth=block.shape[1] - k_row,
        axis=1,
    )[:, -k_row:]
    row_score = float(row_top.mean())

    # Top-k por coluna.
    col_top = np.partition(
        block,
        kth=block.shape[0] - k_col,
        axis=0,
    )[-k_col:, :]
    col_score = float(col_top.mean())

    return 0.5 * (row_score + col_score)


def semantic_metrics_for_orientation(
    sim_ij: np.ndarray,
    fg_i: np.ndarray,
    fg_j: np.ndarray,
    topk: int,
) -> dict[str, float]:
    """
    Métricas semânticas para uma hipótese de foreground.

    fgfg:
      coerência cross-image entre patches propostos como FG.

    bgbg:
      coerência cross-image entre patches propostos como BG.

    fgbg / bgfg:
      coerência entre regiões semanticamente opostas.

    semantic_margin:
      FG-FG deve ser mais coerente que FG-BG/BG-FG.
      Não exige que o fundo seja menos coerente que o foreground.
    """
    bg_i = ~fg_i
    bg_j = ~fg_j

    fgfg = topk_cross_similarity(sim_ij, fg_i, fg_j, topk)
    bgbg = topk_cross_similarity(sim_ij, bg_i, bg_j, topk)
    fgbg = topk_cross_similarity(sim_ij, fg_i, bg_j, topk)
    bgfg = topk_cross_similarity(sim_ij, bg_i, fg_j, topk)

    semantic_margin = fgfg - 0.5 * (fgbg + bgfg)

    return {
        "fgfg": float(fgfg),
        "bgbg": float(bgbg),
        "fgbg": float(fgbg),
        "bgfg": float(bgfg),
        "semantic_margin": float(semantic_margin),
    }


# ---------------------------------------------------------------------------
# Decisões CLS / Spatial / Auto
# ---------------------------------------------------------------------------

def cls_choice_and_confidence(
    side0_i: np.ndarray,
    side0_j: np.ndarray,
    sal_i: np.ndarray,
    sal_j: np.ndarray,
) -> tuple[int, float, float, float]:
    """
    Retorna:
      choice: 0 para side0, 1 para complemento
      confidence: margem normalizada pela dispersão da saliência
      mean0 / mean1
    """
    sal_all = np.concatenate([sal_i, sal_j])
    side0_all = np.concatenate([side0_i, side0_j])

    mean0 = safe_mean(sal_all[side0_all])
    mean1 = safe_mean(sal_all[~side0_all])

    choice = 0 if mean0 >= mean1 else 1

    scale = float(np.std(sal_all)) + 1e-8
    confidence = abs(mean0 - mean1) / scale

    return choice, float(confidence), float(mean0), float(mean1)


def spatial_choice_and_confidence(
    side0_i: np.ndarray,
    side0_j: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    patch_grid: tuple[int, int],
) -> tuple[int, float, dict[str, float], dict[str, float]]:
    score0, comp0 = pair_spatial_score(
        side0_i, side0_j, idx_i, idx_j, patch_grid
    )
    score1, comp1 = pair_spatial_score(
        ~side0_i, ~side0_j, idx_i, idx_j, patch_grid
    )

    choice = 0 if score0 >= score1 else 1

    confidence = abs(score0 - score1) / (
        abs(score0) + abs(score1) + 1e-8
    )

    return choice, float(confidence), comp0, comp1


def make_auto_recommendation(
    cls_choice: int,
    spatial_choice: int,
    cls_confidence: float,
    spatial_confidence: float,
    semantic0: dict[str, float],
    semantic1: dict[str, float],
    semantic_margin_min: float,
) -> tuple[int, str, float]:
    """
    Regra experimental NÃO SUPERVISIONADA.

    1. Se CLS e Spatial concordam, usa consenso.
    2. Se discordam e a diferença de semantic_margin é suficientemente clara,
       usa a hipótese semanticamente melhor.
    3. Se a diferença semântica é pequena, usa o método com maior confiança
       interna normalizada.

    Retorna:
      choice (0/1), source, confidence
    """
    if cls_choice == spatial_choice:
        confidence = max(cls_confidence, spatial_confidence)
        return cls_choice, "consensus", float(confidence)

    sem0 = semantic0["semantic_margin"]
    sem1 = semantic1["semantic_margin"]
    sem_delta = abs(sem0 - sem1)

    if sem_delta >= semantic_margin_min:
        choice = 0 if sem0 >= sem1 else 1
        return choice, "semantic", float(sem_delta)

    if cls_confidence >= spatial_confidence:
        return cls_choice, "cls_confidence", float(cls_confidence)

    return spatial_choice, "spatial_confidence", float(spatial_confidence)


# ---------------------------------------------------------------------------
# Análise principal
# ---------------------------------------------------------------------------

def analyze(
    features_path: Path,
    correspondence_path: Path,
    semantic_topk: int,
    semantic_margin_min: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feats = load_npz(features_path)
    corr = load_npz(correspondence_path)

    patch_features = feats["patch_features"].astype(np.float32)
    patch_grid = tuple(int(x) for x in feats["patch_grid"])
    resize_info = (
        feats["resize_info"]
        if "resize_info" in feats.files
        else None
    )

    pair_votes = corr["pair_votes"].astype(np.float32)
    top_pairs = corr["top_pairs"]
    cls_saliency = corr["cls_saliency"].astype(np.float32)

    N, P, D = patch_features.shape

    if pair_votes.shape[1:] != (N, P):
        raise ValueError(
            "Shape de pair_votes incompatível com features: "
            f"{pair_votes.shape} vs N={N}, P={P}"
        )

    content_mask = compute_content_mask(resize_info, patch_grid)
    if content_mask is None:
        content_mask = np.ones((N, P), dtype=bool)

    # Normaliza uma única vez para todas as comparações semânticas.
    patch_norm = sk_normalize(
        patch_features.reshape(-1, D),
        axis=1,
    ).reshape(N, P, D).astype(np.float32)

    rows: list[dict[str, Any]] = []

    n_agree = 0
    selector_cls_count = 0
    selector_spatial_count = 0
    selector_consensus_count = 0
    selector_semantic_count = 0
    selector_cls_conf_count = 0
    selector_spatial_conf_count = 0

    semantic_advantages = []
    cls_confidences = []
    spatial_confidences = []

    for k, pair in enumerate(top_pairs):
        i = int(pair[0])
        j = int(pair[1])
        cls_sim = float(pair[2])

        idx_i = np.flatnonzero(content_mask[i])
        idx_j = np.flatnonzero(content_mask[j])

        if idx_i.size == 0 or idx_j.size == 0:
            continue

        vote_i_full = pair_votes[k, i] > 0.5
        vote_j_full = pair_votes[k, j] > 0.5

        # side0 = lado salvo no correspondence.npz, restrito ao conteúdo.
        side0_i = vote_i_full[idx_i]
        side0_j = vote_j_full[idx_j]

        sal_i = cls_saliency[i, idx_i]
        sal_j = cls_saliency[j, idx_j]

        cls_choice, cls_conf, cls_mean0, cls_mean1 = (
            cls_choice_and_confidence(
                side0_i,
                side0_j,
                sal_i,
                sal_j,
            )
        )

        spatial_choice, spatial_conf, spatial0, spatial1 = (
            spatial_choice_and_confidence(
                side0_i,
                side0_j,
                idx_i,
                idx_j,
                patch_grid,
            )
        )

        # Uma única matriz de similaridade cross-image por par.
        fi = patch_norm[i, idx_i]
        fj = patch_norm[j, idx_j]
        sim_ij = fi @ fj.T

        semantic0 = semantic_metrics_for_orientation(
            sim_ij,
            side0_i,
            side0_j,
            semantic_topk,
        )
        semantic1 = semantic_metrics_for_orientation(
            sim_ij,
            ~side0_i,
            ~side0_j,
            semantic_topk,
        )

        auto_choice, auto_source, auto_conf = (
            make_auto_recommendation(
                cls_choice,
                spatial_choice,
                cls_conf,
                spatial_conf,
                semantic0,
                semantic1,
                semantic_margin_min,
            )
        )

        if cls_choice == spatial_choice:
            n_agree += 1

        if auto_source == "consensus":
            selector_consensus_count += 1
        elif auto_source == "semantic":
            selector_semantic_count += 1
        elif auto_source == "cls_confidence":
            selector_cls_conf_count += 1
        elif auto_source == "spatial_confidence":
            selector_spatial_conf_count += 1

        # Quantas vezes o seletor final coincide com cada heurística.
        if auto_choice == cls_choice:
            selector_cls_count += 1
        if auto_choice == spatial_choice:
            selector_spatial_count += 1

        semantic_advantages.append(
            semantic1["semantic_margin"]
            - semantic0["semantic_margin"]
        )
        cls_confidences.append(cls_conf)
        spatial_confidences.append(spatial_conf)

        row = {
            "pair_rank": k,
            "image_i": i,
            "image_j": j,
            "cls_pair_similarity": cls_sim,

            "cls_choice": cls_choice,
            "spatial_choice": spatial_choice,
            "cls_spatial_agree": int(cls_choice == spatial_choice),

            "cls_confidence": cls_conf,
            "spatial_confidence": spatial_conf,
            "cls_mean_side0": cls_mean0,
            "cls_mean_side1": cls_mean1,

            "spatial_score_side0": spatial0["score"],
            "spatial_score_side1": spatial1["score"],
            "centrality_side0": spatial0["centrality"],
            "centrality_side1": spatial1["centrality"],
            "compactness_side0": spatial0["compactness"],
            "compactness_side1": spatial1["compactness"],
            "border_contact_side0": spatial0["border_contact"],
            "border_contact_side1": spatial1["border_contact"],

            "semantic_margin_side0": semantic0["semantic_margin"],
            "semantic_margin_side1": semantic1["semantic_margin"],
            "semantic_fgfg_side0": semantic0["fgfg"],
            "semantic_fgfg_side1": semantic1["fgfg"],
            "semantic_bgbg_side0": semantic0["bgbg"],
            "semantic_bgbg_side1": semantic1["bgbg"],
            "semantic_fgbg_side0": semantic0["fgbg"],
            "semantic_fgbg_side1": semantic1["fgbg"],
            "semantic_bgfg_side0": semantic0["bgfg"],
            "semantic_bgfg_side1": semantic1["bgfg"],

            "auto_choice": auto_choice,
            "auto_source": auto_source,
            "auto_confidence": auto_conf,

            # Indica qual método o seletor está seguindo quando CLS/Spatial divergem.
            "auto_matches_cls": int(auto_choice == cls_choice),
            "auto_matches_spatial": int(auto_choice == spatial_choice),
        }

        rows.append(row)

    K = len(rows)

    if K == 0:
        raise RuntimeError("Nenhum par válido foi analisado.")

    disagree_rows = [
        r for r in rows
        if not r["cls_spatial_agree"]
    ]

    n_disagree = len(disagree_rows)

    # Em pares divergentes, quantas vezes o seletor escolhe cada método.
    auto_cls_on_disagreement = sum(
        r["auto_matches_cls"]
        for r in disagree_rows
    )
    auto_spatial_on_disagreement = sum(
        r["auto_matches_spatial"]
        for r in disagree_rows
    )

    if n_disagree == 0:
        collection_recommendation = "consensus"
        recommendation_confidence = 1.0
    else:
        frac_spatial = auto_spatial_on_disagreement / n_disagree
        frac_cls = auto_cls_on_disagreement / n_disagree

        if frac_spatial > frac_cls:
            collection_recommendation = "spatial"
            recommendation_confidence = frac_spatial
        elif frac_cls > frac_spatial:
            collection_recommendation = "cls"
            recommendation_confidence = frac_cls
        else:
            collection_recommendation = "mixed"
            recommendation_confidence = 0.5

    summary = {
        "features_path": str(features_path),
        "correspondence_path": str(correspondence_path),
        "n_images": int(N),
        "n_patches": int(P),
        "n_pairs_analyzed": int(K),

        "semantic_topk": int(semantic_topk),
        "semantic_margin_min": float(semantic_margin_min),

        "cls_spatial_agreement": float(n_agree / K),
        "cls_spatial_disagreement": float(n_disagree / K),

        "cls_confidence_mean": float(np.mean(cls_confidences)),
        "spatial_confidence_mean": float(np.mean(spatial_confidences)),

        "semantic_side1_minus_side0_mean": float(
            np.mean(semantic_advantages)
        ),

        "auto_source_counts": {
            "consensus": int(selector_consensus_count),
            "semantic": int(selector_semantic_count),
            "cls_confidence": int(selector_cls_conf_count),
            "spatial_confidence": int(selector_spatial_conf_count),
        },

        "auto_matches_cls_fraction_all_pairs": float(
            selector_cls_count / K
        ),
        "auto_matches_spatial_fraction_all_pairs": float(
            selector_spatial_count / K
        ),

        "n_disagreement_pairs": int(n_disagree),
        "auto_matches_cls_on_disagreement": int(
            auto_cls_on_disagreement
        ),
        "auto_matches_spatial_on_disagreement": int(
            auto_spatial_on_disagreement
        ),

        "collection_recommendation": collection_recommendation,
        "recommendation_confidence": float(
            recommendation_confidence
        ),

        "note": (
            "Diagnóstico não supervisionado. A recomendação deve ser validada "
            "offline contra os resultados CLS vs Spatial já existentes, mas o "
            "seletor não usa GT como entrada."
        ),
    }

    return summary, rows


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def write_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "orientation_selector_summary.json"
    csv_path = output_dir / "orientation_selector_pairs.csv"

    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if rows:
        with csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n[Salvo] {json_path}")
    print(f"[Salvo] {csv_path}")


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 74)
    print("ORIENTATION SELECTOR — DIAGNÓSTICO NÃO SUPERVISIONADO")
    print("=" * 74)
    print(
        f"Pares analisados          : "
        f"{summary['n_pairs_analyzed']}"
    )
    print(
        f"CLS × Spatial agreement   : "
        f"{summary['cls_spatial_agreement']:.3f}"
    )
    print(
        f"CLS × Spatial disagreement: "
        f"{summary['cls_spatial_disagreement']:.3f}"
    )
    print(
        f"Confiança CLS média       : "
        f"{summary['cls_confidence_mean']:.3f}"
    )
    print(
        f"Confiança Spatial média   : "
        f"{summary['spatial_confidence_mean']:.3f}"
    )
    print(
        f"Pares divergentes         : "
        f"{summary['n_disagreement_pairs']}"
    )
    print(
        f"Auto→CLS nas divergências : "
        f"{summary['auto_matches_cls_on_disagreement']}"
    )
    print(
        f"Auto→Spatial nas diverg.  : "
        f"{summary['auto_matches_spatial_on_disagreement']}"
    )
    print(
        f"Recomendação da coleção   : "
        f"{summary['collection_recommendation']}"
    )
    print(
        f"Confiança da recomendação : "
        f"{summary['recommendation_confidence']:.3f}"
    )
    print("=" * 74)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisa, sem GT, sinais para selecionar orientação CLS ou Spatial."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--features",
        required=True,
        type=Path,
        help="features.npz produzido pela Etapa 1.",
    )

    parser.add_argument(
        "--correspondence",
        required=True,
        type=Path,
        help="correspondence.npz produzido pela Etapa 2.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Diretório de saída da análise.",
    )

    parser.add_argument(
        "--semantic_topk",
        type=int,
        default=5,
        help=(
            "Número de melhores correspondências cross-image por patch "
            "usado na coerência semântica."
        ),
    )

    parser.add_argument(
        "--semantic_margin_min",
        type=float,
        default=0.02,
        help=(
            "Diferença mínima entre semantic_margin das duas hipóteses "
            "para o desempate semântico ser considerado confiável."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary, rows = analyze(
        features_path=args.features,
        correspondence_path=args.correspondence,
        semantic_topk=args.semantic_topk,
        semantic_margin_min=args.semantic_margin_min,
    )

    write_outputs(
        output_dir=args.output,
        summary=summary,
        rows=rows,
    )

    print_summary(summary)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERRO] {exc}")
        sys.exit(1)
