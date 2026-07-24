#!/usr/bin/env python3
"""
=============================================================================
analyze_step3_cosaliency.py — Auditoria da Etapa 3 de um pipeline de cosegmentação
=============================================================================

Lê o cosaliency.npz gerado pela Etapa 3 e calcula métricas para responder:

  1. O clustering global encontrou clusters que correspondem ao objeto? (oracle)
  2. O mapeamento cluster→FG/BG foi feito corretamente?
  3. O refinamento NCut melhorou ou degradou a máscara coarse?
  4. As máscaras finais têm coerência espacial (regiões conectadas)?
  5. A fração FG está numa faixa plausível?
  6. Se houver GT: qual é o IoU real da máscara final (patch e pixel)?
  7. Se houver GT: qual a separação entre erros de clustering vs. mapeamento?
  8. [NOVO] O prior de saliência espacial está ajudando? (centralidade, compacidade,
     comparação score_e2 vs score_espacial vs score_final por cluster)

Funciona com ou sem ground truth.
Compatível com cosaliency.npz gerado tanto pela versão antiga (sem saliency_alpha)
quanto pela nova versão (com cluster_scores_e2, cluster_scores_sp, etc.).

Uso mínimo:
  python analyze_step3_cosaliency.py \
      --cosaliency features/Panda/cosaliency.npz \
      --features   features/Panda/features.npz

Com ground truth:
  python analyze_step3_cosaliency.py \
      --cosaliency features/Panda/cosaliency.npz \
      --features   features/Panda/features.npz \
      --gt_index   gt_index.json \
      --output     features/Panda/step3_quality

Saídas:
  step3_quality.json
  step3_quality_report.md
  viz_mask_quality.png
  viz_cluster_map.png
  viz_coarse_vs_refined.png
  viz_fg_fraction.png
  viz_saliency_decomposition.png   ← novo (sempre gerado se scores disponíveis)
  viz_saliency_spatial_map.png     ← novo (mapa espacial dos scores por cluster)
  viz_gt_iou_curve.png             (se GT disponível)
  viz_oracle_clusters.png          (se GT disponível)
  viz_mapping_accuracy.png         (se GT disponível)

=============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
except Exception:
    plt = None

try:
    from sklearn.metrics import roc_auc_score
    from scipy.ndimage import label as ndimage_label
except Exception:
    roc_auc_score = None
    ndimage_label = None


# ---------------------------------------------------------------------------
# Estruturas
# ---------------------------------------------------------------------------

@dataclass
class Assessment:
    metric: str
    value: Optional[float]
    verdict: str
    indication: str
    suggested_test: str


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


def compact_float(x: Optional[float], nd: int = 4) -> str:
    v = safe_float(x)
    return "n/a" if v is None else f"{v:.{nd}f}"


def assessment_to_dict(a: Assessment) -> Dict[str, Any]:
    return {"metric": a.metric, "value": a.value, "verdict": a.verdict,
            "indication": a.indication, "suggested_test": a.suggested_test}


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)

def compute_content_mask(
    resize_info: Optional[np.ndarray],
    patch_grid: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Identifica quais patches pertencem ao conteúdo real da imagem.

    Retorna:
        (N, P) bool
        True  = conteúdo real
        False = padding

    Retorna None quando resize_info não existe, isto é, quando a Etapa 1
    foi executada com resize='square'.
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
    row_ends = row_starts + patch_h
    col_starts = np.arange(W_p) * patch_w
    col_ends = col_starts + patch_w

    masks = np.zeros((N, H_p, W_p), dtype=bool)

    for i in range(N):
        resized_h, resized_w, pad_top, pad_left, _, _ = [
            int(x) for x in resize_info[i]
        ]

        content_rows = (
            (row_ends > pad_top)
            & (row_starts < pad_top + resized_h)
        )

        content_cols = (
            (col_ends > pad_left)
            & (col_starts < pad_left + resized_w)
        )

        masks[i] = (
            content_rows[:, None]
            & content_cols[None, :]
        )

    return masks.reshape(N, -1)

# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_cosaliency(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"cosaliency.npz não encontrado: {path}")
    data = np.load(path, allow_pickle=True)
    required = ["fg_mask_patch", "fg_mask_coarse", "cluster_labels",
                "cluster_scores", "cluster_is_fg", "n_clusters_used",
                "image_paths", "image_sizes"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"Campos ausentes em cosaliency.npz: {missing}")

    def _opt(key, dtype=np.float32):
        return np.asarray(data[key], dtype=dtype) if key in data.files else None

    return {
        "fg_mask_patch"              : np.asarray(data["fg_mask_patch"],   dtype=np.float32),
        "fg_mask_pixel"              : np.asarray(data["fg_mask_pixel"],   dtype=np.float32)
                                       if "fg_mask_pixel" in data.files else None,
        "fg_mask_coarse"             : np.asarray(data["fg_mask_coarse"],  dtype=np.float32),
        "fg_score_global"            : np.asarray(data["fg_score_global"], dtype=np.float32)
                                       if "fg_score_global" in data.files else None,
        "cluster_labels"             : np.asarray(data["cluster_labels"],  dtype=np.int32),
        "cluster_scores"             : np.asarray(data["cluster_scores"],  dtype=np.float32),
        "cluster_scores_e2"          : _opt("cluster_scores_e2"),
        "cluster_scores_sp"          : _opt("cluster_scores_sp"),
        "cluster_scores_centrality"  : _opt("cluster_scores_centrality"),
        "cluster_scores_compactness" : _opt("cluster_scores_compactness"),
        "cluster_scores_border"      : _opt("cluster_scores_border"),
        "cluster_is_fg"              : np.asarray(data["cluster_is_fg"],   dtype=bool),
        "n_clusters_used"            : int(data["n_clusters_used"]),
        "saliency_alpha"             : float(data["saliency_alpha"]) if "saliency_alpha" in data.files else None,
        "fg_decision_mode"           : str(data["fg_decision_mode"]) if "fg_decision_mode" in data.files else "saliency",
        "seeds"                      : list(data["seeds"]) if "seeds" in data.files else [],
        "image_paths"                : [str(x) for x in data["image_paths"].tolist()],
        "image_sizes"                : np.asarray(data["image_sizes"]),
    }


def load_features(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    if "patch_features" not in data.files:
        return None
    return {
        "patch_features": np.asarray(data["patch_features"], dtype=np.float32),
        "patch_grid": tuple(int(x) for x in data["patch_grid"].tolist()),
        # resize_info só existe se a Etapa 1 rodou com --resize keep_ar.
        # Colunas: resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w
        "resize_info": data["resize_info"] if "resize_info" in data.files else None,
    }


def load_gt_index(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        print(f"[Aviso] gt_index.json não encontrado. Rodando sem GT.")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _path_variants(path_str: str):
    import os
    variants = set()
    raw = str(path_str).replace("\\", "/")
    variants.add(raw); variants.add(raw.lstrip("./"))
    p = Path(path_str)
    try: variants.add(p.as_posix())
    except Exception: pass
    try: variants.add(os.path.relpath(p).replace("\\", "/"))
    except Exception: pass
    for marker in ["icoseg/images/", "images/"]:
        idx = raw.find(marker)
        if idx >= 0:
            variants.add(raw[idx:])
    return {v for v in variants if v}


def load_patch_gt_masks(
    image_paths: List[str],
    patch_grid: Tuple[int, int],
    gt_index: Optional[Dict[str, Any]],
    fg_threshold: float = 0.5,
    resize_info: Optional[np.ndarray] = None,  # (N, 6), só existe no modo keep_ar
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Carrega GT e reduz para a grade de patches.

    Se resize_info for fornecido (Etapa 1 rodada com --resize keep_ar), o GT
    é primeiro redimensionado para o tamanho da imagem DENTRO do canvas (sem
    padding), colado num canvas do mesmo tamanho usado como entrada do DINOv2
    e só então reduzido para a grade de patches. Sem isso, a grade de patches
    (que cobre o canvas com padding) fica desalinhada com o GT (que cobre só
    a imagem original), inflando FN/FP perto das bordas em imagens não-quadradas.
    """
    if gt_index is None:
        return None, {"available": False, "matched": 0, "missing": len(image_paths)}

    mappings = gt_index.get("mappings", {})
    variant_lookup: Dict[str, str] = {}
    basename_to_gt: Dict[str, List[str]] = {}
    for img_p, gt_p in mappings.items():
        for v in _path_variants(img_p):
            variant_lookup[v] = str(gt_p)
        base = Path(img_p).name
        basename_to_gt.setdefault(base, []).append(str(gt_p))

    unique_basename: Dict[str, Optional[str]] = {}
    for base, gts in basename_to_gt.items():
        uniq = sorted(set(gts))
        unique_basename[base] = uniq[0] if len(uniq) == 1 else None

    H_p, W_p = patch_grid
    masks, missing = [], []
    for idx, img_path in enumerate(image_paths):
        gt_path = None
        for v in _path_variants(img_path):
            if v in variant_lookup:
                gt_path = Path(variant_lookup[v]); break
        if gt_path is None:
            s = unique_basename.get(Path(img_path).name)
            gt_path = Path(s) if s else None

        if gt_path is None or not gt_path.exists():
            masks.append(np.zeros((H_p, W_p), dtype=bool))
            missing.append(img_path); continue
        try:
            gt_img = Image.open(gt_path).convert("L")

            if resize_info is not None:
                resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = \
                    [int(x) for x in resize_info[idx]]

                # Redimensiona o GT para o tamanho da imagem sem padding...
                gt_resized = gt_img.resize((resized_w, resized_h), Image.BILINEAR)

                # ...e cola num canvas preto (background) do mesmo tamanho
                # usado como entrada do DINOv2, para alinhar com a grade de patches.
                canvas = Image.new("L", (canvas_w, canvas_h), 0)
                canvas.paste(gt_resized, (pad_left, pad_top))

                arr = np.asarray(canvas.resize((W_p, H_p), Image.BILINEAR),
                                 dtype=np.float32) / 255.0
            else:
                arr = np.asarray(gt_img.resize((W_p, H_p), Image.BILINEAR),
                                 dtype=np.float32) / 255.0

            masks.append(arr >= fg_threshold)
        except Exception as e:
            print(f"[Aviso] Falha lendo GT {gt_path}: {e}")
            masks.append(np.zeros((H_p, W_p), dtype=bool))
            missing.append(img_path)

    patch_gt = np.stack([m.reshape(-1) for m in masks])
    matched = len(image_paths) - len(missing)
    return (patch_gt if matched > 0 else None), {
        "available": matched > 0, "matched": matched,
        "missing": len(missing),
        "fg_patch_fraction": float(patch_gt.mean()) if matched > 0 else None,
    }


def load_pixel_gt_masks(
    image_paths: List[str],
    image_sizes: np.ndarray,
    gt_index: Optional[Dict[str, Any]],
    fg_threshold: float = 0.5,
) -> Optional[List[Optional[np.ndarray]]]:
    if gt_index is None:
        return None
    mappings = gt_index.get("mappings", {})
    variant_lookup: Dict[str, str] = {}
    basename_to_gt: Dict[str, List[str]] = {}
    for img_p, gt_p in mappings.items():
        for v in _path_variants(img_p):
            variant_lookup[v] = str(gt_p)
        base = Path(img_p).name
        basename_to_gt.setdefault(base, []).append(str(gt_p))
    unique_basename = {b: (sorted(set(g))[0] if len(set(g)) == 1 else None)
                       for b, g in basename_to_gt.items()}

    masks = []
    for i, img_path in enumerate(image_paths):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        gt_path = None
        for v in _path_variants(img_path):
            if v in variant_lookup:
                gt_path = Path(variant_lookup[v]); break
        if gt_path is None:
            s = unique_basename.get(Path(img_path).name)
            gt_path = Path(s) if s else None
        if gt_path is None or not gt_path.exists():
            masks.append(None); continue
        try:
            arr = np.asarray(Image.open(gt_path).convert("L").resize(
                (W_i, H_i), Image.BILINEAR), dtype=np.float32) / 255.0
            masks.append(arr >= fg_threshold)
        except Exception:
            masks.append(None)
    return masks


# ---------------------------------------------------------------------------
# Métricas sem GT
# ---------------------------------------------------------------------------

def compute_fg_fraction_metrics(
    fg_mask_patch: np.ndarray,
    fg_mask_coarse: np.ndarray,
    content_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Se content_mask for fornecida, a fração de FG por imagem é calculada só
    sobre patches de conteúdo. Patches de padding são sempre 0 (background)
    por construção no pipeline (fg_mask_coarse_flat[is_padding_flat] = 0.0
    em step3_cosaliency.py) — incluí-los dilui a fração real de FG na área
    que importa, especialmente em coleções com bastante padding.
    """
    N = fg_mask_patch.shape[0]

    if content_mask is not None:
        fracs_refined = np.array([
            fg_mask_patch[i].reshape(-1)[content_mask[i]].mean()
            if content_mask[i].any() else fg_mask_patch[i].reshape(-1).mean()
            for i in range(N)
        ])
        fracs_coarse = np.array([
            fg_mask_coarse[i].reshape(-1)[content_mask[i]].mean()
            if content_mask[i].any() else fg_mask_coarse[i].reshape(-1).mean()
            for i in range(N)
        ])
    else:
        fracs_refined = fg_mask_patch.reshape(N, -1).mean(axis=1)
        fracs_coarse  = fg_mask_coarse.reshape(N, -1).mean(axis=1)

    n_extreme_low  = int(np.sum(fracs_refined < 0.05))
    n_extreme_high = int(np.sum(fracs_refined > 0.90))

    ideal = 0.35
    diff_coarse  = float(np.mean(np.abs(fracs_coarse  - ideal)))
    diff_refined = float(np.mean(np.abs(fracs_refined - ideal)))
    refine_helped = diff_refined < diff_coarse

    return {
        "fg_fraction_padding_excluded": bool(content_mask is not None),
        "fg_fraction_mean":          float(np.mean(fracs_refined)),
        "fg_fraction_std":           float(np.std(fracs_refined)),
        "fg_fraction_min":           float(np.min(fracs_refined)),
        "fg_fraction_max":           float(np.max(fracs_refined)),
        "fg_fraction_extreme_low":   n_extreme_low,
        "fg_fraction_extreme_high":  n_extreme_high,
        "coarse_fraction_mean":      float(np.mean(fracs_coarse)),
        "refine_helped_fraction":    refine_helped,
        "refine_fraction_delta":     safe_float(diff_coarse - diff_refined),
    }


def compute_spatial_coherence(
    fg_mask_patch: np.ndarray,
) -> Dict[str, Any]:
    if ndimage_label is None:
        return {"spatial_coherence_note": "scipy não disponível."}

    N = fg_mask_patch.shape[0]
    n_comps, largest_fracs = [], []

    for i in range(N):
        mask = (fg_mask_patch[i] > 0.5).astype(np.uint8)
        if mask.sum() == 0:
            n_comps.append(0); largest_fracs.append(0.0); continue

        labeled, n = ndimage_label(mask)
        n_comps.append(int(n))
        if n > 0:
            sizes = [int((labeled == k).sum()) for k in range(1, n + 1)]
            largest_fracs.append(float(max(sizes) / mask.sum()))
        else:
            largest_fracs.append(0.0)

    return {
        "spatial_n_components_mean":        safe_float(np.mean(n_comps)),
        "spatial_n_components_std":         safe_float(np.std(n_comps)),
        "spatial_largest_component_frac":   safe_float(np.mean(largest_fracs)),
    }


def compute_cluster_consistency(
    cluster_labels: np.ndarray,
    cluster_scores: np.ndarray,
    cluster_is_fg: np.ndarray,
    n_clusters: int,
    content_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Espera-se receber cluster_labels JÁ com padding marcado como -1 (ver
    cluster_labels_eval em main()), não o array bruto salvo em
    cosaliency.npz. No cosaliency.npz, patches de padding recebem o rótulo
    arbitrário do cluster C0 — se C0 for um cluster de BG, isso infla
    cluster_bg_recurrence_mean artificialmente (o padding "recorre" em
    100% das imagens só por estar sempre presente, não porque o BG real se
    repete). Passar content_mask aqui é uma segunda camada de proteção
    (recomputa a máscara -1 internamente caso o chamador não tenha feito).
    """
    N, P = cluster_labels.shape

    if content_mask is not None:
        labels = cluster_labels.copy()
        labels[~content_mask] = -1
    else:
        labels = cluster_labels

    recurrence = np.array([
        float(np.mean([np.any(labels[i] == c) for i in range(N)]))
        for c in range(n_clusters)
    ])
    fg_recurrence = float(recurrence[cluster_is_fg].mean()) if cluster_is_fg.any() else 0.0
    bg_recurrence = float(recurrence[~cluster_is_fg].mean()) if (~cluster_is_fg).any() else 0.0

    fg_scores_c = cluster_scores[cluster_is_fg]
    bg_scores_c = cluster_scores[~cluster_is_fg]
    score_margin = None
    if len(fg_scores_c) > 0 and len(bg_scores_c) > 0:
        score_margin = float(fg_scores_c.mean() - bg_scores_c.mean())

    n_ambiguous = int(np.sum((cluster_scores > 0.4) & (cluster_scores < 0.6)))

    return {
        "cluster_consistency_padding_excluded": bool(content_mask is not None),
        "cluster_fg_recurrence_mean":  safe_float(fg_recurrence),
        "cluster_bg_recurrence_mean":  safe_float(bg_recurrence),
        "cluster_score_margin_fg_bg":  safe_float(score_margin),
        "cluster_n_ambiguous":         n_ambiguous,
        "cluster_n_fg":                int(cluster_is_fg.sum()),
        "cluster_n_bg":                int((~cluster_is_fg).sum()),
    }


# ---------------------------------------------------------------------------
# Métricas de saliência espacial (NOVO)
# ---------------------------------------------------------------------------

def compute_saliency_metrics(
    cluster_scores_e2  : Optional[np.ndarray],   # (n_clusters,)
    cluster_scores_sp  : Optional[np.ndarray],   # (n_clusters,)
    cluster_scores_cent: Optional[np.ndarray],   # (n_clusters,)
    cluster_scores_comp: Optional[np.ndarray],   # (n_clusters,)
    cluster_scores_fin : np.ndarray,             # (n_clusters,) — score final
    cluster_is_fg      : np.ndarray,             # (n_clusters,)
    saliency_alpha     : Optional[float],
) -> Dict[str, Any]:
    """
    Analisa a qualidade e o impacto do prior de saliência espacial.

    Métricas calculadas:

    1. saliency_available: se o npz foi gerado com a nova versão da E3.

    2. spatial_fg_score_mean / spatial_bg_score_mean:
       Score espacial médio dos clusters FG vs. BG.
       Ideal: FG > BG (o prior está discriminando corretamente).

    3. spatial_margin_fg_bg:
       spatial_fg_score_mean - spatial_bg_score_mean.
       Positivo = prior espacial concorda com a decisão FG/BG.
       Negativo = prior espacial contradiz a E2 (clusters FG são periféricos).

    4. e2_vs_spatial_agreement:
       Fração de clusters onde score_e2 e score_spatial concordam na
       classificação FG/BG (ambos > 0.5 ou ambos <= 0.5).
       Alto = os dois priors convergem; baixo = conflito entre priors.

    5. alpha_sensitivity:
       Clusters que mudariam de decisão FG/BG se α variasse entre 0 e 1.
       Indica quão sensível o resultado é ao valor de saliency_alpha.

    6. centrality_fg_mean / compactness_fg_mean:
       Valores médios de centralidade e compacidade para clusters FG.
       Clusters FG ideais têm alta centralidade E alta compacidade.
    """
    if cluster_scores_e2 is None or cluster_scores_sp is None:
        return {"saliency_available": False,
                "saliency_note": "Arquivo gerado sem prior espacial (versão antiga da E3)."}

    n = len(cluster_scores_fin)
    result: Dict[str, Any] = {"saliency_available": True}
    result["saliency_alpha"] = safe_float(saliency_alpha)

    # Scores médios FG vs BG
    if cluster_is_fg.any():
        result["spatial_fg_score_mean"] = safe_float(float(cluster_scores_sp[cluster_is_fg].mean()))
        result["e2_fg_score_mean"]      = safe_float(float(cluster_scores_e2[cluster_is_fg].mean()))
    if (~cluster_is_fg).any():
        result["spatial_bg_score_mean"] = safe_float(float(cluster_scores_sp[~cluster_is_fg].mean()))
        result["e2_bg_score_mean"]      = safe_float(float(cluster_scores_e2[~cluster_is_fg].mean()))

    sp_fg = result.get("spatial_fg_score_mean")
    sp_bg = result.get("spatial_bg_score_mean")
    if sp_fg is not None and sp_bg is not None:
        result["spatial_margin_fg_bg"] = safe_float(sp_fg - sp_bg)

    # Concordância e2 vs spatial
    e2_decision  = cluster_scores_e2  > 0.5
    sp_decision  = cluster_scores_sp  > 0.5
    agreement    = float(np.mean(e2_decision == sp_decision))
    result["e2_vs_spatial_agreement"] = safe_float(agreement)

    # Clusters que mudariam com α=0 vs α=1
    decision_alpha0 = cluster_scores_e2  > 0.5   # só E2
    decision_alpha1 = cluster_scores_sp  > 0.5   # só espacial
    n_flip = int(np.sum(decision_alpha0 != decision_alpha1))
    result["alpha_sensitivity_n_clusters"] = n_flip

    # Centralidade e compacidade dos FG
    if cluster_scores_cent is not None and cluster_is_fg.any():
        result["centrality_fg_mean"] = safe_float(float(cluster_scores_cent[cluster_is_fg].mean()))
        result["centrality_bg_mean"] = safe_float(float(cluster_scores_cent[~cluster_is_fg].mean())) \
                                       if (~cluster_is_fg).any() else None
    if cluster_scores_comp is not None and cluster_is_fg.any():
        result["compactness_fg_mean"] = safe_float(float(cluster_scores_comp[cluster_is_fg].mean()))
        result["compactness_bg_mean"] = safe_float(float(cluster_scores_comp[~cluster_is_fg].mean())) \
                                        if (~cluster_is_fg).any() else None

    return result


# ---------------------------------------------------------------------------
# Métricas com GT
# ---------------------------------------------------------------------------

def compute_patch_iou_metrics(
    fg_mask_patch: np.ndarray,
    fg_mask_coarse: np.ndarray,
    patch_gt: np.ndarray,
) -> Dict[str, Any]:
    N = fg_mask_patch.shape[0]
    ious_refined, ious_coarse = [], []

    for i in range(N):
        gt_i = patch_gt[i].astype(bool)
        if not gt_i.any() and not (~gt_i).any():
            continue
        ious_refined.append(_iou(fg_mask_patch[i].flatten() > 0.5, gt_i))
        ious_coarse.append(_iou(fg_mask_coarse[i].flatten() > 0.5, gt_i))

    if not ious_refined:
        return {"patch_iou_error": "GT sem foreground/background suficientes."}

    mean_r = float(np.mean(ious_refined))
    mean_c = float(np.mean(ious_coarse))
    refine_delta = mean_r - mean_c

    return {
        "patch_iou_refined_mean":    safe_float(mean_r),
        "patch_iou_refined_std":     safe_float(float(np.std(ious_refined))),
        "patch_iou_coarse_mean":     safe_float(mean_c),
        "patch_iou_delta_refined_minus_coarse": safe_float(refine_delta),
        "refine_helped_n_images":    int(sum(r > c for r, c in zip(ious_refined, ious_coarse))),
        "refine_hurt_n_images":      int(sum(r < c for r, c in zip(ious_refined, ious_coarse))),
    }


def compute_pixel_iou_metrics(
    fg_mask_pixel: Optional[np.ndarray],
    image_sizes: np.ndarray,
    pixel_gt_masks: Optional[List[Optional[np.ndarray]]],
) -> Dict[str, Any]:
    if fg_mask_pixel is None or pixel_gt_masks is None:
        return {"pixel_iou_note": "fg_mask_pixel ou GT pixel indisponível."}

    N = fg_mask_pixel.shape[0]
    ious = []
    for i in range(N):
        if pixel_gt_masks[i] is None:
            continue
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        pred = fg_mask_pixel[i, :H_i, :W_i] > 0.5
        gt   = pixel_gt_masks[i].astype(bool)
        if gt.any() or (~gt).any():
            ious.append(_iou(pred, gt))

    if not ious:
        return {"pixel_iou_note": "Nenhuma imagem com GT pixel válido."}

    return {
        "pixel_iou_mean": safe_float(float(np.mean(ious))),
        "pixel_iou_std":  safe_float(float(np.std(ious))),
        "pixel_iou_min":  safe_float(float(np.min(ious))),
        "pixel_iou_max":  safe_float(float(np.max(ious))),
    }


def compute_oracle_cluster_metrics(
    cluster_labels: np.ndarray,
    cluster_scores: np.ndarray,
    patch_gt: np.ndarray,
    n_clusters: int,
) -> Dict[str, Any]:
    import itertools
    N, P = patch_gt.shape
    labels_flat = cluster_labels.reshape(-1)
    gt_flat = patch_gt.reshape(-1).astype(bool)

    best_global_iou = 0.0
    best_global_subset = []
    if n_clusters <= 10:
        for r in range(1, n_clusters + 1):
            for subset in itertools.combinations(range(n_clusters), r):
                pred = np.isin(labels_flat, subset)
                val = _iou(pred, gt_flat)
                if val > best_global_iou:
                    best_global_iou = val
                    best_global_subset = list(subset)
    else:
        order = np.argsort(cluster_scores)[::-1]
        for k in range(1, n_clusters + 1):
            subset = order[:k].tolist()
            pred = np.isin(labels_flat, subset)
            val = _iou(pred, gt_flat)
            if val > best_global_iou:
                best_global_iou = val
                best_global_subset = subset

    imagewise_ious = []
    for i in range(N):
        gt_i = patch_gt[i].astype(bool)
        li = cluster_labels[i]
        best_i = 0.0
        if n_clusters <= 10:
            for r in range(1, n_clusters + 1):
                for subset in itertools.combinations(range(n_clusters), r):
                    best_i = max(best_i, _iou(np.isin(li, subset), gt_i))
        else:
            for k in range(1, n_clusters + 1):
                subset = np.argsort(cluster_scores)[-k:].tolist()
                best_i = max(best_i, _iou(np.isin(li, subset), gt_i))
        imagewise_ious.append(best_i)

    return {
        "oracle_best_global_subset":     best_global_subset,
        "oracle_best_global_iou":        safe_float(best_global_iou),
        "oracle_imagewise_iou_mean":     safe_float(float(np.mean(imagewise_ious))),
        "oracle_imagewise_iou_std":      safe_float(float(np.std(imagewise_ious))),
    }


def compute_mapping_accuracy(
    cluster_labels: np.ndarray,
    cluster_is_fg: np.ndarray,
    patch_gt: np.ndarray,
    n_clusters: int,
) -> Dict[str, Any]:
    N, P_flat = patch_gt.shape
    labels_flat = cluster_labels.reshape(-1)
    gt_flat = patch_gt.reshape(-1).astype(bool)

    correct = total = 0
    cluster_true_fg = []
    for c in range(n_clusters):
        mask = labels_flat == c
        if not mask.any():
            cluster_true_fg.append(None); continue
        total += 1
        gt_frac = float(gt_flat[mask].mean())
        true_fg = gt_frac > 0.5
        cluster_true_fg.append(true_fg)
        if true_fg == bool(cluster_is_fg[c]):
            correct += 1

    accuracy = float(correct / total) if total > 0 else None
    return {
        "mapping_accuracy":         safe_float(accuracy),
        "mapping_correct_clusters": correct,
        "mapping_total_clusters":   total,
        "cluster_true_fg":          [x for x in cluster_true_fg if x is not None],
    }


def compute_auc_metrics(
    fg_score_global: Optional[np.ndarray],
    patch_gt: np.ndarray,
    content_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Se content_mask for fornecida, a AUC é calculada só sobre patches de
    conteúdo. Padding tem fg_score=0 e GT=0 (background) de forma
    determinística — um bloco grande de pares (score=0, label=0) sempre
    perfeitamente ranqueado no fundo da lista infla a AUC sem refletir a
    qualidade real de separação FG/BG no conteúdo (mesmo efeito já
    corrigido em analyze_step2_correspondence.py).
    """
    if roc_auc_score is None or fg_score_global is None:
        return {"auc_note": "sklearn ou fg_score_global indisponível."}
    if content_mask is not None:
        flat_scores = fg_score_global.reshape(-1)[content_mask.reshape(-1)]
        flat_gt     = patch_gt.reshape(-1)[content_mask.reshape(-1)].astype(int)
    else:
        flat_scores = fg_score_global.reshape(-1)
        flat_gt     = patch_gt.reshape(-1).astype(int)
    if len(np.unique(flat_gt)) < 2:
        return {"auc_note": "GT sem variação suficiente."}
    try:
        auc = float(roc_auc_score(flat_gt, flat_scores))
        return {
            "auc_padding_excluded": bool(content_mask is not None),
            "fg_score_auc_roc": safe_float(auc),
        }
    except Exception:
        return {"auc_note": "Falha no cálculo de AUC."}


def compute_gt_threshold_iou(
    fg_score_global: Optional[np.ndarray],
    patch_gt: np.ndarray,
) -> Dict[str, Any]:
    if fg_score_global is None:
        return {}
    thresholds = np.arange(0.05, 0.96, 0.05)
    N = patch_gt.shape[0]
    mean_ious = []
    for t in thresholds:
        ious = []
        for i in range(N):
            pred = fg_score_global[i].flatten() >= t
            gt   = patch_gt[i].astype(bool)
            if gt.any() or (~gt).any():
                ious.append(_iou(pred, gt))
        mean_ious.append(float(np.mean(ious)) if ious else 0.0)
    best_idx = int(np.argmax(mean_ious))
    return {
        "score_best_iou":             safe_float(mean_ious[best_idx]),
        "score_best_threshold":       safe_float(float(thresholds[best_idx])),
        "score_iou_curve_thresholds": thresholds.tolist(),
        "score_iou_curve_values":     mean_ious,
    }


# ---------------------------------------------------------------------------
# Interpretações heurísticas
# ---------------------------------------------------------------------------

def assess_high_good(metric, value, strong, moderate,
                     smsg, mmsg, wmsg, test) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível", "Não foi possível calcular.", test)
    if v >= strong:   return Assessment(metric, v, "bom",          smsg, test)
    if v >= moderate: return Assessment(metric, v, "intermediário", mmsg, test)
    return Assessment(metric, v, "fraco", wmsg, test)


def assess_low_good(metric, value, good, moderate,
                    gmsg, mmsg, bmsg, test) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível", "Não foi possível calcular.", test)
    if v <= good:     return Assessment(metric, v, "bom",          gmsg, test)
    if v <= moderate: return Assessment(metric, v, "intermediário", mmsg, test)
    return Assessment(metric, v, "alerta", bmsg, test)


def assess_range(metric, value, lo_g, hi_g, lo_w, hi_w,
                 gmsg, mmsg, bmsg, test) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível", "Não foi possível calcular.", test)
    if lo_g <= v <= hi_g: return Assessment(metric, v, "bom",          gmsg, test)
    if lo_w <= v <= hi_w: return Assessment(metric, v, "intermediário", mmsg, test)
    return Assessment(metric, v, "alerta", bmsg, test)


def build_assessments(metrics: Dict[str, Any], has_gt: bool) -> List[Assessment]:
    A = []

    A.append(assess_range(
        "fg_fraction_mean", metrics.get("fg_fraction_mean"),
        lo_g=0.10, hi_g=0.65, lo_w=0.05, hi_w=0.80,
        gmsg="Fração FG dentro de uma faixa plausível para objetos naturais.",
        mmsg="Fração levemente fora do ideal.",
        bmsg="Fração extrema. Clustering pode estar elegendo o fundo como FG.",
        test="Ajuste --fg_threshold ou --n_clusters. Inspecione viz_fg_fraction.png.",
    ))

    A.append(assess_high_good(
        "spatial_largest_component_frac",
        metrics.get("spatial_largest_component_frac"),
        strong=0.70, moderate=0.45,
        smsg="Máscaras FG formam regiões bem conectadas.",
        mmsg="Coesão espacial parcial.",
        wmsg="Máscaras muito fragmentadas.",
        test="Aumente --k_neighbors. Inspecione viz_cluster_map.png."
    ))

    A.append(assess_high_good(
        "cluster_score_margin_fg_bg",
        metrics.get("cluster_score_margin_fg_bg"),
        strong=0.20, moderate=0.08,
        smsg="Boa separação entre scores FG e BG.",
        mmsg="Separação moderada.",
        wmsg="Clusters FG e BG com scores similares — mapeamento instável.",
        test="Inspecione cluster_scores. Tente --n_clusters diferente."
    ))

    # Métricas de saliência (novo — só se disponível)
    if metrics.get("saliency_available"):
        A.append(assess_high_good(
            "spatial_margin_fg_bg",
            metrics.get("spatial_margin_fg_bg"),
            strong=0.10, moderate=0.0,
            smsg="Prior espacial concorda com FG/BG: clusters FG são mais centrais/compactos.",
            mmsg="Prior espacial levemente favorável a FG.",
            wmsg="Prior espacial contradiz a decisão FG/BG. Clusters FG estão na periferia — "
                 "considere aumentar --saliency_alpha.",
            test="Veja viz_saliency_decomposition.png. Se spatial_margin < 0, "
                 "o prior espacial está penalizando o objeto."
        ))

        A.append(assess_high_good(
            "e2_vs_spatial_agreement",
            metrics.get("e2_vs_spatial_agreement"),
            strong=0.80, moderate=0.60,
            smsg="Score da E2 e prior espacial concordam na maioria dos clusters.",
            mmsg="Concordância parcial entre os dois priors.",
            wmsg="Forte conflito entre E2 e prior espacial. "
                 "Verifique se o objeto está centralizado nas imagens.",
            test="Se agreement baixo, tente --saliency_alpha 0.2 para reduzir o peso espacial."
        ))

        A.append(assess_low_good(
            "alpha_sensitivity_n_clusters",
            metrics.get("alpha_sensitivity_n_clusters"),
            good=1, moderate=2,
            gmsg="Resultado robusto ao valor de saliency_alpha.",
            mmsg="Alguns clusters mudam de decisão com α diferente — vale varrer α.",
            bmsg="Resultado muito sensível a α. Faça ablação com α ∈ {0.2, 0.4, 0.6, 0.8}.",
            test="Rode com --saliency_alpha 0.2 e 0.8 e compare viz_overlay.png."
        ))

        A.append(assess_high_good(
            "centrality_fg_mean",
            metrics.get("centrality_fg_mean"),
            strong=0.55, moderate=0.40,
            smsg="Clusters FG têm boa centralidade — objeto perto do centro das imagens.",
            mmsg="Centralidade moderada dos clusters FG.",
            wmsg="Clusters FG são periféricos. O prior de centralidade pode estar penalizando "
                 "objetos que aparecem nas bordas.",
            test="Se objeto é sistematicamente periférico, considere --saliency_alpha 0.2."
        ))

        A.append(assess_high_good(
            "compactness_fg_mean",
            metrics.get("compactness_fg_mean"),
            strong=0.55, moderate=0.40,
            smsg="Clusters FG são espacialmente compactos — bom sinal de objeto coeso.",
            mmsg="Compacidade moderada dos clusters FG.",
            wmsg="Clusters FG são espalhados. Podem incluir regiões de fundo estruturado.",
            test="Inspecione viz_saliency_spatial_map.png para ver a distribuição espacial."
        ))

    if has_gt:
        A.append(assess_high_good(
            "patch_iou_refined_mean", metrics.get("patch_iou_refined_mean"),
            strong=0.55, moderate=0.35,
            smsg="IoU patch-level bom — Etapa 3 não é gargalo.",
            mmsg="IoU moderado. A Etapa 4 precisa refinar bem.",
            wmsg="IoU baixo. Clustering pode ser gargalo.",
            test="Compare com oracle_best_global_iou."
        ))

        A.append(assess_high_good(
            "oracle_best_global_iou", metrics.get("oracle_best_global_iou"),
            strong=0.60, moderate=0.40,
            smsg="Oracle alto — a informação está nos clusters, o problema é mapeamento.",
            mmsg="Oracle moderado.",
            wmsg="Oracle baixo — clustering não encontrou o objeto. Gargalo na E3.",
            test="Se oracle >> iou_real, ajuste --fg_threshold ou --saliency_alpha."
        ))

        A.append(assess_high_good(
            "mapping_accuracy", metrics.get("mapping_accuracy"),
            strong=0.80, moderate=0.60,
            smsg="Mapeamento cluster→FG/BG correto na maioria.",
            mmsg="Mapeamento com erros frequentes.",
            wmsg="Mapeamento ruim — muitos clusters classificados incorretamente.",
            test="Veja viz_mapping_accuracy.png."
        ))

    return A


def infer_bottleneck(metrics: Dict[str, Any], has_gt: bool) -> str:
    iou_r      = safe_float(metrics.get("patch_iou_refined_mean"))
    oracle     = safe_float(metrics.get("oracle_best_global_iou"))
    mapac      = safe_float(metrics.get("mapping_accuracy"))
    sp_margin  = safe_float(metrics.get("spatial_margin_fg_bg"))
    agreement  = safe_float(metrics.get("e2_vs_spatial_agreement"))
    n_flip     = metrics.get("alpha_sensitivity_n_clusters", 0)
    sal_avail  = metrics.get("saliency_available", False)
    polar      = safe_float(metrics.get("spatial_largest_component_frac"))

    if has_gt:
        if iou_r is not None and iou_r >= 0.55:
            return "Diagnóstico: Etapa 3 não é gargalo. IoU patch suficiente para guiar a E4."
        if oracle is not None and oracle >= 0.55 and mapac is not None and mapac < 0.70:
            if sal_avail and sp_margin is not None and sp_margin < 0.0:
                return ("Diagnóstico: clustering tem informação (oracle alto), mas mapeamento falha. "
                        "O prior espacial está penalizando o objeto (spatial_margin < 0) — "
                        "tente --saliency_alpha 0.2 ou verifique se o objeto está centralizado.")
            return ("Diagnóstico: clustering tem informação (oracle alto), mas mapeamento FG/BG é fraco. "
                    "Ajuste --fg_threshold ou --saliency_alpha.")
        if oracle is not None and oracle < 0.40:
            return ("Diagnóstico: oracle baixo — o clustering espectral não está encontrando o objeto. "
                    "Tente aumentar --n_spectral, --k_neighbors, ou --n_clusters.")
        if sal_avail and agreement is not None and agreement < 0.6:
            return ("Diagnóstico: forte conflito entre score E2 e prior espacial. "
                    f"{n_flip} cluster(s) mudariam de decisão com α diferente. "
                    "Faça ablação com --saliency_alpha ∈ {0.2, 0.4, 0.6, 0.8}.")
        if iou_r is not None and 0.35 <= iou_r < 0.55:
            return ("Diagnóstico: resultado intermediário. A E4 (SAM) pode compensar, "
                    "mas seeds de baixa qualidade limitam o resultado final.")
        return ("Diagnóstico: resultado inconclusivo. Use ablação de n_clusters, "
                "k_neighbors, fg_threshold e saliency_alpha.")
    # sem GT
    if sal_avail and sp_margin is not None and sp_margin < -0.10:
        return ("Diagnóstico sem GT: prior espacial contradiz a decisão FG/BG. "
                "Clusters FG são mais periféricos que os BG — "
                "reduza --saliency_alpha ou verifique a centralidade do objeto.")
    if polar is not None and polar < 0.45:
        return "Diagnóstico sem GT: máscaras fragmentadas. Aumente --k_neighbors."
    return ("Diagnóstico sem GT: sinais estruturais dentro do esperado. "
            "Forneça gt_index.json para diagnóstico preciso.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mask_quality(
    fg_mask_patch, fg_mask_pixel, image_paths, image_sizes, output_dir, max_images=8
):
    if plt is None: return
    N_vis = min(len(image_paths), max_images)
    n_cols = 3 if fg_mask_pixel is not None else 2
    fig, axes = plt.subplots(N_vis, n_cols, figsize=(4.5 * n_cols, 3.5 * N_vis))
    if N_vis == 1: axes = axes[np.newaxis, :]
    fig.suptitle("Qualidade das máscaras — Etapa 3", fontsize=11)
    for i in range(N_vis):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        img = Image.open(image_paths[i]).convert("RGB")
        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        axes[i, 0].set_title(Path(image_paths[i]).name[:24], fontsize=7)
        im = axes[i, 1].imshow(fg_mask_patch[i], cmap="RdYlGn",
                               interpolation="nearest", vmin=0, vmax=1)
        axes[i, 1].set_title(f"patch mask  FG%={fg_mask_patch[i].mean()*100:.0f}%", fontsize=7)
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
        if fg_mask_pixel is not None:
            axes[i, 2].imshow(fg_mask_pixel[i, :H_i, :W_i], cmap="RdYlGn", vmin=0, vmax=1)
            axes[i, 2].set_title(f"pixel mask", fontsize=7)
            axes[i, 2].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "viz_mask_quality.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_map(
    cluster_labels, cluster_is_fg, image_paths, patch_grid, output_dir, max_images=8
):
    if plt is None: return
    N_vis = min(len(image_paths), max_images)
    nc = cluster_labels.max() + 1
    if cluster_labels.ndim == 3:
        H_p, W_p = cluster_labels.shape[1], cluster_labels.shape[2]
    else:
        # patch_grid vem sempre do chamador agora — evita assumir grade
        # quadrada (sqrt(P)), que quebra para qualquer H_p != W_p.
        H_p, W_p = patch_grid

    cmap = plt.get_cmap("tab10", nc)
    fig, axes = plt.subplots(1, N_vis, figsize=(3.5 * N_vis, 4))
    if N_vis == 1: axes = [axes]
    legend_patches = [
        mpatches.Patch(color=cmap(c),
                       label=f"C{c} {'FG' if cluster_is_fg[c] else 'BG'}")
        for c in range(min(nc, 10))
    ]
    for i in range(N_vis):
        lmap = cluster_labels[i].reshape(H_p, W_p) if cluster_labels.ndim == 2 else cluster_labels[i]
        axes[i].imshow(lmap.astype(float) / max(nc - 1, 1), cmap=cmap,
                       vmin=0, vmax=1, interpolation="nearest")
        axes[i].set_title(f"img[{i}]", fontsize=8); axes[i].axis("off")
    axes[-1].legend(handles=legend_patches, loc="lower right", fontsize=7)
    fig.suptitle(f"Clusters globais ({nc}) — cores FG/BG", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_cluster_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_coarse_vs_refined(
    fg_mask_patch, fg_mask_coarse, image_paths, patch_gt, output_dir, max_images=6
):
    if plt is None: return
    N_vis = min(len(image_paths), max_images)
    n_cols = 4 if patch_gt is not None else 3
    fig, axes = plt.subplots(N_vis, n_cols, figsize=(3.8 * n_cols, 3.5 * N_vis))
    if N_vis == 1: axes = axes[np.newaxis, :]
    fig.suptitle("Coarse vs. Refined (+ GT se disponível)", fontsize=11)
    for i in range(N_vis):
        img = Image.open(image_paths[i]).convert("RGB")
        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        axes[i, 0].set_title(Path(image_paths[i]).name[:20], fontsize=7)
        axes[i, 1].imshow(fg_mask_coarse[i], cmap="RdYlGn",
                          interpolation="nearest", vmin=0, vmax=1)
        axes[i, 1].set_title(f"Coarse {fg_mask_coarse[i].mean()*100:.0f}%", fontsize=7)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(fg_mask_patch[i], cmap="RdYlGn",
                          interpolation="nearest", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Refined {fg_mask_patch[i].mean()*100:.0f}%", fontsize=7)
        axes[i, 2].axis("off")
        if patch_gt is not None and n_cols == 4:
            H_p, W_p = fg_mask_patch.shape[1], fg_mask_patch.shape[2]
            axes[i, 3].imshow(patch_gt[i].reshape(H_p, W_p), cmap="gray",
                              interpolation="nearest", vmin=0, vmax=1)
            axes[i, 3].set_title("GT patch", fontsize=7)
            axes[i, 3].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "viz_coarse_vs_refined.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fg_fraction(fg_mask_patch, output_dir):
    if plt is None: return
    fracs = fg_mask_patch.reshape(len(fg_mask_patch), -1).mean(axis=1) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(fracs)), fracs, color="steelblue")
    ax.axhline(5,  color="red",    ls="--", lw=1.2, alpha=0.6, label="limite inf (5%)")
    ax.axhline(65, color="orange", ls="--", lw=1.2, alpha=0.6, label="limite sup (65%)")
    ax.set_xlabel("Imagem"); ax.set_ylabel("Fração FG (%)"); ax.set_ylim(0, 100)
    ax.set_title("Fração FG por imagem — máscara refinada")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_fg_fraction.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_saliency_decomposition(
    cluster_scores_e2  : Optional[np.ndarray],
    cluster_scores_sp  : Optional[np.ndarray],
    cluster_scores_cent: Optional[np.ndarray],
    cluster_scores_comp: Optional[np.ndarray],
    cluster_scores_fin : np.ndarray,
    cluster_is_fg      : np.ndarray,
    saliency_alpha     : Optional[float],
    output_dir         : Path,
) -> None:
    """
    Visualização do prior de saliência espacial em dois subplots:

    Esquerdo: Score E2 vs Score Espacial vs Score Final por cluster.
              Permite ver quais clusters são promovidos/rebaixados pelo prior.

    Direito: Decomposição do score espacial em Centralidade e Compacidade.
             Revela se o problema é de posição (periférico) ou forma (disperso).
    """
    if plt is None or cluster_scores_e2 is None or cluster_scores_sp is None:
        return

    nc = len(cluster_scores_fin)
    alpha = saliency_alpha if saliency_alpha is not None else 0.5
    cluster_ids = np.arange(nc)
    bar_w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Decomposição do Prior de Saliência Espacial  (α={alpha:.2f})\n"
        f"score_final = (1−α)·score_E2  +  α·score_espacial",
        fontsize=11
    )

    # Subplot esquerdo: E2 vs Espacial vs Final
    ax = axes[0]
    ax.bar(cluster_ids - bar_w,   cluster_scores_e2,  width=bar_w,
           label="Score E2 (corr. cross-img)", color="#4472C4", alpha=0.85)
    ax.bar(cluster_ids,            cluster_scores_sp,  width=bar_w,
           label="Score Espacial (√central.·compact.)", color="#ED7D31", alpha=0.85)
    ax.bar(cluster_ids + bar_w,   cluster_scores_fin, width=bar_w,
           label="Score Final (decisão)", color="#70AD47", alpha=0.85)
    ax.axhline(0.5, color="black", ls="--", lw=1.2, alpha=0.7, label="threshold=0.5")

    # Anotação de FG/BG
    for c in cluster_ids:
        label_txt = "FG" if cluster_is_fg[c] else "BG"
        color_txt = "#1a7a1a" if cluster_is_fg[c] else "#7a1a1a"
        ax.text(c, 1.02, label_txt, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color_txt)

    ax.set_xticks(cluster_ids)
    ax.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=10)
    ax.set_ylabel("Score [0, 1]"); ax.set_ylim(0, 1.12)
    ax.set_title("Scores por cluster: E2 / Espacial / Final")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)

    # Subplot direito: Centralidade vs Compacidade
    ax2 = axes[1]
    if cluster_scores_cent is not None and cluster_scores_comp is not None:
        ax2.bar(cluster_ids - bar_w / 2, cluster_scores_cent, width=bar_w,
                label="Centralidade (posição)", color="#9B59B6", alpha=0.85)
        ax2.bar(cluster_ids + bar_w / 2, cluster_scores_comp, width=bar_w,
                label="Compacidade (forma)",    color="#E74C3C", alpha=0.85)
        # Pontos do score espacial combinado
        ax2.plot(cluster_ids, cluster_scores_sp, "ko", ms=6, zorder=5,
                 label="Espacial combinado (√C·K)")

        for c in cluster_ids:
            label_txt = "FG" if cluster_is_fg[c] else "BG"
            color_txt = "#1a7a1a" if cluster_is_fg[c] else "#7a1a1a"
            ax2.text(c, 1.02, label_txt, ha="center", va="bottom",
                     fontsize=9, fontweight="bold", color=color_txt)

        ax2.set_xticks(cluster_ids)
        ax2.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=10)
        ax2.set_ylabel("Score [0, 1]"); ax2.set_ylim(0, 1.12)
        ax2.set_title("Decomposição do Score Espacial")
        ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Centralidade/Compacidade\nnão disponíveis",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "viz_saliency_decomposition.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_saliency_spatial_map(
    cluster_labels     : np.ndarray,    # (N, P)
    cluster_scores_sp  : Optional[np.ndarray],  # (n_clusters,)
    cluster_scores_cent: Optional[np.ndarray],
    cluster_scores_comp: Optional[np.ndarray],
    cluster_is_fg      : np.ndarray,
    image_paths        : List[str],
    patch_grid         : Tuple[int, int],
    output_dir         : Path,
    max_images         : int = 6,
) -> None:
    """
    Mapa espacial dos scores por cluster projetado de volta nas imagens.

    Para cada imagem mostra três mapas de calor:
    - Score espacial combinado (√centralidade × compacidade) por patch
    - Centralidade por patch
    - Compacidade por patch

    Permite visualizar onde o prior espacial está concentrado vs. onde
    o objeto realmente aparece — útil para diagnosticar se o prior está
    penalizando o objeto ou o fundo correto.
    """
    if plt is None or cluster_scores_sp is None:
        return

    H_p, W_p = patch_grid
    N_vis = min(len(image_paths), max_images)

    # Mapas de score por patch
    labels_flat = cluster_labels.reshape(-1)
    sp_map = cluster_scores_sp[labels_flat].reshape(len(image_paths), H_p, W_p)

    has_cent = cluster_scores_cent is not None
    has_comp = cluster_scores_comp is not None
    cent_map = cluster_scores_cent[labels_flat].reshape(len(image_paths), H_p, W_p) \
               if has_cent else None
    comp_map = cluster_scores_comp[labels_flat].reshape(len(image_paths), H_p, W_p) \
               if has_comp else None

    n_cols = 1 + (1 if has_cent else 0) + (1 if has_comp else 0) + 1  # img + scores + FG/BG
    fig, axes = plt.subplots(N_vis, n_cols, figsize=(3.8 * n_cols, 3.5 * N_vis))
    if N_vis == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Mapa Espacial do Prior de Saliência por Imagem\n"
                 "Verde escuro = score alto (central/compacto)  |  Vermelho = score baixo",
                 fontsize=11)

    for i in range(N_vis):
        col = 0
        img = Image.open(image_paths[i]).convert("RGB")
        axes[i, col].imshow(img); axes[i, col].axis("off")
        axes[i, col].set_title(Path(image_paths[i]).name[:20], fontsize=7)
        col += 1

        im = axes[i, col].imshow(sp_map[i], cmap="RdYlGn",
                                  interpolation="nearest", vmin=0, vmax=1)
        axes[i, col].set_title("Score Espacial", fontsize=7)
        axes[i, col].axis("off")
        plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)
        col += 1

        if has_cent:
            axes[i, col].imshow(cent_map[i], cmap="Blues",
                                interpolation="nearest", vmin=0, vmax=1)
            axes[i, col].set_title("Centralidade", fontsize=7)
            axes[i, col].axis("off")
            col += 1

        if has_comp:
            axes[i, col].imshow(comp_map[i], cmap="Oranges",
                                interpolation="nearest", vmin=0, vmax=1)
            axes[i, col].set_title("Compacidade", fontsize=7)
            axes[i, col].axis("off")
            col += 1

    fig.tight_layout()
    fig.savefig(output_dir / "viz_saliency_spatial_map.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_gt_iou_curve(metrics, output_dir):
    if plt is None: return
    thrs = metrics.get("score_iou_curve_thresholds")
    vals = metrics.get("score_iou_curve_values")
    if thrs is None or vals is None: return
    best_t = metrics.get("score_best_threshold", 0.5)
    best_v = metrics.get("score_best_iou", 0.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thrs, vals, "o-", color="steelblue", lw=2, ms=5)
    ax.axvline(best_t, color="red",   ls="--", lw=1.5, label=f"thr ótimo={best_t:.2f}")
    ax.axhline(best_v, color="green", ls="--", lw=1.2, label=f"IoU ótimo={best_v:.3f}")
    iou_r = metrics.get("patch_iou_refined_mean")
    if iou_r:
        ax.axhline(iou_r, color="orange", ls=":", lw=1.5, label=f"IoU refined={iou_r:.3f}")
    ax.set_xlabel("Threshold fg_score"); ax.set_ylabel("IoU médio")
    ax.set_title("IoU vs threshold (fg_score_global)")
    ax.set_ylim(0, 1); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_gt_iou_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_clusters(
    cluster_labels, cluster_scores, cluster_is_fg, patch_gt,
    oracle_subset, patch_grid, output_dir, max_images=6
):
    if plt is None: return
    H_p, W_p = patch_grid
    N_vis = min(len(cluster_labels), max_images)
    fig, axes = plt.subplots(N_vis, 3, figsize=(9, 3 * N_vis))
    if N_vis == 1: axes = axes[np.newaxis, :]
    fig.suptitle(f"Oracle clusters vs. mapeamento real vs. GT\n"
                 f"Oracle subset = {oracle_subset}", fontsize=10)
    for i in range(N_vis):
        lmap        = cluster_labels[i].reshape(H_p, W_p)
        pred_real   = cluster_is_fg[cluster_labels[i]].reshape(H_p, W_p)
        pred_oracle = np.isin(cluster_labels[i], oracle_subset).reshape(H_p, W_p)
        gt_map      = patch_gt[i].reshape(H_p, W_p)
        axes[i, 0].imshow(pred_real, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 0].set_title(f"Mapeamento real img[{i}]", fontsize=7); axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_oracle, cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 1].set_title("Oracle subset", fontsize=7); axes[i, 1].axis("off")
        axes[i, 2].imshow(gt_map, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 2].set_title("GT patch", fontsize=7); axes[i, 2].axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "viz_oracle_clusters.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_mapping_accuracy(
    cluster_scores, cluster_is_fg, cluster_true_fg, output_dir
):
    if plt is None: return
    nc = len(cluster_scores)
    colors = []
    for i in range(nc):
        if i >= len(cluster_true_fg):
            colors.append("gray"); continue
        pred = bool(cluster_is_fg[i])
        true = bool(cluster_true_fg[i])
        if pred == true:
            colors.append("#4caf50" if pred else "#2196f3")
        else:
            colors.append("#f44336")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(nc), cluster_scores, color=colors)
    ax.axhline(0.5, color="black", ls="--", lw=1.2, alpha=0.7, label="threshold=0.5")
    ax.set_xticks(range(nc))
    ax.set_xticklabels([f"C{i}" for i in range(nc)])
    ax.set_ylabel("Score FG final"); ax.set_ylim(0, 1)
    ax.set_title("Scores por cluster\n"
                 "Verde=FG correto  Azul=BG correto  Vermelho=erro")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_mapping_accuracy.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def write_report(output_dir, cosaliency_path, metrics, assessments,
                 diagnosis, gt_info, padding_info) -> Path:
    report_path = output_dir / "step3_quality_report.md"
    sal_avail = metrics.get("saliency_available", False)
    alpha_str = f"  {metrics.get('saliency_alpha', 'n/a')}" if sal_avail else "  n/a (versão antiga)"

    lines = [
        "# Relatório de qualidade — Etapa 3 (Co-Saliency via Spectral Clustering)",
        "",
        f"Arquivo analisado: `{cosaliency_path}`",
        "",
        "## Resumo",
        "",
        f"- Imagens: **{metrics.get('n_images')}**",
        f"- Patches por imagem: **{metrics.get('n_patches')}**",
        f"- Clusters usados: **{metrics.get('n_clusters_used')}**",
        f"  ({metrics.get('cluster_n_fg')} FG, {metrics.get('cluster_n_bg')} BG)",
        f"- GT encontrado: **{gt_info.get('matched', 0)} / {metrics.get('n_images')}**",
        f"- Prior espacial: **{'Sim' if sal_avail else 'Não (versão antiga)'}**"
        + (f"  (α={metrics.get('saliency_alpha', 'n/a')})" if sal_avail else ""),
    ]
    if padding_info.get("available"):
        lines.append(
            f"- Padding (--resize keep_ar): **{padding_info['fraction_padding']:.1%} dos patches**, "
            f"mascarado nas métricas: **{'sim' if padding_info.get('masked') else 'não (--no_padding_mask)'}**"
        )
    else:
        lines.append("- Padding: sem `resize_info` disponível (`--features` não informado, ou "
                     "Etapa 1 rodou em modo `square`)")
    lines += [
        "",
        "## Diagnóstico",
        "",
        diagnosis,
        "",
        "## Métricas interpretadas",
        "",
        "| Métrica | Valor | Veredito | Indicação | Teste sugerido |",
        "|---|---:|---|---|---|",
    ]
    for a in assessments:
        lines.append(
            f"| `{a.metric}` | {compact_float(a.value)} | {a.verdict} | "
            f"{a.indication} | {a.suggested_test} |"
        )

    if sal_avail:
        lines += [
            "",
            "## Prior de Saliência Espacial",
            "",
            "| Cluster | Score E2 | Centralidade | Compacidade | Espacial | Final | Decisão |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        nc   = metrics.get("n_clusters_used", 0)
        s_e2 = metrics.get("_cluster_scores_e2_raw", [])
        s_sp = metrics.get("_cluster_scores_sp_raw", [])
        s_ce = metrics.get("_cluster_scores_cent_raw", [])
        s_co = metrics.get("_cluster_scores_comp_raw", [])
        s_fi = metrics.get("_cluster_scores_fin_raw", [])
        s_fg = metrics.get("_cluster_is_fg_raw", [])
        for c in range(len(s_fi)):
            e2  = compact_float(s_e2[c]  if c < len(s_e2)  else None, 3)
            ce  = compact_float(s_ce[c]  if c < len(s_ce)  else None, 3)
            co  = compact_float(s_co[c]  if c < len(s_co)  else None, 3)
            sp  = compact_float(s_sp[c]  if c < len(s_sp)  else None, 3)
            fi  = compact_float(s_fi[c]  if c < len(s_fi)  else None, 3)
            fg  = "**FG**" if (c < len(s_fg) and s_fg[c]) else "BG"
            lines.append(f"| C{c} | {e2} | {ce} | {co} | {sp} | {fi} | {fg} |")

    lines += [
        "",
        "## Arquivos gerados",
        "",
    ]
    for name in ["step3_quality.json", "viz_mask_quality.png", "viz_cluster_map.png",
                 "viz_coarse_vs_refined.png", "viz_fg_fraction.png",
                 "viz_saliency_decomposition.png", "viz_saliency_spatial_map.png",
                 "viz_gt_iou_curve.png", "viz_oracle_clusters.png",
                 "viz_mapping_accuracy.png"]:
        if (output_dir / name).exists():
            lines.append(f"- `{name}`")
    lines += [
        "",
        "## Interpretação do prior espacial",
        "",
        "- `spatial_margin_fg_bg > 0`: prior espacial favorece clusters FG — esperado.",
        "- `spatial_margin_fg_bg < 0`: clusters FG são mais periféricos que BG — prior está errado para este objeto.",
        "- `e2_vs_spatial_agreement`: alta concordância = os dois priors convergem.",
        "- `alpha_sensitivity`: muitos clusters sensíveis a α → faça ablação de --saliency_alpha.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_console_summary(metrics, assessments, diagnosis, padding_info) -> None:
    print("\n" + "=" * 72)
    print("  AUDITORIA DA ETAPA 3 — CO-SALIENCY VIA SPECTRAL CLUSTERING")
    print("=" * 72)
    print(f"  Imagens   : {metrics.get('n_images')}")
    print(f"  Patches   : {metrics.get('n_patches')}")
    print(f"  Clusters  : {metrics.get('n_clusters_used')}  "
          f"({metrics.get('cluster_n_fg')} FG, {metrics.get('cluster_n_bg')} BG)")
    if metrics.get("saliency_available"):
        print(f"  α espacial: {metrics.get('saliency_alpha', 'n/a')}")
    if padding_info.get("available"):
        estado = "mascarado" if padding_info.get("masked") else "NÃO mascarado (--no_padding_mask)"
        print(f"  Padding   : {padding_info['fraction_padding']:.1%} dos patches — {estado}")
    print("-" * 72)
    for a in assessments:
        print(f"\n[{a.verdict.upper()}] {a.metric}: {compact_float(a.value)}")
        print(f"  Indicação     : {a.indication}")
        print(f"  Teste sugerido: {a.suggested_test}")
    print("\n" + "-" * 72)
    print(diagnosis)
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analisa cosaliency.npz gerado pela Etapa 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cosaliency", required=True, type=Path)
    p.add_argument("--features",   type=Path, default=None)
    p.add_argument("--gt_index",   type=Path, default=None)
    p.add_argument("--output",     type=Path, default=None)
    p.add_argument("--fg_threshold", type=float, default=0.5)
    p.add_argument("--max_viz_images", type=int, default=8)
    p.add_argument("--no_plots",   action="store_true")
    p.add_argument("--no_padding_mask", action="store_true",
                   help="Desativa o mascaramento de patches de padding (modo --resize "
                        "keep_ar da Etapa 1) em todas as métricas. Por padrão, o "
                        "mascaramento é ativado sempre que --features apontar para um "
                        "features.npz com resize_info.")
    return p.parse_args()


def main():
    args = parse_args()
    corr_path  = args.cosaliency
    output_dir = args.output or (corr_path.parent / "step3_quality")
    ensure_dir(output_dir)

    print(f"[1/8] Carregando cosaliency: {corr_path}")
    cos = load_cosaliency(corr_path)
    fg_mask_patch  = cos["fg_mask_patch"]
    fg_mask_coarse = cos["fg_mask_coarse"]
    fg_mask_pixel  = cos["fg_mask_pixel"]
    fg_score_global= cos["fg_score_global"]
    cluster_labels = cos["cluster_labels"]
    cluster_scores = cos["cluster_scores"]
    cluster_is_fg  = cos["cluster_is_fg"]
    n_clusters     = cos["n_clusters_used"]
    image_paths    = cos["image_paths"]
    image_sizes    = cos["image_sizes"]
    saliency_alpha = cos["saliency_alpha"]
    fg_decision_mode = cos.get("fg_decision_mode", "saliency")

    c_scores_e2   = cos["cluster_scores_e2"]
    c_scores_sp   = cos["cluster_scores_sp"]
    c_scores_cent = cos["cluster_scores_centrality"]
    c_scores_comp = cos["cluster_scores_compactness"]

    if fg_decision_mode != "saliency":
        print(f"      [Aviso] fg_decision_mode='{fg_decision_mode}' — a decisão FG/BG "
              f"NÃO usou saliency_alpha/cluster_scores_sp nesta execução. As seções deste "
              f"relatório que sugerem ajustar --saliency_alpha não se aplicam; "
              f"cluster_scores_sp ainda é reportado como referência, mas não influenciou "
              f"o resultado.")

    patch_grid = (fg_mask_patch.shape[1], fg_mask_patch.shape[2])
    N = len(image_paths)
    P = fg_mask_patch.shape[1] * fg_mask_patch.shape[2]

    metrics: Dict[str, Any] = {
        "cosaliency_path": str(corr_path),
        "n_images":        int(N),
        "n_patches":       int(P),
        "patch_grid":      list(patch_grid),
        "n_clusters_used": int(n_clusters),
        "saliency_alpha":  safe_float(saliency_alpha),
        # raw arrays para tabela no relatório
        "_cluster_scores_e2_raw":   c_scores_e2.tolist()   if c_scores_e2   is not None else [],
        "_cluster_scores_sp_raw":   c_scores_sp.tolist()   if c_scores_sp   is not None else [],
        "_cluster_scores_cent_raw": c_scores_cent.tolist() if c_scores_cent is not None else [],
        "_cluster_scores_comp_raw": c_scores_comp.tolist() if c_scores_comp is not None else [],
        "_cluster_scores_fin_raw":  cluster_scores.tolist(),
        "_cluster_is_fg_raw":       cluster_is_fg.tolist(),
    }

    # Carrega resize_info e calcula content_mask ANTES de qualquer métrica —
    # não só para o GT (alinhamento), mas também para fg_fraction,
    # consistência de clusters e AUC, que ficam infladas/diluídas por
    # padding se não forem restritas ao conteúdo real. Ver docstrings das
    # funções correspondentes.
    feat_data = load_features(args.features)
    resize_info = feat_data.get("resize_info") if feat_data else None

    content_mask = None
    if not args.no_padding_mask:
        content_mask = compute_content_mask(resize_info, patch_grid)

    padding_info: Dict[str, Any] = {"available": False, "masked": False, "fraction_padding": 0.0}
    if resize_info is not None:
        frac_padding = 1.0 - float((content_mask if content_mask is not None
                                    else compute_content_mask(resize_info, patch_grid)).mean())
        padding_info = {
            "available": True,
            "masked": content_mask is not None,
            "fraction_padding": frac_padding,
        }
        if content_mask is not None:
            print(f"      resize_info encontrado em features.npz (modo keep_ar) — "
                  f"{frac_padding:.1%} dos patches são padding e serão excluídos das "
                  f"métricas (fração FG, consistência de clusters, oracle, mapeamento, AUC).")
        else:
            print(f"      resize_info encontrado, mas --no_padding_mask ativo — "
                  f"{frac_padding:.1%} dos patches de padding NÃO serão excluídos.")
    elif args.features is None:
        print("      [Aviso] --features não informado — se a Etapa 1 rodou com "
              "--resize keep_ar, o GT e as métricas de padding ficarão "
              "desalinhados/inflados. Passe --features para corrigir.")
    metrics["padding_info"] = padding_info

    # Versão dos labels só para avaliação: no cosaliency.npz, patches de
    # padding recebem artificialmente o label C0 — substituímos por -1 para
    # não serem confundidos com integrantes reais do cluster C0.
    cluster_labels_eval = cluster_labels.reshape(N, P).copy()
    if content_mask is not None:
        cluster_labels_eval[~content_mask] = -1

    print("[2/8] Calculando fração FG e comparação coarse/refined...")
    metrics.update(compute_fg_fraction_metrics(fg_mask_patch, fg_mask_coarse, content_mask=content_mask))

    print("[3/8] Calculando coerência espacial...")
    metrics.update(compute_spatial_coherence(fg_mask_patch))

    print("[4/8] Calculando consistência dos clusters...")
    metrics.update(compute_cluster_consistency(
        cluster_labels_eval, cluster_scores, cluster_is_fg, n_clusters, content_mask=content_mask))

    print("[5/8] Calculando métricas de saliência espacial...")
    metrics.update(compute_saliency_metrics(
        c_scores_e2, c_scores_sp, c_scores_cent, c_scores_comp,
        cluster_scores, cluster_is_fg, saliency_alpha
    ))

    print("[6/8] Procurando ground truth...")
    gt_index = load_gt_index(args.gt_index)

    patch_gt, gt_info = load_patch_gt_masks(
        image_paths,
        patch_grid,
        gt_index,
        args.fg_threshold,
        resize_info=resize_info,
    )

    has_gt = (
        patch_gt is not None
        and gt_info.get("matched", 0) > 0
    )

    cluster_true_fg_list = []
    oracle_subset = []

    if has_gt:
        print(f"      GT: {gt_info['matched']} / {N} imagens.")
        print("[7/8] Métricas com GT: IoU, oracle, mapeamento, AUC...")
        metrics.update(compute_patch_iou_metrics(fg_mask_patch, fg_mask_coarse, patch_gt))
        pixel_gts = load_pixel_gt_masks(image_paths, image_sizes, gt_index, args.fg_threshold)
        metrics.update(compute_pixel_iou_metrics(fg_mask_pixel, image_sizes, pixel_gts))
        oracle_m = compute_oracle_cluster_metrics(
            cluster_labels_eval,
            cluster_scores,
            patch_gt,
            n_clusters,
        )
        metrics.update(oracle_m)

        oracle_subset = oracle_m.get(
            "oracle_best_global_subset",
            [],
        )

        mapping_m = compute_mapping_accuracy(
            cluster_labels_eval,
            cluster_is_fg,
            patch_gt,
            n_clusters,
        )
        metrics.update(mapping_m)
        cluster_true_fg_list = mapping_m.get("cluster_true_fg", [])
        metrics.update(compute_auc_metrics(fg_score_global, patch_gt, content_mask=content_mask))
        metrics.update(compute_gt_threshold_iou(fg_score_global, patch_gt))
    else:
        print("      GT indisponível.")

    print("[8/8] Gerando relatório e visualizações...")
    assessments = build_assessments(metrics, has_gt=has_gt)
    diagnosis   = infer_bottleneck(metrics, has_gt=has_gt)

    # Remove raw arrays do JSON final (são grandes e redundantes)
    metrics_json = {k: v for k, v in metrics.items() if not k.startswith("_")}
    json_path = output_dir / "step3_quality.json"
    json_path.write_text(
        json.dumps({"metrics": metrics_json,
                    "assessments": [assessment_to_dict(a) for a in assessments],
                    "diagnosis": diagnosis},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.no_plots:
        plot_mask_quality(fg_mask_patch, fg_mask_pixel, image_paths,
                          image_sizes, output_dir, args.max_viz_images)
        plot_cluster_map(cluster_labels, cluster_is_fg, image_paths, patch_grid,
                         output_dir, args.max_viz_images)
        plot_coarse_vs_refined(fg_mask_patch, fg_mask_coarse, image_paths,
                               patch_gt if has_gt else None, output_dir, args.max_viz_images)
        plot_fg_fraction(fg_mask_patch, output_dir)

        # Plots de saliência espacial (novos)
        plot_saliency_decomposition(
            c_scores_e2, c_scores_sp, c_scores_cent, c_scores_comp,
            cluster_scores, cluster_is_fg, saliency_alpha, output_dir
        )
        plot_saliency_spatial_map(
            cluster_labels, c_scores_sp, c_scores_cent, c_scores_comp,
            cluster_is_fg, image_paths, patch_grid, output_dir, args.max_viz_images
        )

        if has_gt:
            plot_gt_iou_curve(metrics, output_dir)
            if oracle_subset:
                plot_oracle_clusters(
                cluster_labels_eval,
                cluster_scores,
                cluster_is_fg,
                patch_gt,
                oracle_subset,
                patch_grid,
                output_dir,
                args.max_viz_images,
            )
            if cluster_true_fg_list:
                plot_mapping_accuracy(cluster_scores, cluster_is_fg,
                                      cluster_true_fg_list, output_dir)

    report_path = write_report(output_dir, corr_path, metrics, assessments,
                               diagnosis, gt_info, padding_info)
    print_console_summary(metrics, assessments, diagnosis, padding_info)
    print(f"\n[Salvo] JSON     : {json_path}")
    print(f"[Salvo] Relatório: {report_path}")
    print(f"[Salvo] Figuras  : {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido."); sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}"); sys.exit(1)