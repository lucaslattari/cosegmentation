#!/usr/bin/env python3
"""
=============================================================================
analyze_step2_correspondence.py — Auditoria da Etapa 2 de um pipeline de cosegmentação
=============================================================================

Lê o correspondence.npz gerado pela Etapa 2 e calcula métricas para responder:

  1. Os fg_scores estão polarizados (bons) ou achatados em torno de 0.5 (ruins)?
  2. Os pares concordam entre si sobre quais patches são FG?
  3. A correlação entre fg_scores e saliência CLS (Etapa 1) faz sentido?
  4. O resultado é estável ou depende criticamente de poucos pares?
  5. A fração de FG por imagem está em uma faixa razoável?
  6. Se houver GT: o threshold ótimo de fg_scores produz boa segmentação?
  7. Se houver GT: a orientação FG/BG (qual lado do NCut é objeto) está correta?
  8. Se houver GT: uma heurística alternativa de orientação (tamanho, posição
     espacial, ou combinação) teria funcionado melhor do que a saliência CLS
     usada em produção?
  9. O sinal está apenas invertido? Compara AUC/IoU raw, invertidos e oracle.
 10. Quanto de IoU é perdido exclusivamente pela orientação de cada par?
 11. Se a orientação dos pares fosse oracle antes da votação, a agregação seria boa?
 12. Os top-K pares cobrem todas as imagens de forma equilibrada?

Funciona com ou sem ground truth.

Uso mínimo:
  python analyze_step2_correspondence.py \
      --correspondence features/Panda/correspondence.npz

Com ground truth:
  python analyze_step2_correspondence.py \
      --correspondence features/Panda/correspondence.npz \
      --gt_index gt_index.json \
      --output features/Panda/step2_quality

Saídas:
  step2_quality.json
  step2_quality_report.md
  viz_score_polarization.png
  viz_pair_agreement.png
  viz_leave_one_pair_stability.png
  viz_fg_fraction_distribution.png
  viz_gt_iou_curve.png                          (se GT disponível)
  viz_orientation_accuracy.png                  (se GT disponível)
  viz_orientation_heuristics_comparison.png     (se GT disponível)
  viz_polarity_diagnostics.png                  (se GT disponível)
  viz_pair_coverage.png

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
    import matplotlib.gridspec as gridspec
except Exception:
    plt = None

try:
    from sklearn.metrics import roc_auc_score, f1_score
except Exception:
    roc_auc_score = None
    f1_score = None


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class Assessment:
    metric: str
    value: Optional[float]
    verdict: str
    indication: str
    suggested_test: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def compact_float(x: Optional[float], ndigits: int = 4) -> str:
    v = safe_float(x)
    return "n/a" if v is None else f"{v:.{ndigits}f}"


def assessment_to_dict(a: Assessment) -> Dict[str, Any]:
    return {
        "metric": a.metric,
        "value": a.value,
        "verdict": a.verdict,
        "indication": a.indication,
        "suggested_test": a.suggested_test,
    }


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_correspondence(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"correspondence.npz não encontrado: {path}")

    data = np.load(path, allow_pickle=True)
    required = ["fg_scores", "pair_votes", "top_pairs", "cls_saliency",
                "patch_grid", "image_paths"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"Campos ausentes em correspondence.npz: {missing}")

    fg_scores    = np.asarray(data["fg_scores"],    dtype=np.float32)   # (N, P)
    pair_votes   = np.asarray(data["pair_votes"],   dtype=np.float32)   # (K, N, P)
    top_pairs    = np.asarray(data["top_pairs"],    dtype=np.float32)   # (K, 3)
    cls_saliency = np.asarray(data["cls_saliency"], dtype=np.float32)   # (N, P)
    patch_grid   = tuple(int(x) for x in np.asarray(data["patch_grid"]).tolist())
    image_paths  = [str(x) for x in data["image_paths"].tolist()]

    N, P = fg_scores.shape
    H_p, W_p = patch_grid

    if fg_scores.ndim != 2:
        raise ValueError(f"fg_scores deveria ser (N, P), obtido {fg_scores.shape}")
    if pair_votes.shape[1] != N or pair_votes.shape[2] != P:
        raise ValueError(f"pair_votes incompatível: {pair_votes.shape} vs N={N} P={P}")
    if H_p * W_p != P:
        raise ValueError(f"patch_grid {patch_grid} incompatível com P={P}")

    return {
        "fg_scores":    fg_scores,
        "pair_votes":   pair_votes,
        "top_pairs":    top_pairs,
        "cls_saliency": cls_saliency,
        "patch_grid":   patch_grid,
        "image_paths":  image_paths,
    }

def load_resize_info(features_path: Optional[Path]) -> Optional[np.ndarray]:
    """
    Lê resize_info de um features.npz da Etapa 1, se disponível.
    Só existe se a Etapa 1 rodou com --resize keep_ar.
    Colunas: resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w
    """
    if features_path is None or not features_path.exists():
        return None
    data = np.load(features_path, allow_pickle=True)
    if "resize_info" not in data.files:
        return None
    return np.asarray(data["resize_info"])

def load_gt_index(gt_index_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if gt_index_path is None:
        return None
    if not gt_index_path.exists():
        print(f"[Aviso] gt_index.json não encontrado: {gt_index_path}. Rodando sem GT.")
        return None
    with gt_index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _path_variants(path_str: str):
    import os
    variants = set()
    raw = str(path_str).replace("\\", "/")
    variants.add(raw)
    variants.add(raw.lstrip("./"))
    p = Path(path_str)
    try:
        variants.add(p.as_posix())
    except Exception:
        pass
    try:
        variants.add(os.path.relpath(p).replace("\\", "/"))
    except Exception:
        pass
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
    resize_info: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
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
    for img_path, gt_path in mappings.items():
        for v in _path_variants(img_path):
            variant_lookup[v] = str(gt_path)
        base = Path(img_path).name
        basename_to_gt.setdefault(base, []).append(str(gt_path))

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
                gt_path = Path(variant_lookup[v])
                break
        if gt_path is None:
            gt_path_str = unique_basename.get(Path(img_path).name)
            gt_path = Path(gt_path_str) if gt_path_str else None

        if gt_path is None or not gt_path.exists():
            masks.append(np.zeros((H_p, W_p), dtype=bool))
            missing.append(img_path)
            continue

        try:
            gt_img = Image.open(gt_path).convert("L")

            if resize_info is not None:
                resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = \
                    [int(x) for x in resize_info[idx]]

                gt_resized = gt_img.resize((resized_w, resized_h), Image.BILINEAR)
                canvas = Image.new("L", (canvas_w, canvas_h), 0)
                canvas.paste(gt_resized, (pad_left, pad_top))

                mask_small = canvas.resize((W_p, H_p), Image.BILINEAR)
            else:
                mask_small = gt_img.resize((W_p, H_p), Image.BILINEAR)

            arr = np.asarray(mask_small, dtype=np.float32) / 255.0
            masks.append(arr >= fg_threshold)
        except Exception as e:
            print(f"[Aviso] Falha lendo GT {gt_path}: {e}")
            masks.append(np.zeros((H_p, W_p), dtype=bool))
            missing.append(img_path)

    patch_gt = np.stack([m.reshape(-1) for m in masks], axis=0)
    matched = len(image_paths) - len(missing)
    info = {
        "available": matched > 0,
        "matched": matched,
        "missing": len(missing),
        "fg_patch_fraction": float(patch_gt.mean()) if matched > 0 else None,
    }
    return (patch_gt if matched > 0 else None), info


# ---------------------------------------------------------------------------
# Métricas sem GT
# ---------------------------------------------------------------------------

def compute_content_mask(resize_info: "np.ndarray | None",
                         patch_grid: Tuple[int, int]) -> "np.ndarray | None":
    """
    Portado de step2_correspondence.py / step3_cosaliency.py — mesma lógica
    exata. A partir de resize_info (só existe quando a Etapa 1 rodou com
    --resize keep_ar), calcula, por imagem, quais patches da grade caem
    dentro da imagem real e quais caem inteira ou parcialmente na faixa de
    padding.

    Por que isso importa AQUI (não só para o alinhamento do GT): fg_scores
    e pair_votes para patches de padding são forçados a 0 (background) por
    construção em step2_correspondence.py — nunca participam do NCut. Isso
    NÃO é ruído, é um valor determinístico. Se as métricas deste analisador
    não excluírem esses patches, dois problemas surgem:

    1. Métricas de "concordância"/"decisão" ficam infladas: um bloco grande
       de zeros idênticos entre pares/imagens conta como "decidido" ou
       "concordância perfeita" trivialmente, sem refletir nenhum trabalho
       real do NCut.

    2. Em orientation_accuracy e orientation_alternatives, o lado invertido
       (~vote / ~pred) usado como hipótese alternativa passa a incluir TODO
       o padding como se fosse "foreground" — inflando artificialmente o
       número de falsos positivos dessa hipótese e enviesando a comparação
       a favor da orientação já escolhida, mesmo quando ela está errada no
       conteúdo real. Ver correção em compute_orientation_accuracy() e
       compute_orientation_alternatives().

    Retorna None se resize_info for None (Etapa 1 rodou em modo 'square',
    sem padding — não há nada para mascarar).
    """
    if resize_info is None:
        return None

    H_p, W_p = patch_grid
    N = resize_info.shape[0]
    canvas_h, canvas_w = int(resize_info[0, 4]), int(resize_info[0, 5])
    patch_h = canvas_h / H_p
    patch_w = canvas_w / W_p

    row_starts = np.arange(H_p) * patch_h
    col_starts = np.arange(W_p) * patch_w
    row_ends = row_starts + patch_h
    col_ends = col_starts + patch_w

    masks = np.zeros((N, H_p, W_p), dtype=bool)
    for i in range(N):
        resized_h, resized_w, pad_top, pad_left, _, _ = [int(x) for x in resize_info[i]]
        content_row = (row_ends > pad_top) & (row_starts < pad_top + resized_h)
        content_col = (col_ends > pad_left) & (col_starts < pad_left + resized_w)
        masks[i] = content_row[:, None] & content_col[None, :]

    return masks.reshape(N, -1)


def _content_indices(content_mask: "np.ndarray | None", img_idx: int, P: int) -> np.ndarray:
    """Índices de patches de conteúdo para uma imagem, ou todos os P se não houver máscara."""
    if content_mask is None:
        return np.arange(P)
    idx = np.flatnonzero(content_mask[img_idx])
    return idx if idx.size > 0 else np.arange(P)  # salvaguarda: imagem sem conteúdo detectado


def compute_polarization_metrics(fg_scores: np.ndarray,
                                 content_mask: "np.ndarray | None" = None) -> Dict[str, Any]:
    """
    Avalia se os fg_scores estão polarizados (0 ou 1) ou achatados em torno de 0.5.

    Um NCut bem-sucedido produz partição limpa → scores longe de 0.5.
    Scores concentrados em torno de 0.5 indicam que o NCut não encontrou
    estrutura clara, ou que a orientação FG/BG falhou sistematicamente.

    Métricas:
      - fraction_decided: fração de patches com |score - 0.5| > 0.2
        (patches que "tomaram partido" claramente)
      - mean_abs_deviation: desvio médio de 0.5 (quanto os scores se afastam da incerteza)
      - bimodality_score: B de Bimodalidade de Sarle, baseado em skewness e kurtosis.
        B > 0.555 (distribuição uniforme) sugere bimodalidade; B > 0.9 é forte.
    """
    N, P = fg_scores.shape

    if content_mask is not None:
        flat = fg_scores[content_mask]
    else:
        flat = fg_scores.flatten()

    dev = np.abs(flat - 0.5)
    fraction_decided = float(np.mean(dev > 0.2))
    mean_abs_dev = float(np.mean(dev))

    # Bimodalidade de Sarle: B = (skew² + 1) / (kurt + 3(n-1)²/((n-2)(n-3)))
    # Para n grande, simplifica para B = (skew² + 1) / (kurt + 3)
    n = len(flat)
    mu = float(np.mean(flat))
    std = float(np.std(flat))
    if std < 1e-8:
        bimodality = 0.0
    else:
        skew = float(np.mean(((flat - mu) / std) ** 3))
        kurt = float(np.mean(((flat - mu) / std) ** 4))
        # Correção de viés para n finito
        skew_corr = (
            skew * math.sqrt(n * (n - 1)) / (n - 2)
            if n > 2
            else skew
        )

        kurt_excess = kurt - 3.0

        finite_sample_correction = (
            3 * (n - 1) ** 2
            / max((n - 2) * (n - 3), 1)
        )

        bimodality = (
            (skew_corr ** 2 + 1)
            / (kurt_excess + finite_sample_correction)
        )

    # Fração de FG por imagem (idealmente entre 0.15 e 0.70).
    # Calculada só sobre patches de conteúdo: patches de padding são sempre
    # 0 (background) por construção, e diluiriam a fração real de FG na
    # área que de fato importa, especialmente em coleções com bastante
    # padding (imagens muito não-quadradas).
    if content_mask is not None:
        fg_fractions = np.array([
            fg_scores[i][content_mask[i]].mean() if content_mask[i].any() else fg_scores[i].mean()
            for i in range(N)
        ])
    else:
        fg_fractions = fg_scores.mean(axis=1)
    fg_frac_mean = float(np.mean(fg_fractions))
    fg_frac_std  = float(np.std(fg_fractions))
    fg_frac_min  = float(np.min(fg_fractions))
    fg_frac_max  = float(np.max(fg_fractions))

    # Imagens com FG fraction extrema (possível orientação errada)
    n_extreme_low  = int(np.sum(fg_fractions < 0.10))
    n_extreme_high = int(np.sum(fg_fractions > 0.85))

    return {
        "polarization_padding_excluded": bool(content_mask is not None),
        "polarization_fraction_decided": fraction_decided,
        "polarization_mean_abs_deviation": mean_abs_dev,
        "polarization_bimodality_score": safe_float(bimodality),
        "fg_fraction_mean": fg_frac_mean,
        "fg_fraction_std": fg_frac_std,
        "fg_fraction_min": fg_frac_min,
        "fg_fraction_max": fg_frac_max,
        "fg_fraction_extreme_low_count": n_extreme_low,
        "fg_fraction_extreme_high_count": n_extreme_high,
    }


def compute_pair_agreement_metrics(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    content_mask: "np.ndarray | None" = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Mede se os K pares concordam entre si sobre quais patches são FG.

    Para cada imagem i, pega os votos de todos os pares que incluem i
    e calcula a concordância média entre eles (correlação de Pearson entre
    os vetores de voto de pares diferentes para a mesma imagem).

    Concordância alta → o NCut encontra estrutura consistente independente
    do parceiro escolhido para comparação.
    Concordância baixa → resultado instável, muito sensível a qual par é usado.

    Se content_mask for fornecida, a correlação é calculada só sobre
    patches de conteúdo. Patches de padding têm voto sempre 0 em qualquer
    par (forçado por construção), então incluí-los infla a correlação
    trivialmente — um bloco grande de "concordância perfeita" que não
    reflete nenhum trabalho real do NCut.
    """
    K, N, P = pair_votes.shape
    if K < 2:
        return {
            "pair_agreement_mean": None,
            "pair_agreement_std": None,
            "pair_agreement_note": "Menos de 2 pares — concordância indefinida.",
        }

    # Para cada imagem, coleta os vetores de voto dos pares que a incluem
    img_correlations = []
    for i in range(N):
        cidx = _content_indices(content_mask, i, P)

        # Pares que incluem imagem i
        vote_vectors = []
        for k in range(K):
            pi, pj = int(top_pairs[k, 0]), int(top_pairs[k, 1])
            if pi == i or pj == i:
                vote_vectors.append(pair_votes[k, i][cidx])

        if len(vote_vectors) < 2:
            continue

        # Correlação média entre todos os pares de vetores de voto
        corrs = []
        for a in range(len(vote_vectors)):
            for b in range(a + 1, len(vote_vectors)):
                va, vb = vote_vectors[a], vote_vectors[b]
                sa, sb = float(np.std(va)), float(np.std(vb))
                if sa > 1e-6 and sb > 1e-6:
                    corr = float(np.corrcoef(va, vb)[0, 1])
                    corrs.append(corr)

        if corrs:
            img_correlations.append(float(np.mean(corrs)))

    if not img_correlations:
        return {
            "pair_agreement_mean": None,
            "pair_agreement_std": None,
            "pair_agreement_note": "Nenhum par de votos comparável encontrado.",
        }

    return {
        "pair_agreement_padding_excluded": bool(content_mask is not None),
        "pair_agreement_mean": safe_float(np.mean(img_correlations)),
        "pair_agreement_std": safe_float(np.std(img_correlations)),
        "pair_agreement_n_images": len(img_correlations),
    }


def compute_stability_metrics(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    fg_scores: np.ndarray,
    content_mask: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """
    Estabilidade leave-one-pair-out: remove um par por vez e recalcula fg_scores.
    Mede o quanto o resultado muda quando um par é excluído.

    Alta variabilidade → resultado frágil, depende criticamente de poucos pares.
    Baixa variabilidade → resultado robusto mesmo com menos pares.

    Se content_mask for fornecida, o MAE é calculado só sobre patches de
    conteúdo. Em patches de padding, fg_loo e fg_scores são sempre 0
    independente de qual par foi removido (voto de padding é sempre
    forçado a background) — incluí-los no MAE dilui artificialmente o
    valor final, fazendo o resultado parecer mais estável do que é.
    """
    K, N, P = pair_votes.shape
    if K < 3:
        return {
            "stability_loo_mae_mean": None,
            "stability_loo_mae_std": None,
            "stability_note": "Menos de 3 pares — estabilidade LOO indefinida.",
        }

    pairs_list = [(int(top_pairs[k, 0]), int(top_pairs[k, 1]), float(top_pairs[k, 2]))
                  for k in range(K)]

    maes = []
    for k_out in range(K):
        # Recalcula fg_scores sem o par k_out
        remaining = [(k, p) for k, p in enumerate(pairs_list) if k != k_out]
        if not remaining:
            continue

        weight_sum  = np.zeros((N, P), dtype=np.float32)
        fg_weighted = np.zeros((N, P), dtype=np.float32)

        for k, (pi, pj, sim) in remaining:
            w = max(sim, 0.0)
            fg_weighted[pi] += pair_votes[k, pi] * w
            fg_weighted[pj] += pair_votes[k, pj] * w
            weight_sum[pi]  += w
            weight_sum[pj]  += w

        fg_loo = np.divide(
            fg_weighted, weight_sum,
            out=np.full_like(fg_weighted, 0.5),
            where=weight_sum > 0,
        )
        diff = np.abs(fg_loo - fg_scores)
        if content_mask is not None:
            mae = float(np.mean(diff[content_mask]))
        else:
            mae = float(np.mean(diff))
        maes.append(mae)

    return {
        "stability_padding_excluded": bool(content_mask is not None),
        "stability_loo_mae_mean": safe_float(np.mean(maes)),
        "stability_loo_mae_std":  safe_float(np.std(maes)),
        "stability_loo_mae_max":  safe_float(np.max(maes)),
        "stability_n_pairs":      int(K),
    }


def compute_cls_correlation_metrics(
    fg_scores: np.ndarray,
    cls_saliency: np.ndarray,
    content_mask: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """
    Correlação entre fg_scores (Etapa 2) e saliência CLS (Etapa 1).

    Alta correlação positiva → as duas fontes de informação concordam.
    Correlação negativa sistemática → orientação FG/BG provavelmente invertida.
    Correlação próxima de zero → a Etapa 2 não está refinando a saliência CLS.

    Nota: esperamos correlação positiva moderada, não necessariamente alta —
    o NCut pode identificar um objeto diferente do mais saliente pelo CLS.

    Se content_mask for fornecida, a correlação é calculada só sobre patches
    de conteúdo — em patches de padding, fg_scores é sempre 0 e cls_saliency
    tende a ser aproximadamente constante (embedding quase idêntico entre
    padding de imagens diferentes), o que pode distorcer a correlação sem
    refletir a relação real entre as duas etapas na região que importa.
    """
    N, P = fg_scores.shape
    corrs = []
    for i in range(N):
        cidx = _content_indices(content_mask, i, P)
        fg  = fg_scores[i][cidx]
        sal = cls_saliency[i][cidx]
        sf, ss = float(np.std(fg)), float(np.std(sal))
        if sf > 1e-6 and ss > 1e-6:
            corrs.append(float(np.corrcoef(fg, sal)[0, 1]))

    if not corrs:
        return {"cls_correlation_mean": None, "cls_correlation_std": None}

    n_negative = int(sum(1 for c in corrs if c < -0.1))
    return {
        "cls_correlation_padding_excluded": bool(content_mask is not None),
        "cls_correlation_mean": safe_float(np.mean(corrs)),
        "cls_correlation_std":  safe_float(np.std(corrs)),
        "cls_correlation_min":  safe_float(np.min(corrs)),
        "cls_correlation_n_negative_images": n_negative,
    }


# ---------------------------------------------------------------------------
# Métricas com GT
# ---------------------------------------------------------------------------

def _best_iou_over_thresholds(
    scores: np.ndarray,
    patch_gt: np.ndarray,
    content_mask: "np.ndarray | None" = None,
) -> Tuple[float, float, List[float], List[float]]:
    """Calcula IoU médio por threshold, opcionalmente só em patches de conteúdo."""
    N, P = scores.shape
    thresholds = np.arange(0.05, 0.96, 0.05)
    mean_ious: List[float] = []

    for t in thresholds:
        ious = []
        for i in range(N):
            cidx = _content_indices(content_mask, i, P)
            pred = scores[i][cidx] >= t
            gt = patch_gt[i].astype(bool)[cidx]
            inter = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            if union > 0:
                ious.append(float(inter / union))
        mean_ious.append(float(np.mean(ious)) if ious else 0.0)

    best_idx = int(np.argmax(mean_ious))
    return (
        float(mean_ious[best_idx]),
        float(thresholds[best_idx]),
        thresholds.tolist(),
        mean_ious,
    )


def compute_gt_iou_metrics(
    fg_scores: np.ndarray,
    patch_gt: np.ndarray,
    content_mask: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """
    Avalia a qualidade dos fg_scores em DUAS polaridades:

      raw      = fg_scores
      inverted = 1 - fg_scores

    Isso é crucial porque um AUC muito abaixo de 0.5 pode indicar sinal forte,
    porém com polaridade invertida, e não ausência de sinal.

    Retorna métricas compatíveis com a versão anterior (gt_best_iou e
    gt_auc_roc continuam significando a polaridade raw), além de:
      - gt_best_iou_inverted
      - gt_best_iou_oracle_polarity
      - gt_auc_roc_inverted
      - gt_auc_roc_oracle_polarity
      - gt_polarity_iou_gain
      - gt_polarity_auc_gain

    Quando há content_mask, TODAS as comparações raw/inverted são restritas
    aos patches de conteúdo. Isso evita que 1-fg_scores transforme padding
    determinístico (score 0) em foreground artificial.
    """
    N, P = fg_scores.shape

    raw_best_iou, raw_best_thr, thresholds, raw_curve = _best_iou_over_thresholds(
        fg_scores, patch_gt, content_mask
    )
    inv_scores = 1.0 - fg_scores
    inv_best_iou, inv_best_thr, _, inv_curve = _best_iou_over_thresholds(
        inv_scores, patch_gt, content_mask
    )

    # IoU no threshold 0.5 para as duas polaridades
    def mean_iou_at(scores: np.ndarray, threshold: float) -> float:
        vals = []
        for i in range(N):
            cidx = _content_indices(content_mask, i, P)
            pred = scores[i][cidx] >= threshold
            gt = patch_gt[i].astype(bool)[cidx]
            vals.append(_iou(pred, gt))
        return float(np.mean(vals)) if vals else 0.0

    raw_iou_05 = mean_iou_at(fg_scores, 0.5)
    inv_iou_05 = mean_iou_at(inv_scores, 0.5)

    auc_raw = auc_inv = None
    if roc_auc_score is not None:
        if content_mask is not None:
            flat_raw = fg_scores[content_mask]
            flat_inv = inv_scores[content_mask]
            flat_gt = patch_gt[content_mask].astype(int)
        else:
            flat_raw = fg_scores.flatten()
            flat_inv = inv_scores.flatten()
            flat_gt = patch_gt.flatten().astype(int)
        if len(np.unique(flat_gt)) == 2:
            try:
                auc_raw = float(roc_auc_score(flat_gt, flat_raw))
                auc_inv = float(roc_auc_score(flat_gt, flat_inv))
            except Exception:
                pass

    oracle_iou = max(raw_best_iou, inv_best_iou)
    oracle_iou_polarity = "raw" if raw_best_iou >= inv_best_iou else "inverted"

    valid_aucs = [x for x in [auc_raw, auc_inv] if x is not None]
    oracle_auc = max(valid_aucs) if valid_aucs else None
    oracle_auc_polarity = None
    if oracle_auc is not None:
        oracle_auc_polarity = "raw" if (auc_raw is not None and auc_raw >= (auc_inv or -1)) else "inverted"

    return {
        # Compatibilidade com relatórios antigos
        "gt_iou_padding_note": "IoU/AUC raw e invertidos são avaliados apenas em conteúdo quando content_mask está disponível.",
        "gt_auc_padding_excluded": bool(content_mask is not None),
        "gt_best_iou": safe_float(raw_best_iou),
        "gt_best_threshold": safe_float(raw_best_thr),
        "gt_iou_at_threshold_05": safe_float(raw_iou_05),
        "gt_auc_roc": safe_float(auc_raw),
        "gt_iou_curve_thresholds": thresholds,
        "gt_iou_curve_values": raw_curve,

        # Diagnóstico explícito de polaridade
        "gt_best_iou_raw": safe_float(raw_best_iou),
        "gt_best_threshold_raw": safe_float(raw_best_thr),
        "gt_iou_at_threshold_05_raw": safe_float(raw_iou_05),
        "gt_best_iou_inverted": safe_float(inv_best_iou),
        "gt_best_threshold_inverted": safe_float(inv_best_thr),
        "gt_iou_at_threshold_05_inverted": safe_float(inv_iou_05),
        "gt_best_iou_oracle_polarity": safe_float(oracle_iou),
        "gt_best_iou_oracle_polarity_name": oracle_iou_polarity,
        "gt_polarity_iou_gain": safe_float(oracle_iou - raw_best_iou),
        "gt_iou_curve_values_inverted": inv_curve,
        "gt_auc_roc_raw": safe_float(auc_raw),
        "gt_auc_roc_inverted": safe_float(auc_inv),
        "gt_auc_roc_oracle_polarity": safe_float(oracle_auc),
        "gt_auc_roc_oracle_polarity_name": oracle_auc_polarity,
        "gt_polarity_auc_gain": safe_float((oracle_auc - auc_raw) if oracle_auc is not None and auc_raw is not None else None),
    }

def compute_orientation_accuracy(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    patch_gt: np.ndarray,
    content_mask: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """
    Para cada par k e cada imagem (i, j) nesse par, verifica se o voto
    do NCut tem maior overlap com o GT do que o voto invertido.

    orientation_accuracy = fração de (par, imagem) em que o NCut escolheu
    o lado correto (FG) espontaneamente.

    Valores abaixo de 0.5 indicam que a heurística de orientação por
    saliência CLS está sistematicamente falhando.

    CORREÇÃO IMPORTANTE (padding): vote para patches de padding é sempre
    False (background) por construção — nunca participaram do NCut. Isso é
    inofensivo para iou_pos = _iou(vote, gt), já que padding vira
    verdadeiro-negativo (não entra na fórmula de IoU). MAS o voto invertido
    ~vote inclui TODO o padding como se fosse "foreground" — inflando
    artificialmente os falsos positivos da hipótese invertida e enviesando
    a comparação a favor da orientação já escolhida, mesmo quando ela está
    errada no conteúdo real. Em coleções com bastante padding, isso pode
    literalmente inverter o veredito (confirmado por simulação numérica:
    iou_pos >= iou_neg vira True mesmo quando a orientação correta, medida
    só no conteúdo, seria a invertida). Por isso vote/gt são restritos a
    content_mask antes de qualquer comparação.
    """
    K, N, P = pair_votes.shape
    correct = total = 0

    for k in range(K):
        pi, pj = int(top_pairs[k, 0]), int(top_pairs[k, 1])
        for img_idx in [pi, pj]:
            cidx = _content_indices(content_mask, img_idx, P)
            vote = pair_votes[k, img_idx].astype(bool)[cidx]
            gt   = patch_gt[img_idx].astype(bool)[cidx]

            iou_pos = _iou(vote, gt)
            iou_neg = _iou(~vote, gt)

            if iou_pos > 0 or iou_neg > 0:
                total += 1
                if iou_pos >= iou_neg:
                    correct += 1

    accuracy = float(correct / total) if total > 0 else None
    return {
        "orientation_padding_excluded": bool(content_mask is not None),
        "orientation_accuracy": safe_float(accuracy),
        "orientation_correct_pairs": correct,
        "orientation_total_pairs": total,
    }


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)



def compute_pair_orientation_iou_metrics(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    patch_gt: np.ndarray,
    content_mask: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """
    Mede, para cada ocorrência (par, imagem), a IoU da orientação salva,
    da orientação invertida e do oracle entre as duas.

    Isso separa claramente:
      - corte bom + orientação ruim  -> oracle alto, selected baixo, loss alto
      - corte ruim                   -> oracle baixo mesmo após inverter
    """
    K, N, P = pair_votes.shape
    selected_vals, inverted_vals, oracle_vals, losses = [], [], [], []
    details = []

    for k in range(K):
        pi, pj = int(top_pairs[k, 0]), int(top_pairs[k, 1])
        for img_idx in [pi, pj]:
            cidx = _content_indices(content_mask, img_idx, P)
            vote = pair_votes[k, img_idx].astype(bool)[cidx]
            gt = patch_gt[img_idx].astype(bool)[cidx]
            selected = _iou(vote, gt)
            inverted = _iou(~vote, gt)
            oracle = max(selected, inverted)
            loss = oracle - selected
            selected_vals.append(selected)
            inverted_vals.append(inverted)
            oracle_vals.append(oracle)
            losses.append(loss)
            details.append({
                "pair_index": int(k),
                "image_index": int(img_idx),
                "pair_i": int(pi),
                "pair_j": int(pj),
                "selected_iou": selected,
                "inverted_iou": inverted,
                "oracle_iou": oracle,
                "orientation_loss": loss,
                "oracle_prefers_inverted": bool(inverted > selected),
            })

    def stat(vals, fn):
        return safe_float(fn(vals)) if vals else None

    return {
        "pair_orientation_selected_iou_mean": stat(selected_vals, np.mean),
        "pair_orientation_inverted_iou_mean": stat(inverted_vals, np.mean),
        "pair_orientation_oracle_iou_mean": stat(oracle_vals, np.mean),
        "pair_orientation_oracle_iou_median": stat(oracle_vals, np.median),
        "pair_orientation_loss_mean": stat(losses, np.mean),
        "pair_orientation_loss_max": stat(losses, np.max),
        "pair_orientation_inverted_preferred_fraction": safe_float(
            np.mean([d["oracle_prefers_inverted"] for d in details]) if details else None
        ),
        "pair_orientation_details": details,
    }


def compute_pair_coverage_metrics(
    top_pairs: np.ndarray,
    n_images: int,
) -> Dict[str, Any]:
    """Conta quantos pares selecionados incluem cada imagem."""
    counts = np.zeros(n_images, dtype=np.int32)
    for k in range(len(top_pairs)):
        i, j = int(top_pairs[k, 0]), int(top_pairs[k, 1])
        counts[i] += 1
        counts[j] += 1
    return {
        "pair_coverage_per_image": counts.tolist(),
        "pair_coverage_min": int(counts.min()) if len(counts) else 0,
        "pair_coverage_max": int(counts.max()) if len(counts) else 0,
        "pair_coverage_mean": safe_float(counts.mean() if len(counts) else 0.0),
        "pair_coverage_std": safe_float(counts.std() if len(counts) else 0.0),
        "pair_coverage_n_images_without_pairs": int(np.sum(counts == 0)),
        "pair_coverage_n_images_single_pair": int(np.sum(counts == 1)),
    }


def _aggregate_votes_for_diagnostic(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    N: int,
    P: int,
) -> np.ndarray:
    """Mesma agregação ponderada da Etapa 2, usada apenas para diagnóstico."""
    weight_sum = np.zeros((N, P), dtype=np.float32)
    fg_weighted = np.zeros((N, P), dtype=np.float32)
    for k in range(len(top_pairs)):
        i, j, sim = int(top_pairs[k, 0]), int(top_pairs[k, 1]), float(top_pairs[k, 2])
        w = max(sim, 0.0)
        fg_weighted[i] += pair_votes[k, i] * w
        fg_weighted[j] += pair_votes[k, j] * w
        weight_sum[i] += w
        weight_sum[j] += w
    return np.divide(
        fg_weighted,
        weight_sum,
        out=np.full_like(fg_weighted, 0.5),
        where=weight_sum > 0,
    ).astype(np.float32)


def compute_oracle_orientation_aggregation_metrics(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    patch_gt: np.ndarray,
    content_mask: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """
    Upper bound diagnóstico da orientação ANTES da agregação.

    Produz dois oracles:
      1) imagewise: cada imagem de cada par pode inverter independentemente.
      2) pairwise: o par inteiro escolhe uma única polaridade, preservando a
         restrição real de orient_partition() da Etapa 2.

    Depois agrega os votos com a mesma regra ponderada da produção e mede o
    melhor IoU obtido por threshold.
    """
    K, N, P = pair_votes.shape
    oracle_imagewise = pair_votes.copy().astype(np.float32)
    oracle_pairwise = pair_votes.copy().astype(np.float32)
    pairwise_flips = 0
    imagewise_flips = 0

    for k in range(K):
        pi, pj = int(top_pairs[k, 0]), int(top_pairs[k, 1])

        # Oracle imagewise
        for img_idx in [pi, pj]:
            cidx = _content_indices(content_mask, img_idx, P)
            vote_c = pair_votes[k, img_idx].astype(bool)[cidx]
            gt_c = patch_gt[img_idx].astype(bool)[cidx]
            if _iou(~vote_c, gt_c) > _iou(vote_c, gt_c):
                imagewise_flips += 1
                full = np.zeros(P, dtype=np.float32)
                full[cidx] = (~vote_c).astype(np.float32)
                oracle_imagewise[k, img_idx] = full

        # Oracle pairwise: uma única decisão para i+j, como em produção
        selected_pair = 0.0
        inverted_pair = 0.0
        cache = {}
        for img_idx in [pi, pj]:
            cidx = _content_indices(content_mask, img_idx, P)
            vote_c = pair_votes[k, img_idx].astype(bool)[cidx]
            gt_c = patch_gt[img_idx].astype(bool)[cidx]
            selected_pair += _iou(vote_c, gt_c)
            inverted_pair += _iou(~vote_c, gt_c)
            cache[img_idx] = (cidx, vote_c)

        if inverted_pair > selected_pair:
            pairwise_flips += 1
            for img_idx in [pi, pj]:
                cidx, vote_c = cache[img_idx]
                full = np.zeros(P, dtype=np.float32)
                full[cidx] = (~vote_c).astype(np.float32)
                oracle_pairwise[k, img_idx] = full

    scores_imagewise = _aggregate_votes_for_diagnostic(oracle_imagewise, top_pairs, N, P)
    scores_pairwise = _aggregate_votes_for_diagnostic(oracle_pairwise, top_pairs, N, P)

    iw_iou, iw_thr, _, _ = _best_iou_over_thresholds(scores_imagewise, patch_gt, content_mask)
    pw_iou, pw_thr, _, _ = _best_iou_over_thresholds(scores_pairwise, patch_gt, content_mask)

    return {
        "oracle_orientation_imagewise_aggregated_best_iou": safe_float(iw_iou),
        "oracle_orientation_imagewise_aggregated_best_threshold": safe_float(iw_thr),
        "oracle_orientation_imagewise_flip_count": int(imagewise_flips),
        "oracle_orientation_pairwise_aggregated_best_iou": safe_float(pw_iou),
        "oracle_orientation_pairwise_aggregated_best_threshold": safe_float(pw_thr),
        "oracle_orientation_pairwise_flip_count": int(pairwise_flips),
        "oracle_orientation_pairwise_total_pairs": int(K),
    }

def compute_orientation_alternatives(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    patch_gt: np.ndarray,
    cls_saliency: np.ndarray,
    patch_grid: Tuple[int, int],
    baseline_accuracy: Optional[float],
    content_mask: "np.ndarray | None" = None,
) -> Dict[str, Any]:
    """
    Testa heurísticas alternativas de orientação FG/BG contra a heurística
    de produção (saliência CLS média, usada em orient_partition() no
    step2_correspondence.py).

    Para cada par k e cada imagem (i, j) desse par, o corte NCut já produziu
    uma partição binária em dois lados: pair_votes[k, img_idx] e seu
    complemento (~vote). A heurística de produção já escolheu um dos dois
    como FG (é o que está salvo em pair_votes). Aqui, para os MESMOS dois
    lados, recalculamos qual seria escolhido por três critérios alternativos,
    e medimos a fração de casos em que o lado escolhido bate melhor com o GT
    do que o lado oposto — a mesma definição de orientation_accuracy já usada
    para a heurística de produção, o que torna os números diretamente
    comparáveis entre si.

    Critérios testados:
      - size_based: escolhe o lado com MENOS patches marcados como FG.
        Assume que o objeto de interesse costuma ser minoria de pixels da
        imagem (prior comum em cosegmentação; ver Faktor & Irani, ICCV 2013).
      - spatial_based: escolhe o lado mais central e mais compacto. Mesma
        fórmula de compute_spatial_saliency() da Etapa 3
        (score = √(centralidade × compacidade)), aplicada aqui par a par
        em vez de por cluster global.
      - combined: votação por maioria simples entre os três critérios
        (CLS saliency original + size_based + spatial_based). Com 3
        critérios não há empate possível.

    Retorna também qual heurística obteve a maior orientation_accuracy
    nesta coleção e o ganho sobre a heurística de produção (baseline).

    CORREÇÃO IMPORTANTE (padding): vote (e portanto pred_*) para patches de
    padding é sempre False por construção. O lado invertido ~vote/~pred usado
    em toda comparação aqui (size_based via vote.sum() vs (~vote).sum(),
    spatial_based via spatial_score(~vote), e a checagem final iou_pred vs
    iou_other) passaria a incluir TODO o padding como se fosse o lado
    escolhido — inflando artificialmente o tamanho, deslocando o centro de
    massa espacial para a borda do canvas (onde o padding fica), e inflando
    falsos positivos na comparação de IoU. As três heurísticas alternativas
    ficam enviesadas a favor de "não inverter" a orientação da produção,
    mesmo quando a orientação correta no conteúdo real seria a invertida
    (mesmo problema numericamente confirmado em compute_orientation_accuracy).
    Por isso vote/gt/sal e as coordenadas espaciais usadas em spatial_score
    são todas restritas a content_mask antes de qualquer comparação.
    """
    K, N, P = pair_votes.shape
    H_p, W_p = patch_grid

    row_coords_full = np.repeat(np.arange(H_p), W_p) / max(H_p - 1, 1)
    col_coords_full = np.tile(np.arange(W_p), H_p) / max(W_p - 1, 1)
    max_dist = math.sqrt(0.5 ** 2 + 0.5 ** 2)
    dist_center_full = np.sqrt((row_coords_full - 0.5) ** 2 + (col_coords_full - 0.5) ** 2) / max_dist

    def spatial_score(mask: np.ndarray, row_c: np.ndarray, col_c: np.ndarray,
                      dist_c: np.ndarray) -> float:
        if not mask.any():
            return 0.0
        centrality = 1.0 - float(dist_c[mask].mean())
        std_r = float(row_c[mask].std())
        std_c_ = float(col_c[mask].std())
        # 0.289 = desvio padrão máximo de uma distribuição uniforme em [0,1]
        # (mesma normalização usada em compute_spatial_saliency da Etapa 3).
        compactness = 1.0 - min((std_r + std_c_) / 2.0 / 0.289, 1.0)
        return math.sqrt(max(centrality, 0.0) * max(compactness, 0.0))

    def cls_score(mask: np.ndarray, saliency: np.ndarray) -> float:
        return float(saliency[mask].mean()) if mask.any() else 0.0

    counters = {
        "size_based":    [0, 0],  # [corretos, total]
        "spatial_based": [0, 0],
        "combined":      [0, 0],
    }

    for k in range(K):
        pi, pj = int(top_pairs[k, 0]), int(top_pairs[k, 1])
        for img_idx in [pi, pj]:
            cidx = _content_indices(content_mask, img_idx, P)

            vote = pair_votes[k, img_idx].astype(bool)[cidx]
            gt   = patch_gt[img_idx].astype(bool)[cidx]
            sal  = cls_saliency[img_idx][cidx]
            row_c  = row_coords_full[cidx]
            col_c  = col_coords_full[cidx]
            dist_c = dist_center_full[cidx]

            # --- size_based: prefere o lado com menos patches ---
            pred_size = vote if int(vote.sum()) <= int((~vote).sum()) else ~vote

            # --- spatial_based: prefere o lado mais central/compacto ---
            s_pos = spatial_score(vote, row_c, col_c, dist_c)
            s_neg = spatial_score(~vote, row_c, col_c, dist_c)
            pred_spatial = vote if s_pos >= s_neg else ~vote

            # --- combined: maioria entre cls_saliency / size / spatial ---
            votes_for_saved_side = 0
            if cls_score(vote, sal) >= cls_score(~vote, sal):
                votes_for_saved_side += 1
            if int(vote.sum()) <= int((~vote).sum()):
                votes_for_saved_side += 1
            if s_pos >= s_neg:
                votes_for_saved_side += 1
            pred_combined = vote if votes_for_saved_side >= 2 else ~vote

            for name, pred in [
                ("size_based", pred_size),
                ("spatial_based", pred_spatial),
                ("combined", pred_combined),
            ]:
                iou_pred  = _iou(pred, gt)
                iou_other = _iou(~pred, gt)
                if iou_pred > 0 or iou_other > 0:
                    counters[name][1] += 1
                    if iou_pred >= iou_other:
                        counters[name][0] += 1

    result: Dict[str, Any] = {"orientation_alternatives_padding_excluded": bool(content_mask is not None)}
    accuracies: Dict[str, Optional[float]] = {"cls_saliency": safe_float(baseline_accuracy)}
    for name, (correct, total) in counters.items():
        acc = float(correct / total) if total > 0 else None
        result[f"orientation_accuracy_{name}"] = safe_float(acc)
        result[f"orientation_total_pairs_{name}"] = total
        accuracies[name] = safe_float(acc)

    valid = {k: v for k, v in accuracies.items() if v is not None}
    if valid:
        best_name = max(valid, key=valid.get)
        best_acc  = valid[best_name]
        result["orientation_best_heuristic"]    = best_name
        result["orientation_best_accuracy"]     = best_acc
        result["orientation_baseline_accuracy"] = accuracies.get("cls_saliency")
        if accuracies.get("cls_saliency") is not None:
            result["orientation_improvement_over_baseline"] = safe_float(
                best_acc - accuracies["cls_saliency"]
            )

    return result


# ---------------------------------------------------------------------------
# Interpretações heurísticas
# ---------------------------------------------------------------------------

def assess_high_good(
    metric: str, value: Optional[float],
    strong: float, moderate: float,
    strong_msg: str, moderate_msg: str, weak_msg: str,
    suggested_test: str,
) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível",
                          "Não foi possível calcular esta métrica.", suggested_test)
    if v >= strong:
        return Assessment(metric, v, "bom", strong_msg, suggested_test)
    if v >= moderate:
        return Assessment(metric, v, "intermediário", moderate_msg, suggested_test)
    return Assessment(metric, v, "fraco", weak_msg, suggested_test)


def assess_low_good(
    metric: str, value: Optional[float],
    good: float, moderate: float,
    good_msg: str, moderate_msg: str, bad_msg: str,
    suggested_test: str,
) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível",
                          "Não foi possível calcular esta métrica.", suggested_test)
    if v <= good:
        return Assessment(metric, v, "bom", good_msg, suggested_test)
    if v <= moderate:
        return Assessment(metric, v, "intermediário", moderate_msg, suggested_test)
    return Assessment(metric, v, "alerta", bad_msg, suggested_test)


def assess_range(
    metric: str, value: Optional[float],
    lo_good: float, hi_good: float,
    lo_warn: float, hi_warn: float,
    good_msg: str, moderate_msg: str, bad_msg: str,
    suggested_test: str,
) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível",
                          "Não foi possível calcular esta métrica.", suggested_test)
    if lo_good <= v <= hi_good:
        return Assessment(metric, v, "bom", good_msg, suggested_test)
    if lo_warn <= v <= hi_warn:
        return Assessment(metric, v, "intermediário", moderate_msg, suggested_test)
    return Assessment(metric, v, "alerta", bad_msg, suggested_test)


def build_assessments(metrics: Dict[str, Any], has_gt: bool) -> List[Assessment]:
    assessments: List[Assessment] = []

    # --- Polarização ---
    assessments.append(assess_high_good(
        "polarization_fraction_decided",
        metrics.get("polarization_fraction_decided"),
        strong=0.70, moderate=0.45,
        strong_msg="A maioria dos patches tem score longe de 0.5 — o NCut está particionando com clareza.",
        moderate_msg="A partição existe mas é parcialmente indecisa. Pode haver regiões de borda ou fundo variado.",
        weak_msg="Muitos patches ficaram próximos de 0.5 — o NCut não encontrou estrutura clara ou a orientação FG/BG falhou.",
        suggested_test="Teste temperaturas menores (--temperature 0.02) e mais pares (--top_k 15+). Se persistir, investigue a Etapa 1."
    ))

    assessments.append(assess_high_good(
        "polarization_bimodality_score",
        metrics.get("polarization_bimodality_score"),
        strong=0.6, moderate=0.4,
        strong_msg="Distribuição de fg_scores é bimodal — partição FG/BG bem definida.",
        moderate_msg="Bimodalidade fraca; os scores se concentram parcialmente em torno de 0.5.",
        weak_msg="Distribuição unimodal ou plana. O NCut pode estar falhando em separar FG de BG.",
        suggested_test="Veja viz_score_polarization.png. Se o histograma for plano, a temperatura pode estar alta demais."
    ))

    # --- Fração de FG ---
    assessments.append(assess_range(
        "fg_fraction_mean",
        metrics.get("fg_fraction_mean"),
        lo_good=0.15, hi_good=0.65,
        lo_warn=0.05, hi_warn=0.85,
        good_msg="Fração de FG dentro de uma faixa plausível para objetos naturais.",
        moderate_msg="Fração de FG levemente fora do esperado. Pode ser objeto muito grande/pequeno ou orientação parcialmente errada.",
        bad_msg="Fração de FG extrema. Sugere que a heurística de orientação está sistematicamente elegendo o lado errado.",
        suggested_test="Veja viz_fg_fraction_distribution.png. Tente --top_k maior para agregar mais votos e reduzir outliers de orientação."
    ))

    assessments.append(assess_low_good(
        "fg_fraction_extreme_low_count",
        metrics.get("fg_fraction_extreme_low_count"),
        good=0, moderate=2,
        good_msg="Nenhuma imagem com fração FG extremamente baixa.",
        moderate_msg="Algumas imagens com FG < 10% — possível falha de orientação em casos difíceis.",
        bad_msg="Várias imagens com FG < 10%. A heurística de orientação por saliência CLS pode estar falhando nessas imagens.",
        suggested_test="Inspecione as imagens com FG fraction baixa em viz_fg_fraction_distribution.png."
    ))

    assessments.append(assess_low_good(
        "fg_fraction_extreme_high_count",
        metrics.get("fg_fraction_extreme_high_count"),
        good=0, moderate=2,
        good_msg="Nenhuma imagem com fração FG extremamente alta.",
        moderate_msg="Algumas imagens com FG > 85% — o objeto pode ser muito dominante ou a orientação está invertida.",
        bad_msg="Várias imagens com FG > 85%. O NCut pode estar elegendo o fundo como FG nessas imagens.",
        suggested_test="Compare com cls_saliency nessas imagens — se o objeto for realmente dominante, pode ser esperado."
    ))

    # --- Concordância entre pares ---
    assessments.append(assess_high_good(
        "pair_agreement_mean",
        metrics.get("pair_agreement_mean"),
        strong=0.50, moderate=0.25,
        strong_msg="Os pares concordam consistentemente sobre quais patches são FG — resultado robusto.",
        moderate_msg="Concordância moderada. O resultado final agrega bem, mas cada par individualmente é menos confiável.",
        weak_msg="Baixa concordância entre pares. O NCut está encontrando estruturas diferentes em cada par — sinal fraco ou ruído.",
        suggested_test="Veja viz_pair_agreement.png. Tente aumentar top_k para suavizar via votação."
    ))

    # --- Estabilidade ---
    assessments.append(assess_low_good(
        "stability_loo_mae_mean",
        metrics.get("stability_loo_mae_mean"),
        good=0.05, moderate=0.12,
        good_msg="Resultado estável: remover um par muda pouco os fg_scores.",
        moderate_msg="Estabilidade moderada. Alguns pares têm influência desproporcional.",
        bad_msg="Resultado instável: remover um par muda significativamente os fg_scores. Poucos pares dominam o resultado.",
        suggested_test="Aumente --top_k para distribuir a influência entre mais pares."
    ))

    # --- Correlação com saliência CLS ---
    assessments.append(assess_high_good(
        "cls_correlation_mean",
        metrics.get("cls_correlation_mean"),
        strong=0.35, moderate=0.10,
        strong_msg="fg_scores correlacionam bem com a saliência CLS — as duas etapas concordam.",
        moderate_msg="Correlação fraca com saliência CLS. A Etapa 2 está encontrando estrutura diferente da Etapa 1.",
        weak_msg="Correlação muito baixa ou negativa. Pode indicar orientação FG/BG sistematicamente errada.",
        suggested_test="Veja viz_score_correlation.png da Etapa 2. Se correlação for negativa em muitas imagens, a orientação está invertida."
    ))

    assessments.append(assess_low_good(
        "cls_correlation_n_negative_images",
        metrics.get("cls_correlation_n_negative_images"),
        good=0, moderate=2,
        good_msg="Nenhuma imagem com correlação negativa entre fg_scores e saliência CLS.",
        moderate_msg="Algumas imagens com correlação negativa — orientação possivelmente invertida nesses casos.",
        bad_msg="Várias imagens com correlação negativa. A heurística de orientação está falhando sistematicamente.",
        suggested_test="Inspecione as imagens problemáticas: compare a saliência CLS com o fg_score_2d visualmente."
    ))

    # --- Com GT ---
    if has_gt:
        assessments.append(assess_high_good(
            "gt_best_iou",
            metrics.get("gt_best_iou"),
            strong=0.55, moderate=0.35,
            strong_msg="O threshold ótimo de fg_scores produz boa segmentação em nível de patch — upper bound saudável para a Etapa 3.",
            moderate_msg="IoU ótimo moderado. A Etapa 3 terá margem limitada para melhorar.",
            weak_msg="IoU ótimo baixo mesmo com threshold ajustado. A Etapa 2 é gargalo: os fg_scores não contêm sinal suficiente.",
            suggested_test="Compare gt_best_iou com oracle_imagewise_best_subset_iou da Etapa 1. Se E1 for alto e E2 for baixo, o gargalo está no NCut."
        ))

        assessments.append(assess_high_good(
            "gt_auc_roc",
            metrics.get("gt_auc_roc"),
            strong=0.80, moderate=0.65,
            strong_msg="fg_scores ranqueiam bem patches FG acima de BG — boa qualidade de sinal para a Etapa 3.",
            moderate_msg="AUC moderado. O sinal existe mas é ruidoso.",
            weak_msg="AUC próximo de 0.5 — os fg_scores não discriminam FG de BG. A Etapa 2 não está adicionando informação útil.",
            suggested_test="Se AUC for baixo mesmo com boas features da Etapa 1, revise temperature e top_k."
        ))

        assessments.append(assess_high_good(
            "gt_best_iou_oracle_polarity",
            metrics.get("gt_best_iou_oracle_polarity"),
            strong=0.55, moderate=0.35,
            strong_msg="Ao permitir apenas inverter globalmente a polaridade dos fg_scores, surge boa sobreposição com o GT.",
            moderate_msg="A inversão global melhora parcialmente o resultado, mas ainda há limitação estrutural no sinal.",
            weak_msg="Nem a melhor polaridade global recupera boa IoU; o problema provavelmente vai além de uma simples inversão FG/BG.",
            suggested_test="Compare raw vs inverted em viz_polarity_diagnostics.png e, em seguida, o oracle de orientação antes da agregação."
        ))

        assessments.append(assess_high_good(
            "gt_auc_roc_oracle_polarity",
            metrics.get("gt_auc_roc_oracle_polarity"),
            strong=0.80, moderate=0.65,
            strong_msg="Os fg_scores contêm forte sinal discriminativo em pelo menos uma das duas polaridades.",
            moderate_msg="Existe sinal discriminativo moderado após escolher a melhor polaridade.",
            weak_msg="Mesmo escolhendo a melhor polaridade, o ranking FG/BG continua fraco.",
            suggested_test="Se raw AUC < 0.5 e inverted AUC > 0.8, trate primeiro como falha de orientação, não como ausência de sinal."
        ))

        assessments.append(assess_high_good(
            "orientation_accuracy",
            metrics.get("orientation_accuracy"),
            strong=0.75, moderate=0.55,
            strong_msg="A heurística de orientação FG/BG (saliência CLS) está elegendo o lado correto na maioria dos pares.",
            moderate_msg="Orientação correta em pouco mais da metade dos pares — funciona, mas com erros frequentes.",
            weak_msg="A heurística de orientação está falhando: o NCut escolhe o lado errado com frequência. Considere estratégias de orientação alternativas.",
            suggested_test="Veja viz_orientation_accuracy.png. Se orientation_accuracy < 0.5, o sinal de saliência CLS pode estar invertido ou ausente."
        ))

        if metrics.get("orientation_best_heuristic") is not None:
            best_h      = metrics.get("orientation_best_heuristic")
            best_acc    = metrics.get("orientation_best_accuracy")
            improvement = metrics.get("orientation_improvement_over_baseline")
            imp_str     = f"{improvement:+.1%}" if isinstance(improvement, (int, float)) else "n/a"

            if best_h == "cls_saliency":
                assessments.append(Assessment(
                    "orientation_best_heuristic", best_acc, "bom",
                    "A heurística de produção (saliência CLS) já é a melhor entre as testadas para esta coleção.",
                    "Nenhuma ação necessária na heurística de orientação desta classe."
                ))
            elif improvement is not None and improvement >= 0.10:
                assessments.append(Assessment(
                    "orientation_best_heuristic", best_acc, "alerta",
                    f"A heurística '{best_h}' supera a saliência CLS em {imp_str} de orientation_accuracy "
                    f"nesta coleção — indício de que o objeto de interesse não é o mais saliente da cena "
                    f"(comum em multidões, objetos pequenos ou cenas com clutter dominante).",
                    f"Rode o mesmo diagnóstico em outras classes com o mesmo perfil (objeto pequeno/disperso). "
                    f"Se o padrão se repetir, considere trocar orient_partition() no step2_correspondence.py "
                    f"para usar '{best_h}' (ou a combinação) em vez de só saliência CLS."
                ))
            else:
                assessments.append(Assessment(
                    "orientation_best_heuristic", best_acc, "intermediário",
                    f"A heurística '{best_h}' é ligeiramente melhor que a saliência CLS "
                    f"(ganho de {imp_str}), mas a diferença é pequena para esta coleção.",
                    "Ganho marginal isolado; valide em mais classes antes de trocar a heurística em produção."
                ))

    return assessments


def infer_bottleneck(metrics: Dict[str, Any], has_gt: bool) -> str:
    polar   = safe_float(metrics.get("polarization_fraction_decided"))
    agree   = safe_float(metrics.get("pair_agreement_mean"))
    stab    = safe_float(metrics.get("stability_loo_mae_mean"))
    corr    = safe_float(metrics.get("cls_correlation_mean"))
    n_neg   = metrics.get("cls_correlation_n_negative_images", 0)
    best_iou = safe_float(metrics.get("gt_best_iou"))
    orient   = safe_float(metrics.get("orientation_accuracy"))
    oracle_pol_iou = safe_float(metrics.get("gt_best_iou_oracle_polarity"))
    oracle_pol_auc = safe_float(metrics.get("gt_auc_roc_oracle_polarity"))
    oracle_pair_agg = safe_float(metrics.get("oracle_orientation_pairwise_aggregated_best_iou"))

    if has_gt:
        if best_iou is not None and best_iou < 0.35 and oracle_pol_auc is not None and oracle_pol_auc >= 0.80:
            if oracle_pair_agg is not None and oracle_pair_agg >= 0.55:
                return (
                    "Diagnóstico: forte evidência de falha de orientação FG/BG. "
                    "A polaridade raw é ruim, mas a polaridade oracle preserva forte sinal discriminativo, "
                    "e corrigir a orientação por par antes da agregação produz bom upper bound. "
                    "Priorize a heurística orient_partition() antes de alterar features ou o NCut."
                )
            return (
                "Diagnóstico: os fg_scores parecem conter sinal forte com polaridade invertida, "
                "mas corrigir somente a orientação pode não ser suficiente. Compare o oracle de "
                "orientação por par com o oracle de polaridade global para separar falha de orientação "
                "de falha estrutural do corte/agregação."
            )
        if best_iou is not None and best_iou >= 0.55:
            return (
                "Diagnóstico: Etapa 2 não é gargalo. Os fg_scores têm upper bound "
                "suficiente para guiar a Etapa 3. Se a segmentação final for ruim, "
                "investigue o thresholding/SAM na Etapa 3."
            )
        if best_iou is not None and best_iou < 0.35:
            if orient is not None and orient < 0.55:
                extra = ""
                best_h = metrics.get("orientation_best_heuristic")
                improvement = metrics.get("orientation_improvement_over_baseline")
                best_acc = metrics.get("orientation_best_accuracy")
                if (best_h and best_h != "cls_saliency"
                        and improvement is not None and improvement > 0
                        and best_acc is not None):
                    extra = (
                        f" A heurística alternativa '{best_h}' testada neste diagnóstico obteve "
                        f"orientation_accuracy {best_acc:.2f} ({improvement:+.1%} sobre a saliência "
                        f"CLS) — candidata a substituir a orientação atual nesta classe."
                    )
                return (
                    "Diagnóstico: gargalo provável na orientação FG/BG. "
                    "O NCut está particionando, mas escolhendo o lado errado com frequência. "
                    "Considere uma heurística de orientação alternativa (ex: menor componente, "
                    "restrição espacial central, ou saliência de múltiplas camadas)." + extra
                )
            return (
                "Diagnóstico: IoU máximo baixo mesmo com threshold ajustado. "
                "O NCut pode não estar encontrando o objeto de interesse. "
                "Teste temperature menor, mais pares, ou features de outra camada do DINOv2."
            )
        if best_iou is not None and 0.35 <= best_iou < 0.55:
            return (
                "Diagnóstico: Etapa 2 com resultado intermediário. "
                "O sinal existe mas não é forte. "
                "A Etapa 3 precisará de prompts mais robustos ou refinamento iterativo."
            )

    # Sem GT
    if polar is not None and polar < 0.45:
        return (
            "Diagnóstico sem GT: polarização baixa — o NCut está produzindo partições indecisas. "
            "Teste temperature=0.02 e top_k maior. Se a Etapa 1 tiver cross_nn baixo, o problema começa lá."
        )
    if n_neg is not None and n_neg > 2:
        return (
            "Diagnóstico sem GT: múltiplas imagens com correlação negativa entre fg_scores e saliência CLS. "
            "A heurística de orientação pode estar falhando. Inspecione as imagens problemáticas."
        )
    if agree is not None and agree < 0.25:
        return (
            "Diagnóstico sem GT: baixa concordância entre pares. "
            "Cada par encontra uma estrutura diferente — o objeto comum pode não dominar "
            "as correspondências. Aumente top_k e verifique a qualidade das features da Etapa 1."
        )
    if stab is not None and stab > 0.12:
        return (
            "Diagnóstico sem GT: resultado instável. Remover um par altera "
            "significativamente os fg_scores. Aumente top_k."
        )
    return (
        "Diagnóstico sem GT: sinais não supervisionados dentro do esperado. "
        "Para confirmar se a Etapa 2 é gargalo, rode com gt_index.json."
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_score_polarization(
    fg_scores: np.ndarray,
    output_dir: Path,
    max_images: int = 8,
) -> None:
    if plt is None:
        return
    N, P = fg_scores.shape
    N_vis = min(N, max_images)

    fig, axes = plt.subplots(2, N_vis, figsize=(2.8 * N_vis, 6))
    if N_vis == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("Polarização dos fg_scores — distribuição por imagem", fontsize=11)

    for i in range(N_vis):
        scores = fg_scores[i]
        axes[0, i].hist(scores, bins=30, range=(0, 1), color="steelblue", edgecolor="none")
        axes[0, i].axvline(0.5, color="red", lw=1.2, ls="--", alpha=0.7)
        axes[0, i].set_title(f"img[{i}]  μ={scores.mean():.2f}", fontsize=8)
        axes[0, i].set_xlabel("fg_score", fontsize=7)
        axes[0, i].set_ylabel("patches", fontsize=7)

        decided = np.abs(scores - 0.5) > 0.2
        axes[1, i].pie(
            [decided.sum(), (~decided).sum()],
            labels=["decidido", "incerto"],
            colors=["#4caf50", "#e0e0e0"],
            autopct="%1.0f%%",
            textprops={"fontsize": 7},
        )

    fig.tight_layout()
    fig.savefig(output_dir / "viz_score_polarization.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_pair_agreement(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    patch_grid: Tuple[int, int],
    image_paths: List[str],
    output_dir: Path,
    max_images: int = 4,
) -> None:
    if plt is None:
        return
    K, N, P = pair_votes.shape
    H_p, W_p = patch_grid
    N_vis = min(N, max_images)

    fig, axes = plt.subplots(N_vis, min(K, 5) + 1,
                             figsize=(2.5 * (min(K, 5) + 1), 2.5 * N_vis))
    if N_vis == 1:
        axes = axes[np.newaxis, :]
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle("Votos individuais por par vs. fg_score agregado", fontsize=10)

    for i in range(N_vis):
        col = 0
        for k in range(min(K, 5)):
            pi, pj = int(top_pairs[k, 0]), int(top_pairs[k, 1])
            if pi == i or pj == i:
                vote_map = pair_votes[k, i].reshape(H_p, W_p)
                axes[i, col].imshow(vote_map, cmap="RdYlGn",
                                    interpolation="nearest", vmin=0, vmax=1)
                axes[i, col].set_title(f"par ({pi},{pj})", fontsize=7)
                axes[i, col].axis("off")
                col += 1

        # Última coluna: fg_score agregado (recalculado dos votos disponíveis)
        agg = pair_votes[:, i, :].mean(axis=0).reshape(H_p, W_p)
        axes[i, -1].imshow(agg, cmap="RdYlGn",
                           interpolation="nearest", vmin=0, vmax=1)
        axes[i, -1].set_title(f"img[{i}] agregado", fontsize=7)
        axes[i, -1].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "viz_pair_agreement.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_stability(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    fg_scores: np.ndarray,
    output_dir: Path,
) -> None:
    if plt is None:
        return
    K, N, P = pair_votes.shape
    if K < 3:
        return

    pairs_list = [(int(top_pairs[k, 0]), int(top_pairs[k, 1]), float(top_pairs[k, 2]))
                  for k in range(K)]
    maes = []
    for k_out in range(K):
        remaining = [(k, p) for k, p in enumerate(pairs_list) if k != k_out]
        if not remaining:
            continue
        weight_sum  = np.zeros((N, P), dtype=np.float32)
        fg_weighted = np.zeros((N, P), dtype=np.float32)
        for k, (pi, pj, sim) in remaining:
            w = max(sim, 0.0)
            fg_weighted[pi] += pair_votes[k, pi] * w
            fg_weighted[pj] += pair_votes[k, pj] * w
            weight_sum[pi]  += w
            weight_sum[pj]  += w
        fg_loo = np.divide(
            fg_weighted, weight_sum,
            out=np.full_like(fg_weighted, 0.5),
            where=weight_sum > 0,
        )
        maes.append(float(np.mean(np.abs(fg_loo - fg_scores))))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(K), maes, color="coral")
    ax.set_xlabel("Par removido (índice)")
    ax.set_ylabel("MAE dos fg_scores")
    ax.set_title("Estabilidade leave-one-pair-out\n(barra alta = par com influência desproporcional)")
    ax.axhline(float(np.mean(maes)), color="gray", ls="--", lw=1.2, label=f"média={np.mean(maes):.4f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_leave_one_pair_stability.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fg_fraction_distribution(fg_scores: np.ndarray, output_dir: Path) -> None:
    if plt is None:
        return
    fractions = fg_scores.mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(fractions)), fractions * 100, color="steelblue")
    ax.axhline(15, color="red",   ls="--", lw=1.2, alpha=0.6, label="limite inferior (15%)")
    ax.axhline(65, color="orange", ls="--", lw=1.2, alpha=0.6, label="limite superior (65%)")
    ax.set_xlabel("Imagem")
    ax.set_ylabel("Fração FG (%)")
    ax.set_title("Fração de patches classificados como FG por imagem")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_fg_fraction_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_gt_iou_curve(metrics: Dict[str, Any], output_dir: Path) -> None:
    if plt is None:
        return
    thrs = metrics.get("gt_iou_curve_thresholds")
    vals = metrics.get("gt_iou_curve_values")
    if thrs is None or vals is None:
        return
    best_t = metrics.get("gt_best_threshold", 0.5)
    best_v = metrics.get("gt_best_iou", 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thrs, vals, "o-", color="steelblue", lw=2, ms=5)
    ax.axvline(best_t, color="red", ls="--", lw=1.5, label=f"threshold ótimo={best_t:.2f}")
    ax.axhline(best_v, color="green", ls="--", lw=1.2, label=f"IoU ótimo={best_v:.3f}")
    ax.set_xlabel("Threshold de fg_score")
    ax.set_ylabel("IoU médio")
    ax.set_title("IoU vs. threshold de fg_score (upper bound da Etapa 2)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_gt_iou_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_orientation_accuracy(
    pair_votes: np.ndarray,
    top_pairs: np.ndarray,
    patch_gt: np.ndarray,
    patch_grid: Tuple[int, int],
    output_dir: Path,
    max_pairs: int = 6,
) -> None:
    if plt is None:
        return
    K, N, P = pair_votes.shape
    H_p, W_p = patch_grid
    K_vis = min(K, max_pairs)

    fig, axes = plt.subplots(K_vis, 4, figsize=(11, 3 * K_vis))
    if K_vis == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Orientação NCut vs. GT — par por par", fontsize=11)

    col_labels = ["Voto NCut (img i)", "GT (img i)", "Voto NCut (img j)", "GT (img j)"]
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=8)

    for k in range(K_vis):
        pi, pj = int(top_pairs[k, 0]), int(top_pairs[k, 1])
        sim = float(top_pairs[k, 2])
        for col, (img_idx, label) in enumerate([(pi, f"img[{pi}]"), (pi, "GT"), (pj, f"img[{pj}]"), (pj, "GT")]):
            ax = axes[k, col]
            if col % 2 == 0:
                ax.imshow(pair_votes[k, img_idx].reshape(H_p, W_p),
                          cmap="RdYlGn", interpolation="nearest", vmin=0, vmax=1)
            else:
                ax.imshow(patch_gt[img_idx].reshape(H_p, W_p),
                          cmap="gray", interpolation="nearest", vmin=0, vmax=1)
            ax.axis("off")
        axes[k, 0].set_ylabel(f"par {k}\nsim={sim:.3f}", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "viz_orientation_accuracy.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_orientation_heuristics_comparison(metrics: Dict[str, Any], output_dir: Path) -> None:
    """
    Barra comparativa: orientation_accuracy da heurística de produção
    (saliência CLS) vs. as três alternativas testadas (size_based,
    spatial_based, combined). A barra vencedora fica destacada em verde.
    """
    if plt is None:
        return
    names_values = [
        ("cls_saliency\n(produção)", metrics.get("orientation_baseline_accuracy")),
        ("size_based",               metrics.get("orientation_accuracy_size_based")),
        ("spatial_based",            metrics.get("orientation_accuracy_spatial_based")),
        ("combined",                 metrics.get("orientation_accuracy_combined")),
    ]
    names_values = [(n, v) for n, v in names_values if v is not None]
    if not names_values:
        return

    names  = [n for n, _ in names_values]
    values = [v for _, v in names_values]
    best_idx = int(np.argmax(values))
    colors = ["#2ecc71" if i == best_idx else "#95a5a6" for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, values, color=colors, edgecolor="white")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="acaso (0.50)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("orientation_accuracy")
    ax.set_title(
        "Comparação de heurísticas de orientação FG/BG\n(verde = melhor nesta coleção)",
        fontsize=10
    )
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_orientation_heuristics_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)



def plot_pair_coverage(metrics: Dict[str, Any], output_dir: Path) -> None:
    if plt is None:
        return
    counts = metrics.get("pair_coverage_per_image")
    if not counts:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(counts)), counts, color="steelblue")
    ax.axhline(float(np.mean(counts)), color="gray", ls="--", lw=1.2,
               label=f"média={np.mean(counts):.2f}")
    ax.set_xlabel("Imagem")
    ax.set_ylabel("Número de pares selecionados")
    ax.set_title("Cobertura dos top-K pares por imagem")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_pair_coverage.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_polarity_diagnostics(metrics: Dict[str, Any], output_dir: Path) -> None:
    if plt is None:
        return
    labels = ["raw", "inverted", "oracle"]
    ious = [
        metrics.get("gt_best_iou_raw"),
        metrics.get("gt_best_iou_inverted"),
        metrics.get("gt_best_iou_oracle_polarity"),
    ]
    aucs = [
        metrics.get("gt_auc_roc_raw"),
        metrics.get("gt_auc_roc_inverted"),
        metrics.get("gt_auc_roc_oracle_polarity"),
    ]
    if all(v is None for v in ious + aucs):
        return

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, [0 if v is None else v for v in ious], width, label="best IoU")
    ax.bar(x + width/2, [0 if v is None else v for v in aucs], width, label="AUC ROC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Diagnóstico de polaridade: raw vs. invertida")
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_polarity_diagnostics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    corr_path: Path,
    metrics: Dict[str, Any],
    assessments: List[Assessment],
    diagnosis: str,
    gt_info: Dict[str, Any],
    padding_info: Dict[str, Any],
) -> Path:
    report_path = output_dir / "step2_quality_report.md"
    lines: List[str] = []
    lines.append("# Relatório de qualidade — Etapa 2 (Cross-Image Correspondence)")
    lines.append("")
    lines.append(f"Arquivo analisado: `{corr_path}`")
    lines.append("")
    lines.append("## Resumo")
    lines.append("")
    lines.append(f"- Imagens: **{metrics.get('n_images')}**")
    lines.append(f"- Patches por imagem: **{metrics.get('n_patches')}**")
    lines.append(f"- Pares processados: **{metrics.get('n_pairs')}**")
    lines.append(f"- Grade: **{metrics.get('patch_grid')}**")
    lines.append(f"- GT encontrado: **{gt_info.get('matched', 0)} / {metrics.get('n_images')}**")
    if padding_info.get("available"):
        lines.append(
            f"- Padding (--resize keep_ar): **{padding_info['fraction_padding']:.1%} dos patches**, "
            f"mascarado nas métricas: **{'sim' if padding_info.get('masked') else 'não (--no_padding_mask)'}**"
        )
    else:
        lines.append("- Padding: sem `resize_info` disponível (`--features` não informado, ou "
                     "Etapa 1 rodou em modo `square`) — nada a mascarar ou não foi possível checar")
    lines.append("")
    lines.append("## Diagnóstico")
    lines.append("")
    lines.append(diagnosis)
    lines.append("")
    lines.append("## Métricas interpretadas")
    lines.append("")
    lines.append("| Métrica | Valor | Veredito | Indicação | Teste sugerido |")
    lines.append("|---|---:|---|---|---|")
    for a in assessments:
        lines.append(
            f"| `{a.metric}` | {compact_float(a.value)} | {a.verdict} | "
            f"{a.indication} | {a.suggested_test} |"
        )
    lines.append("")

    if metrics.get("orientation_best_heuristic") is not None:
        lines.append("## Heurísticas de orientação alternativas")
        lines.append("")
        lines.append(
            "Testa, para os mesmos dois lados do corte NCut já calculados pela Etapa 2, "
            "critérios de orientação FG/BG diferentes da saliência CLS usada em produção. "
            "Ver `compute_orientation_alternatives()` em `analyze_step2_correspondence.py`."
        )
        lines.append("")
        lines.append("| Heurística | orientation_accuracy |")
        lines.append("|---|---:|")
        for label, key in [
            ("cls_saliency (produção)", "orientation_baseline_accuracy"),
            ("size_based",              "orientation_accuracy_size_based"),
            ("spatial_based",           "orientation_accuracy_spatial_based"),
            ("combined",                "orientation_accuracy_combined"),
        ]:
            lines.append(f"| {label} | {compact_float(metrics.get(key))} |")
        lines.append("")
        best_h = metrics.get("orientation_best_heuristic")
        improvement = metrics.get("orientation_improvement_over_baseline")
        imp_str = f"{improvement:+.1%}" if isinstance(improvement, (int, float)) else "n/a"
        lines.append(f"**Melhor heurística nesta coleção:** `{best_h}` (ganho de {imp_str} sobre a produção).")
        lines.append("")

    if metrics.get("gt_best_iou_oracle_polarity") is not None:
        lines.append("## Diagnóstico de polaridade")
        lines.append("")
        lines.append("| Métrica | Raw | Invertida | Oracle |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| Melhor IoU | {compact_float(metrics.get('gt_best_iou_raw'))} | "
            f"{compact_float(metrics.get('gt_best_iou_inverted'))} | "
            f"{compact_float(metrics.get('gt_best_iou_oracle_polarity'))} |"
        )
        lines.append(
            f"| AUC ROC | {compact_float(metrics.get('gt_auc_roc_raw'))} | "
            f"{compact_float(metrics.get('gt_auc_roc_inverted'))} | "
            f"{compact_float(metrics.get('gt_auc_roc_oracle_polarity'))} |"
        )
        lines.append("")
        lines.append(
            f"- IoU médio por ocorrência (par,imagem): selecionado="
            f"**{compact_float(metrics.get('pair_orientation_selected_iou_mean'))}**, "
            f"invertido=**{compact_float(metrics.get('pair_orientation_inverted_iou_mean'))}**, "
            f"oracle=**{compact_float(metrics.get('pair_orientation_oracle_iou_mean'))}**."
        )
        lines.append(
            f"- Perda média atribuível à orientação: "
            f"**{compact_float(metrics.get('pair_orientation_loss_mean'))}**."
        )
        lines.append(
            f"- Oracle de orientação antes da agregação: imagewise IoU="
            f"**{compact_float(metrics.get('oracle_orientation_imagewise_aggregated_best_iou'))}**, "
            f"pairwise IoU=**{compact_float(metrics.get('oracle_orientation_pairwise_aggregated_best_iou'))}**."
        )
        lines.append("")

    lines.append("## Cobertura dos pares")
    lines.append("")
    lines.append(
        f"- Pares por imagem: `{metrics.get('pair_coverage_per_image')}`"
    )
    lines.append(
        f"- Mínimo/máximo/média: **{metrics.get('pair_coverage_min')} / "
        f"{metrics.get('pair_coverage_max')} / {compact_float(metrics.get('pair_coverage_mean'))}**"
    )
    lines.append(
        f"- Imagens sem nenhum par: **{metrics.get('pair_coverage_n_images_without_pairs', 0)}**"
    )
    lines.append("")

    lines.append("## Arquivos gerados")
    lines.append("")
    for name in [
        "step2_quality.json",
        "viz_score_polarization.png",
        "viz_pair_agreement.png",
        "viz_leave_one_pair_stability.png",
        "viz_fg_fraction_distribution.png",
        "viz_gt_iou_curve.png",
        "viz_orientation_accuracy.png",
        "viz_orientation_heuristics_comparison.png",
        "viz_polarity_diagnostics.png",
        "viz_pair_coverage.png",
    ]:
        p = output_dir / name
        if p.exists():
            lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Observações")
    lines.append("")
    lines.append("- Os limiares são heurísticos; use em ablação comparativa, não como valores absolutos.")
    lines.append("- `gt_best_iou` é o upper bound da Etapa 2 antes do SAM/refinamento.")
    lines.append("- Se `orientation_accuracy` < 0.55, considere substituir a heurística de orientação por saliência CLS.")
    lines.append("- Se `pair_agreement_mean` for alto mas `gt_best_iou` for baixo, o gargalo está antes da Etapa 2.")
    lines.append("- As heurísticas alternativas são testadas post-hoc sobre os mesmos votos já salvos; "
                 "elas indicam potencial de melhoria, mas não são aplicadas automaticamente ao correspondence.npz.")
    if padding_info.get("available") and padding_info.get("masked"):
        lines.append(
            "- Patches de padding (modo `--resize keep_ar` na Etapa 1) foram excluídos de todas as "
            "métricas: polarização, concordância entre pares, estabilidade, correlação com CLS, AUC "
            "com GT, e — mais importante — `orientation_accuracy` e as heurísticas alternativas. Sem "
            "essa exclusão, o lado invertido (~vote) usado nessas comparações de orientação absorve "
            "todo o padding como se fosse foreground, inflando falsos positivos da hipótese invertida "
            "e enviesando o resultado a favor de nunca inverter — mesmo quando a orientação correta no "
            "conteúdo real seria a invertida. Use `--no_padding_mask` para reproduzir o comportamento antigo."
        )
    elif padding_info.get("available") and not padding_info.get("masked"):
        lines.append(
            "- `--no_padding_mask` estava ativo: as métricas acima incluem patches de padding sem "
            "exclusão. Em particular, `orientation_accuracy` e as heurísticas alternativas podem estar "
            "enviesadas a favor da orientação de produção — reveja com o mascaramento ativado antes de "
            "confiar nesses números."
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_console_summary(
    metrics: Dict[str, Any],
    assessments: List[Assessment],
    diagnosis: str,
    padding_info: Dict[str, Any],
) -> None:
    print("\n" + "=" * 72)
    print("  AUDITORIA DA ETAPA 2 — CROSS-IMAGE CORRESPONDENCE")
    print("=" * 72)
    print(f"  Imagens       : {metrics.get('n_images')}")
    print(f"  Grade patches : {metrics.get('patch_grid')} = {metrics.get('n_patches')} patches/img")
    print(f"  Pares         : {metrics.get('n_pairs')}")
    if padding_info.get("available"):
        estado = "mascarado" if padding_info.get("masked") else "NÃO mascarado (--no_padding_mask)"
        print(f"  Padding       : {padding_info['fraction_padding']:.1%} dos patches — {estado}")
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analisa a qualidade da correspondência gerada pela Etapa 2 (correspondence.npz).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--correspondence", required=True, type=Path,
                   help="Caminho para correspondence.npz gerado pela Etapa 2.")
    p.add_argument("--features", type=Path, default=None,
                   help="Caminho para features.npz da Etapa 1. Opcional, mas necessário "
                        "para alinhar corretamente o GT quando a Etapa 1 rodou com "
                        "--resize keep_ar.")
    p.add_argument("--gt_index", type=Path, default=None,
                   help="Caminho para gt_index.json. Opcional.")
    p.add_argument("--output", type=Path, default=None,
                   help="Diretório de saída. Default: <dir_correspondence>/step2_quality.")
    p.add_argument("--fg_threshold", type=float, default=0.5,
                   help="Limiar para binarizar máscaras GT na grade de patches.")
    p.add_argument("--max_viz_images", type=int, default=8,
                   help="Máximo de imagens nas visualizações.")
    p.add_argument("--no_plots", action="store_true",
                   help="Não gerar visualizações.")
    p.add_argument("--no_orientation_alternatives", action="store_true",
                   help="Não testa heurísticas alternativas de orientação FG/BG "
                        "(size_based, spatial_based, combined). Por padrão, são "
                        "testadas sempre que há GT disponível.")
    p.add_argument("--no_padding_mask", action="store_true",
                   help="Desativa o mascaramento de patches de padding (modo --resize "
                        "keep_ar da Etapa 1) em todas as métricas. Por padrão, o "
                        "mascaramento é ativado sempre que --features apontar para um "
                        "features.npz com resize_info (mesmo comportamento de "
                        "step2_correspondence.py, step1's analyzer, etc.). Requer "
                        "--features para funcionar; sem --features não há resize_info "
                        "disponível e o aviso já emitido sobre GT desalinhado se aplica "
                        "aqui também às métricas não supervisionadas.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corr_path  = args.correspondence
    output_dir = args.output or (corr_path.parent / "step2_quality")
    ensure_dir(output_dir)

    print(f"[1/6] Carregando correspondências: {corr_path}")
    data = load_correspondence(corr_path)
    fg_scores    = data["fg_scores"]
    pair_votes   = data["pair_votes"]
    top_pairs    = data["top_pairs"]
    cls_saliency = data["cls_saliency"]
    patch_grid   = data["patch_grid"]
    image_paths  = data["image_paths"]

    N, P = fg_scores.shape
    K    = len(top_pairs)
    metrics: Dict[str, Any] = {
        "correspondence_path": str(corr_path),
        "n_images":    int(N),
        "n_patches":   int(P),
        "n_pairs":     int(K),
        "patch_grid":  list(patch_grid),
    }
    metrics.update(compute_pair_coverage_metrics(top_pairs, N))

    # Carrega resize_info ANTES de qualquer métrica (não só para o GT): as
    # métricas não supervisionadas (polarização, concordância, estabilidade,
    # correlação com CLS) também são infladas por patches de padding se não
    # forem excluídos — ver docstrings das funções correspondentes.
    resize_info = load_resize_info(args.features)
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
                  f"métricas (incluindo GT e orientação).")
        else:
            print(f"      resize_info encontrado, mas --no_padding_mask ativo — "
                  f"{frac_padding:.1%} dos patches de padding NÃO serão excluídos.")
    elif args.features is None:
        print("      [Aviso] --features não informado — se a Etapa 1 rodou com "
              "--resize keep_ar, o GT E as métricas não supervisionadas ficarão "
              "desalinhados/inflados por padding. Passe --features para corrigir.")
    metrics["padding_info"] = padding_info

    print("[2/6] Calculando polarização e fração FG...")
    metrics.update(compute_polarization_metrics(fg_scores, content_mask=content_mask))

    print("[3/6] Calculando concordância entre pares e estabilidade...")
    metrics.update(compute_pair_agreement_metrics(pair_votes, top_pairs, content_mask=content_mask))
    metrics.update(compute_stability_metrics(pair_votes, top_pairs, fg_scores, content_mask=content_mask))

    print("[4/6] Calculando correlação com saliência CLS...")
    metrics.update(compute_cls_correlation_metrics(fg_scores, cls_saliency, content_mask=content_mask))

    print("[5/6] Procurando ground truth...")
    gt_index = load_gt_index(args.gt_index)
    patch_gt, gt_info = load_patch_gt_masks(
        image_paths, patch_grid, gt_index, fg_threshold=args.fg_threshold,
        resize_info=resize_info,
    )
    metrics["gt_info"] = gt_info
    has_gt = patch_gt is not None and gt_info.get("matched", 0) > 0

    if has_gt:
        print(f"      GT encontrado para {gt_info['matched']} / {N} imagens.")
        print("[6/6] Calculando métricas com GT: IoU curve, AUC, orientação...")
        metrics.update(compute_gt_iou_metrics(fg_scores, patch_gt, content_mask=content_mask))
        metrics.update(compute_orientation_accuracy(pair_votes, top_pairs, patch_gt, content_mask=content_mask))
        metrics.update(compute_pair_orientation_iou_metrics(
            pair_votes, top_pairs, patch_gt, content_mask=content_mask
        ))
        metrics.update(compute_oracle_orientation_aggregation_metrics(
            pair_votes, top_pairs, patch_gt, content_mask=content_mask
        ))
        if not args.no_orientation_alternatives:
            print("      Testando heurísticas alternativas de orientação "
                  "(size_based, spatial_based, combined)...")
            metrics.update(compute_orientation_alternatives(
                pair_votes, top_pairs, patch_gt, cls_saliency, patch_grid,
                baseline_accuracy=metrics.get("orientation_accuracy"),
                content_mask=content_mask,
            ))
    else:
        print("      GT indisponível. Seguindo apenas com métricas sem GT.")

    assessments = build_assessments(metrics, has_gt=has_gt)
    diagnosis   = infer_bottleneck(metrics, has_gt=has_gt)

    json_path = output_dir / "step2_quality.json"
    json_path.write_text(
        json.dumps({"metrics": metrics,
                    "assessments": [assessment_to_dict(a) for a in assessments],
                    "diagnosis": diagnosis},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.no_plots:
        plot_score_polarization(fg_scores, output_dir, max_images=args.max_viz_images)
        plot_pair_agreement(pair_votes, top_pairs, patch_grid, image_paths,
                            output_dir, max_images=min(4, args.max_viz_images))
        plot_stability(pair_votes, top_pairs, fg_scores, output_dir)
        plot_fg_fraction_distribution(fg_scores, output_dir)
        plot_pair_coverage(metrics, output_dir)
        if has_gt:
            plot_polarity_diagnostics(metrics, output_dir)
            plot_gt_iou_curve(metrics, output_dir)
            plot_orientation_accuracy(pair_votes, top_pairs, patch_gt,
                                      patch_grid, output_dir)
            if metrics.get("orientation_best_heuristic") is not None:
                plot_orientation_heuristics_comparison(metrics, output_dir)

    report_path = write_report(output_dir, corr_path, metrics, assessments, diagnosis, gt_info, padding_info)
    print_console_summary(metrics, assessments, diagnosis, padding_info)
    print(f"\n[Salvo] JSON     : {json_path}")
    print(f"[Salvo] Relatório: {report_path}")
    print(f"[Salvo] Figuras  : {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)