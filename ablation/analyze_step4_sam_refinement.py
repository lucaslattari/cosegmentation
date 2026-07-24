#!/usr/bin/env python3
"""
=============================================================================
analyze_step4_sam_refinement.py — Auditoria da Etapa 4 de um pipeline de cosegmentação
=============================================================================

Lê o refined_masks.npz gerado pela Etapa 4 e calcula métricas para responder:

  1. O SAM teve alta confiança nas máscaras geradas (iou_scores_sam)?
  2. O refinamento SAM melhorou ou degradou a máscara da Etapa 3?
  3. A coesão semântica cross-image foi preservada após o refinamento?
  4. A fração FG está em faixa plausível?
  5. Se houver GT: qual é o IoU real contra o ground truth?
  6. Se houver GT: qual foi o ganho líquido da E4 sobre a E3?
  7. Se houver GT: qual modo de prompting funciona melhor?

Funciona com ou sem ground truth.
Requer: refined_masks.npz (E4) + cosaliency.npz (E3, para comparação).

Uso mínimo:
  python analyze_step4_sam_refinement.py \
      --refined    features/Panda/refined_masks.npz \
      --cosaliency features/Panda/cosaliency.npz

Com ground truth:
  python analyze_step4_sam_refinement.py \
      --refined    features/Panda/refined_masks.npz \
      --cosaliency features/Panda/cosaliency.npz \
      --gt_index   gt_index.json \
      --output     features/Panda/step4_quality

Saídas:
  step4_quality.json
  step4_quality_report.md
  viz_sam_confidence.png
  viz_e3_vs_e4_comparison.png
  viz_fg_fraction.png
  viz_gt_iou_comparison.png      (se GT disponível)
  viz_per_image_gain.png         (se GT disponível)

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
except Exception:
    plt = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

try:
    from scipy.ndimage import label as ndimage_label
except Exception:
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


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_refined(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"refined_masks.npz não encontrado: {path}")
    data = np.load(path, allow_pickle=True)
    required = ["refined_masks", "iou_scores_sam", "fg_coherence",
                "prompt_mode", "image_paths", "image_sizes"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"Campos ausentes em refined_masks.npz: {missing}")
    return {
        "refined_masks" : np.asarray(data["refined_masks"],  dtype=np.float32),
        "iou_scores_sam": np.asarray(data["iou_scores_sam"], dtype=np.float32),
        # Campos novos; fg_coherence continua como fallback para .npz antigos.
        "fg_coherence"  : float(data["fg_coherence_e4"])
                          if "fg_coherence_e4" in data.files
                          else float(data["fg_coherence"]),
        "fg_coherence_e3": float(data["fg_coherence_e3"])
                           if "fg_coherence_e3" in data.files else None,
        "fg_coherence_e4": float(data["fg_coherence_e4"])
                           if "fg_coherence_e4" in data.files
                           else float(data["fg_coherence"]),
        "fg_coherence_delta": float(data["fg_coherence_delta"])
                              if "fg_coherence_delta" in data.files else None,
        "prompt_mode"   : str(data["prompt_mode"]),
        "image_paths"   : [str(x) for x in data["image_paths"].tolist()],
        "image_sizes"   : np.asarray(data["image_sizes"]),
    }


def load_cosaliency(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    result = {}
    if "fg_mask_pixel" in data.files:
        result["fg_mask_pixel"] = np.asarray(data["fg_mask_pixel"], dtype=np.float32)
    if "image_sizes" in data.files:
        result["image_sizes"] = np.asarray(data["image_sizes"])
    return result if result else None


def load_gt_index(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        print("[Aviso] gt_index.json não encontrado. Rodando sem GT.")
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
        if idx >= 0: variants.add(raw[idx:])
    return {v for v in variants if v}


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
            arr = np.asarray(
                Image.open(gt_path).convert("L").resize((W_i, H_i), Image.BILINEAR),
                dtype=np.float32
            ) / 255.0
            masks.append(arr >= fg_threshold)
        except Exception:
            masks.append(None)
    return masks


# ---------------------------------------------------------------------------
# Métricas sem GT
# ---------------------------------------------------------------------------

def compute_sam_confidence_metrics(
    iou_scores_sam: np.ndarray,
) -> Dict[str, Any]:
    """
    iou_scores_sam é a confiança interna do SAM — não é IoU real contra GT,
    mas é um indicador confiável de qualidade do prompt recebido.
    Valores baixos (<0.6) sinalizam que o prompt foi ambíguo ou ruim.
    """
    n_high     = int(np.sum(iou_scores_sam >= 0.80))
    n_medium   = int(np.sum((iou_scores_sam >= 0.60) & (iou_scores_sam < 0.80)))
    n_low      = int(np.sum(iou_scores_sam < 0.60))

    return {
        "sam_iou_mean":    safe_float(float(np.mean(iou_scores_sam))),
        "sam_iou_std":     safe_float(float(np.std(iou_scores_sam))),
        "sam_iou_min":     safe_float(float(np.min(iou_scores_sam))),
        "sam_iou_max":     safe_float(float(np.max(iou_scores_sam))),
        "sam_iou_n_high":  n_high,
        "sam_iou_n_medium":n_medium,
        "sam_iou_n_low":   n_low,
    }


def compute_fg_fraction_metrics(
    refined_masks: np.ndarray,
    image_sizes: np.ndarray,
) -> Dict[str, Any]:
    N = len(refined_masks)
    fracs = []
    for i in range(N):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        fracs.append(float(refined_masks[i, :H_i, :W_i].mean()))

    fracs = np.array(fracs)
    return {
        "fg_fraction_mean":         safe_float(float(np.mean(fracs))),
        "fg_fraction_std":          safe_float(float(np.std(fracs))),
        "fg_fraction_min":          safe_float(float(np.min(fracs))),
        "fg_fraction_max":          safe_float(float(np.max(fracs))),
        "fg_fraction_extreme_low":  int(np.sum(fracs < 0.02)),
        "fg_fraction_extreme_high": int(np.sum(fracs > 0.90)),
    }


def compute_spatial_coherence(
    refined_masks: np.ndarray,
    image_sizes: np.ndarray,
) -> Dict[str, Any]:
    """Coesão espacial das máscaras refinadas — componentes conexas."""
    if ndimage_label is None:
        return {"spatial_coherence_note": "scipy não disponível."}

    N = len(refined_masks)
    n_comps, largest_fracs = [], []
    for i in range(N):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        mask = (refined_masks[i, :H_i, :W_i] > 0.5).astype(np.uint8)
        if mask.sum() == 0:
            n_comps.append(0); largest_fracs.append(0.0); continue
        labeled, n = ndimage_label(mask)
        n_comps.append(int(n))
        if n > 0:
            sizes = [(labeled == k).sum() for k in range(1, n + 1)]
            largest_fracs.append(float(max(sizes) / mask.sum()))
        else:
            largest_fracs.append(0.0)

    return {
        "spatial_n_components_mean":      safe_float(float(np.mean(n_comps))),
        "spatial_largest_component_frac": safe_float(float(np.mean(largest_fracs))),
    }


def compute_e3_vs_e4_comparison(
    refined_masks: np.ndarray,
    cosaliency: Optional[Dict[str, Any]],
    image_sizes: np.ndarray,
) -> Dict[str, Any]:
    """
    Compara fração FG e coesão espacial entre E3 e E4 sem usar GT.
    Uma boa E4 deve ter fg_fraction mais moderada (SAM tende a ser mais preciso)
    e maior coerência espacial (bordas limpas).
    """
    if cosaliency is None or "fg_mask_pixel" not in cosaliency:
        return {"e3_vs_e4_note": "cosaliency.npz não fornecido ou sem fg_mask_pixel."}

    fg_e3 = cosaliency["fg_mask_pixel"]
    N = min(len(refined_masks), len(fg_e3))

    fracs_e4, fracs_e3 = [], []
    for i in range(N):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        fracs_e4.append(float(refined_masks[i, :H_i, :W_i].mean()))
        e3_i = fg_e3[i, :H_i, :W_i] if fg_e3.shape[1] >= H_i else fg_e3[i]
        fracs_e3.append(float(e3_i.mean()))

    frac_delta = float(np.mean(fracs_e4)) - float(np.mean(fracs_e3))

    return {
        "e3_fg_fraction_mean":          safe_float(float(np.mean(fracs_e3))),
        "e4_fg_fraction_mean":          safe_float(float(np.mean(fracs_e4))),
        "fg_fraction_delta_e4_minus_e3": safe_float(frac_delta),
    }


# ---------------------------------------------------------------------------
# Métricas com GT
# ---------------------------------------------------------------------------

def compute_gt_iou_metrics(
    refined_masks: np.ndarray,
    image_sizes: np.ndarray,
    pixel_gt_masks: List[Optional[np.ndarray]],
) -> Dict[str, Any]:
    """IoU pixel-level real contra GT."""
    ious = []
    for i, gt in enumerate(pixel_gt_masks):
        if gt is None:
            continue
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        pred = refined_masks[i, :H_i, :W_i] > 0.5
        if gt.any() or (~gt).any():
            ious.append(_iou(pred, gt.astype(bool)))

    if not ious:
        return {"gt_iou_note": "Nenhuma imagem com GT pixel válido."}

    return {
        "gt_iou_mean": safe_float(float(np.mean(ious))),
        "gt_iou_std":  safe_float(float(np.std(ious))),
        "gt_iou_min":  safe_float(float(np.min(ious))),
        "gt_iou_max":  safe_float(float(np.max(ious))),
        "gt_iou_n_above_05":  int(sum(v >= 0.50 for v in ious)),
        "gt_iou_n_above_07":  int(sum(v >= 0.70 for v in ious)),
    }


def compute_e3_vs_e4_gt_gain(
    refined_masks: np.ndarray,
    cosaliency: Optional[Dict[str, Any]],
    image_sizes: np.ndarray,
    pixel_gt_masks: List[Optional[np.ndarray]],
) -> Dict[str, Any]:
    """
    Ganho líquido da E4 sobre a E3 em IoU real.
    Separa imagens onde SAM melhorou vs. degradou.
    """
    if cosaliency is None or "fg_mask_pixel" not in cosaliency:
        return {"e3_vs_e4_gt_note": "cosaliency.npz não fornecido."}

    fg_e3 = cosaliency["fg_mask_pixel"]
    N = min(len(refined_masks), len(fg_e3))

    ious_e3, ious_e4 = [], []
    for i in range(N):
        gt = pixel_gt_masks[i] if i < len(pixel_gt_masks) else None
        if gt is None:
            continue
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        pred_e4 = refined_masks[i, :H_i, :W_i] > 0.5
        e3_i    = fg_e3[i, :H_i, :W_i] if fg_e3.shape[1] >= H_i else fg_e3[i]
        pred_e3 = e3_i > 0.5
        gt_bool = gt.astype(bool)
        if gt_bool.any() or (~gt_bool).any():
            ious_e3.append(_iou(pred_e3, gt_bool))
            ious_e4.append(_iou(pred_e4, gt_bool))

    if not ious_e3:
        return {"e3_vs_e4_gt_note": "Sem imagens comparáveis."}

    gains = [e4 - e3 for e4, e3 in zip(ious_e4, ious_e3)]
    n_improved = int(sum(g > 0.01 for g in gains))
    n_degraded = int(sum(g < -0.01 for g in gains))
    n_neutral  = len(gains) - n_improved - n_degraded

    return {
        "e3_iou_mean":              safe_float(float(np.mean(ious_e3))),
        "e4_iou_mean":              safe_float(float(np.mean(ious_e4))),
        "iou_gain_mean":            safe_float(float(np.mean(gains))),
        "iou_gain_std":             safe_float(float(np.std(gains))),
        "n_images_improved":        n_improved,
        "n_images_degraded":        n_degraded,
        "n_images_neutral":         n_neutral,
        "per_image_iou_e3":         ious_e3,
        "per_image_iou_e4":         ious_e4,
    }


# ---------------------------------------------------------------------------
# Interpretações heurísticas
# ---------------------------------------------------------------------------

def assess_high_good(metric, value, strong, moderate,
                     smsg, mmsg, wmsg, test) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível", "Não foi possível calcular.", test)
    if v >= strong:   return Assessment(metric, v, "bom",           smsg, test)
    if v >= moderate: return Assessment(metric, v, "intermediário", mmsg, test)
    return Assessment(metric, v, "fraco", wmsg, test)


def assess_low_good(metric, value, good, moderate,
                    gmsg, mmsg, bmsg, test) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível", "Não foi possível calcular.", test)
    if v <= good:     return Assessment(metric, v, "bom",           gmsg, test)
    if v <= moderate: return Assessment(metric, v, "intermediário", mmsg, test)
    return Assessment(metric, v, "alerta", bmsg, test)


def assess_range(metric, value, lo_g, hi_g, lo_w, hi_w,
                 gmsg, mmsg, bmsg, test) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível", "Não foi possível calcular.", test)
    if lo_g <= v <= hi_g: return Assessment(metric, v, "bom",           gmsg, test)
    if lo_w <= v <= hi_w: return Assessment(metric, v, "intermediário", mmsg, test)
    return Assessment(metric, v, "alerta", bmsg, test)


def build_assessments(metrics: Dict[str, Any], has_gt: bool) -> List[Assessment]:
    A = []

    A.append(assess_high_good(
        "sam_iou_mean", metrics.get("sam_iou_mean"),
        strong=0.85, moderate=0.70,
        smsg="SAM com alta confiança nas máscaras — os prompts da E3 estão sendo bem interpretados.",
        mmsg="Confiança SAM moderada. Alguns prompts podem ser ambíguos.",
        wmsg="Confiança SAM baixa — seeds da E3 estão produzindo prompts ruins. Gargalo pode estar na E3.",
        test="Verifique viz_sam_confidence.png. Se muitas imagens tiverem IoU < 0.6, tente --prompt_mode box_only."
    ))

    A.append(assess_low_good(
        "sam_iou_n_low", metrics.get("sam_iou_n_low"),
        good=0, moderate=2,
        gmsg="Nenhuma imagem com confiança SAM muito baixa.",
        mmsg="Algumas imagens com baixa confiança SAM — seeds problemáticas nesses casos.",
        bmsg="Várias imagens com confiança SAM baixa. Os prompts da E3 precisam de revisão.",
        test="Inspecione as imagens com IoU < 0.6 em viz_sam_confidence.png."
    ))

    A.append(assess_range(
        "fg_fraction_mean", metrics.get("fg_fraction_mean"),
        lo_g=0.05, hi_g=0.65, lo_w=0.02, hi_w=0.80,
        gmsg="Fração FG dentro de uma faixa plausível após refinamento SAM.",
        mmsg="Fração levemente fora do ideal — SAM pode estar sobre ou sub-segmentando.",
        bmsg="Fração FG extrema após SAM. Verificar se o SAM está recebendo bounding boxes corretas.",
        test="Verifique viz_fg_fraction.png. Bboxes muito grandes ou pontos BG errados causam over-segmentation."
    ))

    A.append(assess_high_good(
        "spatial_largest_component_frac",
        metrics.get("spatial_largest_component_frac"),
        strong=0.80, moderate=0.60,
        smsg="Máscaras SAM espacialmente coesas — bordas limpas e objeto contíguo.",
        mmsg="Coesão espacial moderada. Pode haver fragmentos residuais.",
        wmsg="Máscaras fragmentadas mesmo após SAM. Verificar se bounding boxes estão corretas.",
        test="SAM geralmente produz máscaras coesas. Fragmentação indica bbox muito imprecisa."
    ))

    A.append(assess_high_good(
        "fg_coherence_e4", metrics.get("fg_coherence_e4"),
        strong=0.70, moderate=0.50,
        smsg="Alta coesão entre os protótipos de foreground das imagens após o SAM.",
        mmsg="Coesão semântica moderada entre os foregrounds refinados.",
        wmsg="Os protótipos de foreground das imagens estão pouco semelhantes.",
        test="Consulte também fg_coherence_delta e o IoU real; os limiares são heurísticos."
    ))

    coherence_delta = safe_float(
        metrics.get("fg_coherence_delta")
    )

    if coherence_delta is None:
        A.append(Assessment(
            metric="fg_coherence_delta",
            value=None,
            verdict="indisponível",
            indication=(
                "Não foi possível comparar a coesão semântica "
                "das Etapas 3 e 4."
            ),
            suggested_test=(
                "Verifique se refined_masks.npz contém "
                "fg_coherence_e3, fg_coherence_e4 e "
                "fg_coherence_delta."
            ),
        ))

    elif coherence_delta >= 0.01:
        A.append(Assessment(
            metric="fg_coherence_delta",
            value=coherence_delta,
            verdict="bom",
            indication=(
                "O SAM aumentou a coesão semântica "
                "cross-image em relação à E3."
            ),
            suggested_test=(
                "Confirme se o ganho de IoU também foi positivo."
            ),
        ))

    elif coherence_delta >= -0.01:
        A.append(Assessment(
            metric="fg_coherence_delta",
            value=coherence_delta,
            verdict="bom",
            indication=(
                "O SAM preservou a coesão semântica da E3. "
                "A pequena variação está dentro da tolerância de ±0,01."
            ),
            suggested_test=(
                "Nenhuma ação necessária. Avalie o ganho de IoU "
                "para verificar se o refinamento melhorou os contornos."
            ),
        ))

    elif coherence_delta >= -0.03:
        A.append(Assessment(
            metric="fg_coherence_delta",
            value=coherence_delta,
            verdict="intermediário",
            indication=(
                "O SAM reduziu levemente a coesão semântica "
                "em relação à E3."
            ),
            suggested_test=(
                "Se o ganho de IoU continuar positivo, a redução pode "
                "ser apenas consequência do refinamento das bordas. "
                "Caso contrário, revise o modo de prompting."
            ),
        ))

    else:
        A.append(Assessment(
            metric="fg_coherence_delta",
            value=coherence_delta,
            verdict="alerta",
            indication=(
                "O SAM reduziu significativamente a coesão "
                "semântica cross-image."
            ),
            suggested_test=(
                "Compare box_only, box_points e box_points_mask. "
                "Verifique também se o ganho de IoU foi negativo."
            ),
        ))

    if has_gt:
        A.append(assess_high_good(
            "gt_iou_mean", metrics.get("gt_iou_mean"),
            strong=0.65, moderate=0.45,
            smsg="IoU final bom — o pipeline completo E1→E4 está produzindo segmentações precisas.",
            mmsg="IoU moderado. O resultado é utilizável mas tem margem de melhora.",
            wmsg="IoU baixo mesmo após SAM. O problema pode estar nos prompts (E3) ou na qualidade das seeds.",
            test="Compare com E3 pixel_iou. Se E4 < E3, o SAM está degradando — tente box_only ou aumente n_iter."
        ))

        A.append(assess_high_good(
            "iou_gain_mean", metrics.get("iou_gain_mean"),
            strong=0.05, moderate=0.00,
            smsg="SAM melhorou o IoU em relação à E3 — refinamento foi positivo.",
            mmsg="Ganho nulo ou marginal. SAM manteve mas não melhorou a qualidade da E3.",
            wmsg="SAM degradou o IoU em relação à E3 — os prompts estão atrapalhando.",
            test="Tente --prompt_mode box_only para ver se o problema é nos pontos ou na máscara coarse."
        ))

        A.append(assess_low_good(
            "n_images_degraded", metrics.get("n_images_degraded"),
            good=0, moderate=2,
            gmsg="Nenhuma imagem degradada pelo SAM.",
            mmsg="Algumas imagens tiveram IoU reduzido pelo SAM — seeds problemáticas nesses casos.",
            bmsg="Muitas imagens degradadas. O modo de prompting pode não ser adequado para esta coleção.",
            test="Veja viz_per_image_gain.png para identificar as imagens problemáticas."
        ))

    return A


def infer_bottleneck(metrics: Dict[str, Any], has_gt: bool) -> str:
    sam_iou  = safe_float(metrics.get("sam_iou_mean"))
    gt_iou   = safe_float(metrics.get("gt_iou_mean"))
    gain     = safe_float(metrics.get("iou_gain_mean"))
    n_deg    = metrics.get("n_images_degraded", 0)
    coherence = safe_float(metrics.get("fg_coherence_e4"))
    coherence_delta = safe_float(metrics.get("fg_coherence_delta"))
    n_low    = metrics.get("sam_iou_n_low", 0)

    if has_gt:
        if gt_iou is not None and gt_iou >= 0.65:
            return ("Diagnóstico: pipeline completo funcionando bem. "
                    "IoU final acima de 0.65 é um resultado sólido para cosegmentação não supervisionada.")
        if gain is not None and gain < -0.03:
            return ("Diagnóstico: SAM está degradando a segmentação da E3. "
                    "O gargalo está nos prompts — seeds da E3 são ruins. "
                    "Tente --prompt_mode box_only ou revise n_clusters/fg_threshold na E3.")
        if sam_iou is not None and sam_iou < 0.70 and n_low > 1:
            return ("Diagnóstico: SAM com baixa confiança em várias imagens — seeds ambíguas. "
                    "Isso geralmente indica bounding boxes imprecisas ou pontos FG/BG contraditórios. "
                    "Revise a E3 ou use --prompt_mode box_only.")
        if gt_iou is not None and 0.45 <= gt_iou < 0.65:
            return ("Diagnóstico: resultado intermediário. "
                    "O pipeline funciona mas há margem de melhora. "
                    "Tente --n_iter 2 (refinamento iterativo) ou ajuste os parâmetros da E3.")
        return ("Diagnóstico: IoU abaixo do esperado. "
                "Verifique se o gargalo está na E3 (oracle baixo) ou nos prompts da E4.")

    # Sem GT
    if sam_iou is not None and sam_iou < 0.70:
        return ("Diagnóstico sem GT: confiança SAM baixa — prompts provavelmente ruins. "
                "Revise seeds da E3 ou use --prompt_mode box_only.")
    if coherence_delta is not None and coherence_delta < -0.03:
        return ("Diagnóstico sem GT: o SAM reduziu a coesão semântica em relação à E3. "
                "Revise o modo de prompting e inspecione as máscaras por imagem.")
    if coherence is not None and coherence < 0.50:
        return ("Diagnóstico sem GT: coesão cross-image baixa após SAM. "
                "O SAM pode estar segmentando objetos diferentes em cada imagem.")
    return ("Diagnóstico sem GT: sinais dentro do esperado. "
            "Forneça gt_index.json para diagnóstico preciso do ganho sobre a E3.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_sam_confidence(
    iou_scores_sam: np.ndarray,
    image_paths: List[str],
    fg_coherence: float,
    output_dir: Path,
) -> None:
    if plt is None: return
    N = len(iou_scores_sam)
    names  = [Path(p).name[:22] for p in image_paths]
    colors = ["#2ecc71" if v >= 0.80 else "#f39c12" if v >= 0.60 else "#e74c3c"
              for v in iou_scores_sam]

    fig, ax = plt.subplots(figsize=(9, max(3, N * 0.45 + 1.5)))
    bars = ax.barh(range(N), iou_scores_sam, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(N)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("IoU Score (confiança interna SAM 2)", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.80, color="gray", ls="--", lw=1, alpha=0.6, label="0.80 (bom)")
    ax.axvline(0.60, color="gray", ls=":",  lw=1, alpha=0.6, label="0.60 (razoável)")
    ax.legend(fontsize=8)
    ax.set_title(
        f"Confiança SAM 2 por imagem\n"
        f"verde≥0.80  laranja≥0.60  vermelho<0.60 | "
        f"FG coherence E4={fg_coherence:.4f}",
        fontsize=9
    )
    for bar, v in zip(bars, iou_scores_sam):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_sam_confidence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_e3_vs_e4(
    refined_masks: np.ndarray,
    cosaliency: Optional[Dict[str, Any]],
    image_paths: List[str],
    image_sizes: np.ndarray,
    output_dir: Path,
    max_images: int = 8,
) -> None:
    if plt is None: return
    N_vis = min(len(image_paths), max_images)
    has_e3 = cosaliency is not None and "fg_mask_pixel" in cosaliency
    n_cols = 3 if has_e3 else 2

    fig, axes = plt.subplots(N_vis, n_cols, figsize=(4.5 * n_cols, 3.5 * N_vis))
    if N_vis == 1: axes = axes[np.newaxis, :]
    fig.suptitle("Comparação E3 (coarse) → E4 (SAM refinada)", fontsize=11)

    for i in range(N_vis):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        img = Image.open(image_paths[i]).convert("RGB")
        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        axes[i, 0].set_title(Path(image_paths[i]).name[:22], fontsize=7)

        if has_e3:
            fg_e3 = cosaliency["fg_mask_pixel"]
            e3_i = fg_e3[i, :H_i, :W_i] if fg_e3.shape[1] >= H_i else fg_e3[i]
            axes[i, 1].imshow(e3_i, cmap="RdYlGn", vmin=0, vmax=1)
            axes[i, 1].set_title(f"E3  FG%={e3_i.mean()*100:.0f}%", fontsize=7)
            axes[i, 1].axis("off")

        col = 2 if has_e3 else 1
        r_i = refined_masks[i, :H_i, :W_i]
        axes[i, col].imshow(r_i, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, col].set_title(f"E4 SAM  FG%={r_i.mean()*100:.0f}%", fontsize=7)
        axes[i, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "viz_e3_vs_e4_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fg_fraction(
    refined_masks: np.ndarray,
    image_sizes: np.ndarray,
    e3_fracs: Optional[List[float]],
    output_dir: Path,
) -> None:
    if plt is None: return
    N = len(refined_masks)
    fracs_e4 = [float(refined_masks[i, :int(image_sizes[i,0]), :int(image_sizes[i,1])].mean())
                for i in range(N)]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(N)
    width = 0.35
    if e3_fracs is not None:
        ax.bar(x - width/2, [f*100 for f in e3_fracs], width,
               label="E3 (coarse)", color="steelblue", alpha=0.7)
        ax.bar(x + width/2, [f*100 for f in fracs_e4], width,
               label="E4 (SAM)", color="#2ecc71", alpha=0.8)
    else:
        ax.bar(x, [f*100 for f in fracs_e4], color="#2ecc71", alpha=0.8)
    ax.axhline(5,  color="red",    ls="--", lw=1.2, alpha=0.5, label="limite inf (5%)")
    ax.axhline(65, color="orange", ls="--", lw=1.2, alpha=0.5, label="limite sup (65%)")
    ax.set_xlabel("Imagem"); ax.set_ylabel("Fração FG (%)"); ax.set_ylim(0, 100)
    ax.set_title("Fração FG: E3 vs E4")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_fg_fraction.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_gt_iou_comparison(
    metrics: Dict[str, Any],
    image_paths: List[str],
    output_dir: Path,
) -> None:
    if plt is None: return
    ious_e3 = metrics.get("per_image_iou_e3", [])
    ious_e4 = metrics.get("per_image_iou_e4", [])
    if not ious_e3 or not ious_e4: return

    N = min(len(ious_e3), len(ious_e4))
    x = np.arange(N)
    width = 0.35
    names = [Path(image_paths[i]).name[:18] if i < len(image_paths) else str(i)
             for i in range(N)]

    fig, ax = plt.subplots(figsize=(max(8, N * 0.9), 5))
    ax.bar(x - width/2, ious_e3, width, label="E3 (coarse)", color="steelblue", alpha=0.7)
    ax.bar(x + width/2, ious_e4, width, label="E4 (SAM)",    color="#2ecc71",   alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("IoU vs GT"); ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_title(
        f"IoU real (vs GT) por imagem: E3 vs E4\n"
        f"Ganho médio = {metrics.get('iou_gain_mean', 0):.4f}",
        fontsize=9
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_gt_iou_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_per_image_gain(
    metrics: Dict[str, Any],
    image_paths: List[str],
    output_dir: Path,
) -> None:
    if plt is None: return
    ious_e3 = metrics.get("per_image_iou_e3", [])
    ious_e4 = metrics.get("per_image_iou_e4", [])
    if not ious_e3 or not ious_e4: return

    N = min(len(ious_e3), len(ious_e4))
    gains = [ious_e4[i] - ious_e3[i] for i in range(N)]
    names = [Path(image_paths[i]).name[:22] if i < len(image_paths) else str(i)
             for i in range(N)]
    colors = ["#2ecc71" if g > 0.01 else "#e74c3c" if g < -0.01 else "#95a5a6"
              for g in gains]

    fig, ax = plt.subplots(figsize=(9, max(3, N * 0.45 + 1.5)))
    bars = ax.barh(range(N), gains, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(N)); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Ganho IoU (E4 − E3)")
    ax.set_title("Ganho por imagem (verde=melhora, vermelho=piora, cinza=neutro)", fontsize=9)
    for bar, g in zip(bars, gains):
        ax.text(g + (0.005 if g >= 0 else -0.005),
                bar.get_y() + bar.get_height() / 2,
                f"{g:+.3f}", va="center", ha="left" if g >= 0 else "right", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "viz_per_image_gain.png", dpi=120, bbox_inches="tight")
    plt.close(fig)



# ---------------------------------------------------------------------------
# Plots com GT — overlay visual e mapa de erro por pixel
# ---------------------------------------------------------------------------

def plot_gt_overlay(
    refined_masks: np.ndarray,
    image_paths: List[str],
    image_sizes: np.ndarray,
    pixel_gt_masks: List[Optional[np.ndarray]],
    output_dir: Path,
    max_images: int = 8,
) -> None:
    """
    Para cada imagem mostra 3 colunas:
      1. Imagem original com contorno do GT em amarelo
      2. Overlay da máscara E4 em verde semitransparente + contorno GT
      3. Mapa de erro por pixel:
           Verde  = TP (acertou FG)
           Azul   = FN (era FG, previu BG)
           Vermelho = FP (era BG, previu FG)
           Cinza  = TN (acertou BG)
    """
    if plt is None: return
    N_vis = min(len(image_paths), max_images)
    valid = [i for i in range(N_vis) if i < len(pixel_gt_masks) and pixel_gt_masks[i] is not None]
    if not valid: return

    fig, axes = plt.subplots(len(valid), 3, figsize=(13, 4 * len(valid)))
    if len(valid) == 1: axes = axes[np.newaxis, :]
    fig.suptitle(
        "Comparação visual com GT — Etapa 4\n"
        "Esq: original + contorno GT  |  Centro: overlay E4 + contorno GT  |"
        "  Dir: mapa de erro (verde=TP  azul=FN  vermelho=FP)",
        fontsize=10, y=1.01
    )

    for row, i in enumerate(valid):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        img_arr = np.array(Image.open(image_paths[i]).convert("RGB"))
        pred    = (refined_masks[i, :H_i, :W_i] > 0.5)
        gt      = pixel_gt_masks[i].astype(bool)

        iou = _iou(pred, gt)

        # Contorno GT: borda da máscara GT em amarelo (dilatação simples)
        try:
            from scipy.ndimage import binary_dilation
            contour = binary_dilation(gt, iterations=2) & ~gt
        except Exception:
            contour = np.zeros_like(gt)

        # Col 0: imagem + contorno GT
        img_with_contour = img_arr.copy()
        img_with_contour[contour] = [255, 220, 0]   # amarelo
        axes[row, 0].imshow(img_with_contour)
        axes[row, 0].set_title(f"{Path(image_paths[i]).name[:24]}\nGT contorno (amarelo)",
                               fontsize=7)
        axes[row, 0].axis("off")

        # Col 1: overlay E4 verde + contorno GT amarelo
        overlay = img_arr.astype(np.float32)
        green   = np.zeros_like(img_arr, dtype=np.float32); green[:,:,1] = 255
        alpha   = pred[:,:,None].astype(np.float32) * 0.5
        overlay = (overlay * (1 - alpha) + green * alpha).clip(0, 255).astype(np.uint8)
        overlay[contour] = [255, 220, 0]
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f"E4 (verde) + GT (amarelo)\nIoU={iou:.3f}", fontsize=7)
        axes[row, 1].axis("off")

        # Col 2: mapa de erro TP/FP/FN/TN
        error_map = np.zeros((H_i, W_i, 3), dtype=np.uint8)
        tp = pred & gt;   error_map[tp]  = [46, 204, 113]   # verde
        fp = pred & ~gt;  error_map[fp]  = [231, 76, 60]    # vermelho
        fn = ~pred & gt;  error_map[fn]  = [52, 152, 219]   # azul
        tn = ~pred & ~gt; error_map[tn]  = [200, 200, 200]  # cinza
        axes[row, 2].imshow(error_map)
        n_tp, n_fp, n_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
        axes[row, 2].set_title(
            f"TP={n_tp}  FP={n_fp}  FN={n_fn}\n"
            f"verde=TP  vermelho=FP  azul=FN",
            fontsize=7
        )
        axes[row, 2].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "viz_gt_overlay.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_segmented_with_gt(
    refined_masks: np.ndarray,
    image_paths: List[str],
    image_sizes: np.ndarray,
    pixel_gt_masks: List[Optional[np.ndarray]],
    output_dir: Path,
    max_images: int = 8,
) -> None:
    """
    Exporta imagens individuais em output_dir/segmented_with_gt/:
      <nome>_result.png  — recorte do objeto segmentado (RGBA)
      <nome>_gt.png      — recorte do GT (RGBA)
      <nome>_compare.png — lado a lado: original | E4 | GT
    """
    if plt is None: return
    seg_dir = output_dir / "segmented_with_gt"
    seg_dir.mkdir(parents=True, exist_ok=True)

    N = min(len(image_paths), max_images)
    print(f"\n[Export GT] Salvando comparações em {seg_dir}/")

    for i in range(N):
        if i >= len(pixel_gt_masks) or pixel_gt_masks[i] is None:
            continue
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        stem    = Path(image_paths[i]).stem
        img_arr = np.array(Image.open(image_paths[i]).convert("RGB"))
        pred    = (refined_masks[i, :H_i, :W_i] > 0.5).astype(np.uint8)
        gt      = pixel_gt_masks[i].astype(np.uint8)
        iou     = _iou(pred.astype(bool), gt.astype(bool))

        # Cutout E4 (RGBA)
        rgba_e4 = np.zeros((H_i, W_i, 4), dtype=np.uint8)
        rgba_e4[:,:,:3] = img_arr
        rgba_e4[:,:,3]  = pred * 255
        Image.fromarray(rgba_e4, "RGBA").save(seg_dir / f"{stem}_result.png")

        # Cutout GT (RGBA)
        rgba_gt = np.zeros((H_i, W_i, 4), dtype=np.uint8)
        rgba_gt[:,:,:3] = img_arr
        rgba_gt[:,:,3]  = gt * 255
        Image.fromarray(rgba_gt, "RGBA").save(seg_dir / f"{stem}_gt.png")

        # Comparação lado a lado: original | E4 overlay | GT overlay
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{stem}  —  IoU={iou:.3f}", fontsize=10)

        axes[0].imshow(img_arr); axes[0].set_title("Original", fontsize=8); axes[0].axis("off")

        overlay_e4 = img_arr.astype(np.float32)
        green = np.zeros_like(img_arr, dtype=np.float32); green[:,:,1] = 255
        alpha = pred[:,:,None].astype(np.float32) * 0.5
        overlay_e4 = (overlay_e4 * (1-alpha) + green * alpha).clip(0,255).astype(np.uint8)
        axes[1].imshow(overlay_e4)
        axes[1].set_title(f"E4 (SAM)  FG%={pred.mean()*100:.0f}%", fontsize=8)
        axes[1].axis("off")

        overlay_gt = img_arr.astype(np.float32)
        yellow = np.zeros_like(img_arr, dtype=np.float32)
        yellow[:,:,0] = 255; yellow[:,:,1] = 220
        alpha_gt = gt[:,:,None].astype(np.float32) * 0.5
        overlay_gt = (overlay_gt * (1-alpha_gt) + yellow * alpha_gt).clip(0,255).astype(np.uint8)
        axes[2].imshow(overlay_gt)
        axes[2].set_title(f"GT  FG%={gt.mean()*100:.0f}%", fontsize=8)
        axes[2].axis("off")

        fig.tight_layout()
        fig.savefig(seg_dir / f"{stem}_compare.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"  {stem}  IoU={iou:.3f}")

    print(f"[Export GT] Concluído → {seg_dir}/")


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    refined_path: Path,
    metrics: Dict[str, Any],
    assessments: List[Assessment],
    diagnosis: str,
    gt_matched: int,
    n_images: int,
) -> Path:
    report_path = output_dir / "step4_quality_report.md"
    lines = [
        "# Relatório de qualidade — Etapa 4 (SAM 2 Refinement)",
        "",
        f"Arquivo analisado: `{refined_path}`",
        "",
        "## Resumo",
        "",
        f"- Imagens: **{n_images}**",
        f"- Modo de prompt: **{metrics.get('prompt_mode', 'n/a')}**",
        f"- IoU SAM médio: **{compact_float(metrics.get('sam_iou_mean'))}**",
        f"- FG coherence E3: **{compact_float(metrics.get('fg_coherence_e3'))}**",
        f"- FG coherence E4: **{compact_float(metrics.get('fg_coherence_e4'))}**",
        f"- Delta coherence E4−E3: **{compact_float(metrics.get('fg_coherence_delta'))}**",
        f"- GT encontrado: **{gt_matched} / {n_images}**",
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
    lines += ["", "## Arquivos gerados", ""]
    for name in ["step4_quality.json", "viz_sam_confidence.png",
                 "viz_e3_vs_e4_comparison.png", "viz_fg_fraction.png",
                 "viz_gt_iou_comparison.png", "viz_per_image_gain.png",
                 "viz_gt_overlay.png", "segmented_with_gt/"]:
        p = output_dir / name
        if p.exists():
            lines.append(f"- `{name}`")
    lines += [
        "",
        "## Observações",
        "",
        "- `iou_scores_sam` é confiança interna do SAM, não IoU real contra GT.",
        "- `fg_coherence_e3/e4` compara protótipos DINOv2 do foreground entre imagens.",
        "- `fg_coherence_delta` mede a mudança semântica causada pelo SAM (E4−E3).",
        "- Os limiares de coesão por protótipos são heurísticos e devem ser calibrados em mais coleções.",
        "- Se `iou_gain_mean` for negativo, o gargalo está nos prompts (seeds da E3).",
        "- Tente `--prompt_mode box_only` se pontos FG/BG estiverem degradando o resultado.",
        "- `--n_iter 2` aplica refinamento iterativo e geralmente melhora bordas.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_console_summary(metrics, assessments, diagnosis) -> None:
    print("\n" + "=" * 72)
    print("  AUDITORIA DA ETAPA 4 — SAM 2 REFINEMENT")
    print("=" * 72)
    print(f"  Modo prompt  : {metrics.get('prompt_mode', 'n/a')}")
    print(f"  IoU SAM médio: {compact_float(metrics.get('sam_iou_mean'))}")
    print(f"  FG coherence E3 : {compact_float(metrics.get('fg_coherence_e3'))}")
    print(f"  FG coherence E4 : {compact_float(metrics.get('fg_coherence_e4'))}")
    print(f"  Delta E4-E3     : {compact_float(metrics.get('fg_coherence_delta'))}")
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
        description="Analisa refined_masks.npz gerado pela Etapa 4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--refined",     required=True, type=Path,
                   help="refined_masks.npz da Etapa 4.")
    p.add_argument("--cosaliency",  type=Path, default=None,
                   help="cosaliency.npz da E3 (para comparação E3 vs E4).")
    p.add_argument("--gt_index",    type=Path, default=None)
    p.add_argument("--output",      type=Path, default=None)
    p.add_argument("--fg_threshold", type=float, default=0.5)
    p.add_argument("--max_viz_images", type=int, default=8)
    p.add_argument("--no_plots",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    refined_path = args.refined
    output_dir   = args.output or (refined_path.parent / "step4_quality")
    ensure_dir(output_dir)

    print(f"[1/6] Carregando refined_masks: {refined_path}")
    ref = load_refined(refined_path)
    refined_masks  = ref["refined_masks"]
    iou_scores_sam = ref["iou_scores_sam"]
    fg_coherence_e3 = ref.get("fg_coherence_e3")
    fg_coherence_e4 = ref.get("fg_coherence_e4", ref["fg_coherence"])
    fg_coherence_delta = ref.get("fg_coherence_delta")
    prompt_mode    = ref["prompt_mode"]
    image_paths    = ref["image_paths"]
    image_sizes    = ref["image_sizes"]
    N              = len(image_paths)

    metrics: Dict[str, Any] = {
        "refined_path": str(refined_path),
        "n_images":     int(N),
        "prompt_mode":  prompt_mode,
        # Alias antigo preservado no JSON para consumidores legados.
        "fg_coherence": safe_float(fg_coherence_e4),
        "fg_coherence_e3": safe_float(fg_coherence_e3),
        "fg_coherence_e4": safe_float(fg_coherence_e4),
        "fg_coherence_delta": safe_float(fg_coherence_delta),
    }

    print(f"[2/6] Calculando confiança SAM e fração FG...")
    metrics.update(compute_sam_confidence_metrics(iou_scores_sam))
    metrics.update(compute_fg_fraction_metrics(refined_masks, image_sizes))
    metrics.update(compute_spatial_coherence(refined_masks, image_sizes))

    print(f"[3/6] Carregando cosaliency (E3) para comparação...")
    cosaliency = load_cosaliency(args.cosaliency)
    metrics.update(compute_e3_vs_e4_comparison(refined_masks, cosaliency, image_sizes))

    print(f"[4/6] Procurando ground truth...")
    gt_index     = load_gt_index(args.gt_index)
    pixel_gt     = load_pixel_gt_masks(image_paths, image_sizes, gt_index, args.fg_threshold)
    gt_matched   = sum(1 for m in pixel_gt if m is not None) if pixel_gt else 0
    metrics["gt_matched"] = gt_matched
    has_gt = gt_matched > 0

    if has_gt:
        print(f"      GT: {gt_matched} / {N} imagens.")
        print(f"[5/6] Calculando IoU real e ganho E3→E4...")
        metrics.update(compute_gt_iou_metrics(refined_masks, image_sizes, pixel_gt))
        gain_m = compute_e3_vs_e4_gt_gain(
            refined_masks, cosaliency, image_sizes, pixel_gt)
        # Separa listas (não serializam bem em JSON) das métricas escalares
        per_img_e3 = gain_m.pop("per_image_iou_e3", [])
        per_img_e4 = gain_m.pop("per_image_iou_e4", [])
        metrics.update(gain_m)
        metrics["per_image_iou_e3"] = per_img_e3
        metrics["per_image_iou_e4"] = per_img_e4
    else:
        print("      GT indisponível.")
        per_img_e3, per_img_e4 = [], []

    print(f"[6/6] Gerando relatório e visualizações...")
    assessments = build_assessments(metrics, has_gt=has_gt)
    diagnosis   = infer_bottleneck(metrics, has_gt=has_gt)

    # Serializa JSON (remove listas longas de per_image para não poluir)
    json_metrics = {k: v for k, v in metrics.items()
                    if k not in ("per_image_iou_e3", "per_image_iou_e4")}
    json_path = output_dir / "step4_quality.json"
    json_path.write_text(
        json.dumps({"metrics": json_metrics,
                    "assessments": [assessment_to_dict(a) for a in assessments],
                    "diagnosis": diagnosis},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not args.no_plots:
        plot_sam_confidence(iou_scores_sam, image_paths, fg_coherence_e4, output_dir)
        plot_e3_vs_e4(refined_masks, cosaliency, image_paths, image_sizes,
                      output_dir, args.max_viz_images)
        e3_fracs = None
        if cosaliency and "fg_mask_pixel" in cosaliency:
            e3_fracs = [float(cosaliency["fg_mask_pixel"][i,
                              :int(image_sizes[i,0]), :int(image_sizes[i,1])].mean())
                        for i in range(min(N, len(cosaliency["fg_mask_pixel"])))]
        plot_fg_fraction(refined_masks, image_sizes, e3_fracs, output_dir)
        if has_gt:
            metrics["per_image_iou_e3"] = per_img_e3
            metrics["per_image_iou_e4"] = per_img_e4
            plot_gt_iou_comparison(metrics, image_paths, output_dir)
            plot_per_image_gain(metrics, image_paths, output_dir)
            plot_gt_overlay(refined_masks, image_paths, image_sizes,
                            pixel_gt, output_dir, args.max_viz_images)
            plot_segmented_with_gt(refined_masks, image_paths, image_sizes,
                                   pixel_gt, output_dir, args.max_viz_images)

    report_path = write_report(output_dir, refined_path, metrics, assessments,
                               diagnosis, gt_matched, N)
    print_console_summary(metrics, assessments, diagnosis)
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
