#!/usr/bin/env python3
"""
analyze_e3_cluster_selection.py
===============================

Ablação/diagnóstico pós-hoc da Etapa 3 (E3) do pipeline de cosegmentação.

NÃO reexecuta E1/E2/E2B/E3 e NÃO roda SAM.
Usa os artefatos já existentes em cada classe:

    features.npz
    correspondence_bg.npz   (se existir; senão correspondence.npz)
    cosaliency.npz

Objetivos
---------
Separar três modos de falha da E3:

A) SELECTION FAILURE
   Os clusters globais contêm o objeto, mas a regra atual escolhe clusters ruins.
   Sinal:
       oracle global (1/2/3 clusters) alto
       current baixo

B) CLUSTER CONSISTENCY FAILURE
   Em cada imagem existe algum cluster bom, mas o ID do cluster do objeto muda
   entre imagens. Uma única seleção global de clusters não funciona.
   Sinal:
       oracle por imagem alto
       oracle global baixo

C) CLUSTERING FAILURE
   Nem escolhendo o melhor cluster por imagem o objeto é bem isolado.
   Sinal:
       oracle por imagem baixo

Além dos oracles, simula seletores NÃO supervisionados sobre os clusters já
produzidos pela E3:

    current_cluster          seleção cluster_is_fg gravada pela E3
    current_pixel            fg_mask_pixel oficial gravada pela E3
    e2b_top1 / e2b_top2      ranking pela média do score E2B no cluster
    final_top1 / final_top2  ranking pelo cluster_scores da E3
    e2b_sp_top1/top2         0.7*E2B + 0.3*spatial
    e2b_border_top1/top2     0.7*E2B + 0.3*(1-border_fraction)
    hybrid_top1/top2         0.6*E2B + 0.2*spatial + 0.2*(1-border)
    fallback_e2b_t001/t002/t005
                             mantém current_cluster, mas se a fração FG atual
                             for menor que 1%, 2% ou 5%, usa e2b_top2.

Os GTs são usados SOMENTE para avaliação/oracle diagnóstico.
Nenhuma regra simulada usa GT para escolher clusters.

Saídas
------
e3_selector_summary.csv
e3_cluster_diagnostics.csv
e3_per_image.csv
e3_results.json

Uso PASCAL:
    python ablation/analyze_e3_cluster_selection.py \
        --features-root features/pascal_coseg_bg_recurrence_l0p50 \
        --gt-index datasets/PASCAL/gt_index.json \
        --classes person boat bottle bird bus car horse \
        --output ablation_results/e3_cluster_pascal

Uso iCoseg:
    python ablation/analyze_e3_cluster_selection.py \
        --features-root features/icoseg38_bg_recurrence_l0p50 \
        --gt-index gt_index.json \
        --classes "Pandas" "Agra Taj Mahal" "Alaskan Brown Bear" Balloon Kite Statue \
        --output ablation_results/e3_cluster_icoseg
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluate import (  # noqa: E402
    build_gt_mapping,
    get_gt_index_multi,
    load_gt_index,
    load_mask,
)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def safe_name(name: str) -> str:
    return "".join(
        ch if (ch.isalnum() or ch in "_-") else "_"
        for ch in name
    ).strip("_")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return inter / union if union else 0.0


def resize_bool(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    arr = np.asarray(mask, dtype=bool)
    if arr.shape == (h, w):
        return arr
    img = Image.fromarray(arr.astype(np.uint8) * 255, mode="L")
    out = img.resize((w, h), Image.NEAREST)
    return np.asarray(out, dtype=np.uint8) > 128


def patch_mask_to_gt(
    patch_mask: np.ndarray,
    gt_hw: tuple[int, int],
    resize_row: np.ndarray | None,
) -> np.ndarray:
    """Reconstrói patch-grid no domínio do GT, removendo padding keep_ar."""
    h_gt, w_gt = gt_hw
    patch_img = Image.fromarray(
        np.asarray(patch_mask, dtype=np.uint8) * 255,
        mode="L",
    )

    if resize_row is None:
        out = patch_img.resize((w_gt, h_gt), Image.NEAREST)
        return np.asarray(out, dtype=np.uint8) > 128

    vals = [int(round(float(x))) for x in np.asarray(resize_row).tolist()]
    if len(vals) < 6:
        out = patch_img.resize((w_gt, h_gt), Image.NEAREST)
        return np.asarray(out, dtype=np.uint8) > 128

    resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = vals[:6]
    if min(resized_h, resized_w, canvas_h, canvas_w) <= 0:
        out = patch_img.resize((w_gt, h_gt), Image.NEAREST)
        return np.asarray(out, dtype=np.uint8) > 128

    canvas = patch_img.resize((canvas_w, canvas_h), Image.NEAREST)
    left = max(0, pad_left)
    top = max(0, pad_top)
    right = min(canvas_w, pad_left + resized_w)
    bottom = min(canvas_h, pad_top + resized_h)

    if right <= left or bottom <= top:
        out = patch_img.resize((w_gt, h_gt), Image.NEAREST)
        return np.asarray(out, dtype=np.uint8) > 128

    content = canvas.crop((left, top, right, bottom))
    out = content.resize((w_gt, h_gt), Image.NEAREST)
    return np.asarray(out, dtype=np.uint8) > 128


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Carregamento e alinhamento
# ---------------------------------------------------------------------------

def load_class_artifacts(class_dir: Path) -> dict[str, Any]:
    feat = load_npz(class_dir / "features.npz")
    cos = load_npz(class_dir / "cosaliency.npz")

    corr_path = class_dir / "correspondence_bg.npz"
    if not corr_path.exists():
        corr_path = class_dir / "correspondence.npz"
    corr = load_npz(corr_path)

    image_paths = [str(x) for x in feat["image_paths"].tolist()]
    image_sizes = np.asarray(feat["image_sizes"])
    resize_info = (
        np.asarray(feat["resize_info"])
        if "resize_info" in feat
        else None
    )

    labels = np.asarray(cos["cluster_labels"], dtype=np.int32)

    # O step3 atual pode salvar cluster_labels de duas formas:
    #   (N, H, W)  -> grid explícito
    #   (N, P)     -> patches achatados, com P = H*W
    # No PASCAL observado: (N, 1369), isto é, 37*37.
    if labels.ndim == 2:
        n_images, n_patches = labels.shape

        # Tenta recuperar o grid primeiro do cosaliency.npz e depois
        # do correspondence_bg/correspondence.
        patch_grid = None
        if "patch_grid" in cos:
            patch_grid = tuple(int(x) for x in np.asarray(cos["patch_grid"]).tolist())
        elif "patch_grid" in corr:
            patch_grid = tuple(int(x) for x in np.asarray(corr["patch_grid"]).tolist())

        if patch_grid is None or len(patch_grid) != 2:
            side = int(round(np.sqrt(n_patches)))
            if side * side != n_patches:
                raise ValueError(
                    "cluster_labels está achatado e não foi possível inferir "
                    f"patch_grid para P={n_patches}"
                )
            hp, wp = side, side
        else:
            hp, wp = patch_grid

        if hp * wp != n_patches:
            raise ValueError(
                f"patch_grid={hp}x{wp} incompatível com "
                f"cluster_labels P={n_patches}"
            )

        labels = labels.reshape(n_images, hp, wp)

    elif labels.ndim == 3:
        n_images, hp, wp = labels.shape

    else:
        raise ValueError(
            "cluster_labels deveria ser (N,P) ou (N,H,W), "
            f"obtido {labels.shape}"
        )

    if n_images != len(image_paths):
        raise ValueError(
            f"N de cluster_labels={n_images} != image_paths={len(image_paths)}"
        )

    k = int(np.asarray(cos["n_clusters_used"]).item())
    cluster_is_fg = np.asarray(cos["cluster_is_fg"], dtype=bool).reshape(-1)
    if len(cluster_is_fg) != k:
        raise ValueError(
            f"cluster_is_fg tem {len(cluster_is_fg)} valores, esperado {k}"
        )

    # Score E2B por patch usado para recalcular a evidência de cada cluster.
    if "fg_scores_2d" in corr:
        e2b = np.asarray(corr["fg_scores_2d"], dtype=np.float64)
    elif "fg_scores" in corr:
        e2b = np.asarray(corr["fg_scores"], dtype=np.float64).reshape(
            n_images, hp, wp
        )
    else:
        raise KeyError(
            f"{corr_path} sem fg_scores_2d/fg_scores."
        )

    if e2b.shape != labels.shape:
        if e2b.size == labels.size:
            e2b = e2b.reshape(labels.shape)
        else:
            raise ValueError(
                f"E2B shape={e2b.shape} incompatível com labels={labels.shape}"
            )

    current_pixel = (
        np.asarray(cos["fg_mask_pixel"])
        if "fg_mask_pixel" in cos
        else None
    )

    stored = {}
    for key in [
        "cluster_scores",
        "cluster_scores_e2",
        "cluster_scores_sp",
        "cluster_scores_centrality",
        "cluster_scores_compactness",
    ]:
        if key in cos:
            arr = np.asarray(cos[key], dtype=np.float64).reshape(-1)
            stored[key] = arr if len(arr) == k else None
        else:
            stored[key] = None

    return {
        "image_paths": image_paths,
        "image_sizes": image_sizes,
        "resize_info": resize_info,
        "labels": labels,
        "e2b": e2b,
        "current_pixel": current_pixel,
        "cluster_is_fg": cluster_is_fg,
        "k": k,
        "stored": stored,
        "corr_path": str(corr_path),
        "cos_path": str(class_dir / "cosaliency.npz"),
    }


def load_gts(
    image_paths: list[str],
    gt_index: dict,
    class_name: str,
) -> tuple[list[np.ndarray], list[str]]:
    gt_multi = get_gt_index_multi(gt_index)
    mapping, not_found, ambiguous = build_gt_mapping(
        image_paths,
        gt_multi,
        collection_hint=class_name,
    )
    if not_found:
        raise RuntimeError(
            f"{class_name}: {len(not_found)} imagens sem GT."
        )
    if ambiguous:
        raise RuntimeError(
            f"{class_name}: {len(ambiguous)} imagens com GT ambíguo."
        )

    by_image = {str(k): str(v) for k, v in mapping.items()}
    gts = []
    gt_paths = []
    for p in image_paths:
        gt_path = Path(by_image[str(p)])
        gt = load_mask(gt_path).astype(bool)
        gts.append(gt)
        gt_paths.append(str(gt_path))
    return gts, gt_paths


# ---------------------------------------------------------------------------
# Estatísticas dos clusters
# ---------------------------------------------------------------------------

def compute_cluster_stats(
    labels: np.ndarray,
    e2b: np.ndarray,
    k: int,
    stored: dict[str, np.ndarray | None],
    border_width: int,
) -> list[dict[str, Any]]:
    n, h, w = labels.shape

    border = np.zeros((h, w), dtype=bool)
    bw = max(1, min(border_width, h // 2, w // 2))
    border[:bw, :] = True
    border[-bw:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True
    border3 = np.broadcast_to(border, labels.shape)

    rows = []
    total = labels.size

    for c in range(k):
        m = labels == c
        count = int(m.sum())
        if count == 0:
            e2_mean = 0.0
            border_frac = 1.0
            image_coverage = 0.0
        else:
            e2_mean = float(e2b[m].mean())
            border_frac = float(border3[m].mean())
            image_coverage = float(
                np.mean([np.any(labels[i] == c) for i in range(n)])
            )

        row = {
            "cluster_id": c,
            "patch_fraction": count / total,
            "image_coverage": image_coverage,
            "e2b_mean_recomputed": e2_mean,
            "border_fraction": border_frac,
        }

        for key, arr in stored.items():
            row[key] = (
                float(arr[c])
                if arr is not None and c < len(arr)
                else None
            )
        rows.append(row)

    return rows


def selector_sets(
    stats: list[dict[str, Any]],
    cluster_is_fg: np.ndarray,
) -> dict[str, set[int]]:
    k = len(stats)

    e2 = minmax(np.array([r["e2b_mean_recomputed"] for r in stats]))
    border_good = 1.0 - minmax(
        np.array([r["border_fraction"] for r in stats])
    )

    stored_sp = np.array([
        r["cluster_scores_sp"]
        if r["cluster_scores_sp"] is not None
        else 0.0
        for r in stats
    ])
    sp = minmax(stored_sp)

    stored_final = np.array([
        r["cluster_scores"]
        if r["cluster_scores"] is not None
        else 0.0
        for r in stats
    ])
    final = minmax(stored_final)

    scores = {
        "e2b": e2,
        "final": final,
        "e2b_sp": 0.7 * e2 + 0.3 * sp,
        "e2b_border": 0.7 * e2 + 0.3 * border_good,
        "hybrid": 0.6 * e2 + 0.2 * sp + 0.2 * border_good,
    }

    out: dict[str, set[int]] = {
        "current_cluster": set(
            int(i) for i in np.flatnonzero(cluster_is_fg)
        )
    }

    for name, score in scores.items():
        order = np.argsort(-score)
        out[f"{name}_top1"] = {int(order[0])}
        out[f"{name}_top2"] = set(int(x) for x in order[: min(2, k)])

    return out


# ---------------------------------------------------------------------------
# Avaliação dos seletores/oracles
# ---------------------------------------------------------------------------

def masks_for_cluster_set(
    labels: np.ndarray,
    selected: set[int],
    gts: list[np.ndarray],
    resize_info: np.ndarray | None,
) -> list[np.ndarray]:
    out = []
    for i in range(len(gts)):
        patch = np.isin(labels[i], list(selected))
        rr = resize_info[i] if resize_info is not None else None
        out.append(patch_mask_to_gt(patch, gts[i].shape, rr))
    return out


def mean_iou(masks: list[np.ndarray], gts: list[np.ndarray]) -> float:
    return float(np.mean([iou(p, g) for p, g in zip(masks, gts)]))


def per_image_ious(
    masks: list[np.ndarray],
    gts: list[np.ndarray],
) -> list[float]:
    return [iou(p, g) for p, g in zip(masks, gts)]


def current_pixel_masks(
    pixel: np.ndarray | None,
    gts: list[np.ndarray],
    image_sizes: np.ndarray,
) -> list[np.ndarray] | None:
    if pixel is None:
        return None
    if len(pixel) != len(gts):
        return None

    masks = []
    for i, gt in enumerate(gts):
        arr = np.asarray(pixel[i])
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        h_i, w_i = [int(x) for x in image_sizes[i][:2]]
        arr = arr[:h_i, :w_i]
        masks.append(resize_bool(arr > 0.5, gt.shape))
    return masks


def global_oracle(
    labels: np.ndarray,
    gts: list[np.ndarray],
    resize_info: np.ndarray | None,
    k: int,
    max_union: int,
) -> dict[str, Any]:
    results = {}

    for r in range(1, min(max_union, k) + 1):
        best_score = -1.0
        best_set: tuple[int, ...] | None = None

        for combo in itertools.combinations(range(k), r):
            masks = masks_for_cluster_set(
                labels,
                set(combo),
                gts,
                resize_info,
            )
            score = mean_iou(masks, gts)
            if score > best_score:
                best_score = score
                best_set = combo

        results[f"oracle_global_{r}"] = {
            "iou": best_score,
            "clusters": list(best_set) if best_set else [],
        }

    return results


def per_image_oracle(
    labels: np.ndarray,
    gts: list[np.ndarray],
    resize_info: np.ndarray | None,
    k: int,
    max_union: int,
) -> dict[str, Any]:
    n = len(gts)
    per_r: dict[int, list[float]] = {
        r: [] for r in range(1, min(max_union, k) + 1)
    }
    best_sets: dict[int, list[list[int]]] = {
        r: [] for r in per_r
    }

    # Pré-calcula a máscara GT-space de cada cluster por imagem.
    cluster_masks: list[list[np.ndarray]] = []
    for i in range(n):
        rr = resize_info[i] if resize_info is not None else None
        per_cluster = []
        for c in range(k):
            per_cluster.append(
                patch_mask_to_gt(
                    labels[i] == c,
                    gts[i].shape,
                    rr,
                )
            )
        cluster_masks.append(per_cluster)

    for i in range(n):
        for r in per_r:
            best = -1.0
            best_combo: tuple[int, ...] | None = None
            for combo in itertools.combinations(range(k), r):
                pred = np.zeros_like(gts[i], dtype=bool)
                for c in combo:
                    pred |= cluster_masks[i][c]
                score = iou(pred, gts[i])
                if score > best:
                    best = score
                    best_combo = combo
            per_r[r].append(best)
            best_sets[r].append(list(best_combo) if best_combo else [])

    out = {}
    for r, vals in per_r.items():
        out[f"oracle_per_image_{r}"] = {
            "iou_mean": float(np.mean(vals)),
            "iou_std": float(np.std(vals)),
            "best_sets": best_sets[r],
        }
    return out


def classify_diagnosis(
    current: float,
    global_best: float,
    per_image_best: float,
) -> str:
    # Heurística diagnóstica. GT é usado somente para categorizar a falha.
    if current >= 0.45:
        return "e3_reasonable"
    if global_best >= 0.45 and global_best - current >= 0.15:
        return "selection_failure"
    if per_image_best >= 0.45 and global_best < 0.35:
        return "cluster_consistency_failure"
    if per_image_best < 0.30:
        return "clustering_failure"
    if per_image_best - global_best >= 0.15:
        return "mixed_consistency_failure"
    return "intermediate"


# ---------------------------------------------------------------------------
# Classe
# ---------------------------------------------------------------------------

def analyze_class(
    features_root: Path,
    gt_index: dict,
    class_name: str,
    border_width: int,
    oracle_max_union: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    class_dir = features_root / safe_name(class_name)
    a = load_class_artifacts(class_dir)
    gts, gt_paths = load_gts(a["image_paths"], gt_index, class_name)

    stats = compute_cluster_stats(
        a["labels"],
        a["e2b"],
        a["k"],
        a["stored"],
        border_width,
    )

    selectors = selector_sets(stats, a["cluster_is_fg"])
    selector_results: dict[str, dict[str, Any]] = {}
    selector_masks: dict[str, list[np.ndarray]] = {}

    for name, selected in selectors.items():
        masks = masks_for_cluster_set(
            a["labels"],
            selected,
            gts,
            a["resize_info"],
        )
        selector_masks[name] = masks
        selector_results[name] = {
            "iou_mean": mean_iou(masks, gts),
            "selected_clusters": sorted(selected),
            "fg_fraction_patch": float(
                np.mean(np.isin(a["labels"], list(selected)))
            ),
        }

    pix_masks = current_pixel_masks(
        a["current_pixel"],
        gts,
        a["image_sizes"],
    )
    if pix_masks is not None:
        selector_masks["current_pixel"] = pix_masks
        selector_results["current_pixel"] = {
            "iou_mean": mean_iou(pix_masks, gts),
            "selected_clusters": None,
            "fg_fraction_patch": None,
        }

    # Fallback adaptativo: só troca se a máscara cluster atual colapsar.
    current_fg_fraction = selector_results["current_cluster"][
        "fg_fraction_patch"
    ]
    for threshold in [0.01, 0.02, 0.05]:
        name = f"fallback_e2b_t{int(threshold*1000):03d}"
        if current_fg_fraction < threshold:
            base = "e2b_top2"
        else:
            base = "current_cluster"
        selector_masks[name] = selector_masks[base]
        selector_results[name] = {
            "iou_mean": selector_results[base]["iou_mean"],
            "selected_clusters": selector_results[base]["selected_clusters"],
            "fg_fraction_patch": selector_results[base]["fg_fraction_patch"],
            "fallback_triggered": base != "current_cluster",
        }

    glob = global_oracle(
        a["labels"],
        gts,
        a["resize_info"],
        a["k"],
        oracle_max_union,
    )
    per_img_oracle = per_image_oracle(
        a["labels"],
        gts,
        a["resize_info"],
        a["k"],
        oracle_max_union,
    )

    current_iou = selector_results.get(
        "current_pixel",
        selector_results["current_cluster"],
    )["iou_mean"]

    global_best = max(
        v["iou"] for v in glob.values()
    )
    per_image_best = max(
        v["iou_mean"] for v in per_img_oracle.values()
    )

    summary = {
        "class": class_name,
        "n_images": len(gts),
        "n_clusters": a["k"],
        "correspondence_source": a["corr_path"],
        "cosaliency_path": a["cos_path"],
        "current_pixel_iou": selector_results.get(
            "current_pixel", {}
        ).get("iou_mean"),
        "current_cluster_iou": selector_results[
            "current_cluster"
        ]["iou_mean"],
        "current_fg_fraction_patch": current_fg_fraction,
        "oracle_global_best_iou": global_best,
        "oracle_per_image_best_iou": per_image_best,
        "diagnosis": classify_diagnosis(
            current_iou,
            global_best,
            per_image_best,
        ),
    }

    for key, value in glob.items():
        summary[f"{key}_iou"] = value["iou"]
        summary[f"{key}_clusters"] = ",".join(
            str(x) for x in value["clusters"]
        )

    for key, value in per_img_oracle.items():
        summary[f"{key}_iou"] = value["iou_mean"]

    for name, result in selector_results.items():
        summary[f"selector_{name}_iou"] = result["iou_mean"]

    # Diagnóstico por cluster.
    cluster_rows = []
    for row in stats:
        c = row["cluster_id"]
        cmasks = masks_for_cluster_set(
            a["labels"],
            {c},
            gts,
            a["resize_info"],
        )
        r = {
            "class": class_name,
            **row,
            "mean_iou_cluster_alone": mean_iou(cmasks, gts),
            "selected_current": bool(a["cluster_is_fg"][c]),
        }
        cluster_rows.append(r)

    # Resultado por imagem para os seletores principais.
    per_image_rows = []
    names_for_per_image = [
        "current_pixel",
        "current_cluster",
        "e2b_top1",
        "e2b_top2",
        "e2b_sp_top1",
        "e2b_sp_top2",
        "e2b_border_top1",
        "e2b_border_top2",
        "hybrid_top1",
        "hybrid_top2",
        "fallback_e2b_t010",
        "fallback_e2b_t020",
        "fallback_e2b_t050",
    ]
    names_for_per_image = [
        n for n in names_for_per_image if n in selector_masks
    ]

    oracle_pi_key = f"oracle_per_image_{min(oracle_max_union, a['k'])}"
    oracle_pi_vals = per_img_oracle[oracle_pi_key]["best_sets"]

    for i, img_path in enumerate(a["image_paths"]):
        row = {
            "class": class_name,
            "index": i,
            "image_name": Path(img_path).name,
            "image_path": img_path,
            "gt_path": gt_paths[i],
        }
        for name in names_for_per_image:
            row[f"iou_{name}"] = iou(
                selector_masks[name][i],
                gts[i],
            )

        # melhores IoUs por imagem para 1..max_union
        for r in range(1, min(oracle_max_union, a["k"]) + 1):
            key = f"oracle_per_image_{r}"
            row[f"iou_{key}"] = per_img_oracle[key]["best_sets"][i]
            # acima guarda conjunto; corrigimos abaixo com score recalculado
            combo = per_img_oracle[key]["best_sets"][i]
            pred = np.zeros_like(gts[i], dtype=bool)
            rr = a["resize_info"][i] if a["resize_info"] is not None else None
            for c in combo:
                pred |= patch_mask_to_gt(
                    a["labels"][i] == c,
                    gts[i].shape,
                    rr,
                )
            row[f"iou_{key}"] = iou(pred, gts[i])
            row[f"clusters_{key}"] = ",".join(str(c) for c in combo)

        per_image_rows.append(row)

    return summary, cluster_rows, per_image_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnóstico de seleção/clustering da E3."
    )
    p.add_argument("--features-root", required=True, type=Path)
    p.add_argument("--gt-index", required=True, type=Path)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--border-width", type=int, default=2)
    p.add_argument(
        "--oracle-max-union",
        type=int,
        default=3,
        help="Testa oracle global e por imagem com união de até N clusters.",
    )
    args = p.parse_args()

    features_root = args.features_root.resolve()
    gt_index_path = args.gt_index.resolve()
    output = args.output.resolve()

    if not features_root.exists():
        raise SystemExit(
            f"[ERRO] features-root inexistente: {features_root}"
        )
    if not gt_index_path.exists():
        raise SystemExit(
            f"[ERRO] gt-index inexistente: {gt_index_path}"
        )

    gt_index = load_gt_index(gt_index_path)

    summaries = []
    clusters = []
    per_image = []

    print("=" * 118)
    print("ABLAÇÃO E3 — CLUSTER ORACLE + SELETORES NÃO SUPERVISIONADOS")
    print("=" * 118)
    print(f"Features root : {features_root}")
    print(f"GT index      : {gt_index_path}")
    print(f"Classes       : {', '.join(args.classes)}")
    print(f"Oracle união  : até {args.oracle_max_union} clusters")

    for class_name in args.classes:
        print("\n" + "-" * 118)
        print(f"Classe: {class_name}")
        print("-" * 118)
        try:
            summary, crows, prows = analyze_class(
                features_root,
                gt_index,
                class_name,
                args.border_width,
                args.oracle_max_union,
            )
        except Exception as exc:
            print(f"[ERRO] {class_name}: {exc}")
            continue

        summaries.append(summary)
        clusters.extend(crows)
        per_image.extend(prows)

        print(
            f"  current_pixel          : "
            f"{summary.get('current_pixel_iou')}"
        )
        print(
            f"  current_cluster        : "
            f"{summary['current_cluster_iou']:.4f}"
        )
        print(
            f"  FG atual (patch frac.) : "
            f"{summary['current_fg_fraction_patch']:.2%}"
        )
        print(
            f"  oracle global best     : "
            f"{summary['oracle_global_best_iou']:.4f}"
        )
        print(
            f"  oracle per-image best  : "
            f"{summary['oracle_per_image_best_iou']:.4f}"
        )
        print(f"  diagnóstico            : {summary['diagnosis']}")
        print()
        for name in [
            "e2b_top1",
            "e2b_top2",
            "e2b_sp_top1",
            "e2b_sp_top2",
            "e2b_border_top1",
            "e2b_border_top2",
            "hybrid_top1",
            "hybrid_top2",
            "fallback_e2b_t010",
            "fallback_e2b_t020",
            "fallback_e2b_t050",
        ]:
            key = f"selector_{name}_iou"
            if key in summary:
                print(f"  {name:<24}: {summary[key]:.4f}")

    output.mkdir(parents=True, exist_ok=True)

    summary_csv = output / "e3_selector_summary.csv"
    cluster_csv = output / "e3_cluster_diagnostics.csv"
    per_image_csv = output / "e3_per_image.csv"
    json_path = output / "e3_results.json"

    write_csv(summary_csv, summaries)
    write_csv(cluster_csv, clusters)
    write_csv(per_image_csv, per_image)

    json_path.write_text(
        json.dumps(
            {
                "features_root": str(features_root),
                "gt_index": str(gt_index_path),
                "classes": summaries,
                "cluster_diagnostics": clusters,
                "per_image": per_image,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 118)
    print("RESUMO")
    print("=" * 118)
    print(
        f"{'Classe':<20}"
        f"{'Current':>10}"
        f"{'G.Orcl':>10}"
        f"{'PI.Orcl':>10}"
        f"{'E2B1':>10}"
        f"{'E2B2':>10}"
        f"{'Hybrid2':>10}"
        f"  Diagnóstico"
    )
    print("-" * 118)

    for s in summaries:
        current = (
            s["current_pixel_iou"]
            if s["current_pixel_iou"] is not None
            else s["current_cluster_iou"]
        )
        print(
            f"{s['class']:<20}"
            f"{current:>10.4f}"
            f"{s['oracle_global_best_iou']:>10.4f}"
            f"{s['oracle_per_image_best_iou']:>10.4f}"
            f"{s.get('selector_e2b_top1_iou', 0.0):>10.4f}"
            f"{s.get('selector_e2b_top2_iou', 0.0):>10.4f}"
            f"{s.get('selector_hybrid_top2_iou', 0.0):>10.4f}"
            f"  {s['diagnosis']}"
        )

    print(f"\n[Salvo] {summary_csv}")
    print(f"[Salvo] {cluster_csv}")
    print(f"[Salvo] {per_image_csv}")
    print(f"[Salvo] {json_path}")


if __name__ == "__main__":
    main()
