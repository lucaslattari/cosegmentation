#!/usr/bin/env python3
"""
Ablation E2 — Adaptive recurring-background suppression
=======================================================

Objetivo
--------
Testar uma correção pós-E2 / pré-E3 que penaliza patches parecidos com um
background recorrente estimado a partir das bordas das imagens, preservando
o pipeline baseline (spatial + degree_alpha=0.10).

A ideia é atacar falhas como:
- céu azul recorrente em Airshows;
- gramado recorrente em Woman Soccer;
- chão recorrente em Monks;

sem aplicar uma penalização global fixa. A força efetiva é modulada por:
1) consistência cross-image das bordas (background_confidence);
2) sobreposição entre os patches FG mais confiáveis e a borda
   (fg_border_fraction), reduzindo a penalização quando o FG parece tocar
   as bordas com frequência.

Variantes padrão
----------------
baseline
recurrence_l0p25
recurrence_l0p50
multi_k2_l0p25
multi_k2_l0p50

Classes alvo
------------
- Woman Soccer Players-Soccer-Dustin Hall-white
- Airshows-helicopter
- Monks-LAO PDR 2008-Rolandito
- Airshows-planes
- Airshows-Huntsville Airshow Flying-ocreationsphotography (finally back after computer

Controles fortes / anti-regressão
---------------------------------
- Ferrari 599 GTB Fiorano-599-auTomoTiVe FreAk !
- Kendo-EKC 2008 Helsinki-SanSuiSo
- Gymnastics
- Elephants-safari June 2008-timd2000
- Statue of Liberty-Liberty Enlightening the World (Groups)-EricaJoy
- Stonehenge
- Red Sox Players - Red Sox Yankees Game-JB Imaging
- Speedskating-Worldcup Final 2008-Guido Jansen

Uso típico
----------
A partir da raiz do projeto:

python ablation/ablation_e2_background_suppression.py --run-downstream

Por padrão, o script usa como baseline:
  features/icoseg_full_e2_spatial_degree_a0p10
  results/icoseg_full_e2_spatial_degree_a0p10

Saídas:
  ablation_results/e2_background_suppression/
    prepared/<variant>/<classe>/
    results/<variant>/<classe>/metrics.json
    ablation_summary.csv
    ablation_summary.json

Para só gerar correspondence.npz corrigidos, sem E3→E5:
  python ablation/ablation_e2_background_suppression.py --prepare-only

Para rodar apenas as classes-alvo:
  python ablation/ablation_e2_background_suppression.py \
      --run-downstream --group target

Notas
-----
- O baseline não é recalculado: as métricas existentes do baseline são lidas
  diretamente de results/icoseg_full_e2_spatial_degree_a0p10.
- As variantes alteram somente fg_scores/fg_scores_2d do correspondence.npz.
  Todos os demais campos da E2 são preservados.
- O experimento é deliberadamente separado do pipeline principal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception as exc:
    raise SystemExit(
        "scikit-learn é necessário para a variante multi-prototype. "
        "Instale/ative o mesmo ambiente usado pelo pipeline.\n"
        f"Erro original: {exc}"
    )


# ---------------------------------------------------------------------------
# Configuração experimental
# ---------------------------------------------------------------------------

TARGET_CLASSES = [
    "Woman Soccer Players-Soccer-Dustin Hall-white",
    "Airshows-helicopter",
    "Monks-LAO PDR 2008-Rolandito",
    "Airshows-planes",
    "Airshows-Huntsville Airshow Flying-ocreationsphotography (finally back after computer",
]

CONTROL_CLASSES = [
    "Ferrari 599 GTB Fiorano-599-auTomoTiVe FreAk !",
    "Kendo-EKC 2008 Helsinki-SanSuiSo",
    "Gymnastics",
    "Elephants-safari June 2008-timd2000",
    "Statue of Liberty-Liberty Enlightening the World (Groups)-EricaJoy",
    "Stonehenge",
    "Red Sox Players - Red Sox Yankees Game-JB Imaging",
    "Speedskating-Worldcup Final 2008-Guido Jansen",
]


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str          # baseline | recurrence | multi
    lam: float = 0.0
    k: int = 1


VARIANTS = [
    Variant("baseline", "baseline", 0.0, 1),
    Variant("recurrence_l0p25", "recurrence", 0.25, 1),
    Variant("recurrence_l0p50", "recurrence", 0.50, 1),
    Variant("multi_k2_l0p25", "multi", 0.25, 2),
    Variant("multi_k2_l0p50", "multi", 0.50, 2),
]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def safe_name(class_name: str) -> str:
    """Replica a sanitização usada pelo run_pipeline*.sh."""
    return "".join(ch for ch in class_name.replace(" ", "_") if ch.isalnum() or ch in "_-")


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


def robust_01(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Normalização robusta por imagem usando percentis 5–95."""
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        xi = x[i]
        valid = np.ones_like(xi, dtype=bool) if mask is None else mask[i].astype(bool)
        vals = xi[valid]
        if vals.size == 0:
            continue
        lo, hi = np.percentile(vals, [5, 95])
        if hi <= lo + 1e-8:
            out[i, valid] = 0.0
        else:
            out[i, valid] = np.clip((xi[valid] - lo) / (hi - lo), 0.0, 1.0)
    return out


def border_mask(grid: tuple[int, int], width: int) -> np.ndarray:
    h, w = grid
    width = max(1, min(width, h // 2, w // 2))
    m = np.zeros((h, w), dtype=bool)
    m[:width, :] = True
    m[-width:, :] = True
    m[:, :width] = True
    m[:, -width:] = True
    return m.reshape(-1)


def content_mask_from_resize_info(
    resize_info: np.ndarray | None,
    grid: tuple[int, int],
) -> np.ndarray | None:
    """
    Aproxima a máscara de conteúdo usada na E2.
    resize_info: [resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w].
    Um patch é válido quando o centro cai dentro da região da imagem real.
    """
    if resize_info is None:
        return None

    h_p, w_p = grid
    masks = []
    for row in resize_info:
        resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = map(float, row)
        ys = (np.arange(h_p) + 0.5) * (canvas_h / h_p)
        xs = (np.arange(w_p) + 0.5) * (canvas_w / w_p)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        valid = (
            (yy >= pad_top)
            & (yy < pad_top + resized_h)
            & (xx >= pad_left)
            & (xx < pad_left + resized_w)
        )
        masks.append(valid.reshape(-1))
    return np.stack(masks, axis=0)


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def save_npz_dict(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def run_command(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("  ▶", " ".join(map(str, cmd)))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_iou(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        agg = data.get("aggregate", {})
        iou = agg.get("iou", {})
        if isinstance(iou, dict):
            return float(iou.get("mean"))
        if iou is not None:
            return float(iou)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Background recorrente
# ---------------------------------------------------------------------------

def border_centroids(
    patch_features: np.ndarray,
    border: np.ndarray,
    content_mask: np.ndarray | None,
) -> np.ndarray:
    """
    Centróide L2-normalizado da borda de cada imagem.
    Serve como representação barata do fundo periférico de cada imagem.
    """
    n, p, d = patch_features.shape
    centroids = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        valid = border.copy()
        if content_mask is not None:
            valid &= content_mask[i]
        feats = patch_features[i, valid]
        if feats.size == 0:
            feats = patch_features[i]
        c = feats.mean(axis=0)
        centroids[i] = c / max(np.linalg.norm(c), 1e-12)
    return centroids


def background_confidence(centroids: np.ndarray) -> float:
    """
    Consistência cross-image das bordas.

    A confiança é baseada na média de similaridade cosseno off-diagonal.
    Fazemos um mapeamento suave:
      mean_sim <= 0.20 -> 0
      mean_sim >= 0.80 -> 1
    """
    if len(centroids) < 2:
        return 0.0
    sim = centroids @ centroids.T
    vals = sim[np.triu_indices(len(centroids), k=1)]
    mean_sim = float(vals.mean()) if vals.size else 0.0
    return float(np.clip((mean_sim - 0.20) / 0.60, 0.0, 1.0))


def border_recurrence_weights(
    patch_features: np.ndarray,
    centroids: np.ndarray,
    border: np.ndarray,
    content_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mede recorrência de cada patch de borda comparando-o aos centróides de
    borda das OUTRAS imagens.

    Retorna:
      border_feats   : (M, D)
      weights        : (M,), recorrência [0,1]

    É uma aproximação eficiente de "patch de borda recorrente entre imagens":
    evita comparar todos os patches de borda contra todos os patches de todas
    as outras imagens, mas preserva o caráter cross-image.
    """
    n, p, d = patch_features.shape
    all_feats = []
    all_scores = []

    for i in range(n):
        valid = border.copy()
        if content_mask is not None:
            valid &= content_mask[i]
        feats = patch_features[i, valid]
        if feats.size == 0:
            continue

        if n > 1:
            other = np.delete(centroids, i, axis=0)
            sims = feats @ other.T
            # Um patch é recorrente quando é parecido com o background de
            # várias outras imagens. Média dos top-3 centróides disponíveis.
            k = min(3, sims.shape[1])
            top = np.partition(sims, -k, axis=1)[:, -k:]
            score = top.mean(axis=1)
        else:
            score = feats @ centroids[i]

        all_feats.append(feats)
        all_scores.append(score)

    if not all_feats:
        return (
            np.empty((0, patch_features.shape[-1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    feats = np.concatenate(all_feats, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    # Mapeia similaridade para [0,1] com percentis robustos.
    lo, hi = np.percentile(scores, [10, 90])
    if hi <= lo + 1e-8:
        weights = np.ones_like(scores, dtype=np.float32)
    else:
        weights = np.clip((scores - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    return feats, weights


def build_background_prototypes(
    border_feats: np.ndarray,
    recurrence_weights: np.ndarray,
    mode: str,
    k: int,
    random_state: int = 0,
) -> np.ndarray:
    """
    recurrence:
      1 protótipo = média ponderada pela recorrência.

    multi:
      KMeans ponderado indiretamente por amostragem dos patches mais recorrentes.
      Usamos os 60% patches de borda com maior recorrência para reduzir ruído.
    """
    if border_feats.shape[0] == 0:
        raise RuntimeError("Nenhum patch de borda válido para construir background.")

    if mode == "recurrence":
        w = recurrence_weights + 1e-6
        proto = np.average(border_feats, axis=0, weights=w)
        return l2_normalize(proto[None, :], axis=1).astype(np.float32)

    if mode == "multi":
        threshold = np.quantile(recurrence_weights, 0.40)
        selected = border_feats[recurrence_weights >= threshold]
        if len(selected) < k:
            selected = border_feats
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(selected)
        return l2_normalize(model.cluster_centers_.astype(np.float32), axis=1)

    raise ValueError(f"Modo desconhecido: {mode}")


def fg_border_fraction(
    fg_scores: np.ndarray,
    border: np.ndarray,
    content_mask: np.ndarray | None,
    top_fraction: float = 0.10,
) -> float:
    """
    Fração dos patches FG mais confiáveis que cai na borda.
    Se alta, reduzimos a supressão para proteger classes cujo objeto toca bordas.
    """
    fractions = []
    for i in range(fg_scores.shape[0]):
        valid = np.ones(fg_scores.shape[1], dtype=bool)
        if content_mask is not None:
            valid &= content_mask[i]
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        n_top = max(1, int(math.ceil(top_fraction * idx.size)))
        vals = fg_scores[i, idx]
        top_idx = idx[np.argpartition(vals, -n_top)[-n_top:]]
        fractions.append(float(border[top_idx].mean()))
    return float(np.mean(fractions)) if fractions else 0.0


def suppress_background(
    patch_features: np.ndarray,
    fg_scores: np.ndarray,
    prototypes: np.ndarray,
    border: np.ndarray,
    content_mask: np.ndarray | None,
    lam: float,
    bg_conf: float,
    adaptive: bool = True,
    top_fg_fraction: float = 0.10,
) -> tuple[np.ndarray, dict]:
    """
    s'_fg = s_fg - lambda_eff * bg_score

    bg_score é a máxima similaridade com os protótipos de background,
    robustamente normalizada por imagem.

    lambda_eff = lambda * bg_confidence * (1 - fg_border_fraction)
    quando adaptive=True.
    """
    bg_raw = np.einsum("npd,kd->npk", patch_features, prototypes).max(axis=2)
    bg_score = robust_01(bg_raw, content_mask)

    fg_border = fg_border_fraction(
        fg_scores, border, content_mask, top_fraction=top_fg_fraction
    )

    if adaptive:
        lambda_eff = lam * bg_conf * max(0.0, 1.0 - fg_border)
    else:
        lambda_eff = lam

    corrected = fg_scores.astype(np.float32) - float(lambda_eff) * bg_score

    # Mantém padding em zero e reescala por imagem para [0,1].
    corrected = robust_01(corrected, content_mask)
    if content_mask is not None:
        corrected = np.where(content_mask, corrected, 0.0)

    diagnostics = {
        "lambda_requested": float(lam),
        "lambda_effective": float(lambda_eff),
        "background_confidence": float(bg_conf),
        "fg_border_fraction": float(fg_border),
        "bg_score_mean": float(bg_score.mean()),
        "bg_score_std": float(bg_score.std()),
    }
    return corrected.astype(np.float32), diagnostics


# ---------------------------------------------------------------------------
# Preparação de uma classe/variante
# ---------------------------------------------------------------------------

def prepare_variant(
    class_name: str,
    variant: Variant,
    baseline_features_root: Path,
    output_root: Path,
    border_width: int,
    adaptive: bool,
    top_fg_fraction: float,
    force: bool,
) -> tuple[Path, dict]:
    sname = safe_name(class_name)
    src_dir = baseline_features_root / sname
    features_path = src_dir / "features.npz"
    corr_path = src_dir / "correspondence.npz"

    if not features_path.exists():
        raise FileNotFoundError(f"features.npz ausente: {features_path}")
    if not corr_path.exists():
        raise FileNotFoundError(f"correspondence.npz ausente: {corr_path}")

    dst_dir = output_root / "prepared" / variant.name / sname
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_features = dst_dir / "features.npz"
    dst_corr = dst_dir / "correspondence.npz"
    diag_path = dst_dir / "background_suppression.json"

    if dst_corr.exists() and diag_path.exists() and not force:
        print(f"  ↷ preparado já existe: {dst_corr}")
        return dst_dir, json.loads(diag_path.read_text(encoding="utf-8"))

    # Symlink relativo/absoluto quando possível; fallback para cópia.
    if not dst_features.exists():
        try:
            dst_features.symlink_to(features_path.resolve())
        except OSError:
            shutil.copy2(features_path, dst_features)

    feat = load_npz_dict(features_path)
    corr = load_npz_dict(corr_path)

    patch_features = feat["patch_features"].astype(np.float32)
    patch_features = l2_normalize(patch_features, axis=2).astype(np.float32)

    fg_scores = corr["fg_scores"].astype(np.float32)
    grid = tuple(int(x) for x in corr["patch_grid"])
    h_p, w_p = grid

    resize_info = feat.get("resize_info")
    content_mask = content_mask_from_resize_info(resize_info, grid)
    bmask = border_mask(grid, border_width)

    if variant.mode == "baseline":
        corrected = fg_scores.copy()
        diagnostics = {
            "variant": variant.name,
            "class": class_name,
            "mode": "baseline",
            "border_width": border_width,
            "lambda_requested": 0.0,
            "lambda_effective": 0.0,
            "background_confidence": None,
            "fg_border_fraction": None,
            "n_prototypes": 0,
        }
    else:
        centroids = border_centroids(patch_features, bmask, content_mask)
        bg_conf = background_confidence(centroids)
        bfeats, weights = border_recurrence_weights(
            patch_features, centroids, bmask, content_mask
        )
        prototypes = build_background_prototypes(
            bfeats, weights, variant.mode, variant.k
        )
        corrected, suppression_diag = suppress_background(
            patch_features=patch_features,
            fg_scores=fg_scores,
            prototypes=prototypes,
            border=bmask,
            content_mask=content_mask,
            lam=variant.lam,
            bg_conf=bg_conf,
            adaptive=adaptive,
            top_fg_fraction=top_fg_fraction,
        )
        diagnostics = {
            "variant": variant.name,
            "class": class_name,
            "mode": variant.mode,
            "border_width": border_width,
            "adaptive": adaptive,
            "top_fg_fraction": top_fg_fraction,
            "n_prototypes": int(len(prototypes)),
            **suppression_diag,
        }

    out_corr = dict(corr)
    out_corr["fg_scores"] = corrected
    out_corr["fg_scores_2d"] = corrected.reshape(corrected.shape[0], h_p, w_p)

    # Metadados extras não quebram consumidores que ignoram chaves desconhecidas.
    out_corr["bg_suppression_variant"] = np.array(variant.name)
    out_corr["bg_suppression_lambda"] = np.array(variant.lam, dtype=np.float32)
    out_corr["bg_suppression_adaptive"] = np.array(adaptive)

    save_npz_dict(dst_corr, out_corr)
    diag_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"  ✓ {variant.name}: λ_req={diagnostics.get('lambda_requested', 0):.3f} "
        f"λ_eff={diagnostics.get('lambda_effective', 0):.3f} "
        f"bg_conf={diagnostics.get('background_confidence')}"
    )
    return dst_dir, diagnostics


# ---------------------------------------------------------------------------
# E3 → E5 + avaliação
# ---------------------------------------------------------------------------

def run_downstream(
    project_root: Path,
    prepared_dir: Path,
    result_dir: Path,
    class_name: str,
    python: str,
    sam_model: str,
    force: bool,
    dry_run: bool,
) -> None:
    pipeline = project_root / "pipeline"
    feat = prepared_dir / "features.npz"
    corr = prepared_dir / "correspondence.npz"
    cosal = prepared_dir / "cosaliency.npz"
    refined = prepared_dir / "refined_masks.npz"
    iterative = prepared_dir / "iterative_masks.npz"

    step3 = pipeline / "step3_cosaliency.py"
    step4 = pipeline / "step4_sam_refinement.py"
    step5 = pipeline / "step5_iterative.py"
    evaluate = pipeline / "evaluate.py"
    gt_index = project_root / "gt_index.json"

    required = [step3, step4, step5, evaluate, gt_index]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Arquivos necessários ausentes: " + ", ".join(map(str, missing)))

    # E3
    if force or not cosal.exists():
        run_command([
            python, str(step3),
            "--features", str(feat),
            "--correspondence", str(corr),
            "--output", str(prepared_dir),
            "--n_clusters", "8",
            "--k_neighbors", "15",
            "--n_spectral", "8",
            "--saliency_alpha", "0.5",
            "--fg_decision_mode", "saliency",
        ], cwd=project_root, dry_run=dry_run)
    else:
        print("  ↷ E3 já existe")

    # E4
    if force or not refined.exists() or not (prepared_dir / "segmented_e4").exists():
        run_command([
            python, str(step4),
            "--features", str(feat),
            "--cosaliency", str(cosal),
            "--output", str(prepared_dir),
            "--model", sam_model,
            "--prompt_mode", "box_points_mask",
            "--n_iter", "2",
            "--export",
        ], cwd=project_root, dry_run=dry_run)
    else:
        print("  ↷ E4 já existe")

    # E5
    if force or not iterative.exists() or not (prepared_dir / "segmented_e5").exists():
        run_command([
            python, str(step5),
            "--features", str(feat),
            "--refined", str(refined),
            "--output", str(prepared_dir),
            "--model", sam_model,
            "--prompt_mode", "box_points_mask",
            "--max_iter", "20",
            "--converge_thresh", "0.99",
            "--n_fg_pts", "8",
            "--n_bg_pts", "8",
            "--no_mask_history",
            "--export",
        ], cwd=project_root, dry_run=dry_run)
    else:
        print("  ↷ E5 já existe")

    # Avaliação
    metrics = result_dir / "metrics.json"
    if force or not metrics.exists():
        result_dir.mkdir(parents=True, exist_ok=True)
        run_command([
            python, str(evaluate),
            "--masks_dir", str(prepared_dir / "segmented_e5"),
            "--gt_index", str(gt_index),
            "--output", str(result_dir),
            "--collection", class_name,
        ], cwd=project_root, dry_run=dry_run)
    else:
        print("  ↷ avaliação já existe")


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------

def build_summary(
    classes: list[tuple[str, str]],
    variants: list[Variant],
    baseline_results_root: Path,
    ablation_results_root: Path,
    output_root: Path,
) -> None:
    rows = []

    baseline_iou = {}
    for class_name, group in classes:
        sname = safe_name(class_name)
        iou = read_iou(baseline_results_root / sname / "metrics.json")
        baseline_iou[class_name] = iou

    for variant in variants:
        for class_name, group in classes:
            sname = safe_name(class_name)
            if variant.mode == "baseline":
                iou = baseline_iou[class_name]
            else:
                iou = read_iou(
                    ablation_results_root / variant.name / sname / "metrics.json"
                )
            base = baseline_iou[class_name]
            delta = None if iou is None or base is None else iou - base
            rows.append({
                "variant": variant.name,
                "group": group,
                "class": class_name,
                "iou": iou,
                "baseline_iou": base,
                "delta_iou": delta,
            })

    # Agregados por variante/grupo.
    aggregate = []
    for variant in variants:
        for group in ("target", "control", "all"):
            selected = [
                r for r in rows
                if r["variant"] == variant.name
                and (group == "all" or r["group"] == group)
                and r["iou"] is not None
            ]
            if not selected:
                continue
            deltas = [r["delta_iou"] for r in selected if r["delta_iou"] is not None]
            aggregate.append({
                "variant": variant.name,
                "group": group,
                "n": len(selected),
                "mean_iou": float(np.mean([r["iou"] for r in selected])),
                "mean_delta_iou": float(np.mean(deltas)) if deltas else None,
                "n_improved_gt_0p02": sum(d > 0.02 for d in deltas),
                "n_worsened_lt_m0p02": sum(d < -0.02 for d in deltas),
                "worst_delta_iou": float(min(deltas)) if deltas else None,
                "best_delta_iou": float(max(deltas)) if deltas else None,
            })

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "ablation_summary.csv"
    json_path = output_root / "ablation_summary.json"
    agg_csv = output_root / "ablation_aggregate.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "group", "class", "iou", "baseline_iou", "delta_iou"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with agg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant", "group", "n", "mean_iou", "mean_delta_iou",
                "n_improved_gt_0p02", "n_worsened_lt_m0p02",
                "worst_delta_iou", "best_delta_iou",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate)

    json_path.write_text(
        json.dumps({"rows": rows, "aggregate": aggregate}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print("RESUMO DA ABLAÇÃO")
    print("=" * 92)
    print(f"{'VARIANTE':<24} {'GRUPO':<9} {'N':>3} {'IoU MÉDIA':>11} {'Δ MÉDIO':>11} {'MELH.':>7} {'PIOR.':>7} {'PIOR Δ':>10}")
    print("-" * 92)
    for a in aggregate:
        mean_delta = a["mean_delta_iou"]
        worst = a["worst_delta_iou"]
        print(
            f"{a['variant']:<24} {a['group']:<9} {a['n']:>3d} "
            f"{a['mean_iou']:>11.4f} "
            f"{(mean_delta if mean_delta is not None else float('nan')):>+11.4f} "
            f"{a['n_improved_gt_0p02']:>7d} "
            f"{a['n_worsened_lt_m0p02']:>7d} "
            f"{(worst if worst is not None else float('nan')):>+10.4f}"
        )

    print(f"\nCSV por classe : {csv_path}")
    print(f"CSV agregado   : {agg_csv}")
    print(f"JSON completo  : {json_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablação de supressão adaptativa de background recorrente pós-E2."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Raiz do projeto. Default: dois níveis acima do script (ablation/e2b/ -> raiz).",
    )
    parser.add_argument(
        "--baseline-features-root",
        type=Path,
        default=None,
        help="Default: <project>/features/icoseg_full_e2_spatial_degree_a0p10",
    )
    parser.add_argument(
        "--baseline-results-root",
        type=Path,
        default=None,
        help="Default: <project>/results/icoseg_full_e2_spatial_degree_a0p10",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: <project>/ablation_results/e2_background_suppression",
    )
    parser.add_argument(
        "--group",
        choices=["target", "control", "all"],
        default="all",
        help="Quais classes executar.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[v.name for v in VARIANTS],
        help="Subconjunto de variantes.",
    )
    parser.add_argument("--border-width", type=int, default=2)
    parser.add_argument("--top-fg-fraction", type=float, default=0.10)
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Desativa gating adaptativo; usa lambda fixo.",
    )
    parser.add_argument(
        "--run-downstream",
        action="store_true",
        help="Executa E3→E5 + avaliação para as variantes não-baseline.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Apenas gera correspondence.npz modificados.",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--sam-model",
        default="facebook/sam2.1-hiera-large",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()

    baseline_features_root = (
        args.baseline_features_root.resolve()
        if args.baseline_features_root
        else project_root / "features" / "icoseg_full_e2_spatial_degree_a0p10"
    )
    baseline_results_root = (
        args.baseline_results_root.resolve()
        if args.baseline_results_root
        else project_root / "results" / "icoseg_full_e2_spatial_degree_a0p10"
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else project_root / "ablation_results" / "e2_background_suppression"
    )

    by_name = {v.name: v for v in VARIANTS}
    unknown = [name for name in args.variants if name not in by_name]
    if unknown:
        raise SystemExit(
            f"Variantes desconhecidas: {unknown}\n"
            f"Disponíveis: {', '.join(by_name)}"
        )
    variants = [by_name[name] for name in args.variants]

    classes: list[tuple[str, str]] = []
    if args.group in ("target", "all"):
        classes.extend((c, "target") for c in TARGET_CLASSES)
    if args.group in ("control", "all"):
        classes.extend((c, "control") for c in CONTROL_CLASSES)

    print("=" * 88)
    print("ABLAÇÃO E2 — ADAPTIVE RECURRING-BACKGROUND SUPPRESSION")
    print("=" * 88)
    print(f"Projeto              : {project_root}")
    print(f"Baseline features    : {baseline_features_root}")
    print(f"Baseline results     : {baseline_results_root}")
    print(f"Saída                : {output_root}")
    print(f"Grupo                : {args.group} ({len(classes)} classes)")
    print(f"Border width         : {args.border_width}")
    print(f"Adaptive             : {not args.no_adaptive}")
    print(f"Top FG fraction      : {args.top_fg_fraction}")
    print(f"Run downstream       : {args.run_downstream and not args.prepare_only}")
    print(f"Variantes            : {', '.join(v.name for v in variants)}")

    if not baseline_features_root.exists():
        raise SystemExit(f"Baseline features não encontrado: {baseline_features_root}")

    ablation_results_root = output_root / "results"

    for variant in variants:
        print("\n" + "=" * 88)
        print(f"VARIANTE: {variant.name}")
        print("=" * 88)

        for class_name, group in classes:
            print(f"\n[{group.upper()}] {class_name}")

            # Baseline não precisa ser preparado nem recalculado para downstream.
            if variant.mode == "baseline":
                base_metric = (
                    baseline_results_root
                    / safe_name(class_name)
                    / "metrics.json"
                )
                print(f"  baseline metrics: {base_metric}")
                continue

            try:
                prepared_dir, diag = prepare_variant(
                    class_name=class_name,
                    variant=variant,
                    baseline_features_root=baseline_features_root,
                    output_root=output_root,
                    border_width=args.border_width,
                    adaptive=not args.no_adaptive,
                    top_fg_fraction=args.top_fg_fraction,
                    force=args.force,
                )

                if args.run_downstream and not args.prepare_only:
                    result_dir = (
                        ablation_results_root
                        / variant.name
                        / safe_name(class_name)
                    )
                    run_downstream(
                        project_root=project_root,
                        prepared_dir=prepared_dir,
                        result_dir=result_dir,
                        class_name=class_name,
                        python=args.python,
                        sam_model=args.sam_model,
                        force=args.force,
                        dry_run=args.dry_run,
                    )
            except Exception as exc:
                print(f"  [ERRO] {class_name} / {variant.name}: {exc}")
                if args.force:
                    raise

    if args.run_downstream and not args.prepare_only and not args.dry_run:
        build_summary(
            classes=classes,
            variants=variants,
            baseline_results_root=baseline_results_root,
            ablation_results_root=ablation_results_root,
            output_root=output_root,
        )
    else:
        print(
            "\nPreparação concluída. Para executar E3→E5 e avaliação, rode novamente "
            "com --run-downstream."
        )


if __name__ == "__main__":
    main()
