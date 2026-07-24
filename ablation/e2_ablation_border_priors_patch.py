#!/usr/bin/env python3
"""
Ablacao E2: priors geometricos/background sem alterar o pipeline principal.

Entrada:
- Um ou mais diretorios de classe contendo correspondence.npz e features.npz
  (busca recursiva limitada ao diretorio informado).
- gt_index.json.

Variantes avaliadas post-hoc sobre fg_scores da E2:
1) baseline                  : fg_score original
2) border_distance          : penaliza proximidade da borda do conteudo
3) border_diffusion         : propaga sementes de background da borda usando
                              similaridade DINO entre patches 4-vizinhos
4) rarity                   : favorece patches menos redundantes intra-imagem
5) diffusion_plus_rarity    : combina (3) e (4)

A ideia e medir se um prior ajuda classes com objeto pequeno/fundo dominante
sem piorar classes-controle que ja funcionam bem.

Nao sobrescreve correspondence.npz.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ---------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------

def find_one(root: Path, filename: str) -> Path:
    direct = root / filename
    if direct.exists():
        return direct
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} nao encontrado em {root}")
    if len(matches) > 1:
        # Prefere o arquivo menos profundo.
        matches.sort(key=lambda p: (len(p.relative_to(root).parts), str(p)))
    return matches[0]


def minmax_content(x: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    N, P = x.shape
    for i in range(N):
        valid = mask[i] if mask is not None else np.ones(P, dtype=bool)
        vals = x[i, valid]
        if vals.size == 0:
            continue
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-8:
            out[i, valid] = 0.5
        else:
            out[i, valid] = (vals - lo) / (hi - lo)
    return out


def safe_auc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    if roc_auc_score is None or np.unique(y).size < 2:
        return None
    try:
        return float(roc_auc_score(y.astype(np.uint8), s.astype(np.float64)))
    except Exception:
        return None


def path_variants(path_str: str) -> set[str]:
    import os
    raw = str(path_str).replace("\\", "/")
    variants = {raw, raw.lstrip("./")}
    p = Path(path_str)
    try:
        variants.add(p.as_posix())
    except Exception:
        pass
    try:
        variants.add(os.path.relpath(p).replace("\\", "/"))
    except Exception:
        pass
    for marker in ("icoseg/images/", "images/"):
        idx = raw.find(marker)
        if idx >= 0:
            variants.add(raw[idx:])
    return {v for v in variants if v}


# ---------------------------------------------------------------------
# Carregamento e alinhamento
# ---------------------------------------------------------------------

def load_case(case_dir: Path) -> Dict[str, Any]:
    corr_path = find_one(case_dir, "correspondence.npz")
    feat_path = find_one(case_dir, "features.npz")

    corr = np.load(corr_path, allow_pickle=True)
    feat = np.load(feat_path, allow_pickle=True)

    fg_scores = np.asarray(corr["fg_scores"], dtype=np.float32)
    patch_grid = tuple(int(x) for x in np.asarray(corr["patch_grid"]).tolist())
    image_paths = [str(x) for x in corr["image_paths"].tolist()]
    patch_features = np.asarray(feat["patch_features"], dtype=np.float32)
    resize_info = np.asarray(feat["resize_info"]) if "resize_info" in feat.files else None

    if patch_features.shape[:2] != fg_scores.shape:
        raise ValueError(
            f"Shapes incompativeis em {case_dir}: "
            f"patch_features={patch_features.shape}, fg_scores={fg_scores.shape}"
        )

    return {
        "case_dir": case_dir,
        "corr_path": corr_path,
        "feat_path": feat_path,
        "fg_scores": fg_scores,
        "patch_grid": patch_grid,
        "image_paths": image_paths,
        "patch_features": patch_features,
        "resize_info": resize_info,
    }


def compute_content_mask(
    resize_info: Optional[np.ndarray],
    patch_grid: Tuple[int, int],
    n_images: int,
) -> Optional[np.ndarray]:
    if resize_info is None:
        return None

    H, W = patch_grid
    canvas_h, canvas_w = int(resize_info[0, 4]), int(resize_info[0, 5])
    patch_h = canvas_h / H
    patch_w = canvas_w / W

    row_starts = np.arange(H) * patch_h
    col_starts = np.arange(W) * patch_w
    row_ends = row_starts + patch_h
    col_ends = col_starts + patch_w

    masks = np.zeros((n_images, H, W), dtype=bool)
    for i in range(n_images):
        resized_h, resized_w, pad_top, pad_left, _, _ = [int(x) for x in resize_info[i]]
        rr = (row_ends > pad_top) & (row_starts < pad_top + resized_h)
        cc = (col_ends > pad_left) & (col_starts < pad_left + resized_w)
        masks[i] = rr[:, None] & cc[None, :]
    return masks.reshape(n_images, -1)


def load_gt_index(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_patch_gt(
    image_paths: List[str],
    patch_grid: Tuple[int, int],
    gt_index: Dict[str, Any],
    resize_info: Optional[np.ndarray],
    fg_threshold: float,
) -> Tuple[np.ndarray, int]:
    mappings = gt_index.get("mappings", {})
    variant_lookup: Dict[str, str] = {}
    basename_to_gt: Dict[str, List[str]] = {}

    for img_path, gt_path in mappings.items():
        for v in path_variants(img_path):
            variant_lookup[v] = str(gt_path)
        basename_to_gt.setdefault(Path(img_path).name, []).append(str(gt_path))

    unique_basename: Dict[str, Optional[str]] = {}
    for base, gts in basename_to_gt.items():
        uniq = sorted(set(gts))
        unique_basename[base] = uniq[0] if len(uniq) == 1 else None

    H, W = patch_grid
    masks = []
    matched = 0

    for idx, img_path in enumerate(image_paths):
        gt_path: Optional[Path] = None
        for v in path_variants(img_path):
            if v in variant_lookup:
                gt_path = Path(variant_lookup[v])
                break
        if gt_path is None:
            candidate = unique_basename.get(Path(img_path).name)
            gt_path = Path(candidate) if candidate else None

        if gt_path is None or not gt_path.exists():
            masks.append(np.zeros((H, W), dtype=bool))
            continue

        gt_img = Image.open(gt_path).convert("L")
        if resize_info is not None:
            resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = [
                int(x) for x in resize_info[idx]
            ]
            gt_resized = gt_img.resize((resized_w, resized_h), Image.BILINEAR)
            canvas = Image.new("L", (canvas_w, canvas_h), 0)
            canvas.paste(gt_resized, (pad_left, pad_top))
            small = canvas.resize((W, H), Image.BILINEAR)
        else:
            small = gt_img.resize((W, H), Image.BILINEAR)

        arr = np.asarray(small, dtype=np.float32) / 255.0
        masks.append(arr >= fg_threshold)
        matched += 1

    return np.stack([m.reshape(-1) for m in masks]), matched


# ---------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------

def boundary_seed_mask(content_2d: np.ndarray) -> np.ndarray:
    """Patches de conteudo que tocam o contorno externo da regiao valida."""
    H, W = content_2d.shape
    seeds = np.zeros((H, W), dtype=bool)
    for r in range(H):
        for c in range(W):
            if not content_2d[r, c]:
                continue
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= H or cc < 0 or cc >= W or not content_2d[rr, cc]:
                    seeds[r, c] = True
                    break
    return seeds


def border_distance_prior(content_mask: Optional[np.ndarray],
                          patch_grid: Tuple[int, int],
                          n_images: int,
                          tau: float = 0.18) -> np.ndarray:
    """
    B em [0,1]: 1 na borda do conteudo, decai exponencialmente para o interior.
    """
    H, W = patch_grid
    out = np.zeros((n_images, H, W), dtype=np.float32)

    for i in range(n_images):
        content = (
            content_mask[i].reshape(H, W)
            if content_mask is not None
            else np.ones((H, W), dtype=bool)
        )
        seeds = np.argwhere(boundary_seed_mask(content))
        valid = np.argwhere(content)
        if len(seeds) == 0 or len(valid) == 0:
            continue

        # Distancia Manhattan normalizada ao maior eixo.
        scale = max(H, W)
        for r, c in valid:
            d = np.min(np.abs(seeds[:, 0] - r) + np.abs(seeds[:, 1] - c)) / scale
            out[i, r, c] = math.exp(-d / max(tau, 1e-6))

    return out.reshape(n_images, -1)


def border_diffusion_prior(
    patch_features: np.ndarray,
    content_mask: Optional[np.ndarray],
    patch_grid: Tuple[int, int],
    steps: int = 24,
    alpha: float = 0.88,
    temperature: float = 0.15,
) -> np.ndarray:
    """
    Propaga sementes de background da borda em um grafo 4-vizinho.

    Cada atualizacao usa media ponderada dos vizinhos por similaridade DINO.
    Seeds de borda ficam fixas em 1. O resultado B alto indica forte conexao
    feature-wise com a borda.
    """
    N, P, D = patch_features.shape
    H, W = patch_grid
    out = np.zeros((N, P), dtype=np.float32)

    for i in range(N):
        content = (
            content_mask[i].reshape(H, W)
            if content_mask is not None
            else np.ones((H, W), dtype=bool)
        )
        seeds = boundary_seed_mask(content)

        f = patch_features[i].reshape(H, W, D).astype(np.float64)
        norms = np.linalg.norm(f, axis=2, keepdims=True)
        f = f / np.maximum(norms, 1e-8)

        B = np.zeros((H, W), dtype=np.float64)
        B[seeds] = 1.0

        for _ in range(steps):
            new = B.copy()
            for r in range(H):
                for c in range(W):
                    if not content[r, c] or seeds[r, c]:
                        continue
                    vals, weights = [], []
                    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and content[rr, cc]:
                            sim = float(np.dot(f[r, c], f[rr, cc]))
                            # Peso relativo estavel: sim em [-1,1].
                            w = math.exp((sim - 1.0) / max(temperature, 1e-6))
                            vals.append(B[rr, cc])
                            weights.append(w)
                    if weights and sum(weights) > 1e-12:
                        neigh = float(np.dot(vals, weights) / sum(weights))
                        new[r, c] = alpha * neigh + (1.0 - alpha) * B[r, c]
            B = new
            B[seeds] = 1.0
            B[~content] = 0.0

        out[i] = B.reshape(-1).astype(np.float32)

    return out


def rarity_prior(
    patch_features: np.ndarray,
    content_mask: Optional[np.ndarray],
    k: int = 20,
) -> np.ndarray:
    """
    R em [0,1]: patches menos semelhantes aos k vizinhos mais proximos
    intra-imagem recebem maior raridade.

    Usa similaridade coseno. Exclui o proprio patch.
    """
    N, P, D = patch_features.shape
    out = np.zeros((N, P), dtype=np.float32)

    for i in range(N):
        valid = content_mask[i] if content_mask is not None else np.ones(P, dtype=bool)
        idx = np.flatnonzero(valid)
        if idx.size < 2:
            continue

        f = patch_features[i, idx].astype(np.float64)
        f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-8)
        sim = f @ f.T
        np.fill_diagonal(sim, -np.inf)

        kk = min(k, max(1, idx.size - 1))
        top = np.partition(sim, -kk, axis=1)[:, -kk:]
        redundancy = np.mean(top, axis=1)
        raw = 1.0 - redundancy

        lo, hi = float(raw.min()), float(raw.max())
        rarity = (raw - lo) / (hi - lo + 1e-8)
        out[i, idx] = rarity.astype(np.float32)

    return out


def apply_variant(
    fg: np.ndarray,
    variant: str,
    border_dist: np.ndarray,
    border_diff: np.ndarray,
    rarity: np.ndarray,
    lam: float,
    beta: float,
    content_mask: Optional[np.ndarray],
) -> np.ndarray:
    if variant == "baseline":
        s = fg.copy()
    elif variant == "border_distance":
        s = fg * (1.0 - lam * border_dist)
    elif variant == "border_diffusion":
        s = fg * (1.0 - lam * border_diff)
    elif variant == "rarity":
        s = fg + beta * rarity
    elif variant == "diffusion_plus_rarity":
        s = fg * (1.0 - lam * border_diff) + beta * rarity
    else:
        raise ValueError(variant)

    if content_mask is not None:
        s = s.copy()
        s[~content_mask] = 0.0

    # Normalizacao por imagem evita alterar a escala de threshold apenas
    # por multiplicacao; a ordenacao relativa e o que queremos testar.
    return minmax_content(s, content_mask)


# ---------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------

def confusion_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    tn = int(np.sum(~y & ~pred))
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
    }


def boundary_connectivity(mask: np.ndarray,
                          content: np.ndarray,
                          patch_grid: Tuple[int, int]) -> float:
    """
    Fracao do foreground previsto que pertence a componentes 4-conectados
    que tocam a borda da regiao de conteudo.
    """
    H, W = patch_grid
    m = mask.reshape(H, W).astype(bool)
    c = content.reshape(H, W).astype(bool)
    fg_total = int(np.sum(m & c))
    if fg_total == 0:
        return 0.0

    seeds = boundary_seed_mask(c)
    visited = np.zeros((H, W), dtype=bool)
    stack = [(int(r), int(col)) for r, col in np.argwhere(m & c & seeds)]
    for r, col in stack:
        visited[r, col] = True

    p = 0
    while p < len(stack):
        r, col = stack[p]
        p += 1
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr, cc = r + dr, col + dc
            if (
                0 <= rr < H and 0 <= cc < W
                and c[rr, cc] and m[rr, cc] and not visited[rr, cc]
            ):
                visited[rr, cc] = True
                stack.append((rr, cc))

    return float(np.sum(visited) / fg_total)


def evaluate_scores(
    scores: np.ndarray,
    patch_gt: np.ndarray,
    content_mask: Optional[np.ndarray],
    patch_grid: Tuple[int, int],
    thresholds: np.ndarray,
) -> Dict[str, Any]:
    N, P = scores.shape
    valid = content_mask if content_mask is not None else np.ones((N, P), dtype=bool)

    y_all = patch_gt[valid]
    s_all = scores[valid]
    auc = safe_auc(y_all, s_all)

    best = None
    for t in thresholds:
        m = confusion_metrics(y_all, s_all >= t)
        if best is None or m["iou"] > best["iou"]:
            best = {**m, "threshold": float(t)}

    assert best is not None
    tbest = best["threshold"]

    per_img_iou = []
    per_img_oracle_iou = []
    bc = []
    fg_frac = []

    for i in range(N):
        vi = valid[i]
        yi = patch_gt[i, vi]
        si = scores[i, vi]
        pred = si >= tbest
        per_img_iou.append(confusion_metrics(yi, pred)["iou"])

        oiou = 0.0
        for t in thresholds:
            oiou = max(oiou, confusion_metrics(yi, si >= t)["iou"])
        per_img_oracle_iou.append(oiou)

        full_pred = np.zeros(P, dtype=bool)
        full_pred[vi] = pred
        bc.append(boundary_connectivity(full_pred, valid[i], patch_grid))
        fg_frac.append(float(np.mean(pred)) if pred.size else 0.0)

    return {
        "auc_roc": auc,
        "best_iou": best["iou"],
        "best_threshold": best["threshold"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "accuracy": best["accuracy"],
        "mean_image_iou_at_global_threshold": float(np.mean(per_img_iou)),
        "mean_image_oracle_iou": float(np.mean(per_img_oracle_iou)),
        "boundary_connectivity_mean": float(np.mean(bc)),
        "fg_fraction_mean": float(np.mean(fg_frac)),
    }


# ---------------------------------------------------------------------
# Visualizacao
# ---------------------------------------------------------------------

def save_preview(
    out_path: Path,
    scores_by_variant: Dict[str, np.ndarray],
    patch_gt: np.ndarray,
    content_mask: Optional[np.ndarray],
    patch_grid: Tuple[int, int],
    max_images: int = 6,
) -> None:
    if plt is None:
        return
    H, W = patch_grid
    variants = list(scores_by_variant.keys())
    n = min(max_images, patch_gt.shape[0])
    cols = 1 + len(variants)

    fig, axes = plt.subplots(n, cols, figsize=(3.2 * cols, 3.0 * n), squeeze=False)
    for i in range(n):
        axes[i, 0].imshow(patch_gt[i].reshape(H, W), vmin=0, vmax=1)
        axes[i, 0].set_title(f"GT img[{i}]")
        axes[i, 0].axis("off")
        for j, name in enumerate(variants, start=1):
            arr = scores_by_variant[name][i].reshape(H, W)
            if content_mask is not None:
                arr = arr.copy()
                arr[~content_mask[i].reshape(H, W)] = np.nan
            axes[i, j].imshow(arr, vmin=0, vmax=1)
            axes[i, j].set_title(name)
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablacao post-hoc de priors geometricos/background sobre fg_scores da E2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cases", nargs="+", required=True, type=Path,
                   help="Diretorios de classes; cada um deve conter features.npz e correspondence.npz.")
    p.add_argument("--gt_index", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75])
    p.add_argument("--betas", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    p.add_argument("--border_tau", type=float, default=0.18)
    p.add_argument("--diffusion_steps", type=int, default=24)
    p.add_argument("--diffusion_alpha", type=float, default=0.88)
    p.add_argument("--diffusion_temperature", type=float, default=0.15)
    p.add_argument("--rarity_k", type=int, default=20)
    p.add_argument("--gt_threshold", type=float, default=0.5)
    p.add_argument("--threshold_steps", type=int, default=201)
    p.add_argument("--max_preview_images", type=int, default=6)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gt_index = load_gt_index(args.gt_index)
    thresholds = np.linspace(0.0, 1.0, args.threshold_steps)

    all_rows: List[Dict[str, Any]] = []

    for case_dir in args.cases:
        print("\n" + "=" * 88)
        print(f"[CASE] {case_dir}")
        case = load_case(case_dir)
        fg = case["fg_scores"]
        N, P = fg.shape
        H, W = case["patch_grid"]

        content_mask = compute_content_mask(case["resize_info"], (H, W), N)
        patch_gt, matched = load_patch_gt(
            case["image_paths"], (H, W), gt_index, case["resize_info"], args.gt_threshold
        )
        if matched == 0:
            print("  [AVISO] Nenhum GT encontrado; caso ignorado.")
            continue
        print(f"  GT: {matched}/{N}")

        print("  Calculando prior de distancia da borda...")
        bd = border_distance_prior(content_mask, (H, W), N, tau=args.border_tau)

        print("  Calculando prior de difusao da borda...")
        bdiff = border_diffusion_prior(
            case["patch_features"], content_mask, (H, W),
            steps=args.diffusion_steps,
            alpha=args.diffusion_alpha,
            temperature=args.diffusion_temperature,
        )

        print("  Calculando prior de raridade...")
        rar = rarity_prior(case["patch_features"], content_mask, k=args.rarity_k)

        variants: List[Tuple[str, float, float]] = [("baseline", 0.0, 0.0)]
        for lam in args.lambdas:
            variants += [
                ("border_distance", lam, 0.0),
                ("border_diffusion", lam, 0.0),
            ]
        for beta in args.betas:
            variants.append(("rarity", 0.0, beta))
        for lam in args.lambdas:
            for beta in args.betas:
                variants.append(("diffusion_plus_rarity", lam, beta))

        scores_cache: Dict[str, np.ndarray] = {}
        baseline_metrics = None

        for variant, lam, beta in variants:
            label = variant
            if variant in ("border_distance", "border_diffusion"):
                label += f"_l{lam:g}"
            elif variant == "rarity":
                label += f"_b{beta:g}"
            elif variant == "diffusion_plus_rarity":
                label += f"_l{lam:g}_b{beta:g}"

            scores = apply_variant(fg, variant, bd, bdiff, rar, lam, beta, content_mask)
            metrics = evaluate_scores(scores, patch_gt, content_mask, (H, W), thresholds)

            if variant == "baseline":
                baseline_metrics = metrics

            row = {
                "case": case_dir.name,
                "case_dir": str(case_dir),
                "variant": variant,
                "label": label,
                "lambda": lam,
                "beta": beta,
                "gt_matched": matched,
                **metrics,
            }
            all_rows.append(row)

            if label in (
                "baseline",
                "border_diffusion_l0.5",
                "rarity_b0.25",
                "diffusion_plus_rarity_l0.5_b0.25",
            ):
                scores_cache[label] = scores

        # Deltas do caso.
        case_rows = [r for r in all_rows if r["case_dir"] == str(case_dir)]
        assert baseline_metrics is not None
        for r in case_rows:
            r["delta_best_iou_vs_baseline"] = r["best_iou"] - baseline_metrics["best_iou"]
            r["delta_auc_vs_baseline"] = (
                None if r["auc_roc"] is None or baseline_metrics["auc_roc"] is None
                else r["auc_roc"] - baseline_metrics["auc_roc"]
            )

        best = max(case_rows, key=lambda r: r["best_iou"])
        print(
            f"  Baseline IoU={baseline_metrics['best_iou']:.4f} "
            f"AUC={baseline_metrics['auc_roc'] if baseline_metrics['auc_roc'] is not None else float('nan'):.4f}"
        )
        print(
            f"  Melhor: {best['label']} | IoU={best['best_iou']:.4f} "
            f"(delta={best['delta_best_iou_vs_baseline']:+.4f}) | "
            f"AUC={best['auc_roc'] if best['auc_roc'] is not None else float('nan'):.4f}"
        )

        if not args.no_plots and scores_cache:
            preview_path = args.output / f"{case_dir.name}_preview.png"
            save_preview(
                preview_path, scores_cache, patch_gt, content_mask, (H, W),
                max_images=args.max_preview_images
            )

    if not all_rows:
        print("[ERRO] Nenhum caso avaliado.")
        sys.exit(1)

    # CSV detalhado.
    csv_path = args.output / "e2_geometry_ablation_results.csv"
    fields = list(all_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # Sumario por variante agregando casos.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        grouped.setdefault(r["label"], []).append(r)

    summary = []
    for label, rows in grouped.items():
        summary.append({
            "label": label,
            "n_cases": len(rows),
            "mean_best_iou": float(np.mean([r["best_iou"] for r in rows])),
            "mean_delta_iou": float(np.mean([r["delta_best_iou_vs_baseline"] for r in rows])),
            "median_delta_iou": float(np.median([r["delta_best_iou_vs_baseline"] for r in rows])),
            "n_improved": int(sum(r["delta_best_iou_vs_baseline"] > 1e-6 for r in rows)),
            "n_worsened": int(sum(r["delta_best_iou_vs_baseline"] < -1e-6 for r in rows)),
            "mean_auc": float(np.mean([r["auc_roc"] for r in rows if r["auc_roc"] is not None])),
            "mean_boundary_connectivity": float(np.mean([r["boundary_connectivity_mean"] for r in rows])),
        })
    summary.sort(key=lambda x: x["mean_delta_iou"], reverse=True)

    summary_path = args.output / "e2_geometry_ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    json_path = args.output / "e2_geometry_ablation_results.json"
    json_path.write_text(
        json.dumps({"results": all_rows, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("TOP variantes agregadas:")
    for s in summary[:10]:
        print(
            f"  {s['label']:<36} "
            f"deltaIoU={s['mean_delta_iou']:+.4f} "
            f"melhorou={s['n_improved']}/{s['n_cases']} "
            f"piorou={s['n_worsened']}/{s['n_cases']}"
        )
    print(f"\n[Salvo] {csv_path}")
    print(f"[Salvo] {summary_path}")
    print(f"[Salvo] {json_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {type(e).__name__}: {e}")
        raise
