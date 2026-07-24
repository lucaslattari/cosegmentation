#!/usr/bin/env python3
"""
step2b_background_suppression.py
================================

Etapa opcional entre E2 e E3 para suprimir background recorrente estimado
a partir das bordas das imagens.

Entrada:
  - features.npz       (E1)
  - correspondence.npz (E2)

Saída:
  - correspondence_bg.npz
  - background_suppression.json

Modos:
  none        : copia scores sem alteração
  recurrence  : um protótipo recorrente de background

Proteção FG-vs-BG:
  gamma=0.0   : recurrence_l0p50 validado na primeira ablação
  gamma=0.75  : variante protegida que reduziu regressões na segunda ablação

A correção aplicada é, conceitualmente:

  lambda_eff =
      lambda
      * background_confidence
      * (1 - fg_border_fraction)
      * (1 - gamma * fg_bg_overlap)

  fg_score' = fg_score - lambda_eff * bg_score

Depois, os scores são normalizados robustamente por imagem para [0,1].

Este script NÃO usa ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(denom, eps)


def robust_01(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
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
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def border_centroids(
    patch_features: np.ndarray,
    border: np.ndarray,
    content_mask: np.ndarray | None,
) -> np.ndarray:
    n, _, d = patch_features.shape
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
    n = patch_features.shape[0]
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

    lo, hi = np.percentile(scores, [10, 90])
    if hi <= lo + 1e-8:
        weights = np.ones_like(scores, dtype=np.float32)
    else:
        weights = np.clip((scores - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    return feats, weights


def build_recurring_background_prototype(
    border_feats: np.ndarray,
    recurrence_weights: np.ndarray,
) -> np.ndarray:
    if border_feats.shape[0] == 0:
        raise RuntimeError("Nenhum patch de borda válido para construir background.")
    w = recurrence_weights + 1e-6
    proto = np.average(border_feats, axis=0, weights=w)
    return l2_normalize(proto[None, :], axis=1).astype(np.float32)


def fg_border_fraction(
    fg_scores: np.ndarray,
    border: np.ndarray,
    content_mask: np.ndarray | None,
    top_fraction: float,
) -> float:
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
    prototype: np.ndarray,
    border: np.ndarray,
    content_mask: np.ndarray | None,
    lam: float,
    gamma: float,
    top_fg_fraction: float,
) -> tuple[np.ndarray, dict]:
    bg_raw = np.einsum("npd,kd->npk", patch_features, prototype).max(axis=2)
    bg_score = robust_01(bg_raw, content_mask)

    centroids = border_centroids(patch_features, border, content_mask)
    bg_conf = background_confidence(centroids)
    fg_border = fg_border_fraction(
        fg_scores, border, content_mask, top_fraction=top_fg_fraction
    )

    overlaps = []
    for i in range(fg_scores.shape[0]):
        valid = np.ones(fg_scores.shape[1], dtype=bool)
        if content_mask is not None:
            valid &= content_mask[i]
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        n_top = max(1, int(math.ceil(top_fg_fraction * idx.size)))
        vals = fg_scores[i, idx]
        top_idx = idx[np.argpartition(vals, -n_top)[-n_top:]]
        overlaps.append(float(bg_score[i, top_idx].mean()))
    fg_bg_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    protection = float(np.clip(1.0 - gamma * fg_bg_overlap, 0.0, 1.0))
    lambda_eff = (
        lam
        * bg_conf
        * max(0.0, 1.0 - fg_border)
        * protection
    )

    corrected = fg_scores.astype(np.float32) - float(lambda_eff) * bg_score
    corrected = robust_01(corrected, content_mask)
    if content_mask is not None:
        corrected = np.where(content_mask, corrected, 0.0)

    diagnostics = {
        "mode": "recurrence",
        "lambda_requested": float(lam),
        "lambda_effective": float(lambda_eff),
        "gamma": float(gamma),
        "background_confidence": float(bg_conf),
        "fg_border_fraction": float(fg_border),
        "fg_bg_overlap": float(fg_bg_overlap),
        "fg_protection_factor": float(protection),
        "top_fg_fraction": float(top_fg_fraction),
        "bg_score_mean": float(bg_score.mean()),
        "bg_score_std": float(bg_score.std()),
    }
    return corrected.astype(np.float32), diagnostics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E2B: supressão adaptativa de background recorrente."
    )
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--correspondence", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--mode",
        choices=["none", "recurrence"],
        default="recurrence",
    )
    p.add_argument("--lambda-bg", type=float, default=0.50)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--border-width", type=int, default=2)
    p.add_argument("--top-fg-fraction", type=float, default=0.10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.lambda_bg):
        raise SystemExit("--lambda-bg deve ser >= 0")
    if not (0.0 <= args.gamma <= 1.0):
        raise SystemExit("--gamma deve estar em [0,1]")
    if not (0.0 < args.top_fg_fraction <= 1.0):
        raise SystemExit("--top-fg-fraction deve estar em (0,1]")

    feat = load_npz_dict(args.features)
    corr = load_npz_dict(args.correspondence)

    required_feat = {"patch_features"}
    required_corr = {"fg_scores", "patch_grid"}
    missing_feat = required_feat - set(feat)
    missing_corr = required_corr - set(corr)
    if missing_feat:
        raise KeyError(f"features.npz sem chaves: {sorted(missing_feat)}")
    if missing_corr:
        raise KeyError(f"correspondence.npz sem chaves: {sorted(missing_corr)}")

    patch_features = l2_normalize(
        feat["patch_features"].astype(np.float32), axis=2
    ).astype(np.float32)
    fg_scores = corr["fg_scores"].astype(np.float32)
    grid = tuple(int(x) for x in corr["patch_grid"])
    h_p, w_p = grid

    if fg_scores.ndim != 2:
        raise ValueError(f"fg_scores deve ser (N,P), recebido {fg_scores.shape}")
    if patch_features.shape[:2] != fg_scores.shape:
        raise ValueError(
            "patch_features e fg_scores incompatíveis: "
            f"{patch_features.shape[:2]} vs {fg_scores.shape}"
        )
    if h_p * w_p != fg_scores.shape[1]:
        raise ValueError(
            f"patch_grid={grid} incompatível com P={fg_scores.shape[1]}"
        )

    resize_info = feat.get("resize_info")
    content_mask = content_mask_from_resize_info(resize_info, grid)
    bmask = border_mask(grid, args.border_width)

    if args.mode == "none":
        corrected = fg_scores.copy()
        diagnostics = {
            "mode": "none",
            "lambda_requested": 0.0,
            "lambda_effective": 0.0,
            "gamma": 0.0,
            "border_width": int(args.border_width),
            "top_fg_fraction": float(args.top_fg_fraction),
        }
    else:
        centroids = border_centroids(patch_features, bmask, content_mask)
        bfeats, weights = border_recurrence_weights(
            patch_features, centroids, bmask, content_mask
        )
        prototype = build_recurring_background_prototype(bfeats, weights)
        corrected, diagnostics = suppress_background(
            patch_features=patch_features,
            fg_scores=fg_scores,
            prototype=prototype,
            border=bmask,
            content_mask=content_mask,
            lam=args.lambda_bg,
            gamma=args.gamma,
            top_fg_fraction=args.top_fg_fraction,
        )
        diagnostics["border_width"] = int(args.border_width)
        diagnostics["n_background_prototypes"] = 1

    out_corr = dict(corr)
    out_corr["fg_scores"] = corrected
    out_corr["fg_scores_2d"] = corrected.reshape(
        corrected.shape[0], h_p, w_p
    )
    out_corr["bg_suppression_mode"] = np.array(args.mode)
    out_corr["bg_suppression_lambda"] = np.array(
        args.lambda_bg, dtype=np.float32
    )
    out_corr["bg_suppression_gamma"] = np.array(
        args.gamma, dtype=np.float32
    )

    args.output.mkdir(parents=True, exist_ok=True)
    out_npz = args.output / "correspondence_bg.npz"
    np.savez_compressed(out_npz, **out_corr)

    diag_path = args.output / "background_suppression.json"
    diag_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("E2B — BACKGROUND SUPPRESSION")
    print("=" * 72)
    print(f"mode                  : {args.mode}")
    print(f"lambda                : {args.lambda_bg:.4f}")
    print(f"gamma                 : {args.gamma:.4f}")
    print(f"correspondence input  : {args.correspondence}")
    print(f"correspondence output : {out_npz}")
    if args.mode != "none":
        print(
            f"lambda efetivo        : "
            f"{diagnostics['lambda_effective']:.6f}"
        )
        print(
            f"background confidence : "
            f"{diagnostics['background_confidence']:.6f}"
        )
        print(
            f"FG/BG overlap         : "
            f"{diagnostics['fg_bg_overlap']:.6f}"
        )
    print(f"diagnóstico           : {diag_path}")


if __name__ == "__main__":
    main()
