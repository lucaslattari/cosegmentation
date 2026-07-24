#!/usr/bin/env python3
"""
Diagnostico/ablacao da Etapa 2 antes do NCut.

Objetivo
--------
Responder, usando GT APENAS PARA DIAGNOSTICO:

1) Os patches de foreground encontram patches de foreground na outra imagem?
2) O background domina as correspondencias apenas por "massa" (muito mais patches)?
3) Mutual nearest neighbors melhora a pureza FG<->FG?
4) Reponderacao por raridade/IDF melhora a relacao FG-FG sem destruir BG-BG?
5) Normalizacao por grau reduz a dominancia do background?

O script NAO altera o pipeline e NAO sobrescreve arquivos.

Entradas por caso:
  features.npz
  correspondence.npz   (usa top_pairs para analisar exatamente os pares da E2)
  gt_index.json

Saidas:
  e2_match_diagnostics_pairs.csv
  e2_match_diagnostics_cases.csv
  e2_match_diagnostics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def find_one(root: Path, filename: str) -> Path:
    p = root / filename
    if p.exists():
        return p
    matches = sorted(root.rglob(filename), key=lambda x: (len(x.relative_to(root).parts), str(x)))
    if not matches:
        raise FileNotFoundError(f"{filename} nao encontrado em {root}")
    return matches[0]


def path_variants(path_str: str) -> set[str]:
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


def load_gt_index(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_case(case_dir: Path) -> Dict[str, Any]:
    fpath = find_one(case_dir, "features.npz")
    cpath = find_one(case_dir, "correspondence.npz")

    f = np.load(fpath, allow_pickle=True)
    c = np.load(cpath, allow_pickle=True)

    patch_features = np.asarray(f["patch_features"], dtype=np.float32)
    patch_grid = tuple(int(x) for x in np.asarray(f["patch_grid"]).tolist())
    image_paths = [str(x) for x in f["image_paths"].tolist()]
    resize_info = np.asarray(f["resize_info"]) if "resize_info" in f.files else None
    top_pairs = np.asarray(c["top_pairs"])

    return {
        "features_path": fpath,
        "correspondence_path": cpath,
        "patch_features": patch_features,
        "patch_grid": patch_grid,
        "image_paths": image_paths,
        "resize_info": resize_info,
        "top_pairs": top_pairs,
    }


# ---------------------------------------------------------------------
# Padding / GT
# ---------------------------------------------------------------------

def compute_content_mask(
    resize_info: Optional[np.ndarray],
    patch_grid: Tuple[int, int],
    n_images: int,
) -> np.ndarray:
    H, W = patch_grid
    if resize_info is None:
        return np.ones((n_images, H * W), dtype=bool)

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


def build_gt_lookup(gt_index: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    mappings = gt_index.get("mappings", {})
    variants: Dict[str, str] = {}
    basename: Dict[str, List[str]] = {}
    for img_path, gt_path in mappings.items():
        for v in path_variants(img_path):
            variants[v] = str(gt_path)
        basename.setdefault(Path(img_path).name, []).append(str(gt_path))

    unique_base: Dict[str, Optional[str]] = {}
    for base, vals in basename.items():
        u = sorted(set(vals))
        unique_base[base] = u[0] if len(u) == 1 else None
    return variants, unique_base


def load_patch_gt(
    image_paths: List[str],
    patch_grid: Tuple[int, int],
    gt_index: Dict[str, Any],
    resize_info: Optional[np.ndarray],
    threshold: float,
) -> Tuple[np.ndarray, int]:
    lookup, base_lookup = build_gt_lookup(gt_index)
    H, W = patch_grid
    masks = []
    matched = 0

    for i, img_path in enumerate(image_paths):
        gt_path: Optional[Path] = None
        for v in path_variants(img_path):
            if v in lookup:
                gt_path = Path(lookup[v])
                break
        if gt_path is None:
            cand = base_lookup.get(Path(img_path).name)
            if cand:
                gt_path = Path(cand)

        if gt_path is None or not gt_path.exists():
            masks.append(np.zeros(H * W, dtype=bool))
            continue

        gt_img = Image.open(gt_path).convert("L")
        if resize_info is not None:
            resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = [
                int(x) for x in resize_info[i]
            ]
            resized = gt_img.resize((resized_w, resized_h), Image.BILINEAR)
            canvas = Image.new("L", (canvas_w, canvas_h), 0)
            canvas.paste(resized, (pad_left, pad_top))
            small = canvas.resize((W, H), Image.BILINEAR)
        else:
            small = gt_img.resize((W, H), Image.BILINEAR)

        arr = np.asarray(small, dtype=np.float32) / 255.0
        masks.append((arr.reshape(-1) >= threshold))
        matched += 1

    return np.stack(masks), matched


# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------

def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def topk_indices(sim: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sim.shape[1])
    part = np.argpartition(sim, -k, axis=1)[:, -k:]
    vals = np.take_along_axis(sim, part, axis=1)
    order = np.argsort(vals, axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


def correspondence_stats(
    sim: np.ndarray,
    gt_src: np.ndarray,
    gt_dst: np.ndarray,
    ks: List[int],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    fg_src = np.flatnonzero(gt_src)
    bg_src = np.flatnonzero(~gt_src)

    # Top-k label purity
    for k in ks:
        idx = topk_indices(sim, k)
        labels = gt_dst[idx]

        if fg_src.size:
            out[f"fg_to_fg_hit_at_{k}"] = float(np.mean(np.any(labels[fg_src], axis=1)))
            out[f"fg_to_fg_precision_at_{k}"] = float(np.mean(labels[fg_src]))
        else:
            out[f"fg_to_fg_hit_at_{k}"] = float("nan")
            out[f"fg_to_fg_precision_at_{k}"] = float("nan")

        if bg_src.size:
            bg_labels = ~labels[bg_src]
            out[f"bg_to_bg_hit_at_{k}"] = float(np.mean(np.any(bg_labels, axis=1)))
            out[f"bg_to_bg_precision_at_{k}"] = float(np.mean(bg_labels))
        else:
            out[f"bg_to_bg_hit_at_{k}"] = float("nan")
            out[f"bg_to_bg_precision_at_{k}"] = float("nan")

    # Melhor match FG vs BG para cada source-FG.
    if fg_src.size and np.any(gt_dst) and np.any(~gt_dst):
        best_fg = np.max(sim[np.ix_(fg_src, np.flatnonzero(gt_dst))], axis=1)
        best_bg = np.max(sim[np.ix_(fg_src, np.flatnonzero(~gt_dst))], axis=1)
        margin = best_fg - best_bg
        out["fg_match_margin_mean"] = float(np.mean(margin))
        out["fg_match_margin_positive_fraction"] = float(np.mean(margin > 0))
        out["fg_best_fg_similarity_mean"] = float(np.mean(best_fg))
        out["fg_best_bg_similarity_mean"] = float(np.mean(best_bg))
    else:
        out["fg_match_margin_mean"] = float("nan")
        out["fg_match_margin_positive_fraction"] = float("nan")
        out["fg_best_fg_similarity_mean"] = float("nan")
        out["fg_best_bg_similarity_mean"] = float("nan")

    return out


def mutual_nn_purity(sim: np.ndarray, gt_a: np.ndarray, gt_b: np.ndarray) -> Dict[str, float]:
    a2b = np.argmax(sim, axis=1)
    b2a = np.argmax(sim, axis=0)
    pairs = [(a, int(b)) for a, b in enumerate(a2b) if b2a[b] == a]

    if not pairs:
        return {
            "mnn_count": 0,
            "mnn_fg_fg_fraction": float("nan"),
            "mnn_bg_bg_fraction": float("nan"),
            "mnn_cross_fraction": float("nan"),
            "mnn_fg_recall": 0.0,
        }

    aa = np.array([p[0] for p in pairs], dtype=int)
    bb = np.array([p[1] for p in pairs], dtype=int)

    fgfg = gt_a[aa] & gt_b[bb]
    bgbg = (~gt_a[aa]) & (~gt_b[bb])
    cross = ~(fgfg | bgbg)

    denom_fg = max(int(np.sum(gt_a)), 1)
    return {
        "mnn_count": int(len(pairs)),
        "mnn_fg_fg_fraction": float(np.mean(fgfg)),
        "mnn_bg_bg_fraction": float(np.mean(bgbg)),
        "mnn_cross_fraction": float(np.mean(cross)),
        "mnn_fg_recall": float(np.sum(fgfg) / denom_fg),
    }


def soft_mass_stats(
    sim: np.ndarray,
    gt_a: np.ndarray,
    gt_b: np.ndarray,
    temperature: float,
    degree_normalize: bool = False,
    rarity_weight: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Dict[str, float]:
    # Mesmo formato de afinidade exponencial, estabilizado pelo max global.
    w = np.exp((sim - 1.0) / max(temperature, 1e-6))

    if rarity_weight is not None:
        ra, rb = rarity_weight
        w = w * ra[:, None] * rb[None, :]

    if degree_normalize:
        da = np.maximum(w.sum(axis=1), 1e-12)
        db = np.maximum(w.sum(axis=0), 1e-12)
        w = w / np.sqrt(da[:, None] * db[None, :])

    ia_fg = np.flatnonzero(gt_a)
    ia_bg = np.flatnonzero(~gt_a)
    ib_fg = np.flatnonzero(gt_b)
    ib_bg = np.flatnonzero(~gt_b)

    def block_sum(ra: np.ndarray, cb: np.ndarray) -> float:
        if ra.size == 0 or cb.size == 0:
            return 0.0
        return float(w[np.ix_(ra, cb)].sum())

    fgfg = block_sum(ia_fg, ib_fg)
    fgbg = block_sum(ia_fg, ib_bg)
    bgfg = block_sum(ia_bg, ib_fg)
    bgbg = block_sum(ia_bg, ib_bg)
    total = max(fgfg + fgbg + bgfg + bgbg, 1e-12)

    # "Purity" de massa originada em FG: quanto vai para FG.
    fg_source_total = max(fgfg + fgbg, 1e-12)

    return {
        "mass_fgfg_fraction_total": fgfg / total,
        "mass_fgbg_fraction_total": fgbg / total,
        "mass_bgfg_fraction_total": bgfg / total,
        "mass_bgbg_fraction_total": bgbg / total,
        "mass_fg_source_purity": fgfg / fg_source_total,
        "mass_fgfg_to_bgbg_ratio": fgfg / max(bgbg, 1e-12),
    }


def feature_rarity(f: np.ndarray, k: int) -> np.ndarray:
    """
    Peso em [1,2]: patches redundantes ~1, patches raros ~2.
    Usado apenas como ablação diagnóstica da massa de afinidade.
    """
    f = l2_normalize(f.astype(np.float64))
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)
    kk = min(k, max(1, len(f) - 1))
    top = np.partition(sim, -kk, axis=1)[:, -kk:]
    redundancy = np.mean(top, axis=1)
    rarity = 1.0 - redundancy
    lo, hi = float(rarity.min()), float(rarity.max())
    r = (rarity - lo) / (hi - lo + 1e-8)
    return 1.0 + r


def analyze_direction(
    fa: np.ndarray,
    fb: np.ndarray,
    gta: np.ndarray,
    gtb: np.ndarray,
    ks: List[int],
    temperature: float,
    rarity_k: int,
) -> Dict[str, float]:
    fa = l2_normalize(fa.astype(np.float64))
    fb = l2_normalize(fb.astype(np.float64))
    sim = fa @ fb.T

    out = correspondence_stats(sim, gta, gtb, ks)
    out.update(mutual_nn_purity(sim, gta, gtb))

    raw = soft_mass_stats(sim, gta, gtb, temperature, False, None)
    out.update({f"raw_{k}": v for k, v in raw.items()})

    deg = soft_mass_stats(sim, gta, gtb, temperature, True, None)
    out.update({f"degree_{k}": v for k, v in deg.items()})

    ra = feature_rarity(fa, rarity_k)
    rb = feature_rarity(fb, rarity_k)
    rar = soft_mass_stats(sim, gta, gtb, temperature, False, (ra, rb))
    out.update({f"rarity_{k}": v for k, v in rar.items()})

    deg_rar = soft_mass_stats(sim, gta, gtb, temperature, True, (ra, rb))
    out.update({f"degree_rarity_{k}": v for k, v in deg_rar.items()})

    return out


def nanmean_rows(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, (int, float)) and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnostica correspondencias FG/BG antes do NCut da E2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cases", nargs="+", required=True, type=Path)
    p.add_argument("--gt_index", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--topk_matches", nargs="+", type=int, default=[1, 5, 10])
    p.add_argument("--temperature", type=float, default=0.05,
                   help="Temperatura usada apenas para analisar massa de afinidade.")
    p.add_argument("--rarity_k", type=int, default=20)
    p.add_argument("--gt_threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gt_index = load_gt_index(args.gt_index)

    pair_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []

    for case_dir in args.cases:
        print("\n" + "=" * 92)
        print(f"[CASE] {case_dir.name}")
        data = load_case(case_dir)

        feats = data["patch_features"]
        N, P, D = feats.shape
        grid = data["patch_grid"]
        content = compute_content_mask(data["resize_info"], grid, N)
        gt, matched = load_patch_gt(
            data["image_paths"], grid, gt_index, data["resize_info"], args.gt_threshold
        )
        print(f"  GT encontrado: {matched}/{N}")
        print(f"  Pares E2: {len(data['top_pairs'])}")

        start_idx = len(pair_rows)

        for rank, pair in enumerate(data["top_pairs"]):
            i, j = int(pair[0]), int(pair[1])
            cls_sim = float(pair[2]) if len(pair) >= 3 else float("nan")

            vi = np.flatnonzero(content[i])
            vj = np.flatnonzero(content[j])
            if vi.size == 0 or vj.size == 0:
                continue

            fi = feats[i, vi]
            fj = feats[j, vj]
            gi = gt[i, vi]
            gj = gt[j, vj]

            # Analisa nos dois sentidos e agrega.
            ij = analyze_direction(fi, fj, gi, gj, args.topk_matches, args.temperature, args.rarity_k)
            ji = analyze_direction(fj, fi, gj, gi, args.topk_matches, args.temperature, args.rarity_k)

            row: Dict[str, Any] = {
                "case": case_dir.name,
                "pair_rank": rank,
                "image_i": i,
                "image_j": j,
                "cls_similarity": cls_sim,
                "fg_fraction_i": float(np.mean(gi)),
                "fg_fraction_j": float(np.mean(gj)),
                "n_valid_i": int(len(vi)),
                "n_valid_j": int(len(vj)),
            }

            keys = sorted(set(ij) | set(ji))
            for k in keys:
                vals = [x for x in (ij.get(k), ji.get(k))
                        if isinstance(x, (int, float)) and np.isfinite(x)]
                row[k] = float(np.mean(vals)) if vals else None

            pair_rows.append(row)

        rows = pair_rows[start_idx:]
        if not rows:
            print("  [AVISO] Nenhum par analisado.")
            continue

        # Sumario focado nas metricas que decidem o proximo experimento.
        summary: Dict[str, Any] = {
            "case": case_dir.name,
            "n_images": N,
            "n_pairs": len(rows),
            "gt_matched": matched,
            "mean_fg_fraction": float(np.mean(gt[content])),
        }

        important = [
            "fg_to_fg_hit_at_1",
            "fg_to_fg_precision_at_1",
            "fg_to_fg_hit_at_5",
            "fg_to_fg_precision_at_5",
            "fg_match_margin_mean",
            "fg_match_margin_positive_fraction",
            "mnn_fg_fg_fraction",
            "mnn_cross_fraction",
            "mnn_fg_recall",
            "raw_mass_fg_source_purity",
            "raw_mass_fgfg_fraction_total",
            "raw_mass_bgbg_fraction_total",
            "raw_mass_fgfg_to_bgbg_ratio",
            "degree_mass_fg_source_purity",
            "degree_mass_fgfg_to_bgbg_ratio",
            "rarity_mass_fg_source_purity",
            "rarity_mass_fgfg_to_bgbg_ratio",
            "degree_rarity_mass_fg_source_purity",
            "degree_rarity_mass_fgfg_to_bgbg_ratio",
        ]
        for key in important:
            summary[key] = nanmean_rows(rows, key)

        case_rows.append(summary)

        print(f"  FG->FG hit@1        : {summary['fg_to_fg_hit_at_1']:.4f}")
        print(f"  FG->FG precision@5  : {summary['fg_to_fg_precision_at_5']:.4f}")
        print(f"  FG match margin > 0 : {summary['fg_match_margin_positive_fraction']:.4f}")
        print(f"  MNN FG-FG fraction  : {summary['mnn_fg_fg_fraction']:.4f}")
        print(f"  Raw FG-source purity: {summary['raw_mass_fg_source_purity']:.4f}")
        print(f"  Degree purity       : {summary['degree_mass_fg_source_purity']:.4f}")
        print(f"  Rarity purity       : {summary['rarity_mass_fg_source_purity']:.4f}")
        print(f"  Degree+rarity purity: {summary['degree_rarity_mass_fg_source_purity']:.4f}")

    if not pair_rows:
        raise RuntimeError("Nenhum par analisado.")

    pair_csv = args.output / "e2_match_diagnostics_pairs.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for r in pair_rows for k in r.keys()})
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pair_rows)

    case_csv = args.output / "e2_match_diagnostics_cases.csv"
    with case_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(case_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(case_rows)

    out_json = args.output / "e2_match_diagnostics.json"
    out_json.write_text(
        json.dumps({"cases": case_rows, "pairs": pair_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 92)
    print(f"[Salvo] {pair_csv}")
    print(f"[Salvo] {case_csv}")
    print(f"[Salvo] {out_json}")


if __name__ == "__main__":
    main()
