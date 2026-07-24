#!/usr/bin/env python3
"""
analyze_e3_adaptive_bypass.py
=============================

Ablação barata da E3, sem reexecutar E1/E2/E2B/E3 e sem SAM.

Testa duas famílias de máscaras derivadas DIRETAMENTE dos scores E2B:
  - top-10/20/30% dos patches válidos por imagem;
  - thresholds absolutos (default: 0.30, 0.40, 0.50, 0.60, 0.70).

Para cada máscara E2B testa:
  direct_*   : substitui E3 em todas as imagens (diagnóstico).
  adaptive_* : mantém a máscara E3 atual e usa E2B apenas nas imagens em
               que a fração FG da E3 <= --trigger-fg-frac (default 1%).

O objetivo do adaptive_* é preservar classes boas do iCoseg/PASCAL:
se a E3 não colapsa, a máscara original é mantida.

GT é usado SOMENTE para medir IoU da ablação.

Exemplo PASCAL:
  python ablation/analyze_e3_adaptive_bypass.py \
      --features-root features/pascal_coseg_bg_recurrence_l0p50 \
      --gt-index datasets/PASCAL/gt_index.json \
      --classes person boat bottle bus \
      --output ablation_results/e3_bypass_pascal

Exemplo iCoseg:
  python ablation/analyze_e3_adaptive_bypass.py \
      --features-root features/icoseg38_bg_recurrence_l0p50 \
      --gt-index gt_index.json \
      --classes "Pandas" "Agra Taj Mahal" Kite Statue \
      --output ablation_results/e3_bypass_icoseg
"""

from __future__ import annotations

import argparse
import csv
import json
import re
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


def safe_name(name: str) -> str:
    # Mesmo comportamento prático do run_pipeline.sh:
    # espaços -> "_" e remove pontuação não suportada.
    s = name.replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in "_-")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def content_mask_from_resize_info(
    resize_info: np.ndarray | None,
    patch_grid: tuple[int, int],
    n_images: int,
) -> np.ndarray:
    hp, wp = patch_grid
    if resize_info is None:
        return np.ones((n_images, hp, wp), dtype=bool)

    masks = np.zeros((n_images, hp, wp), dtype=bool)
    for i in range(n_images):
        vals = [float(x) for x in np.asarray(resize_info[i]).tolist()]
        if len(vals) < 6:
            masks[i] = True
            continue

        resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = vals[:6]
        ys = (np.arange(hp) + 0.5) * canvas_h / hp
        xs = (np.arange(wp) + 0.5) * canvas_w / wp
        valid_y = (ys >= pad_top) & (ys < pad_top + resized_h)
        valid_x = (xs >= pad_left) & (xs < pad_left + resized_w)
        masks[i] = valid_y[:, None] & valid_x[None, :]
    return masks


def patch_mask_to_gt(
    patch_mask: np.ndarray,
    gt_hw: tuple[int, int],
    resize_row: np.ndarray | None,
) -> np.ndarray:
    h_gt, w_gt = gt_hw
    patch_img = Image.fromarray(
        np.asarray(patch_mask, dtype=np.uint8) * 255,
        mode="L",
    )

    if resize_row is None:
        return np.asarray(
            patch_img.resize((w_gt, h_gt), Image.NEAREST),
            dtype=np.uint8,
        ) > 128

    vals = [int(round(float(x))) for x in np.asarray(resize_row).tolist()]
    if len(vals) < 6:
        return np.asarray(
            patch_img.resize((w_gt, h_gt), Image.NEAREST),
            dtype=np.uint8,
        ) > 128

    resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = vals[:6]
    if min(resized_h, resized_w, canvas_h, canvas_w) <= 0:
        return np.asarray(
            patch_img.resize((w_gt, h_gt), Image.NEAREST),
            dtype=np.uint8,
        ) > 128

    canvas = patch_img.resize((canvas_w, canvas_h), Image.NEAREST)
    left = max(0, pad_left)
    top = max(0, pad_top)
    right = min(canvas_w, pad_left + resized_w)
    bottom = min(canvas_h, pad_top + resized_h)

    if right <= left or bottom <= top:
        return np.asarray(
            patch_img.resize((w_gt, h_gt), Image.NEAREST),
            dtype=np.uint8,
        ) > 128

    content = canvas.crop((left, top, right, bottom))
    return np.asarray(
        content.resize((w_gt, h_gt), Image.NEAREST),
        dtype=np.uint8,
    ) > 128


def pixel_mask_to_gt(
    pixel_mask: np.ndarray,
    image_hw: tuple[int, int],
    gt_hw: tuple[int, int],
) -> np.ndarray:
    h_img, w_img = image_hw
    arr = np.asarray(pixel_mask)
    arr = np.squeeze(arr)
    arr = arr[:h_img, :w_img] > 0.5

    if arr.shape == gt_hw:
        return arr

    h_gt, w_gt = gt_hw
    img = Image.fromarray(arr.astype(np.uint8) * 255, mode="L")
    return np.asarray(
        img.resize((w_gt, h_gt), Image.NEAREST),
        dtype=np.uint8,
    ) > 128


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return inter / union if union else 0.0


def top_fraction_mask(
    scores: np.ndarray,
    valid: np.ndarray,
    fraction: float,
) -> np.ndarray:
    out = np.zeros_like(valid, dtype=bool)
    idx = np.flatnonzero(valid.reshape(-1))
    if len(idx) == 0:
        return out

    n_take = max(1, int(np.ceil(float(fraction) * len(idx))))
    flat_scores = scores.reshape(-1)
    chosen_local = np.argpartition(
        flat_scores[idx],
        -n_take,
    )[-n_take:]
    chosen = idx[chosen_local]
    out.reshape(-1)[chosen] = True
    return out


def abs_threshold_mask(
    scores: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (scores >= threshold) & valid


def frac_tag(x: float) -> str:
    return f"{int(round(x * 100)):02d}"


def abs_tag(x: float) -> str:
    return f"{x:.2f}".replace(".", "p")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_class(
    features_root: Path,
    class_name: str,
) -> dict[str, Any]:
    class_dir = features_root / safe_name(class_name)
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
    patch_grid = tuple(int(x) for x in np.asarray(feat["patch_grid"]).tolist())
    n = len(image_paths)
    hp, wp = patch_grid

    if "fg_scores_2d" in corr:
        e2b = np.asarray(corr["fg_scores_2d"], dtype=np.float32)
    elif "fg_scores" in corr:
        e2b = np.asarray(corr["fg_scores"], dtype=np.float32).reshape(
            n, hp, wp
        )
    else:
        raise KeyError(f"{corr_path} sem fg_scores/fg_scores_2d")

    if e2b.shape != (n, hp, wp):
        e2b = e2b.reshape(n, hp, wp)

    if "fg_mask_patch" in cos:
        current_patch = np.asarray(cos["fg_mask_patch"]) > 0.5
        if current_patch.shape != (n, hp, wp):
            current_patch = current_patch.reshape(n, hp, wp)
    else:
        current_patch = np.zeros((n, hp, wp), dtype=bool)

    current_pixel = (
        np.asarray(cos["fg_mask_pixel"])
        if "fg_mask_pixel" in cos
        else None
    )

    valid = content_mask_from_resize_info(
        resize_info,
        patch_grid,
        n,
    )

    return {
        "class_dir": class_dir,
        "image_paths": image_paths,
        "image_sizes": image_sizes,
        "resize_info": resize_info,
        "patch_grid": patch_grid,
        "e2b": e2b,
        "current_patch": current_patch,
        "current_pixel": current_pixel,
        "valid": valid,
        "corr_path": corr_path,
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

    by_img = {str(k): str(v) for k, v in mapping.items()}
    gts, paths = [], []
    for p in image_paths:
        gt_path = Path(by_img[str(p)])
        gts.append(load_mask(gt_path).astype(bool))
        paths.append(str(gt_path))
    return gts, paths


def analyze_class(
    features_root: Path,
    gt_index: dict,
    class_name: str,
    trigger_fg_frac: float,
    top_fractions: list[float],
    abs_thresholds: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    d = load_class(features_root, class_name)
    gts, gt_paths = load_gts(d["image_paths"], gt_index, class_name)
    n = len(gts)

    current_masks_gt: list[np.ndarray] = []
    current_fg_fracs: list[float] = []

    for i in range(n):
        valid_i = d["valid"][i]
        denom = max(1, int(valid_i.sum()))
        current_fg_fracs.append(
            float((d["current_patch"][i] & valid_i).sum()) / denom
        )

        if d["current_pixel"] is not None:
            h_i, w_i = [int(x) for x in d["image_sizes"][i][:2]]
            current_masks_gt.append(
                pixel_mask_to_gt(
                    d["current_pixel"][i],
                    (h_i, w_i),
                    gts[i].shape,
                )
            )
        else:
            rr = (
                d["resize_info"][i]
                if d["resize_info"] is not None
                else None
            )
            current_masks_gt.append(
                patch_mask_to_gt(
                    d["current_patch"][i],
                    gts[i].shape,
                    rr,
                )
            )

    variants: dict[str, dict[str, Any]] = {
        "current": {
            "kind": "current",
            "value": None,
        }
    }

    for q in top_fractions:
        variants[f"direct_top{frac_tag(q)}"] = {
            "kind": "top",
            "value": q,
            "adaptive": False,
        }
        variants[f"adaptive_top{frac_tag(q)}"] = {
            "kind": "top",
            "value": q,
            "adaptive": True,
        }

    for t in abs_thresholds:
        tag = abs_tag(t)
        variants[f"direct_abs{tag}"] = {
            "kind": "abs",
            "value": t,
            "adaptive": False,
        }
        variants[f"adaptive_abs{tag}"] = {
            "kind": "abs",
            "value": t,
            "adaptive": True,
        }

    summary_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []

    variant_ious: dict[str, list[float]] = {
        name: [] for name in variants
    }
    variant_fg: dict[str, list[float]] = {
        name: [] for name in variants
    }
    variant_trigger: dict[str, list[bool]] = {
        name: [] for name in variants
    }

    for i in range(n):
        base_row: dict[str, Any] = {
            "class": class_name,
            "index": i,
            "image_name": Path(d["image_paths"][i]).name,
            "image_path": d["image_paths"][i],
            "gt_path": gt_paths[i],
            "current_fg_fraction": current_fg_fracs[i],
            "collapse_trigger": current_fg_fracs[i] <= trigger_fg_frac,
        }

        for name, spec in variants.items():
            if name == "current":
                pred = current_masks_gt[i]
                triggered = False
            else:
                adaptive = bool(spec["adaptive"])
                triggered = (
                    adaptive
                    and current_fg_fracs[i] <= trigger_fg_frac
                )

                if adaptive and not triggered:
                    pred = current_masks_gt[i]
                else:
                    if spec["kind"] == "top":
                        patch = top_fraction_mask(
                            d["e2b"][i],
                            d["valid"][i],
                            float(spec["value"]),
                        )
                    else:
                        patch = abs_threshold_mask(
                            d["e2b"][i],
                            d["valid"][i],
                            float(spec["value"]),
                        )

                    rr = (
                        d["resize_info"][i]
                        if d["resize_info"] is not None
                        else None
                    )
                    pred = patch_mask_to_gt(
                        patch,
                        gts[i].shape,
                        rr,
                    )

            score = iou(pred, gts[i])
            variant_ious[name].append(score)
            variant_fg[name].append(float(pred.mean()))
            variant_trigger[name].append(triggered)
            base_row[f"iou_{name}"] = score
            base_row[f"fg_fraction_{name}"] = float(pred.mean())
            base_row[f"triggered_{name}"] = triggered

        per_image_rows.append(base_row)

    current_mean = float(np.mean(variant_ious["current"]))

    for name in variants:
        mean_iou = float(np.mean(variant_ious[name]))
        triggered = int(sum(variant_trigger[name]))
        summary_rows.append({
            "class": class_name,
            "variant": name,
            "n_images": n,
            "iou_mean": mean_iou,
            "iou_std": float(np.std(variant_ious[name])),
            "delta_vs_current": mean_iou - current_mean,
            "fg_fraction_mean": float(np.mean(variant_fg[name])),
            "trigger_count": triggered,
            "trigger_rate": triggered / n if n else 0.0,
            "trigger_fg_frac": trigger_fg_frac,
            "correspondence_source": str(d["corr_path"]),
        })

    return summary_rows, per_image_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-root", required=True, type=Path)
    p.add_argument("--gt-index", required=True, type=Path)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--trigger-fg-frac", type=float, default=0.01)
    p.add_argument(
        "--top-fractions",
        nargs="+",
        type=float,
        default=[0.10, 0.20, 0.30],
    )
    p.add_argument(
        "--abs-thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.40, 0.50, 0.60, 0.70],
    )
    args = p.parse_args()

    features_root = args.features_root.resolve()
    gt_index_path = args.gt_index.resolve()
    output = args.output.resolve()

    gt_index = load_gt_index(gt_index_path)

    summaries: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []

    print("=" * 110)
    print("E3 ADAPTIVE BYPASS — DIAGNÓSTICO SEM SAM")
    print("=" * 110)
    print(f"Features root  : {features_root}")
    print(f"Trigger FG     : <= {args.trigger_fg_frac:.2%}")
    print(f"Top fractions  : {args.top_fractions}")
    print(f"Abs thresholds : {args.abs_thresholds}")
    print(f"Classes        : {', '.join(args.classes)}")

    for class_name in args.classes:
        print("\n" + "-" * 110)
        print(f"Classe: {class_name}")
        print("-" * 110)
        try:
            srows, prows = analyze_class(
                features_root,
                gt_index,
                class_name,
                args.trigger_fg_frac,
                args.top_fractions,
                args.abs_thresholds,
            )
        except Exception as exc:
            print(f"[ERRO] {class_name}: {exc}")
            continue

        summaries.extend(srows)
        per_image.extend(prows)

        srows_sorted = sorted(
            srows,
            key=lambda r: r["iou_mean"],
            reverse=True,
        )
        current = next(r for r in srows if r["variant"] == "current")
        print(f"  Current E3 IoU : {current['iou_mean']:.4f}")
        print(
            f"  Imagens colapsadas (<= trigger): "
            f"{sum(p['collapse_trigger'] for p in prows)}/{len(prows)}"
        )
        print("  Melhores variantes:")
        for r in srows_sorted[:6]:
            print(
                f"    {r['variant']:<22} "
                f"IoU={r['iou_mean']:.4f} "
                f"Δ={r['delta_vs_current']:+.4f} "
                f"trigger={r['trigger_rate']:.1%}"
            )

    # Agregado macro por variante.
    macro_rows: list[dict[str, Any]] = []
    variants = sorted(set(r["variant"] for r in summaries))
    for variant in variants:
        rows = [r for r in summaries if r["variant"] == variant]
        if not rows:
            continue
        macro_rows.append({
            "class": "MACRO",
            "variant": variant,
            "n_images": sum(int(r["n_images"]) for r in rows),
            "iou_mean": float(np.mean([r["iou_mean"] for r in rows])),
            "iou_std": float(np.std([r["iou_mean"] for r in rows])),
            "delta_vs_current": float(
                np.mean([r["delta_vs_current"] for r in rows])
            ),
            "fg_fraction_mean": float(
                np.mean([r["fg_fraction_mean"] for r in rows])
            ),
            "trigger_count": sum(int(r["trigger_count"]) for r in rows),
            "trigger_rate": float(
                np.mean([r["trigger_rate"] for r in rows])
            ),
            "trigger_fg_frac": args.trigger_fg_frac,
            "correspondence_source": "",
        })

    all_summary_rows = summaries + macro_rows

    output.mkdir(parents=True, exist_ok=True)
    summary_csv = output / "e3_adaptive_bypass_summary.csv"
    per_image_csv = output / "e3_adaptive_bypass_per_image.csv"
    json_path = output / "e3_adaptive_bypass_results.json"

    write_csv(summary_csv, all_summary_rows)
    write_csv(per_image_csv, per_image)
    json_path.write_text(
        json.dumps(
            {
                "features_root": str(features_root),
                "gt_index": str(gt_index_path),
                "trigger_fg_frac": args.trigger_fg_frac,
                "top_fractions": args.top_fractions,
                "abs_thresholds": args.abs_thresholds,
                "summary": all_summary_rows,
                "per_image": per_image,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("RANKING MACRO")
    print("=" * 110)
    for r in sorted(macro_rows, key=lambda x: x["iou_mean"], reverse=True):
        print(
            f"{r['variant']:<24} "
            f"IoU={r['iou_mean']:.4f} "
            f"Δ={r['delta_vs_current']:+.4f} "
            f"trigger médio={r['trigger_rate']:.1%}"
        )

    print(f"\n[Salvo] {summary_csv}")
    print(f"[Salvo] {per_image_csv}")
    print(f"[Salvo] {json_path}")


if __name__ == "__main__":
    main()
