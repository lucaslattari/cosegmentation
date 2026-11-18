#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluate import build_gt_mapping, get_gt_index_multi, load_gt_index, load_mask  # noqa: E402


def safe_name(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in name).strip("_")


def load_e2(class_dir: Path) -> dict[str, Any]:
    corr_path = class_dir / "correspondence.npz"
    feat_path = class_dir / "features.npz"

    if not corr_path.exists():
        raise FileNotFoundError(f"correspondence.npz ausente: {corr_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"features.npz ausente: {feat_path}")

    corr = np.load(corr_path, allow_pickle=True)
    feat = np.load(feat_path, allow_pickle=True)

    if "fg_scores_2d" in corr.files:
        fg2d = np.asarray(corr["fg_scores_2d"], dtype=np.float32)
    else:
        fg = np.asarray(corr["fg_scores"], dtype=np.float32)
        grid = tuple(int(x) for x in corr["patch_grid"])
        fg2d = fg.reshape(len(fg), *grid)

    image_paths = [str(x) for x in corr["image_paths"].tolist()]
    resize_info = np.asarray(feat["resize_info"]) if "resize_info" in feat.files else None

    if len(image_paths) != fg2d.shape[0]:
        raise ValueError(f"N incompatível: image_paths={len(image_paths)} fg_scores_2d={fg2d.shape}")

    return {
        "corr_path": str(corr_path),
        "feat_path": str(feat_path),
        "fg2d": fg2d,
        "image_paths": image_paths,
        "resize_info": resize_info,
        "orientation": str(corr["orientation"].item()) if "orientation" in corr.files else "unknown",
        "pair_selection_mode": str(corr["pair_selection_mode"].item()) if "pair_selection_mode" in corr.files else "global_top_k",
        "k_per_image": int(corr["k_per_image"].item()) if "k_per_image" in corr.files else None,
        "n_pairs": int(len(corr["top_pairs"])) if "top_pairs" in corr.files else None,
    }


def mask_from_patch_grid(
    patch_binary: np.ndarray,
    gt_hw: tuple[int, int],
    resize_row: np.ndarray | None,
) -> np.ndarray:
    h_gt, w_gt = gt_hw
    patch_img = Image.fromarray((patch_binary.astype(np.uint8) * 255), mode="L")

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


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return inter / union if union else 0.0


def accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(pred.astype(bool) == gt.astype(bool)))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0

    def std(key: str) -> float:
        return float(np.std([float(r[key]) for r in rows])) if rows else 0.0

    n_comp_better = sum(r["oracle_side"] == "complement" for r in rows)
    n_raw_better = sum(r["oracle_side"] == "raw" for r in rows)
    n_tie = sum(r["oracle_side"] == "tie" for r in rows)

    return {
        "n_images": len(rows),
        "iou_raw_mean": avg("iou_raw"),
        "iou_raw_std": std("iou_raw"),
        "iou_complement_mean": avg("iou_complement"),
        "iou_complement_std": std("iou_complement"),
        "iou_oracle_mean": avg("iou_oracle"),
        "iou_oracle_std": std("iou_oracle"),
        "accuracy_raw_mean": avg("accuracy_raw"),
        "accuracy_complement_mean": avg("accuracy_complement"),
        "accuracy_oracle_iou_side_mean": avg("accuracy_oracle_iou_side"),
        "n_complement_better_iou": n_comp_better,
        "n_raw_better_iou": n_raw_better,
        "n_tie_iou": n_tie,
        "complement_better_fraction": n_comp_better / len(rows) if rows else 0.0,
        "oracle_gain_iou": avg("iou_oracle") - avg("iou_raw"),
    }


def classify_failure(summary: dict[str, Any]) -> str:
    raw = summary["iou_raw_mean"]
    oracle = summary["iou_oracle_mean"]
    gain = summary["oracle_gain_iou"]

    if raw < 0.20 and oracle >= 0.45 and gain >= 0.20:
        return "orientation_failure"
    if raw < 0.20 and oracle < 0.30:
        return "structural_failure"
    if gain >= 0.15:
        return "orientation_mixed"
    if raw >= 0.45:
        return "e2_reasonable"
    return "intermediate"


def analyze_class(
    features_root: Path,
    gt_index: dict,
    class_name: str,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    class_dir = features_root / safe_name(class_name)
    data = load_e2(class_dir)

    image_paths = data["image_paths"]
    gt_multi = get_gt_index_multi(gt_index)
    mapping, not_found, ambiguous = build_gt_mapping(
        image_paths,
        gt_multi,
        collection_hint=class_name,
    )

    path_to_idx = {str(p): i for i, p in enumerate(image_paths)}
    rows: list[dict[str, Any]] = []

    for img_path, gt_path in mapping.items():
        i = path_to_idx[str(img_path)]
        gt = load_mask(Path(gt_path))
        h_gt, w_gt = gt.shape

        score = data["fg2d"][i]
        raw_patch = score >= threshold
        resize_row = data["resize_info"][i] if data["resize_info"] is not None else None
        raw = mask_from_patch_grid(raw_patch, (h_gt, w_gt), resize_row)
        comp = ~raw

        raw_iou = iou(raw, gt)
        comp_iou = iou(comp, gt)
        raw_acc = accuracy(raw, gt)
        comp_acc = accuracy(comp, gt)

        eps = 1e-12
        if comp_iou > raw_iou + eps:
            side = "complement"
            oracle_iou = comp_iou
            oracle_acc = comp_acc
        elif raw_iou > comp_iou + eps:
            side = "raw"
            oracle_iou = raw_iou
            oracle_acc = raw_acc
        else:
            side = "tie"
            oracle_iou = raw_iou
            oracle_acc = max(raw_acc, comp_acc)

        rows.append({
            "class": class_name,
            "index": i,
            "image_name": Path(img_path).name,
            "image_path": str(img_path),
            "gt_path": str(gt_path),
            "threshold": threshold,
            "iou_raw": raw_iou,
            "iou_complement": comp_iou,
            "iou_oracle": oracle_iou,
            "oracle_side": side,
            "accuracy_raw": raw_acc,
            "accuracy_complement": comp_acc,
            "accuracy_oracle_iou_side": oracle_acc,
            "fg_raw_fraction": float(raw.mean()),
            "fg_gt_fraction": float(gt.mean()),
        })

    rows.sort(key=lambda r: r["index"])
    summary = summarize_rows(rows)
    summary.update({
        "class": class_name,
        "features_root": str(features_root),
        "correspondence": data["corr_path"],
        "features": data["feat_path"],
        "threshold": threshold,
        "orientation_recorded": data["orientation"],
        "pair_selection_mode": data["pair_selection_mode"],
        "k_per_image": data["k_per_image"],
        "n_pairs": data["n_pairs"],
        "n_not_found_gt": len(not_found),
        "n_ambiguous_gt": len(ambiguous),
    })
    summary["diagnosis"] = classify_failure(summary)
    return summary, rows


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico E2: raw vs complemento vs oracle por IoU.")
    parser.add_argument("--features-root", required=True, type=Path)
    parser.add_argument("--gt-index", required=True, type=Path)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    features_root = args.features_root.resolve()
    gt_index_path = args.gt_index.resolve()
    output = args.output.resolve()

    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("[ERRO] --threshold deve estar em [0,1].")

    gt_index = load_gt_index(gt_index_path)

    all_summaries = []
    all_rows = []

    print("=" * 100)
    print("ABLAÇÃO E2 — RAW vs COMPLEMENTO vs ORACLE DE ORIENTAÇÃO")
    print("=" * 100)
    print(f"Features root : {features_root}")
    print(f"GT index      : {gt_index_path}")
    print(f"Threshold     : {args.threshold:.3f}")
    print(f"Classes       : {', '.join(args.classes)}")

    for class_name in args.classes:
        print("\n" + "-" * 100)
        print(f"Classe: {class_name}")
        print("-" * 100)
        try:
            summary, rows = analyze_class(features_root, gt_index, class_name, args.threshold)
        except Exception as exc:
            print(f"[ERRO] {class_name}: {exc}")
            continue

        all_summaries.append(summary)
        all_rows.extend(rows)

        print(f"  Imagens                : {summary['n_images']}")
        print(f"  Pares E2               : {summary['n_pairs']}")
        print(f"  Seleção de pares       : {summary['pair_selection_mode']}")
        print(f"  Orientação registrada  : {summary['orientation_recorded']}")
        print(f"  IoU RAW                : {summary['iou_raw_mean']:.4f}")
        print(f"  IoU COMPLEMENTO        : {summary['iou_complement_mean']:.4f}")
        print(f"  IoU ORACLE             : {summary['iou_oracle_mean']:.4f}")
        print(f"  Ganho oracle vs raw    : {summary['oracle_gain_iou']:+.4f}")
        print(f"  Complemento melhor     : {summary['n_complement_better_iou']}/{summary['n_images']} ({summary['complement_better_fraction']:.1%})")
        print(f"  Diagnóstico            : {summary['diagnosis']}")

    output.mkdir(parents=True, exist_ok=True)
    summary_csv = output / "e2_orientation_summary.csv"
    per_image_csv = output / "e2_orientation_per_image.csv"
    result_json = output / "e2_orientation_results.json"

    save_csv(summary_csv, all_summaries)
    save_csv(per_image_csv, all_rows)

    result_json.write_text(
        json.dumps(
            {
                "features_root": str(features_root),
                "gt_index": str(gt_index_path),
                "threshold": args.threshold,
                "classes": all_summaries,
                "per_image": all_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("RESUMO")
    print("=" * 100)
    print(f"{'Classe':<16}{'IoU raw':>10}{'IoU comp':>10}{'IoU oracle':>12}{'Δ oracle':>11}{'Comp melhor':>14}  Diagnóstico")
    print("-" * 100)
    for s in all_summaries:
        print(
            f"{s['class']:<16}"
            f"{s['iou_raw_mean']:>10.4f}"
            f"{s['iou_complement_mean']:>10.4f}"
            f"{s['iou_oracle_mean']:>12.4f}"
            f"{s['oracle_gain_iou']:>+11.4f}"
            f"{s['complement_better_fraction']:>13.1%}"
            f"  {s['diagnosis']}"
        )

    print(f"\n[Salvo] {summary_csv}")
    print(f"[Salvo] {per_image_csv}")
    print(f"[Salvo] {result_json}")


if __name__ == "__main__":
    main()
