#!/usr/bin/env python3
"""
Color Contrast Hypothesis Test
================================
Tests whether foreground-to-background color contrast (object vs. sky)
explains the residual IoU gap left unexplained by object size alone --
specifically the Kite-Colt_Park vs. Airshows-helicopter pair, which have
nearly identical foreground area (8.3% vs 8.9%) but very different IoU
(0.841 vs 0.556).

-------------------------------------------------------------------------
METHOD
-------------------------------------------------------------------------
For each image in each group:
    fg_color = mean RGB inside the ground-truth foreground mask
    bg_color = mean RGB inside a "sky proxy" region: the top 30% of the
               image, EXCLUDING any pixels that are part of the GT mask
               (so the object itself doesn't contaminate the sky estimate
               if it happens to sit near the top of the frame)
    contrast = Euclidean distance between fg_color and bg_color in RGB
               space (0-441 range; higher = more visually distinct)

This is a simple, defensible proxy -- not a perceptual color-difference
metric (e.g. CIEDE2000 in Lab space would be more accurate perceptually,
but RGB Euclidean distance is standard, requires no extra dependencies,
and is sufficient to test the directional hypothesis: does the low-IoU
Airshows-helicopter group have LOWER contrast than the high-IoU
Kite-Colt_Park group, despite near-identical object size?).

Group-level mean contrast is then reported alongside group IoU and
fg_gt_ratio (already established) for you to inspect the pattern.

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    python color_contrast_hypothesis.py \\
        --features-root features/icoseg38_topk250 \\
        --results-root results/icoseg38_topk250 \\
        --groups "Kite-Margate_Kite_Festival_08-only_lines" "Kite-kitekid" \\
                 "Kite-Colt_Park_6-1-08-rreed3" "Kite-Brighton_kite_Festival-jadepike4" \\
                 "Airshows-planes" "Airshows-helicopter" \\
                 "Airshows-Huntsville_Airshow_Flying-ocreationsphotography_finally_back_after_computer" \\
        --output color_contrast_results.json

Send the resulting color_contrast_results.json back for validation --
it's a compact summary (group-level aggregates + per-image detail), safe
to share.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Requires Pillow: pip install pillow --break-system-packages")


def load_group_per_image(features_root: Path, results_root: Path, group: str):
    """Same matching logic validated for the Airshows diagnostic: metrics.json
    per_image `image_name` has a `_mask` suffix that correspondence.npz's
    `image_paths` does not."""
    metrics_path = results_root / group / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    per_image = {}
    for pi in metrics["per_image"]:
        stem = Path(pi["image_name"]).stem
        if stem.endswith("_mask"):
            stem = stem[: -len("_mask")]
        per_image[stem] = pi

    corr = np.load(features_root / group / "correspondence.npz", allow_pickle=True)
    image_paths = corr["image_paths"]

    rows = []
    n_unmatched = 0
    for img_path in image_paths:
        stem = Path(str(img_path)).stem
        pi = per_image.get(stem)
        if pi is None:
            n_unmatched += 1
            continue
        rows.append({
            "stem": stem,
            "image_path": str(img_path),
            "gt_path": pi["gt_path"],
            "iou": pi["iou"],
        })
    if n_unmatched:
        print(f"  [!] {group}: {n_unmatched} images unmatched, skipped")
    return rows


def compute_contrast(image_path: str, gt_path: str):
    image = np.array(Image.open(image_path).convert("RGB")).astype(float)
    gt = np.array(Image.open(gt_path).convert("L").resize(
        (image.shape[1], image.shape[0]))) > 127

    if gt.sum() == 0:
        return None  # empty GT, can't compute foreground color

    h = image.shape[0]
    top_region = np.zeros(gt.shape, dtype=bool)
    top_region[: int(h * 0.30), :] = True
    sky_proxy = top_region & ~gt

    if sky_proxy.sum() == 0:
        return None  # object fills entire top 30%, no sky proxy available

    fg_color = image[gt].mean(axis=0)
    bg_color = image[sky_proxy].mean(axis=0)
    contrast = float(np.linalg.norm(fg_color - bg_color))

    return {
        "contrast": contrast,
        "fg_color_rgb": fg_color.tolist(),
        "bg_color_rgb": bg_color.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--output", default="color_contrast_results.json")
    args = parser.parse_args()

    features_root = Path(args.features_root)
    results_root = Path(args.results_root)

    group_summaries = {}
    all_per_image = []

    for group in args.groups:
        print(f"Processing {group} ...")
        rows = load_group_per_image(features_root, results_root, group)

        contrasts = []
        for r in rows:
            result = compute_contrast(r["image_path"], r["gt_path"])
            if result is None:
                continue
            contrasts.append(result["contrast"])
            all_per_image.append({
                "group": group, "stem": r["stem"], "iou": r["iou"],
                "contrast": result["contrast"],
            })

        if contrasts:
            group_summaries[group] = {
                "n_images_used": len(contrasts),
                "mean_contrast": float(np.mean(contrasts)),
                "std_contrast": float(np.std(contrasts)),
            }
            print(f"  -> mean contrast = {group_summaries[group]['mean_contrast']:.1f} "
                  f"(n={len(contrasts)})")
        else:
            print(f"  [!] no valid images for {group}")

    output = {"group_summaries": group_summaries, "per_image": all_per_image}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=1)

    print(f"\nWrote results to {args.output}")
    print("\n=== SUMMARY (group mean contrast, sorted) ===")
    for group, s in sorted(group_summaries.items(), key=lambda kv: kv[1]["mean_contrast"]):
        print(f"  {group:<55} contrast={s['mean_contrast']:6.1f}  n={s['n_images_used']}")


if __name__ == "__main__":
    main()
