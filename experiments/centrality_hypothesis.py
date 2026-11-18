#!/usr/bin/env python3
"""
Centrality Hypothesis Test
============================
Tests whether pottedplant's total-miss images (IoU=0) tend to have their
ground-truth object positioned farther from the image center than images
where the pipeline succeeds -- motivated by the pipeline's use of a
centrality-scoring component (cluster_scores_centrality, observed in
cosaliency.npz during earlier investigation) that may bias the E3 stage
toward centrally-located objects.

-------------------------------------------------------------------------
METHOD
-------------------------------------------------------------------------
For each pottedplant image:
    1. Load the ground-truth mask.
    2. Compute the centroid of the GT foreground region.
    3. Compute the normalized distance from that centroid to the image
       center: distance / (half-diagonal of the image), so the result
       is roughly in [0, 1] regardless of image size/aspect ratio
       (0 = object centroid exactly at image center; larger = more
       peripheral).

Then compare this distance between the 15 total-miss images (IoU=0) and
the 26 remaining (non-zero IoU) images, and additionally compute the
correlation between centrality distance and IoU across all 41 images.

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    python centrality_hypothesis.py \\
        --pascal-per-image pascal_per_image.json

No other files needed -- this only reads GT masks via the gt_path field
already present in pascal_per_image.json.
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Requires Pillow: pip install pillow --break-system-packages")


def gt_centrality_distance(gt_path: str):
    gt = np.array(Image.open(gt_path).convert("L")) > 127
    h, w = gt.shape
    ys, xs = np.where(gt)
    if len(ys) == 0:
        return None  # empty GT, shouldn't happen but guard anyway

    centroid_y, centroid_x = ys.mean(), xs.mean()
    center_y, center_x = h / 2, w / 2

    dist = np.hypot(centroid_y - center_y, centroid_x - center_x)
    half_diag = np.hypot(h / 2, w / 2)
    return dist / half_diag


def pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = (sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y)) ** 0.5
    return num / den if den else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pascal-per-image", required=True)
    args = parser.parse_args()

    with open(args.pascal_per_image) as f:
        data = json.load(f)

    rows = [r for r in data["per_image"] if r["class"] == "pottedplant"]
    print(f"n pottedplant images: {len(rows)}")

    results = []
    for r in rows:
        gt_path = r["gt_path"]
        if not Path(gt_path).exists():
            print(f"  [!] {r['image_name']}: gt_path not found, skipping")
            continue
        dist = gt_centrality_distance(gt_path)
        if dist is None:
            print(f"  [!] {r['image_name']}: empty GT mask, skipping")
            continue
        results.append({"image_name": r["image_name"], "iou": r["iou"], "centrality_dist": dist})

    total_miss = [r for r in results if r["iou"] == 0.0]
    success = [r for r in results if r["iou"] > 0.0]

    print(f"\nProcessed {len(results)}/{len(rows)} images successfully")
    print(f"  total-miss (IoU=0): n={len(total_miss)}")
    print(f"  success (IoU>0):    n={len(success)}")

    if total_miss and success:
        mean_tm = statistics.mean([r["centrality_dist"] for r in total_miss])
        mean_s = statistics.mean([r["centrality_dist"] for r in success])
        std_tm = statistics.stdev([r["centrality_dist"] for r in total_miss]) if len(total_miss) > 1 else float("nan")
        std_s = statistics.stdev([r["centrality_dist"] for r in success]) if len(success) > 1 else float("nan")
        print(f"\nMean centrality distance (0=centered, ~1=corner):")
        print(f"  total-miss images: {mean_tm:.4f} (std={std_tm:.4f})")
        print(f"  success images:    {mean_s:.4f} (std={std_s:.4f})")
        print(f"  difference (total-miss minus success): {mean_tm - mean_s:+.4f}")

    all_dist = [r["centrality_dist"] for r in results]
    all_iou = [r["iou"] for r in results]
    print(f"\ncorr(centrality_dist, iou) across all {len(results)} images = {pearson(all_dist, all_iou):.3f}")
    print("(negative correlation expected if the centrality hypothesis holds: "
          "farther from center -> lower IoU)")

    print("\n=== Per-image detail, sorted by centrality distance (most peripheral first) ===")
    for r in sorted(results, key=lambda r: -r["centrality_dist"]):
        flag = "TOTAL MISS" if r["iou"] == 0.0 else f"iou={r['iou']:.3f}"
        print(f"  {r['image_name']:<20} centrality_dist={r['centrality_dist']:.4f}  {flag}")


if __name__ == "__main__":
    main()
