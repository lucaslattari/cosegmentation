#!/usr/bin/env python3
"""
Concatenates per-image metrics from all 20 PASCAL-VOC class folders into a
single file, for cross-class per-image analysis (mirroring what was done
for the Airshows iCoseg groups).

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    python concat_pascal_metrics.py \\
        --results-root results/pascal_coseg_e2_spatial_degree_a0p10_bg_recurrence_l0p50_g0p00_e3bypass_top20_t0 \\
        --output pascal_per_image.json

If your PASCAL results folder has a different name, check
resumo_agregado.json's "metrics_path" field for the correct root -- it was
"results/pascal_coseg_e2_spatial_degree_a0p10_bg_recurrence_l0p50_g0p00_e3bypass_top20_t0/<class>/metrics.json"
as of the last export.

-------------------------------------------------------------------------
WHAT IT DOES
-------------------------------------------------------------------------
For each of the 20 class subfolders, loads the root-level metrics.json
(the final-stage result, same file used for the numbers reported in the
paper -- NOT the e4/ or e5/ subfolder variants) and extracts, per image:
    class, image_name, accuracy, precision, recall, iou, tp, tn, fp, fn,
    n_pixels, fg_pred_ratio, fg_gt_ratio, mask_path, gt_path

The oracle_* fields are dropped to keep the output file smaller, since
they aren't needed for the per-image failure-mode analysis. Re-add them
in FIELDS_TO_KEEP below if you want them.

Prints a summary (image count per class, total) to the console so you can
sanity-check before sending the file over.
"""

import argparse
import json
from pathlib import Path

FIELDS_TO_KEEP = [
    "accuracy", "precision", "recall", "iou", "tp", "tn", "fp", "fn",
    "n_pixels", "fg_pred_ratio", "fg_gt_ratio",
    "image_name", "mask_path", "gt_path",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", default="pascal_per_image.json")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"[!] results-root not found: {results_root}\n"
                          f"    Check resumo_agregado.json's 'metrics_path' field for the correct folder name.")

    class_dirs = sorted([d for d in results_root.iterdir() if d.is_dir()])
    if not class_dirs:
        raise SystemExit(f"[!] no subfolders found under {results_root}")

    all_rows = []
    summary = []

    for cls_dir in class_dirs:
        metrics_path = cls_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"  [!] skipping {cls_dir.name}: no metrics.json found")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        per_image = metrics.get("per_image", [])
        for pi in per_image:
            row = {"class": cls_dir.name}
            for field in FIELDS_TO_KEEP:
                row[field] = pi.get(field)
            all_rows.append(row)

        summary.append((cls_dir.name, len(per_image)))

    output = {
        "n_classes": len(summary),
        "n_images_total": len(all_rows),
        "per_image": all_rows,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=1)

    print(f"\nWrote {len(all_rows)} image records across {len(summary)} classes to {args.output}")
    print("\nPer-class image counts:")
    for name, n in summary:
        print(f"  {name:<15} {n}")

    import os
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nOutput file size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
