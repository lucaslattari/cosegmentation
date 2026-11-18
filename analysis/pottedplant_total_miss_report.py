#!/usr/bin/env python3
"""
Pottedplant Total-Miss Report Generator
==========================================
Generates an HTML report with diff overlays (TP=green, FP=red, FN=blue)
for all 15 pottedplant images with IoU=0 (total miss), to determine
whether the "object-identity confusion" mechanism observed in one
representative case (img_1005) generalizes across the full total-miss
set, or whether other mechanisms (e.g. the large-background-structure
pattern seen in Airshows) are also present.

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    python pottedplant_total_miss_report.py \\
        --images-root datasets/PASCAL/images \\
        --output-dir pottedplant_report

Writes pottedplant_report/report.html plus one diff-overlay PNG per
image under pottedplant_report/assets/.

-------------------------------------------------------------------------
WHAT TO LOOK FOR WHEN REVIEWING report.html
-------------------------------------------------------------------------
For each image, check whether the false-positive (red) region corresponds
to:
    (a) a second, visually distinct plant-like object elsewhere in the
        scene (same mechanism as img_1005 -- "object-identity confusion")
    (b) a large, relatively uniform background structure unrelated to any
        plant (closer to the Airshows mechanism)
    (c) something else entirely (e.g. scattered small regions, no single
        dominant false-positive blob)

No further script run is needed after this -- classification is done by
eye from the report, then reported back as counts per category.

-------------------------------------------------------------------------
NOTE ON PATHS
-------------------------------------------------------------------------
mask_path and gt_path are reconstructed from the same run-folder
convention confirmed for img_1005 earlier. If any path doesn't exist,
the script skips that image and prints a warning rather than failing
outright, so a partial report is still generated.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Requires Pillow: pip install pillow --break-system-packages")


TOTAL_MISS_IMAGES = [
    "img_1005", "img_1006", "img_1007", "img_1010", "img_1012",
    "img_1015", "img_1017", "img_1018", "img_1028", "img_1033",
    "img_1035", "img_1036", "img_1037", "img_1038", "img_1039",
]

BASE_RUN = ("pascal_coseg_e2_spatial_degree_a0p10_bg_recurrence_l0p50_g0p00"
            "_e3bypass_top20_t0")


def resolve_paths(name: str, project_root: Path):
    gt_path = project_root / "datasets" / "PASCAL" / "GT" / "pottedplant" / f"{name}.png"
    mask_path = (project_root / "features" / BASE_RUN / "pottedplant"
                 / "segmented_e5" / f"{name}_mask.png")
    return gt_path, mask_path


def compute_diagnostics(pred_mask, gt_mask):
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    return {"tp": pred & gt, "fp": pred & ~gt, "fn": ~pred & gt}


def overlay_diff(image, diag):
    overlay = image.copy().astype(float)
    colors = {"tp": (0, 200, 0), "fp": (220, 30, 30), "fn": (30, 60, 220)}
    for key, color in colors.items():
        mask = diag[key]
        for c in range(3):
            overlay[..., c] = np.where(mask, 0.5 * overlay[..., c] + 0.5 * color[c], overlay[..., c])
    return Image.fromarray(overlay.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-root", required=True,
                         help="e.g. datasets/PASCAL/images")
    parser.add_argument("--project-root", default=".",
                         help="Repo root containing both datasets/ and features/ (default: current dir)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    images_root = Path(args.images_root) / "pottedplant"
    project_root = Path(args.project_root)

    output_dir = Path(args.output_dir)
    (output_dir / "assets").mkdir(parents=True, exist_ok=True)

    rows = []
    for name in TOTAL_MISS_IMAGES:
        img_path = images_root / f"{name}.jpg"
        gt_path, mask_path = resolve_paths(name, project_root)

        if not img_path.exists():
            print(f"[!] {name}: original image not found at {img_path}, skipping")
            continue
        if not gt_path.exists():
            print(f"[!] {name}: gt not found at {gt_path}, skipping")
            continue
        if not mask_path.exists():
            print(f"[!] {name}: mask not found at {mask_path}, skipping")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        pred_mask = np.array(Image.open(mask_path).convert("L").resize(
            (image.shape[1], image.shape[0]))) > 127
        gt_mask = np.array(Image.open(gt_path).convert("L").resize(
            (image.shape[1], image.shape[0]))) > 127

        diag = compute_diagnostics(pred_mask, gt_mask)
        diff_img = overlay_diff(image, diag)

        asset_name = f"{name}_diff.png"
        diff_img.save(output_dir / "assets" / asset_name)

        fp_frac = diag["fp"].mean()
        rows.append({"name": name, "asset": f"assets/{asset_name}", "fp_frac": fp_frac})
        print(f"  {name}: wrote diff overlay (fp_fraction={fp_frac:.3f})")

    # Write HTML report
    parts = ["<html><head><meta charset='utf-8'><title>Pottedplant Total-Miss Report</title>",
             "<style>body{font-family:sans-serif;margin:20px} "
             ".grid{display:flex;flex-wrap:wrap;gap:12px} "
             ".cell{width:260px;text-align:center;font-size:13px} "
             "img{max-width:250px}</style></head><body>",
             "<h1>Pottedplant Total-Miss Report (15 images, IoU=0)</h1>",
             "<p>TP=green, FP=red, FN=blue. For each image, classify the false-positive "
             "region as: (a) a second plant-like object [object-identity confusion], "
             "(b) a large uniform background structure [Airshows-like mechanism], or "
             "(c) other/no clear pattern.</p>",
             "<div class='grid'>"]

    for r in rows:
        parts.append(f"<div class='cell'><b>{r['name']}</b><br>"
                      f"<img src='{r['asset']}'><br>fp_fraction={r['fp_frac']:.3f}</div>")

    parts.append("</div></body></html>")
    (output_dir / "report.html").write_text("\n".join(parts), encoding="utf-8")

    print(f"\nDone. {len(rows)}/{len(TOTAL_MISS_IMAGES)} images processed.")
    print(f"Report at: {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
