#!/usr/bin/env python3
"""
Qualitative Pipeline Progression Figure Generator
====================================================
Generates a 6-panel horizontal figure showing the pipeline's progression
through its stages for a single representative image from the strongest
iCoseg group (Kite-Margate), for Section VI (Qualitative Results).

Panels: Original | E2B (fg_scores_2d heatmap) | E3 (fg_mask_pixel) |
        E4 (refined_masks) | E5 (final_masks) | Diff vs. GT (TP/FP/FN)

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
Step 1 -- list per-image IoU to pick a representative example (not
necessarily the single best image in the group, to avoid cherry-picking
the most flattering case):

    python qualitative_progression.py list \\
        --results-root results/icoseg38_topk250 \\
        --group "Kite-Margate_Kite_Festival_08-only_lines"

Step 2 -- generate the figure for a chosen image (use the stem printed
by `list`, e.g. an image whose IoU is close to the group's mean):

    python qualitative_progression.py report \\
        --features-root features/icoseg38_topk250 \\
        --results-root results/icoseg38_topk250 \\
        --group "Kite-Margate_Kite_Festival_08-only_lines" \\
        --image-stem <stem_from_list> \\
        --output figures/qualitative_progression.png

-------------------------------------------------------------------------
KNOWN UNCERTAINTY
-------------------------------------------------------------------------
This script assumes the same file schema already confirmed for other
iCoseg groups (Airshows): correspondence_bg.npz has `fg_scores_2d`,
cosaliency.npz has `fg_mask_pixel`, refined_masks.npz has
`refined_masks`, iterative_masks.npz has `final_masks`, and metrics.json
has per_image entries with `image_name`/`mask_path`/`gt_path` (with the
`_mask` suffix quirk on `image_name` already handled below). If
Kite-Margate's files differ, `list` will fail loudly and tell you where.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Requires Pillow: pip install pillow --break-system-packages")


def cmd_list(args):
    results_root = Path(args.results_root)
    metrics_path = results_root / args.group / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    per_image = metrics["per_image"]
    ious = [pi["iou"] for pi in per_image]
    mean_iou = np.mean(ious)
    print(f"Group: {args.group}")
    print(f"n_images={len(per_image)}  mean_iou={mean_iou:.4f}\n")

    print(f"{'image_name':<30} {'iou':>8} {'|iou-mean|':>12}")
    for pi in sorted(per_image, key=lambda p: abs(p["iou"] - mean_iou)):
        stem = Path(pi["image_name"]).stem
        if stem.endswith("_mask"):
            stem = stem[: -len("_mask")]
        print(f"{stem:<30} {pi['iou']:>8.4f} {abs(pi['iou']-mean_iou):>12.4f}")

    print("\nPick an image near the top of this list (closest to mean_iou) "
          "for the `report` step -- avoids cherry-picking the single best case.")


def load_stage_data(features_root: Path, results_root: Path, group: str, image_stem: str):
    corr_bg = np.load(features_root / group / "correspondence_bg.npz", allow_pickle=True)
    cosal = np.load(features_root / group / "cosaliency.npz", allow_pickle=True)
    refined = np.load(features_root / group / "refined_masks.npz", allow_pickle=True)
    iterative = np.load(features_root / group / "iterative_masks.npz", allow_pickle=True)

    image_paths = corr_bg["image_paths"]
    idx = None
    for i, p in enumerate(image_paths):
        if image_stem in str(p):
            idx = i
            break
    if idx is None:
        raise SystemExit(f"[!] image_stem '{image_stem}' not found in group image_paths")

    with open(results_root / group / "metrics.json") as f:
        metrics = json.load(f)
    pi = None
    for entry in metrics["per_image"]:
        stem = Path(entry["image_name"]).stem
        if stem.endswith("_mask"):
            stem = stem[: -len("_mask")]
        if stem == image_stem:
            pi = entry
            break
    if pi is None:
        raise SystemExit(f"[!] image_stem '{image_stem}' not found in metrics.json per_image")

    return {
        "image_path": str(image_paths[idx]),
        "fg_scores_2d": corr_bg["fg_scores_2d"][idx],       # E2B
        "fg_mask_pixel": cosal["fg_mask_pixel"][idx],       # E3
        "refined_mask": refined["refined_masks"][idx],      # E4
        "final_mask": iterative["final_masks"][idx],        # E5
        "mask_path": pi["mask_path"],
        "gt_path": pi["gt_path"],
        "iou": pi["iou"],
        "precision": pi["precision"],
        "recall": pi["recall"],
    }


def normalize_to_uint8(arr):
    arr = arr.astype(float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def heatmap_panel(arr_2d, out_size):
    """Simple red-scale heatmap for a 2D score map, resized to out_size."""
    norm = normalize_to_uint8(arr_2d)
    img = Image.fromarray(norm).resize(out_size, Image.NEAREST)
    arr = np.array(img)
    rgb = np.stack([arr, arr // 3, arr // 3], axis=-1).astype(np.uint8)  # reddish heatmap
    return Image.fromarray(rgb)


def mask_panel(mask_2d, out_size, threshold=0.5):
    """Binary or soft mask rendered as white-on-black, resized to out_size."""
    norm = normalize_to_uint8(mask_2d) if mask_2d.max() <= 1.0 and mask_2d.dtype != bool else \
        (mask_2d.astype(np.uint8) * 255)
    img = Image.fromarray(norm).resize(out_size, Image.NEAREST)
    rgb = np.stack([np.array(img)] * 3, axis=-1)
    return Image.fromarray(rgb)


def diff_panel(image, pred_mask, gt_mask):
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
    overlay = image.copy().astype(float)
    colors = {"tp": (0, 200, 0), "fp": (220, 30, 30), "fn": (30, 60, 220)}
    for key, mask in zip(colors, [tp, fp, fn]):
        color = colors[key]
        for c in range(3):
            overlay[..., c] = np.where(mask, 0.5 * overlay[..., c] + 0.5 * color[c], overlay[..., c])
    return Image.fromarray(overlay.astype(np.uint8))


def cmd_report(args):
    features_root = Path(args.features_root)
    results_root = Path(args.results_root)

    data = load_stage_data(features_root, results_root, args.group, args.image_stem)

    image = np.array(Image.open(data["image_path"]).convert("RGB"))
    out_size = (image.shape[1], image.shape[0])

    gt_mask = np.array(Image.open(data["gt_path"]).convert("L").resize(out_size)) > 127
    pred_mask = np.array(Image.open(data["mask_path"]).convert("L").resize(out_size)) > 127

    panels = [
        ("Original", Image.fromarray(image)),
        ("S2B (fg score)", heatmap_panel(data["fg_scores_2d"], out_size)),
        ("S3 (co-saliency)", mask_panel(data["fg_mask_pixel"], out_size)),
        ("S4 (SAM2 refined)", mask_panel(data["refined_mask"], out_size)),
        ("S5 (final)", Image.fromarray((pred_mask * 255).astype(np.uint8)).convert("RGB")),
        ("vs. GT", diff_panel(image, pred_mask, gt_mask)),
    ]

    # Stitch panels horizontally with small labels using PIL only (no matplotlib dependency)
    pad = 4
    label_h = 36  # increased to fit larger font
    panel_w, panel_h = out_size
    total_w = len(panels) * panel_w + (len(panels) - 1) * pad
    total_h = panel_h + label_h
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except OSError:
        try:
            font = ImageFont.load_default(size=26)  # Pillow >=10.1
        except TypeError:
            font = ImageFont.load_default()  # older Pillow, fixed small size fallback

    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, panel_img in panels:
        canvas.paste(panel_img.resize(out_size), (x, label_h))
        draw.text((x + 4, 4), label, fill=(0, 0, 0), font=font)
        x += panel_w + pad

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    def resized_fraction(arr_2d, out_size):
        img = Image.fromarray(normalize_to_uint8(arr_2d)).resize(out_size, Image.NEAREST)
        return (np.array(img) > 127).mean()

    print(f"\nPer-stage foreground fraction (for prose, resized to native resolution):")
    print(f"  E2B mean fg_score (raw, not resized -- continuous, see note): {data['fg_scores_2d'].mean():.4f}")
    print(f"  E3 fg_mask_pixel coverage: {resized_fraction(data['fg_mask_pixel'], out_size):.4f}")
    print(f"  E4 refined_mask coverage:  {resized_fraction(data['refined_mask'], out_size):.4f}")
    print(f"  E5 final_mask coverage:    {pred_mask.mean():.4f}")
    print(f"  GT coverage:               {gt_mask.mean():.4f}")
    print(f"\nFinal metrics for this image: iou={data['iou']:.4f} "
          f"precision={data['precision']:.4f} recall={data['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--results-root", required=True)
    p_list.add_argument("--group", required=True)
    p_list.set_defaults(func=cmd_list)

    p_report = sub.add_parser("report")
    p_report.add_argument("--features-root", required=True)
    p_report.add_argument("--results-root", required=True)
    p_report.add_argument("--group", required=True)
    p_report.add_argument("--image-stem", required=True)
    p_report.add_argument("--output", required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
