#!/usr/bin/env python3
"""
Airshows E2B Diagnostic v2 -- rewritten against the CONFIRMED file schema
from `inspect` output (2026-08). No more guessed filenames/keys.

Tests the hypothesis that recurring sky/background is competing with small
aircraft silhouettes for co-segmentation likelihood in the Airshows-* iCoseg
groups, using THREE independent lines of evidence per image:

  1. Final prediction vs. GT (mask_path / gt_path from metrics.json)
     -> where are the false positives, and are they concentrated in the
        upper half / low-color-variance (sky-like) regions?

  2. E2B effect (correspondence.npz vs correspondence_bg.npz `fg_scores_2d`)
     -> did background suppression actually lower foreground likelihood in
        the sky region, or did it (as aggregate stats already suggest for
        the Huntsville subgroup) fail to, or even push it the wrong way?

  3. E3 empty-foreground bypass flag (cosaliency.npz `empty_fg_bypass_triggered`)
     -> flagged directly per image, since this is a binary "the clustering
        stage gave up" signal worth seeing image-by-image, not just as an
        aggregate count.

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    python airshows_e2b_diagnostic_v2.py \\
        --features-root features/icoseg38_topk250 \\
        --results-root results/icoseg38_topk250 \\
        --classes "Airshows-planes" "Airshows-helicopter" \\
                  "Airshows-Huntsville_Airshow_Flying-ocreationsphotography_finally_back_after_computer" \\
        --output-dir ./airshows_report

No --images-root needed -- original image paths are read directly from
features.npz's `image_paths` field.

-------------------------------------------------------------------------
ONE REMAINING UNCERTAINTY
-------------------------------------------------------------------------
`metrics.json`'s `per_image` order is ASSUMED to match `image_name` exactly
matching the basename of paths in `features.npz`'s `image_paths` -- the
script matches by basename explicitly (not by list position) to be safe,
but if `image_name` in metrics.json uses a different naming convention
(e.g. without extension, or a hash), matching will silently drop images
and print a warning count at the end. Check that warning count is 0 after
your first run.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Requires Pillow: pip install pillow --break-system-packages")


def load_class_data(features_root: Path, results_root: Path, cls: str):
    """Loads and cross-references everything needed for one class."""
    feat_dir = features_root / cls
    res_dir = results_root / cls

    metrics_path = res_dir / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    per_image = {Path(pi["image_name"]).stem.removesuffix("_mask"): pi for pi in metrics["per_image"]}

    corr = np.load(feat_dir / "correspondence.npz", allow_pickle=True)
    corr_bg = np.load(feat_dir / "correspondence_bg.npz", allow_pickle=True)
    cosal = np.load(feat_dir / "cosaliency.npz", allow_pickle=True)

    image_paths = corr["image_paths"]  # (N,) array of path strings, same order across npz files for this class
    fg_before = corr["fg_scores_2d"]       # (N, 37, 37) before E2B
    fg_after = corr_bg["fg_scores_2d"]     # (N, 37, 37) after E2B
    bypass_flags = cosal["empty_fg_bypass_triggered"]  # (N,) bool
    fg_mask_e3 = cosal["fg_mask_pixel"]    # (N, H, W) E3-stage mask (pre-SAM), for reference

    rows = []
    n_unmatched = 0
    for i, img_path in enumerate(image_paths):
        stem = Path(str(img_path)).stem
        pi = per_image.get(stem)
        if pi is None:
            n_unmatched += 1
            continue
        rows.append({
            "stem": stem,
            "image_path": str(img_path),
            "pred_mask_path": pi["mask_path"],
            "gt_mask_path": pi["gt_path"],
            "precision": pi["precision"],
            "recall": pi["recall"],
            "iou": pi["iou"],
            "fg_before_2d": fg_before[i],
            "fg_after_2d": fg_after[i],
            "e2b_delta_mean": float(np.mean(fg_after[i] - fg_before[i])),
            "bypass_triggered": bool(bypass_flags[i]),
            "e3_fg_area_frac": float(np.mean(fg_mask_e3[i])),
        })

    if n_unmatched:
        print(f"  [!] WARNING: {n_unmatched} images in {cls} could not be matched "
              f"between metrics.json and correspondence.npz -- check image_name "
              f"vs image_paths naming convention.")

    return rows


def compute_mask_diagnostics(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt

    h = pred.shape[0]
    top_half = np.zeros_like(pred)
    top_half[: h // 2, :] = True
    fp_top_half_fraction = (fp & top_half).sum() / max(fp.sum(), 1)

    def region_color_std(mask):
        if mask.sum() == 0:
            return float("nan")
        px = image[mask]
        return float(np.mean(np.std(px.reshape(-1, px.shape[-1]), axis=0)))

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "fp_top_half_fraction": fp_top_half_fraction,
        "fp_color_std": region_color_std(fp),
        "gt_color_std": region_color_std(gt),
    }


def overlay_diff(image: np.ndarray, diag: dict) -> Image.Image:
    overlay = image.copy().astype(float)
    colors = {"tp": (0, 200, 0), "fp": (220, 30, 30), "fn": (30, 60, 220)}
    for key, color in colors.items():
        mask = diag[key]
        for c in range(3):
            overlay[..., c] = np.where(mask, 0.5 * overlay[..., c] + 0.5 * color[c], overlay[..., c])
    return Image.fromarray(overlay.astype(np.uint8))


def e2b_delta_heatmap(fg_before_2d: np.ndarray, fg_after_2d: np.ndarray, out_size) -> Image.Image:
    """Red = E2B raised fg likelihood here, Blue = E2B suppressed it here."""
    delta = fg_after_2d - fg_before_2d  # (37, 37), roughly in [-1, 1]
    delta_img = Image.fromarray(((delta + 1) / 2 * 255).astype(np.uint8)).resize(out_size, Image.NEAREST)
    delta_arr = np.array(delta_img).astype(float)
    rgb = np.zeros((*delta_arr.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((delta_arr - 127) * 2, 0, 255)       # red channel: positive delta
    rgb[..., 2] = np.clip((127 - delta_arr) * 2, 0, 255)       # blue channel: negative delta
    return Image.fromarray(rgb)


def cmd_report(args):
    features_root = Path(args.features_root)
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    (output_dir / "assets").mkdir(parents=True, exist_ok=True)

    sections = []
    for cls in args.classes:
        print(f"Processing {cls} ...")
        rows = load_class_data(features_root, results_root, cls)
        section_rows = []

        for r in rows:
            image = np.array(Image.open(r["image_path"]).convert("RGB"))
            pred_mask = np.array(Image.open(r["pred_mask_path"]).convert("L").resize(
                (image.shape[1], image.shape[0]))) > 127
            gt_mask = np.array(Image.open(r["gt_mask_path"]).convert("L").resize(
                (image.shape[1], image.shape[0]))) > 127

            diag = compute_mask_diagnostics(image, pred_mask, gt_mask)
            diff_img = overlay_diff(image, diag)
            e2b_img = e2b_delta_heatmap(r["fg_before_2d"], r["fg_after_2d"],
                                         (image.shape[1], image.shape[0]))

            base = f"{cls}__{r['stem']}"
            diff_img.save(output_dir / "assets" / f"{base}_diff.png")
            e2b_img.save(output_dir / "assets" / f"{base}_e2b.png")

            section_rows.append({
                "stem": r["stem"], "precision": r["precision"], "recall": r["recall"],
                "iou": r["iou"], "bypass": r["bypass_triggered"],
                "e2b_delta_mean": r["e2b_delta_mean"],
                "fp_top_half_fraction": diag["fp_top_half_fraction"],
                "fp_color_std": diag["fp_color_std"], "gt_color_std": diag["gt_color_std"],
                "diff_asset": f"assets/{base}_diff.png", "e2b_asset": f"assets/{base}_e2b.png",
            })

        section_rows.sort(key=lambda r: r["fp_top_half_fraction"], reverse=True)
        sections.append((cls, section_rows))

    write_html(output_dir / "report.html", sections)
    print(f"\nDone. Report at: {output_dir / 'report.html'}")


def write_html(path: Path, sections):
    parts = ["<html><head><meta charset='utf-8'><title>Airshows E2B Diagnostic v2</title>",
             "<style>body{font-family:sans-serif;margin:20px} table{border-collapse:collapse;width:100%}",
             "td,th{border:1px solid #ccc;padding:6px;text-align:left;font-size:13px;vertical-align:top}",
             "img{max-width:200px;display:block} .bypass{background:#ffe0e0}</style></head><body>",
             "<h1>Airshows E2B Diagnostic v2</h1>",
             "<p>Diff overlay: TP=green, FP=red, FN=blue. E2B delta map: red = foreground "
             "likelihood INCREASED after background suppression, blue = DECREASED "
             "(expect blue in sky/clutter regions if the hypothesis holds). "
             "Rows highlighted pink had the E3 empty-foreground bypass triggered. "
             "Sorted by fp_top_half_fraction (desc).</p>"]

    for cls, rows in sections:
        parts.append(f"<h2>{cls}</h2>")
        parts.append("<table><tr><th>image</th><th>diff (TP/FP/FN)</th><th>E2B delta map</th>"
                      "<th>precision</th><th>recall</th><th>iou</th>"
                      "<th>fp_top_half_frac</th><th>fp_color_std</th><th>gt_color_std</th>"
                      "<th>e2b_delta_mean</th><th>bypass</th></tr>")
        for r in rows:
            row_class = ' class="bypass"' if r["bypass"] else ""
            parts.append(
                f"<tr{row_class}><td>{r['stem']}</td>"
                f"<td><img src='{r['diff_asset']}'></td>"
                f"<td><img src='{r['e2b_asset']}'></td>"
                f"<td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['iou']:.3f}</td>"
                f"<td>{r['fp_top_half_fraction']:.3f}</td>"
                f"<td>{r['fp_color_std']:.1f}</td><td>{r['gt_color_std']:.1f}</td>"
                f"<td>{r['e2b_delta_mean']:+.4f}</td><td>{r['bypass']}</td></tr>"
            )
        parts.append("</table>")

    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-root", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    cmd_report(args)


if __name__ == "__main__":
    main()
