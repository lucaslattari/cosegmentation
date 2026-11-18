#!/usr/bin/env python3
"""
PASCAL Over-/Under-Segmentation Report
========================================
Generates an HTML report showing the most striking over-segmentation and
under-segmentation cases per PASCAL class, ranked by the precision-recall
gap already used to classify failure signatures in the paper text.

    gap = precision - recall
    gap << 0  ->  over-segmentation  (recall high, precision low: object
                  found, but excess area predicted around/beyond it)
    gap >> 0  ->  under-segmentation (precision high, recall low: predicted
                  area is mostly correct, but much of the object is missed)

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    python pascal_over_under_segmentation_report.py \\
        --per-image-json pascal_per_image.json \\
        --images-root datasets/PASCAL/JPEGImages \\
        --classes pottedplant chair diningtable bicycle person \\
        --top-n 4 \\
        --output-dir pascal_seg_report

If you don't know where the original color images live, run once with
--images-root pointing to your best guess; the script will print a
warning per missing image (and still generate the report using the
diff overlay's own dimensions, without the original photo) rather than
crashing. Check the warnings and adjust --images-root if most images are
missing.

--classes defaults to all 20 PASCAL classes if omitted.

-------------------------------------------------------------------------
WHAT EACH ROW SHOWS
-------------------------------------------------------------------------
Original image | diff overlay (TP=green, FP=red, FN=blue) | precision |
recall | iou | gap

Within each class, the report shows the --top-n images with the most
negative gap (clearest over-segmentation) and the --top-n images with the
most positive gap (clearest under-segmentation), so you can visually
compare the two failure signatures side by side.
"""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Requires Pillow: pip install pillow --break-system-packages")


IMAGE_CANDIDATES = [
    "{class_dir}/{name}.jpg", "{class_dir}/{name}.jpeg", "{class_dir}/{name}.png",
    "{name}.jpg", "{name}.jpeg", "{name}.png",
    "JPEGImages/{name}.jpg", "JPEGImages/{name}.jpeg", "JPEGImages/{name}.png",
]


def strip_mask_suffix(name: str) -> str:
    """metrics.json's image_name carries a '_mask' suffix (and possibly an
    extension) that the original color image filename does not have --
    confirmed for both the iCoseg and PASCAL exports."""
    stem = Path(name).stem  # drops extension, if any
    if stem.endswith("_mask"):
        stem = stem[: -len("_mask")]
    return stem


def find_original_image(images_root: Path, cls: str, image_name: str):
    stem = strip_mask_suffix(image_name)
    for pattern in IMAGE_CANDIDATES:
        cand = images_root / pattern.format(class_dir=cls, name=stem)
        if cand.exists():
            return cand
    return None


def compute_diff(image, pred_mask: np.ndarray, gt_mask: np.ndarray):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt

    if image is None:
        # No original photo available -- render the diff on a neutral gray canvas
        canvas = np.full((*pred.shape, 3), 128, dtype=np.uint8).astype(float)
    else:
        canvas = np.array(image.convert("RGB").resize((pred.shape[1], pred.shape[0]))).astype(float)

    colors = {"tp": (0, 200, 0), "fp": (220, 30, 30), "fn": (30, 60, 220)}
    for key, mask, color in [("tp", tp, colors["tp"]), ("fp", fp, colors["fp"]), ("fn", fn, colors["fn"])]:
        for c in range(3):
            canvas[..., c] = np.where(mask, 0.5 * canvas[..., c] + 0.5 * color[c], canvas[..., c])
    return Image.fromarray(canvas.astype(np.uint8))


def process_image(record, images_root, output_dir, asset_prefix, warnings):
    mask_path, gt_path = record["mask_path"], record["gt_path"]
    try:
        pred = np.array(Image.open(mask_path).convert("L")) > 127
        gt_img = Image.open(gt_path).convert("L")
        gt = np.array(gt_img) > 127
    except Exception as e:
        warnings.append(f"  [!] could not load masks for {record['class']}/{record['image_name']}: {e}")
        return None

    if pred.shape != gt.shape:
        pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]))) > 0

    orig = None
    orig_path = find_original_image(images_root, record["class"], record["image_name"]) if images_root else None
    if orig_path is not None:
        orig = Image.open(orig_path)
    elif images_root is not None:
        warnings.append(f"  [!] original image not found for {record['class']}/{record['image_name']} "
                         f"(tried common locations under {images_root})")

    diff_img = compute_diff(orig, pred, gt)
    asset_name = f"{asset_prefix}.png"
    diff_img.save(output_dir / "assets" / asset_name)
    return f"assets/{asset_name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-image-json", required=True)
    parser.add_argument("--images-root", default=None)
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    with open(args.per_image_json) as f:
        data = json.load(f)
    all_rows = data["per_image"]

    classes = args.classes or sorted(set(r["class"] for r in all_rows))
    images_root = Path(args.images_root) if args.images_root else None

    output_dir = Path(args.output_dir)
    (output_dir / "assets").mkdir(parents=True, exist_ok=True)

    warnings = []
    sections = []

    for cls in classes:
        rows = [r for r in all_rows if r["class"] == cls]
        for r in rows:
            r["gap"] = r["precision"] - r["recall"]

        overseg = sorted(rows, key=lambda r: r["gap"])[: args.top_n]          # most negative gap first
        underseg = sorted(rows, key=lambda r: -r["gap"])[: args.top_n]        # most positive gap first

        print(f"Processing {cls}: {len(overseg)} over-seg + {len(underseg)} under-seg examples ...")

        overseg_out, underseg_out = [], []
        for i, r in enumerate(overseg):
            asset = process_image(r, images_root, output_dir, f"{cls}_overseg_{i}", warnings)
            if asset:
                overseg_out.append({**r, "asset": asset})
        for i, r in enumerate(underseg):
            asset = process_image(r, images_root, output_dir, f"{cls}_underseg_{i}", warnings)
            if asset:
                underseg_out.append({**r, "asset": asset})

        sections.append((cls, overseg_out, underseg_out))

    write_html(output_dir / "report.html", sections)

    print(f"\nDone. Report at: {output_dir / 'report.html'}")
    if warnings:
        print(f"\n{len(warnings)} warnings:")
        for w in warnings[:20]:
            print(w)
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")


def write_html(path: Path, sections):
    parts = ["<html><head><meta charset='utf-8'><title>PASCAL Over-/Under-Segmentation Report</title>",
             "<style>body{font-family:sans-serif;margin:20px} table{border-collapse:collapse;width:100%;margin-bottom:20px}",
             "td,th{border:1px solid #ccc;padding:6px;text-align:left;font-size:13px}",
             "img{max-width:220px;display:block} h3{margin-top:30px}</style></head><body>",
             "<h1>PASCAL Over-/Under-Segmentation Report</h1>",
             "<p>Diff overlay: TP=green, FP=red, FN=blue. gap = precision - recall. "
             "Over-segmentation examples are sorted by most negative gap (clearest case first); "
             "under-segmentation examples by most positive gap.</p>"]

    for cls, overseg, underseg in sections:
        parts.append(f"<h2>{cls}</h2>")

        parts.append("<h3>Most over-segmented (gap &lt;&lt; 0)</h3>")
        parts.append("<table><tr><th>image</th><th>diff</th><th>precision</th><th>recall</th>"
                      "<th>iou</th><th>gap</th></tr>")
        for r in overseg:
            parts.append(f"<tr><td>{r['image_name']}</td><td><img src='{r['asset']}'></td>"
                         f"<td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>"
                         f"<td>{r['iou']:.3f}</td><td>{r['gap']:+.3f}</td></tr>")
        parts.append("</table>")

        parts.append("<h3>Most under-segmented (gap &gt;&gt; 0)</h3>")
        parts.append("<table><tr><th>image</th><th>diff</th><th>precision</th><th>recall</th>"
                      "<th>iou</th><th>gap</th></tr>")
        for r in underseg:
            parts.append(f"<tr><td>{r['image_name']}</td><td><img src='{r['asset']}'></td>"
                         f"<td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>"
                         f"<td>{r['iou']:.3f}</td><td>{r['gap']:+.3f}</td></tr>")
        parts.append("</table>")

    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
