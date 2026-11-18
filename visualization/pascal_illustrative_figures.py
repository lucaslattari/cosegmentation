#!/usr/bin/env python3
"""
PASCAL-VOC Illustrative Figure Generator
==========================================
Generates two diff-overlay images (TP=green, FP=red, FN=blue) for the
Per-Class Performance Analysis section: one total-miss example (bimodal
collapse pattern) and one classic over-segmentation example, mirroring
the Airshows figure but WITHOUT any new hypothesis-testing -- these two
images are purely illustrative of patterns already established
numerically in the text.

Pre-selected candidates (from pascal_per_image.json):
    (a) pottedplant / img_1005  -- total miss (TP=0), fg_gt_ratio=0.059
    (b) diningtable / img_1014  -- over-segmentation (precision=0.104,
                                     recall=0.869, IoU=0.102)

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
Step 1 (run first -- locates the original color images, since only
mask_path/gt_path are already known):

    python pascal_illustrative_figures.py inspect \\
        --gt-root datasets/PASCAL/GT \\
        --candidates pottedplant:img_1005 diningtable:img_1014

This prints the known mask_path/gt_path for each candidate and tries a
few common PASCAL-VOC directory conventions to locate the original color
image, reporting what it finds (or doesn't).

Step 2 (once the original-image location is confirmed or given via
--images-root):

    python pascal_illustrative_figures.py report \\
        --images-root datasets/PASCAL/JPEGImages \\
        --candidates pottedplant:img_1005 diningtable:img_1014 \\
        --output-dir figures

Writes two PNGs to --output-dir: pascal_total_miss.png and
pascal_oversegmentation.png -- ready to reference directly in the LaTeX
figure environment.

-------------------------------------------------------------------------
KNOWN UNCERTAINTY
-------------------------------------------------------------------------
The exact folder layout for PASCAL's original color images was not
verified -- only mask_path and gt_path are confirmed (both live under
per-class subfolders: .../GT/<class>/<name>.png and
.../segmented_e5/<name>_mask.png). Standard PASCAL VOC distributions
usually keep original images in a single flat JPEGImages/ folder (not
split by class), so `inspect` tries that convention first, then falls
back to a per-class-subfolder guess. If neither matches, pass the
correct path explicitly via --images-root after checking your dataset
folder yourself.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Requires Pillow: pip install pillow --break-system-packages")


# Pre-filled from pascal_per_image.json -- edit here if you want different
# candidates than the two proposed.
CANDIDATE_INFO = {
    "pottedplant:img_1005": {
        "gt_path": "/mnt/e/github/cosegmentation/datasets/PASCAL/GT/pottedplant/img_1005.png",
        "mask_path": "/mnt/e/github/cosegmentation/features/pascal_coseg_e2_spatial_degree_a0p10_bg_recurrence_l0p50_g0p00_e3bypass_top20_t0/pottedplant/segmented_e5/img_1005_mask.png",
        "label": "total_miss",
    },
    "diningtable:img_1014": {
        "gt_path": None,  # reconstructed at runtime via resolve_paths()
        "mask_path": None,
        "label": "oversegmentation",
    },
}


def resolve_paths(candidate: str, gt_root: str = None):
    """Given 'class:name', return (gt_path, mask_path) -- either from the
    hardcoded table above, or reconstructed from gt_root using the same
    convention observed for pottedplant/img_1005."""
    if candidate in CANDIDATE_INFO and CANDIDATE_INFO[candidate]["gt_path"]:
        return CANDIDATE_INFO[candidate]["gt_path"], CANDIDATE_INFO[candidate]["mask_path"]

    cls, name = candidate.split(":")
    base_run = "/mnt/e/github/cosegmentation/features/pascal_coseg_e2_spatial_degree_a0p10_bg_recurrence_l0p50_g0p00_e3bypass_top20_t0"
    if gt_root is None:
        gt_root = "/mnt/e/github/cosegmentation/datasets/PASCAL/GT"
    gt_path = str(Path(gt_root) / cls / f"{name}.png")
    mask_path = str(Path(base_run) / cls / "segmented_e5" / f"{name}_mask.png")
    return gt_path, mask_path


def find_original_image(images_root: Path, cls: str, name: str):
    """Try a few common conventions for where the original color image lives."""
    candidates = [
        images_root / f"{name}.jpg",                  # flat JPEGImages/ convention
        images_root / f"{name}.png",
        images_root / cls / f"{name}.jpg",             # per-class subfolder convention
        images_root / cls / f"{name}.png",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def cmd_inspect(args):
    images_root = Path(args.images_root) if args.images_root else None
    for candidate in args.candidates:
        cls, name = candidate.split(":")
        gt_path, mask_path = resolve_paths(candidate, args.gt_root)
        print(f"\n{candidate}")
        print(f"  gt_path:   {gt_path}   exists={Path(gt_path).exists()}")
        print(f"  mask_path: {mask_path}   exists={Path(mask_path).exists()}")
        if images_root:
            found = find_original_image(images_root, cls, name)
            print(f"  original image: {'FOUND at ' + str(found) if found else 'NOT FOUND -- tried flat and per-class layouts under ' + str(images_root)}")
        else:
            print("  original image: (--images-root not given, skipping search)")

    print("\nIf 'original image' was NOT FOUND above, locate the correct folder "
          "manually (check your dataset directory) and pass it via --images-root "
          "for the `report` step.")


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


def cmd_report(args):
    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = {"total_miss": "pascal_total_miss.png", "oversegmentation": "pascal_oversegmentation.png"}

    for candidate in args.candidates:
        cls, name = candidate.split(":")
        gt_path, mask_path = resolve_paths(candidate, args.gt_root)
        label = CANDIDATE_INFO.get(candidate, {}).get("label", cls)

        img_path = find_original_image(images_root, cls, name)
        if img_path is None:
            print(f"[!] {candidate}: original image not found under {images_root} -- run `inspect` first")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        pred_mask = np.array(Image.open(mask_path).convert("L").resize(
            (image.shape[1], image.shape[0]))) > 127
        gt_mask = np.array(Image.open(gt_path).convert("L").resize(
            (image.shape[1], image.shape[0]))) > 127

        diag = compute_diagnostics(pred_mask, gt_mask)
        diff_img = overlay_diff(image, diag)

        out_name = filenames.get(label, f"{cls}_{name}.png")
        out_path = output_dir / out_name
        diff_img.save(out_path)
        print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("--images-root", default=None)
    p_inspect.add_argument("--gt-root", default=None)
    p_inspect.add_argument("--candidates", nargs="+", required=True,
                            help="e.g. pottedplant:img_1005 diningtable:img_1014")
    p_inspect.set_defaults(func=cmd_inspect)

    p_report = sub.add_parser("report")
    p_report.add_argument("--images-root", required=True)
    p_report.add_argument("--gt-root", default=None)
    p_report.add_argument("--candidates", nargs="+", required=True)
    p_report.add_argument("--output-dir", required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
