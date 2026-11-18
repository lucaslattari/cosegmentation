#!/usr/bin/env python3
"""
sanity_check_masks.py

Monta paineis visuais (imagem original | GT | mascara prevista | sobreposicao)
para algumas imagens de classes escolhidas, usando os paths ja registrados
em metrics.json (mask_path, gt_path) -- sem precisar adivinhar convencao de
pasta.

Uso:
    python3 sanity_check_masks.py \
        --results_dir results/icoseg38_postfix_baseline \
        --classes "Gymnastics-1" "Ferrari 599 GTB Fiorano-599-auTomoTiVe FreAk !" \
                  "Monks-LAO PDR 2008-Rolandito" "Windmill-Wind Farm-John Alarcon" \
        --n_per_class 2 \
        --output sanity_check_panels

Depois, envie os PNGs gerados em sanity_check_panels/ de volta para eu
inspecionar visualmente.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def find_original_image(gt_path: Path) -> Path | None:
    """Deriva o path da imagem original a partir do path do GT, trocando
    'ground_truth' por 'images' e tentando extensoes comuns."""
    parts = list(gt_path.parts)
    try:
        idx = parts.index("ground_truth")
    except ValueError:
        return None
    parts[idx] = "images"
    base = Path(*parts)
    for ext in [".jpg", ".jpeg", ".JPG", ".png", ".bmp"]:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def make_panel(img_path, gt_path, mask_path, title, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    gt = np.array(Image.open(gt_path).convert("L")) > 127
    mask = np.array(Image.open(mask_path).convert("L")) > 127

    if img_path is not None and img_path.exists():
        img = np.array(Image.open(img_path).convert("RGB"))
        axes[0].imshow(img)
    else:
        axes[0].text(0.5, 0.5, "imagem original\nnao encontrada",
                      ha="center", va="center", transform=axes[0].transAxes)
        img = None
    axes[0].set_title("Imagem original")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title(f"GT (fg={gt.mean():.1%})")
    axes[1].axis("off")

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Predito (fg={mask.mean():.1%})")
    axes[2].axis("off")

    if img is not None:
        overlay = img.copy()
        h, w = mask.shape
        if overlay.shape[:2] != (h, w):
            overlay = np.array(Image.fromarray(overlay).resize((w, h)))
        red = overlay.copy()
        red[mask] = [255, 0, 0]
        blended = (0.5 * overlay + 0.5 * red).astype(np.uint8)
        axes[3].imshow(blended)
    else:
        axes[3].imshow(mask, cmap="gray")
    axes[3].set_title("Predito sobreposto (vermelho)")
    axes[3].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=Path)
    ap.add_argument("--classes", nargs="+", required=True)
    ap.add_argument("--n_per_class", type=int, default=2)
    ap.add_argument("--output", default="sanity_check_panels", type=Path)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    for cls in args.classes:
        safe_cls = "".join(c if c.isalnum() or c in "_- " else "_" for c in cls).replace(" ", "_")
        metrics_path = args.results_dir / safe_cls / "metrics.json"
        if not metrics_path.exists():
            # tenta variantes comuns de sanitizacao
            candidates = list(args.results_dir.glob(f"*{safe_cls[:15]}*/metrics.json"))
            if candidates:
                metrics_path = candidates[0]
            else:
                print(f"[Aviso] Nao achei metrics.json para '{cls}' (tentei {metrics_path})")
                continue

        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_image = data.get("per_image", [])[: args.n_per_class]

        if not per_image:
            print(f"[Aviso] Sem per_image em {metrics_path}")
            continue

        for entry in per_image:
            mask_path = Path(entry["mask_path"])
            gt_path = Path(entry["gt_path"])
            img_path = find_original_image(gt_path)
            iou = entry.get("iou")
            oracle_iou = entry.get("oracle_iou")
            flipped = entry.get("oracle_flip_applied")

            if oracle_iou is not None:
                title = (f"{cls} :: {entry['image_name']} :: "
                          f"iou={iou:.3f} oracle_iou={oracle_iou:.3f} flip={flipped}")
            else:
                title = f"{cls} :: {entry['image_name']} :: iou={iou:.3f}"

            out_name = f"{safe_cls}__{entry['image_name']}.png"
            out_path = args.output / out_name
            try:
                make_panel(img_path, gt_path, mask_path, title, out_path)
                print(f"[OK] {out_path}")
            except Exception as e:
                print(f"[ERRO] {entry['image_name']}: {e}")


if __name__ == "__main__":
    main()
