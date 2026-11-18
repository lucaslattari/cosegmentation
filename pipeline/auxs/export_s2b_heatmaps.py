#!/usr/bin/env python3
"""Exporta heatmaps comparáveis dos fg_scores antes/depois de E2B/S2B.

Uso: python pipeline/export_s2b_heatmaps.py features/<experimento>/<classe>
     python pipeline/export_s2b_heatmaps.py features/<experimento>/<classe> --index 0

Requer: numpy, Pillow, matplotlib. Não modifica os artefatos do pipeline.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def overlay_image(image: Image.Image, patch_map: np.ndarray, resize_row, alpha: float) -> Image.Image:
    """Mapeia grade de patches à imagem original, respeitando padding de keep_ar."""
    if patch_map.ndim != 2:
        raise ValueError(f"Esperado score 2D; obtido {patch_map.shape}")
    heat = Image.fromarray(np.asarray(patch_map, dtype=np.float32), mode="F")
    if resize_row is not None:
        resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = map(int, resize_row[:6])
        heat = heat.resize((canvas_w, canvas_h), Image.Resampling.BILINEAR)
        heat = heat.crop((pad_left, pad_top, pad_left + resized_w, pad_top + resized_h))
    heat = heat.resize(image.size, Image.Resampling.BILINEAR)
    score = np.clip(np.asarray(heat, dtype=np.float32), 0, 1)
    rgb = (plt.get_cmap("RdYlGn")(score)[..., :3] * 255).astype(np.uint8)
    colors = Image.fromarray(rgb, "RGB")
    return Image.blend(image, colors, alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Pasta que contém features.npz e correspondence*.npz")
    parser.add_argument("--index", type=int, default=None, help="Exportar só um índice (base zero); por padrão todas")
    parser.add_argument("--alpha", type=float, default=0.53, help="Opacidade das cores sobre a foto (0 a 1)")
    parser.add_argument("--out", type=Path, default=None, help="Diretório de saída; padrão: <directory>/viz_s2b")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha deve estar entre 0 e 1")
    directory = args.directory.resolve()
    out = args.out or directory / "viz_s2b"
    files = [directory / "features.npz", directory / "correspondence.npz", directory / "correspondence_bg.npz"]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        parser.error("Faltam arquivos: " + ", ".join(missing))

    with np.load(files[0], allow_pickle=True) as features, \
         np.load(files[1], allow_pickle=True) as before, \
         np.load(files[2], allow_pickle=True) as after:
        a = np.asarray(before["fg_scores_2d"])
        b = np.asarray(after["fg_scores_2d"])
        if a.shape != b.shape:
            raise ValueError(f"Incompatibilidade before {a.shape} x after {b.shape}")
        paths = [Path(str(s)) for s in before["image_paths"]]
        if len(paths) != len(a):
            raise ValueError("image_paths incompatível com fg_scores_2d")
        resize_info = features["resize_info"] if "resize_info" in features.files else None
        indices = range(len(paths)) if args.index is None else [args.index]
        if args.index is not None and not 0 <= args.index < len(paths):
            parser.error(f"--index precisa estar entre 0 e {len(paths)-1}")

        out.mkdir(parents=True, exist_ok=True)
        manifest = []
        for i in indices:
            path = paths[i]
            if not path.is_file():
                raise FileNotFoundError(f"Imagem original não encontrada: {path}")
            source = Image.open(path).convert("RGB")
            safe_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", path.stem)[:70]
            prefix = f"{i:03d}_{safe_stem}"
            row = resize_info[i] if resize_info is not None else None
            for tag, score in (("before", a[i]), ("after", b[i])):
                overlay = overlay_image(source, score, row, args.alpha)
                overlay_path = out / f"{prefix}_{tag}_s2b_overlay.png"
                overlay.save(overlay_path)
                # Mapa puro, na grade original de patches, sem interpolação nem reescalonamento.
                heatmap_path = out / f"{prefix}_{tag}_s2b_heatmap.png"
                plt.imsave(heatmap_path, score, cmap="RdYlGn", vmin=0, vmax=1)
            manifest.append(f"{i:03d}\t{path.name}\t{prefix}\n")
            print(f"[{i:03d}] {path.name} -> {prefix}_{{before,after}}_s2b_{{overlay,heatmap}}.png")
        (out / "index.txt").write_text("index\toriginal\tprefix\n" + "".join(manifest), encoding="utf-8")
    print(f"\nSalvos em: {out}")
    print("Escala fixa dos heatmaps: RdYlGn, vmin=0, vmax=1. Verde=maior FG score.")


if __name__ == "__main__":
    main()
