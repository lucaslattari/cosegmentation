#!/usr/bin/env python3
"""Exporta os heatmaps S2 (antes de S2B) e S2B de uma classe cosegmentada.

Uso (executar na raiz do projeto):
    python pipeline/export_s2_heatmaps.py 'features/icoseg38_topk250/Kite-Margate_Kite_Festival_08-only_lines'

Dependencias: numpy, pillow, matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from matplotlib import colormaps

RESAMPLE = Image.Resampling.BILINEAR
CMAP = colormaps["RdYlGn"]  # 0 = vermelho (baixo), 1 = verde (alto)


def locate_image(value: str, root: Path) -> Path | None:
    """Tenta paths gravados como relativos, absolutos, Windows e WSL."""
    raw = str(value).replace("\\", "/")
    candidates = [Path(raw), root / raw]
    if len(raw) > 2 and raw[1:3] == ":/":
        candidates.append(Path("/mnt") / raw[0].lower() / raw[3:])
    for marker in ("datasets/icoseg/", "icoseg/"):
        if marker in raw:
            suffix = raw.split(marker, 1)[1]
            candidates.extend((root / "datasets" / "icoseg" / suffix,
                               root / "icoseg" / suffix))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def score_to_pixels(patch_map: np.ndarray, target_wh: tuple[int, int], resize_row) -> np.ndarray:
    """Reverte o redimensionamento com padding e restaura a geometria da imagem."""
    im = Image.fromarray(np.asarray(patch_map, dtype=np.float32))
    if resize_row is not None:
        rh, rw, pt, pl, ch, cw = map(int, resize_row[:6])
        im = im.resize((cw, ch), RESAMPLE)
        im = im.crop((pl, pt, pl + rw, pt + rh))
    return np.clip(np.asarray(im.resize(target_wh, RESAMPLE), dtype=np.float32), 0, 1)


def heatmap_rgb(pixel_scores: np.ndarray) -> Image.Image:
    rgb = (CMAP(pixel_scores)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def make_overview(rows: list[tuple[int, str, list[Image.Image | None]]], dest: Path):
    headers = ("Imagem original", "S2 (antes)", "S2B (depois)", "S2 overlay", "S2B overlay")
    width, height = 195, 160
    gap, top, left = 14, 56, 70
    canvas = Image.new("RGB", (left + len(headers)*(width+gap), top + len(rows)*(height+32)+20), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), "Foreground scores: vermelho = baixo, verde = alto; escala fixa [0,1]", fill="black")
    for j, title in enumerate(headers):
        draw.text((left + j*(width+gap), 31), title, fill="black")
    for k, (i, basename, images) in enumerate(rows):
        y = top + k*(height+32)
        draw.text((9, y+5), f"#{i:02d}", fill="black")
        for j, im in enumerate(images):
            x = left + j*(width+gap)
            if im is not None:
                thumb = ImageOps.contain(im, (width, height), RESAMPLE)
                canvas.paste(thumb, (x+(width-thumb.width)//2, y+(height-thumb.height)//2))
            else:
                draw.text((x+7, y+50), "Original nao encontrada", fill="black")
        draw.text((left, y+height+4), basename[:115], fill="black")
    canvas.save(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Pasta com features.npz e correspondence*.npz")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Raiz do repositório")
    args = parser.parse_args()
    directory = args.directory.resolve()
    root = args.root.resolve()
    paths_npz = [directory / name for name in
                 ("features.npz", "correspondence.npz", "correspondence_bg.npz")]
    for path in paths_npz:
        if not path.is_file():
            parser.error(f"Arquivo nao encontrado: {path}")

    with np.load(paths_npz[0], allow_pickle=True) as features, \
         np.load(paths_npz[1], allow_pickle=True) as raw, \
         np.load(paths_npz[2], allow_pickle=True) as adjusted:
        a = np.asarray(raw["fg_scores_2d"], dtype=np.float32)
        b = np.asarray(adjusted["fg_scores_2d"], dtype=np.float32)
        images = [str(v) for v in raw["image_paths"]]
        resize_info = features["resize_info"] if "resize_info" in features.files else None
        if a.shape != b.shape or len(images) != len(a):
            raise ValueError("Shapes ou numero de imagens nao correspondem entre S2 e S2B")
        for data, label in ((adjusted, "S2B"), (features, "S1")):
            if "image_paths" in data.files and not np.array_equal(raw["image_paths"], data["image_paths"]):
                raise ValueError(f"A ordem das imagens de {label} difere de S2; nao misturar execucoes")
        if resize_info is not None and len(resize_info) != len(images):
            raise ValueError("resize_info inconsistente com o numero de imagens")

    dest = directory / "viz_s2_slides"
    dest.mkdir(exist_ok=True)
    overview_rows = []
    index_lines = []
    for i, raw_path in enumerate(images):
        path = locate_image(raw_path, root)
        original = Image.open(path).convert("RGB") if path is not None else None
        info = resize_info[i] if resize_info is not None else None
        if original is not None:
            wh = original.size
        elif info is not None:
            wh = (int(info[1]), int(info[0]))
        else:
            wh = (740, 740)
        h1 = heatmap_rgb(score_to_pixels(a[i], wh, info))
        h2 = heatmap_rgb(score_to_pixels(b[i], wh, info))
        name = f"{i:02d}"
        h1.save(dest / f"{name}_s2_raw.png")
        h2.save(dest / f"{name}_s2b_adjusted.png")
        o1 = Image.blend(original, h1, 0.55) if original is not None else None
        o2 = Image.blend(original, h2, 0.55) if original is not None else None
        if o1 is not None:
            o1.save(dest / f"{name}_s2_overlay.png")
            o2.save(dest / f"{name}_s2b_overlay.png")
        overview_rows.append((i, Path(raw_path.replace('\\', '/')).name, [original, h1, h2, o1, o2]))
        diff = float(np.mean(np.abs(a[i] - b[i])))
        index_lines.append(f"#{i:02d}  {raw_path}  |  mean_abs_delta={diff:.6f}")
        print(f"#{i:02d}: {Path(raw_path).name}  (diferenca media S2/S2B: {diff:.4f})")
    make_overview(overview_rows, dest / "overview_s2_vs_s2b.png")
    (dest / "index.txt").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"\nPNG salvos em: {dest}\nAbra overview_s2_vs_s2b.png para escolher a imagem.")


if __name__ == "__main__":
    main()
