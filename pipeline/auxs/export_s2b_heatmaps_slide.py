#!/usr/bin/env python3
"""Exporta S2 versus S2B para slides, com escala consistente [0, 1].

Uso (da raiz do repositório):
  python pipeline/export_s2b_heatmaps_slide.py features/EXPERIMENTO/COLECAO --list
  python pipeline/export_s2b_heatmaps_slide.py features/EXPERIMENTO/COLECAO --index 0
  python pipeline/export_s2b_heatmaps_slide.py features/EXPERIMENTO/COLECAO

Utiliza os NPZ existentes (não recalcula nem modifica o pipeline):
- features.npz
- correspondence.npz (S2)
- correspondence_bg.npz (S2B)

Importante: o pipeline pode renormalizar os scores de S2B por imagem.
As cores exibem os valores SALVOS, e não a penalização subtraída antes da renormalização.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image

CMAP = "RdYlGn"
NORM = Normalize(vmin=0.0, vmax=1.0, clip=True)


def choose_image_path(raw: str, root: Path) -> Path | None:
    """Resolve paths relativos, absolutos, Windows e caminhos de outro checkout."""
    raw = str(raw)
    p = Path(raw)
    candidates = [p, root / p]
    norm = raw.replace("\\", "/")
    if re.match(r"^[a-zA-Z]:/", norm):
        drive = norm[0].lower()
        candidates.append(Path(f"/mnt/{drive}/{norm[3:]}"))
    else:
        candidates.append(Path(norm))
    for marker in ("datasets/icoseg/", "icoseg/"):
        if marker in norm:
            suffix = norm.split(marker, 1)[1]
            candidates.extend([root / "datasets/icoseg" / suffix, root / "icoseg" / suffix])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def to_image_pixels(patch_map: np.ndarray, size: tuple[int, int], resize_row) -> np.ndarray:
    """Inverte keep_ar/padding antes de projetar os scores na imagem original."""
    patch = np.asarray(patch_map, dtype=np.float32)
    if patch.ndim != 2:
        raise ValueError(f"Mapa de scores deve ter 2D, recebido {patch.shape}")
    img = Image.fromarray(patch, mode="F")
    if resize_row is not None:
        rh, rw, pt, pl, ch, cw = map(int, resize_row[:6])
        if min(rh, rw, ch, cw) <= 0:
            raise ValueError(f"resize_info inválido: {resize_row}")
        img = img.resize((cw, ch), Image.Resampling.BILINEAR)
        img = img.crop((pl, pt, pl + rw, pt + rh))
    img = img.resize(size, Image.Resampling.BILINEAR)
    return np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)


def overlay(source: Image.Image, score: np.ndarray, alpha: float) -> Image.Image:
    rgb = (plt.get_cmap(CMAP)(NORM(score))[..., :3] * 255).astype(np.uint8)
    return Image.blend(source, Image.fromarray(rgb, "RGB"), alpha)


def make_slide(original, before, after, target: Path, *, overlays: bool, alpha: float):
    """Figura 3 colunas com uma única barra de cores compartilhada."""
    rgb = np.asarray(original)
    panels = [rgb]
    if overlays:
        panels += [np.asarray(overlay(original, before, alpha)),
                   np.asarray(overlay(original, after, alpha))]
        titles = ["Original", "Raw foreground scores (S2) — overlay",
                  "Adjusted foreground scores (S2B) — overlay"]
    else:
        panels += [before, after]
        titles = ["Original", "Raw foreground scores (S2)",
                  "Adjusted foreground scores (S2B)"]

    fig = plt.figure(figsize=(12.8, 4.5), dpi=180)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.13], hspace=0.12,
                          left=0.018, right=0.982, top=0.92, bottom=0.09, wspace=0.055)
    for c, (panel, title) in enumerate(zip(panels, titles)):
        ax = fig.add_subplot(gs[0, c])
        if not overlays and c > 0:
            ax.imshow(panel, cmap=CMAP, norm=NORM, interpolation="bilinear")
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=11.3, fontweight="semibold", pad=7)
        ax.set_axis_off()
        ax.set_aspect("equal")
    # Barra única: a mesma escala é usada em ambos os mapas.
    cax = fig.add_subplot(gs[1, 1:3])
    bb = cax.get_position()
    cax.set_position([bb.x0 + 0.04, bb.y0 + 0.017, bb.width - 0.08, 0.034])
    mappable = ScalarMappable(norm=NORM, cmap=CMAP)
    mappable.set_array([])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal",
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=9, length=2)
    cb.set_label("Foreground score  |  0 = low (red), 1 = high (green)", fontsize=9, labelpad=3)
    fig.savefig(target, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path, help="Pasta contendo features.npz e correspondence*.npz")
    p.add_argument("--index", type=int, default=None, help="Índice base zero da imagem; padrão: todas")
    p.add_argument("--list", action="store_true", help="Listar os índices/imagens disponíveis, sem exportar")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Raiz do repositório (para localizar imagens)")
    p.add_argument("--alpha", type=float, default=0.53, help="Transparência dos overlays (0..1)")
    p.add_argument("--out", type=Path, default=None, help="Pasta de saída; padrão: <directory>/viz_s2b_slide")
    args = p.parse_args()
    if not 0 <= args.alpha <= 1:
        p.error("--alpha deve estar em [0,1]")
    root = args.root.resolve()
    directory = args.directory.resolve()
    srcs = [directory / "features.npz", directory / "correspondence.npz", directory / "correspondence_bg.npz"]
    missing = [str(path) for path in srcs if not path.is_file()]
    if missing:
        p.error("Arquivos .npz ausentes:\n  " + "\n  ".join(missing))

    with np.load(srcs[0], allow_pickle=True) as feat, \
         np.load(srcs[1], allow_pickle=True) as s2, \
         np.load(srcs[2], allow_pickle=True) as s2b:
        for which, obj in (("S2", s2), ("S2B", s2b)):
            for key in ("fg_scores_2d", "image_paths"):
                if key not in obj.files:
                    p.error(f"{which} não contém {key}")
        a = np.asarray(s2["fg_scores_2d"], dtype=np.float32)
        b = np.asarray(s2b["fg_scores_2d"], dtype=np.float32)
        paths = [str(item) for item in s2["image_paths"]]
        after_paths = [str(item) for item in s2b["image_paths"]]
        if a.shape != b.shape or a.ndim != 3 or len(paths) != a.shape[0]:
            p.error(f"Dimensões incompatíveis: S2={a.shape}, S2B={b.shape}, imagens={len(paths)}")
        if paths != after_paths:
            p.error("A ordem de image_paths é diferente entre S2 e S2B: não misture experimentos.")
        if "image_paths" in feat.files and paths != [str(item) for item in feat["image_paths"]]:
            p.error("A ordem de image_paths de S1 é diferente da ordem de S2.")
        ri = feat["resize_info"] if "resize_info" in feat.files else None
        if ri is not None and len(ri) != len(paths):
            p.error("resize_info tem número incorreto de linhas.")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            p.error("Foram encontrados NaN ou valores infinitos nos mapas.")
        tol = 1e-5
        if a.min() < -tol or b.min() < -tol or a.max() > 1 + tol or b.max() > 1 + tol:
            p.error("Os scores salvos não estão no intervalo [0,1]; confira os NPZ.")
        if args.list:
            for i, raw in enumerate(paths):
                print(f"{i:03d}\t{raw}")
            return
        if args.index is not None and not 0 <= args.index < len(paths):
            p.error(f"--index precisa estar entre 0 e {len(paths)-1}")
        indices = [args.index] if args.index is not None else range(len(paths))
        out = (args.out or directory / "viz_s2b_slide").resolve()
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for i in indices:
            original_path = choose_image_path(paths[i], root)
            if original_path is None:
                p.error(f"Imagem #{i:03d} não localizada: {paths[i]}")
            with Image.open(original_path) as im:
                original = im.convert("RGB")
            row = ri[i] if ri is not None else None
            before = to_image_pixels(a[i], original.size, row)
            after = to_image_pixels(b[i], original.size, row)
            name = re.sub(r"[^a-zA-Z0-9_.-]", "_", original_path.stem)[:65]
            prefix = f"{i:03d}_{name}"

            # Arquivos individuais, com nomenclatura original.
            for tag, patch, pixel in (("before", a[i], before), ("after", b[i], after)):
                plt.imsave(out / f"{prefix}_{tag}_s2b_heatmap.png", patch, cmap=CMAP, vmin=0, vmax=1)
                overlay(original, pixel, args.alpha).save(out / f"{prefix}_{tag}_s2b_overlay.png")
            original.save(out / f"{prefix}_original.png")
            make_slide(original, before, after, out / f"{prefix}_slide_heatmaps.png", overlays=False, alpha=args.alpha)
            make_slide(original, before, after, out / f"{prefix}_slide_overlays.png", overlays=True, alpha=args.alpha)

            rows.append({"index": i, "image": original_path.name, "prefix": prefix,
                         "s2_min": float(a[i].min()), "s2_mean": float(a[i].mean()),
                         "s2_max": float(a[i].max()), "s2b_min": float(b[i].min()),
                         "s2b_mean": float(b[i].mean()), "s2b_max": float(b[i].max())})
            print(f"[{i:03d}] {original_path.name}\n"
                  f"  S2 : min={a[i].min():.3f} mean={a[i].mean():.3f} max={a[i].max():.3f}\n"
                  f"  S2B: min={b[i].min():.3f} mean={b[i].mean():.3f} max={b[i].max():.3f}\n"
                  f"  {out / (prefix + '_slide_heatmaps.png')}")

        with (out / "index.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nImagens salvas em: {out}")
        print("Cores: RdYlGn, valor FIXO 0..1 nas imagens S2/S2B.")
        print("ATENÇÃO: a normalização interna feita por S2B não foi desfeita.")


if __name__ == "__main__":
    main()
