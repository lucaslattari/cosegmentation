#!/usr/bin/env python
"""
Distribuição de tamanhos do iCoseg e granularidade efetiva de patch.

Responde a duas perguntas para o paper:

  1. Qual região da imagem ORIGINAL cada patch de 14x14 do canvas resume?
     Como as imagens são reduzidas para o canvas SxS, um patch cobre
     14 * (maior_lado / S) pixels originais. Quanto maior a imagem de
     entrada, mais grosseiro o patch em termos absolutos.

  2. Que fração da grade de tokens e' padding?
     Depende da razao de aspecto: quanto mais alongada a imagem, mais
     padding o modo keep_ar introduz.

Dois modos de operacao:

  --features  (RECOMENDADO) le resize_info/image_sizes dos features.npz
              ja gerados pela Etapa 1. Geometria exata, sem reimplementar
              o preprocessamento.

  --images    le os headers das imagens e reimplementa o reescalonamento.
              Use quando ainda nao houver features extraidas. O
              arredondamento pode divergir em +-1 px do step1.

Uso:
    python analyze_image_scale_stats.py --features features/
    python analyze_image_scale_stats.py --images /caminho/iCoseg/images/
    python analyze_image_scale_stats.py --features features/ --csv stats.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PATCH = 14
CANVAS = 518
GRID = CANVAS // PATCH  # 37
P_TOTAL = GRID * GRID   # 1369

# Script vive em <repo>/ablation/, features ficam em <repo>/features/<run>/
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = REPO_ROOT / "features" / "icoseg38_final_e3bypass_top20_t0"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------------------
# Politicas de validade (espelham o codigo do pipeline)
# ---------------------------------------------------------------------------

def n_valid_overlap(new_h, new_w, pad_top, pad_left):
    """Politica de sobreposicao: usada em E2, E3 e E6."""
    starts = np.arange(GRID) * PATCH
    ends = starts + PATCH
    rows = (ends > pad_top) & (starts < pad_top + new_h)
    cols = (ends > pad_left) & (starts < pad_left + new_w)
    return int(rows.sum()) * int(cols.sum())


def n_valid_centroid(new_h, new_w, pad_top, pad_left):
    """Politica de centroide: usada no empty-FG bypass (E3) e no E2B."""
    centres = (np.arange(GRID) + 0.5) * PATCH
    rows = (centres >= pad_top) & (centres < pad_top + new_h)
    cols = (centres >= pad_left) & (centres < pad_left + new_w)
    return int(rows.sum()) * int(cols.sum())


# ---------------------------------------------------------------------------
# Coleta
# ---------------------------------------------------------------------------

def collect_from_features(root: Path):
    """Le geometria exata dos features.npz da Etapa 1."""
    npz_paths = sorted(root.rglob("features.npz"))
    if not npz_paths:
        sys.exit(f"[ERRO] nenhum features.npz encontrado em {root}")

    runs = {p.parent.parent for p in npz_paths}
    if len(runs) > 1:
        print(f"  [aviso] {len(runs)} pastas de run encontradas sob {root}.")
        print(f"          Aponte --features para uma run especifica se elas")
        print(f"          usarem img_size diferentes. Duplicatas serao removidas.")

    rows = []
    seen = {}
    n_dup = 0
    n_conflict = 0

    for npz_path in npz_paths:
        cls = npz_path.parent.name
        data = np.load(npz_path, allow_pickle=True)

        if "resize_info" not in data.files:
            print(f"  [aviso] {cls}: sem resize_info (modo 'square'), ignorado")
            continue

        resize_info = data["resize_info"]
        image_sizes = data["image_sizes"]
        paths = data["image_paths"] if "image_paths" in data.files else None

        for i in range(len(resize_info)):
            new_h, new_w, pad_top, pad_left, _, _ = (int(x) for x in resize_info[i])
            orig_h, orig_w = (int(x) for x in image_sizes[i])
            name = Path(str(paths[i])).name if paths is not None else f"{i:04d}"

            key = (cls, name)
            geom = (new_h, new_w, pad_top, pad_left)
            if key in seen:
                n_dup += 1
                if seen[key] != geom:
                    n_conflict += 1
                continue
            seen[key] = geom

            rows.append(dict(
                cls=cls, name=name,
                orig_w=orig_w, orig_h=orig_h,
                new_w=new_w, new_h=new_h,
                pad_top=pad_top, pad_left=pad_left,
            ))

    if n_dup:
        print(f"  [dedup] {n_dup} entradas repetidas removidas")
    if n_conflict:
        print(f"  [ATENCAO] {n_conflict} imagens com geometria DIVERGENTE entre")
        print(f"            runs — ha configuracoes de img_size diferentes.")
        print(f"            Rode apontando para uma unica pasta de run.")
    return rows


def collect_from_images(root: Path):
    """Le apenas o header das imagens e reimplementa o reescalonamento."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("[ERRO] Pillow necessario para --images")

    img_paths = sorted(p for p in root.rglob("*") if p.suffix in IMG_EXTS)
    if not img_paths:
        sys.exit(f"[ERRO] nenhuma imagem encontrada em {root}")

    rows = []
    for p in img_paths:
        with Image.open(p) as im:
            orig_w, orig_h = im.size  # header apenas, sem decodificar
        scale = CANVAS / max(orig_w, orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        rows.append(dict(
            cls=p.parent.name, name=p.name,
            orig_w=orig_w, orig_h=orig_h,
            new_w=new_w, new_h=new_h,
            pad_top=(CANVAS - new_h) // 2,
            pad_left=(CANVAS - new_w) // 2,
        ))
    return rows


# ---------------------------------------------------------------------------
# Derivacao e relatorio
# ---------------------------------------------------------------------------

def enrich(rows):
    for r in rows:
        longest = max(r["orig_w"], r["orig_h"])
        r["longest"] = longest
        # extensao, em pixels da imagem original, coberta por um patch
        r["patch_extent"] = PATCH * longest / CANVAS
        r["aspect"] = longest / min(r["orig_w"], r["orig_h"])
        r["n_overlap"] = n_valid_overlap(
            r["new_h"], r["new_w"], r["pad_top"], r["pad_left"])
        r["n_centroid"] = n_valid_centroid(
            r["new_h"], r["new_w"], r["pad_top"], r["pad_left"])
        r["pad_frac"] = 1.0 - r["n_overlap"] / P_TOTAL
    return rows


def pct(a, q):
    return float(np.percentile(a, q))


def report(rows):
    longest = np.array([r["longest"] for r in rows], dtype=float)
    extent = np.array([r["patch_extent"] for r in rows], dtype=float)
    aspect = np.array([r["aspect"] for r in rows], dtype=float)
    padfrac = np.array([r["pad_frac"] for r in rows], dtype=float)
    n_ovl = np.array([r["n_overlap"] for r in rows], dtype=float)
    n_cen = np.array([r["n_centroid"] for r in rows], dtype=float)

    n = len(rows)
    n_cls = len({r["cls"] for r in rows})

    print()
    print("=" * 68)
    print(f"  {n} imagens em {n_cls} classes  |  canvas {CANVAS}  |  "
          f"grade {GRID}x{GRID} = {P_TOTAL} patches")
    print("=" * 68)

    print("\n[1] Maior lado da imagem original (px)")
    print(f"  min {longest.min():.0f} | p5 {pct(longest,5):.0f} | "
          f"p25 {pct(longest,25):.0f} | mediana {pct(longest,50):.0f} | "
          f"p75 {pct(longest,75):.0f} | p95 {pct(longest,95):.0f} | "
          f"max {longest.max():.0f}")
    n_up = int((longest < CANVAS).sum())
    print(f"  imagens AMPLIADAS (maior lado < {CANVAS}): {n_up} "
          f"({100*n_up/n:.1f}%)")

    print("\n[2] Regiao da imagem ORIGINAL coberta por um patch (px por lado)")
    print(f"  min {extent.min():.1f} | p25 {pct(extent,25):.1f} | "
          f"mediana {pct(extent,50):.1f} | p75 {pct(extent,75):.1f} | "
          f"p95 {pct(extent,95):.1f} | max {extent.max():.1f}")
    print(f"  media {extent.mean():.1f}  ->  patch tipico cobre "
          f"~{extent.mean():.0f}x{extent.mean():.0f} px originais")

    print("\n[3] Razao de aspecto (maior lado / menor lado)")
    print(f"  mediana {pct(aspect,50):.2f} | p95 {pct(aspect,95):.2f} | "
          f"max {aspect.max():.2f}")

    print("\n[4] Padding na grade de tokens (politica de sobreposicao)")
    print(f"  fracao de padding: mediana {100*pct(padfrac,50):.1f}% | "
          f"media {100*padfrac.mean():.1f}% | max {100*padfrac.max():.1f}%")
    print(f"  patches validos: mediana {pct(n_ovl,50):.0f} de {P_TOTAL}")

    print("\n[5] Divergencia entre as duas politicas de validade")
    diff = n_ovl - n_cen
    print(f"  sobreposicao - centroide: mediana {np.median(diff):.0f} | "
          f"media {diff.mean():.1f} | max {diff.max():.0f} patches")
    print(f"  ({100*diff.mean()/P_TOTAL:.1f}% da grade, em media)")

    # classes com os patches mais grosseiros
    per_cls = {}
    for r in rows:
        per_cls.setdefault(r["cls"], []).append(r["patch_extent"])
    ranked = sorted(((np.mean(v), k, len(v)) for k, v in per_cls.items()),
                    reverse=True)
    print("\n[6] Classes com granularidade efetiva mais grosseira")
    for mean_ext, cls, cnt in ranked[:8]:
        print(f"  {cls:<28} {mean_ext:5.1f} px/patch   (n={cnt})")
    print("    ...")
    for mean_ext, cls, cnt in ranked[-3:]:
        print(f"  {cls:<28} {mean_ext:5.1f} px/patch   (n={cnt})")
    print()


def write_csv(rows, path: Path):
    cols = ["cls", "name", "orig_w", "orig_h", "longest", "aspect",
            "new_w", "new_h", "pad_top", "pad_left",
            "patch_extent", "n_overlap", "n_centroid", "pad_frac"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV] {len(rows)} linhas -> {path}")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--features", type=Path, nargs="?", const=DEFAULT_RUN,
                     default=None,
                     help=f"pasta de run com features.npz "
                          f"(default: {DEFAULT_RUN})")
    src.add_argument("--images", type=Path,
                     help="raiz com as imagens do iCoseg (estimativa)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="grava as estatisticas por imagem")
    args = ap.parse_args()

    if args.images:
        rows = collect_from_images(args.images)
        print(f"[fonte] headers de imagem em {args.images} (estimativa)")
    else:
        root = args.features or DEFAULT_RUN
        if not root.exists():
            sys.exit(f"[ERRO] {root} nao existe. Passe --features <pasta_da_run>")
        print(f"[fonte] resize_info de {root}")
        rows = collect_from_features(root)

    if not rows:
        sys.exit("[ERRO] nenhuma imagem coletada")

    rows = enrich(rows)
    report(rows)

    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
