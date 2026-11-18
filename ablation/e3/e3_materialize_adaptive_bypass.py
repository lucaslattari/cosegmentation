#!/usr/bin/env python3
"""
materialize_e3_adaptive_bypass.py
=================================

Cria um novo experimento de features reaproveitando E1/E2/E2B e substituindo
somente a máscara/seeds da E3 por uma variante de bypass E2B.

Variantes:
  adaptive_top10
  adaptive_top20
  adaptive_top30
  adaptive_abs0p30
  adaptive_abs0p40
  adaptive_abs0p50
  adaptive_abs0p60
  adaptive_abs0p70

Também aceita direct_* para diagnóstico, embora o uso recomendado para E4/E5
seja adaptive_*.

A máscara E3 original é mantida nas imagens em que:
  fg_fraction_E3 > --trigger-fg-frac

Nas imagens colapsadas:
  - topXX: usa os XX% patches E2B de maior score;
  - absXpYY: usa fg_scores_E2B >= threshold.

O cosaliency.npz de destino preserva todos os campos originais e atualiza:
  fg_mask_patch
  fg_mask_pixel
  fg_mask_coarse
  seeds
além de adicionar metadados adaptive_bypass_*.

Exemplo:
  python ablation/materialize_e3_adaptive_bypass.py \
      --source-root features/pascal_coseg_bg_recurrence_l0p50 \
      --dest-root features/pascal_coseg_bg_recurrence_l0p50_e3bypass_top20 \
      --variant adaptive_top20 \
      --classes person boat bottle bus
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.step3_cosaliency import extract_seeds, upsample_masks  # noqa: E402


def safe_name(name: str) -> str:
    s = name.replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch in "_-")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def content_mask_from_resize_info(
    resize_info: np.ndarray | None,
    patch_grid: tuple[int, int],
    n_images: int,
) -> np.ndarray:
    hp, wp = patch_grid
    if resize_info is None:
        return np.ones((n_images, hp, wp), dtype=bool)

    masks = np.zeros((n_images, hp, wp), dtype=bool)
    for i in range(n_images):
        vals = [float(x) for x in np.asarray(resize_info[i]).tolist()]
        if len(vals) < 6:
            masks[i] = True
            continue
        resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = vals[:6]
        ys = (np.arange(hp) + 0.5) * canvas_h / hp
        xs = (np.arange(wp) + 0.5) * canvas_w / wp
        masks[i] = (
            ((ys >= pad_top) & (ys < pad_top + resized_h))[:, None]
            & ((xs >= pad_left) & (xs < pad_left + resized_w))[None, :]
        )
    return masks


def top_fraction_mask(
    scores: np.ndarray,
    valid: np.ndarray,
    fraction: float,
) -> np.ndarray:
    out = np.zeros_like(valid, dtype=bool)
    idx = np.flatnonzero(valid.reshape(-1))
    if len(idx) == 0:
        return out
    n_take = max(1, int(np.ceil(fraction * len(idx))))
    flat = scores.reshape(-1)
    chosen_local = np.argpartition(flat[idx], -n_take)[-n_take:]
    out.reshape(-1)[idx[chosen_local]] = True
    return out


def parse_variant(name: str) -> tuple[bool, str, float]:
    m = re.fullmatch(r"(adaptive|direct)_top(\d+)", name)
    if m:
        adaptive = m.group(1) == "adaptive"
        frac = int(m.group(2)) / 100.0
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"Top fraction inválida: {name}")
        return adaptive, "top", frac

    m = re.fullmatch(r"(adaptive|direct)_abs(\d+)p(\d+)", name)
    if m:
        adaptive = m.group(1) == "adaptive"
        value = float(f"{m.group(2)}.{m.group(3)}")
        return adaptive, "abs", value

    raise ValueError(
        "Variante inválida. Exemplos: adaptive_top20, adaptive_abs0p50"
    )


def ensure_symlink(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(src.resolve()), str(dst))


def materialize_class(
    source_root: Path,
    dest_root: Path,
    class_name: str,
    variant: str,
    trigger_fg_frac: float,
    force: bool,
) -> None:
    adaptive, kind, value = parse_variant(variant)

    safe = safe_name(class_name)
    src = source_root / safe
    dst = dest_root / safe

    feat_path = src / "features.npz"
    cos_path = src / "cosaliency.npz"
    corr_path = src / "correspondence_bg.npz"
    if not corr_path.exists():
        corr_path = src / "correspondence.npz"

    feat = load_npz(feat_path)
    cos = load_npz(cos_path)
    corr = load_npz(corr_path)

    image_paths = [str(x) for x in feat["image_paths"].tolist()]
    image_sizes = np.asarray(feat["image_sizes"])
    resize_info = (
        np.asarray(feat["resize_info"])
        if "resize_info" in feat
        else None
    )
    hp, wp = [int(x) for x in np.asarray(feat["patch_grid"]).tolist()]
    n = len(image_paths)

    if "fg_scores_2d" in corr:
        e2b = np.asarray(corr["fg_scores_2d"], dtype=np.float32)
    else:
        e2b = np.asarray(corr["fg_scores"], dtype=np.float32).reshape(
            n, hp, wp
        )
    e2b = e2b.reshape(n, hp, wp)

    current_patch = np.asarray(cos["fg_mask_patch"]) > 0.5
    current_patch = current_patch.reshape(n, hp, wp)

    valid = content_mask_from_resize_info(
        resize_info,
        (hp, wp),
        n,
    )

    final_patch = current_patch.copy()
    triggered = np.zeros(n, dtype=bool)
    current_fg_fraction = np.zeros(n, dtype=np.float32)

    for i in range(n):
        denom = max(1, int(valid[i].sum()))
        current_fg_fraction[i] = (
            (current_patch[i] & valid[i]).sum() / denom
        )

        do_bypass = (not adaptive) or (
            current_fg_fraction[i] <= trigger_fg_frac
        )
        if not do_bypass:
            continue

        triggered[i] = True
        if kind == "top":
            final_patch[i] = top_fraction_mask(
                e2b[i],
                valid[i],
                value,
            )
        else:
            final_patch[i] = (e2b[i] >= value) & valid[i]

    # Usa exatamente as rotinas da E3 para manter compatibilidade espacial
    # e de seeds com step4_sam_refinement.py.
    fg_mask_pixel = upsample_masks(
        final_patch.astype(np.float32),
        image_sizes,
        resize_info=resize_info,
    )
    seeds = extract_seeds(fg_mask_pixel, image_sizes)

    seeds_arr = np.empty(len(seeds), dtype=object)
    for i, seed in enumerate(seeds):
        seeds_arr[i] = seed

    out = dict(cos)
    out["fg_mask_patch"] = final_patch.astype(np.float32)
    out["fg_mask_pixel"] = fg_mask_pixel
    out["fg_mask_coarse"] = final_patch.astype(np.float32)
    out["seeds"] = seeds_arr

    out["adaptive_bypass_variant"] = np.array(variant)
    out["adaptive_bypass_trigger_fg_frac"] = np.array(
        trigger_fg_frac,
        dtype=np.float32,
    )
    out["adaptive_bypass_triggered"] = triggered
    out["adaptive_bypass_trigger_count"] = np.array(
        int(triggered.sum()),
        dtype=np.int32,
    )
    out["adaptive_bypass_current_fg_fraction"] = current_fg_fraction
    out["adaptive_bypass_source"] = np.array(str(corr_path))

    dst.mkdir(parents=True, exist_ok=True)

    # Reaproveita os artefatos prévios sem duplicá-los.
    for filename in [
        "features.npz",
        "correspondence.npz",
        "correspondence_bg.npz",
        "background_suppression.json",
    ]:
        p = src / filename
        if p.exists():
            ensure_symlink(p, dst / filename, force=force)

    out_path = dst / "cosaliency.npz"
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} já existe. Use --force para sobrescrever."
        )
    np.savez_compressed(out_path, **out)

    print(
        f"[OK] {class_name}: {variant} | "
        f"trigger={int(triggered.sum())}/{n} "
        f"({triggered.mean():.1%}) -> {out_path}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", required=True, type=Path)
    p.add_argument("--dest-root", required=True, type=Path)
    p.add_argument("--variant", required=True)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--trigger-fg-frac", type=float, default=0.01)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()

    if source_root == dest_root:
        raise SystemExit(
            "[ERRO] source-root e dest-root devem ser diferentes."
        )

    print("=" * 82)
    print("MATERIALIZAÇÃO E3 ADAPTIVE BYPASS")
    print("=" * 82)
    print(f"Source  : {source_root}")
    print(f"Dest    : {dest_root}")
    print(f"Variant : {args.variant}")
    print(f"Trigger : <= {args.trigger_fg_frac:.2%}")
    print(f"Classes : {', '.join(args.classes)}")

    for class_name in args.classes:
        materialize_class(
            source_root,
            dest_root,
            class_name,
            args.variant,
            args.trigger_fg_frac,
            args.force,
        )


if __name__ == "__main__":
    main()
