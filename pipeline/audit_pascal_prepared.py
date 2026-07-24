#!/usr/bin/env python3
"""
Auditoria estrutural do benchmark PASCAL CoSeg já preparado.

Estrutura esperada:

datasets/PASCAL/
├── images/
│   ├── aeroplane/
│   ├── bicycle/
│   ├── bird/
│   └── ...
└── GT/
    ├── aeroplane/
    ├── bicycle/
    ├── bird/
    └── ...

O script verifica:
  - existência das pastas;
  - conjunto de classes em images/ e GT/;
  - 20 classes PASCAL esperadas;
  - total de imagens e GTs;
  - pareamento imagem ↔ GT por stem;
  - arquivos ausentes/excedentes;
  - imagens/GTs corrompidos;
  - dimensões incompatíveis;
  - máscaras vazias ou totalmente foreground;
  - máscaras binárias ou com valores inesperados;
  - duplicação do mesmo stem entre classes;
  - resumo por classe;
  - validação opcional contra o total esperado de 1037 itens.

Uso:

  python pipeline/audit_pascal_prepared.py

ou explicitando caminhos:

  python pipeline/audit_pascal_prepared.py \
      --images-root /mnt/e/github/cosegmentation/datasets/PASCAL/images \
      --gt-root /mnt/e/github/cosegmentation/datasets/PASCAL/GT \
      --expected-total 1037 \
      --output datasets/PASCAL/audit_pascal

Observação:
  "total" aqui significa soma dos arquivos presentes nas subpastas de classe.
  O script também informa o número de stems únicos globalmente, pois a mesma
  imagem pode eventualmente aparecer em mais de uma classe dependendo de como
  o benchmark foi distribuído.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


EXPECTED_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GT_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

# Apenas para detectar nomes provavelmente digitados/organizados incorretamente.
KNOWN_CLASS_ALIASES = {
    "airplane": "aeroplane",
    "aeroplane": "aeroplane",
    "bycicle": "bicycle",
    "bike": "bicycle",
    "motorcycle": "motorbike",
    "motor cycle": "motorbike",
    "dining table": "diningtable",
    "potted plant": "pottedplant",
    "tv": "tvmonitor",
    "television": "tvmonitor",
}


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name.strip() if ch.isalnum())


EXPECTED_NORMALIZED = {normalize_name(x): x for x in EXPECTED_CLASSES}


def class_alias_suggestion(name: str) -> str | None:
    low = name.strip().lower()
    if low in KNOWN_CLASS_ALIASES:
        target = KNOWN_CLASS_ALIASES[low]
        if target != low:
            return target

    norm = normalize_name(name)
    if norm in EXPECTED_NORMALIZED:
        target = EXPECTED_NORMALIZED[norm]
        if target != name:
            return target

    return None


def list_class_dirs(root: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in sorted(root.iterdir())
        if p.is_dir() and not p.name.startswith(".")
    }


def list_files(root: Path, exts: set[str]) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def stem_map(files: list[Path]) -> tuple[dict[str, Path], dict[str, list[str]]]:
    mapping: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = defaultdict(list)

    for path in files:
        stem = path.stem
        if stem in mapping:
            duplicates[stem].append(str(path))
            if len(duplicates[stem]) == 1:
                duplicates[stem].insert(0, str(mapping[stem]))
        else:
            mapping[stem] = path

    return mapping, dict(duplicates)


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        img.load()
        return img.size  # (w, h)


def analyze_gt(path: Path) -> dict:
    with Image.open(path) as img:
        img.load()
        arr = np.array(img)

    info = {
        "mode": img.mode,
        "shape": list(arr.shape),
        "width": int(img.size[0]),
        "height": int(img.size[1]),
        "is_empty": False,
        "is_full": False,
        "is_binary_like": False,
        "unique_values": None,
        "n_unique": None,
        "foreground_fraction": None,
        "rgb_channels_equal": None,
    }

    # Se RGB/RGBA, verifica se todos os canais RGB são iguais.
    if arr.ndim == 3:
        rgb = arr[..., :3]
        channels_equal = bool(
            np.array_equal(rgb[..., 0], rgb[..., 1])
            and np.array_equal(rgb[..., 1], rgb[..., 2])
        )
        info["rgb_channels_equal"] = channels_equal

        if channels_equal:
            arr2 = rgb[..., 0]
        else:
            # Reduz para ocupação: pixel não preto = foreground candidato.
            arr2 = np.any(rgb != 0, axis=-1).astype(np.uint8)
    else:
        arr2 = arr

    values = np.unique(arr2)
    info["n_unique"] = int(len(values))

    if len(values) <= 20:
        info["unique_values"] = [
            int(v) if np.issubdtype(type(v), np.integer) or isinstance(v, np.integer) else float(v)
            for v in values.tolist()
        ]

    value_set = set(int(v) for v in values.tolist()) if np.issubdtype(arr2.dtype, np.integer) else set()

    # Formatos binários comuns.
    info["is_binary_like"] = (
        value_set.issubset({0, 1})
        or value_set.issubset({0, 255})
        or value_set.issubset({0, 1, 255})
    )

    # Foreground robusto para os formatos binários comuns.
    if info["is_binary_like"]:
        fg = arr2 != 0
    else:
        # Para auditoria, assume background=0 e qualquer outro valor como FG.
        fg = arr2 != 0

    fg_fraction = float(np.mean(fg)) if fg.size else 0.0
    info["foreground_fraction"] = fg_fraction
    info["is_empty"] = fg_fraction == 0.0
    info["is_full"] = fg_fraction == 1.0

    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audita dataset PASCAL CoSeg preparado em pastas por classe."
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("/mnt/e/github/cosegmentation/datasets/PASCAL/images"),
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/mnt/e/github/cosegmentation/datasets/PASCAL/GT"),
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=1037,
        help="Total esperado de itens somados nas pastas de classe (default: 1037).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/e/github/cosegmentation/datasets/PASCAL/audit_pascal"),
    )
    parser.add_argument(
        "--skip-deep-check",
        action="store_true",
        help="Pula abertura de todos os arquivos e análise pixel a pixel dos GTs.",
    )
    args = parser.parse_args()

    images_root = args.images_root.resolve()
    gt_root = args.gt_root.resolve()
    output = args.output.resolve()

    errors: list[str] = []
    warnings: list[str] = []

    if not images_root.is_dir():
        print(f"[ERRO] Pasta de imagens não encontrada: {images_root}")
        raise SystemExit(1)

    if not gt_root.is_dir():
        print(f"[ERRO] Pasta de GT não encontrada: {gt_root}")
        raise SystemExit(1)

    image_classes = list_class_dirs(images_root)
    gt_classes = list_class_dirs(gt_root)

    image_class_names = set(image_classes)
    gt_class_names = set(gt_classes)
    expected_set = set(EXPECTED_CLASSES)

    missing_image_classes = sorted(expected_set - image_class_names)
    missing_gt_classes = sorted(expected_set - gt_class_names)
    extra_image_classes = sorted(image_class_names - expected_set)
    extra_gt_classes = sorted(gt_class_names - expected_set)

    # Sugestões para nomes incorretos.
    alias_warnings = []
    for name in sorted(set(extra_image_classes) | set(extra_gt_classes)):
        suggestion = class_alias_suggestion(name)
        if suggestion:
            alias_warnings.append(f"'{name}' talvez devesse ser '{suggestion}'")

    if missing_image_classes:
        errors.append(
            "Classes esperadas ausentes em images/: "
            + ", ".join(missing_image_classes)
        )
    if missing_gt_classes:
        errors.append(
            "Classes esperadas ausentes em GT/: "
            + ", ".join(missing_gt_classes)
        )
    if extra_image_classes:
        warnings.append(
            "Classes extras em images/: " + ", ".join(extra_image_classes)
        )
    if extra_gt_classes:
        warnings.append(
            "Classes extras em GT/: " + ", ".join(extra_gt_classes)
        )
    warnings.extend(alias_warnings)

    if image_class_names != gt_class_names:
        errors.append(
            "O conjunto de subpastas de classe em images/ e GT/ não é idêntico."
        )

    all_class_names = sorted(image_class_names | gt_class_names)

    per_class = []
    global_image_stems: dict[str, list[str]] = defaultdict(list)
    global_gt_stems: dict[str, list[str]] = defaultdict(list)

    total_images = 0
    total_gts = 0
    total_pairs = 0
    total_missing_gt = 0
    total_missing_image = 0
    total_size_mismatch = 0
    total_corrupt_images = 0
    total_corrupt_gts = 0
    total_empty_masks = 0
    total_full_masks = 0
    total_nonbinary_masks = 0
    total_rgb_non_equal_masks = 0

    mask_format_counter = Counter()
    gt_value_patterns = Counter()

    detailed_issues = []

    print("=" * 96)
    print("AUDITORIA PASCAL CoSeg PREPARADO")
    print("=" * 96)
    print(f"Images root : {images_root}")
    print(f"GT root     : {gt_root}")
    print(f"Alvo total  : {args.expected_total}")
    print()

    for class_name in all_class_names:
        image_dir = image_classes.get(class_name)
        gt_dir = gt_classes.get(class_name)

        image_files = list_files(image_dir, IMAGE_EXTS) if image_dir else []
        gt_files = list_files(gt_dir, GT_EXTS) if gt_dir else []

        image_map, duplicate_image_stems = stem_map(image_files)
        gt_map, duplicate_gt_stems = stem_map(gt_files)

        image_stems = set(image_map)
        gt_stems = set(gt_map)

        missing_gt = sorted(image_stems - gt_stems)
        missing_image = sorted(gt_stems - image_stems)
        paired = sorted(image_stems & gt_stems)

        total_images += len(image_files)
        total_gts += len(gt_files)
        total_pairs += len(paired)
        total_missing_gt += len(missing_gt)
        total_missing_image += len(missing_image)

        for stem in image_stems:
            global_image_stems[stem].append(class_name)
        for stem in gt_stems:
            global_gt_stems[stem].append(class_name)

        class_size_mismatch = 0
        class_corrupt_images = 0
        class_corrupt_gts = 0
        class_empty = 0
        class_full = 0
        class_nonbinary = 0
        class_fg_fractions = []

        if duplicate_image_stems:
            errors.append(
                f"{class_name}: stems duplicados em images/: "
                + ", ".join(sorted(duplicate_image_stems))
            )
        if duplicate_gt_stems:
            errors.append(
                f"{class_name}: stems duplicados em GT/: "
                + ", ".join(sorted(duplicate_gt_stems))
            )

        if not args.skip_deep_check:
            for stem in paired:
                image_path = image_map[stem]
                gt_path = gt_map[stem]

                try:
                    image_size = read_image_size(image_path)
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    class_corrupt_images += 1
                    total_corrupt_images += 1
                    detailed_issues.append({
                        "class": class_name,
                        "stem": stem,
                        "type": "corrupt_image",
                        "path": str(image_path),
                        "detail": str(exc),
                    })
                    continue

                try:
                    gt_info = analyze_gt(gt_path)
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    class_corrupt_gts += 1
                    total_corrupt_gts += 1
                    detailed_issues.append({
                        "class": class_name,
                        "stem": stem,
                        "type": "corrupt_gt",
                        "path": str(gt_path),
                        "detail": str(exc),
                    })
                    continue

                gt_size = (gt_info["width"], gt_info["height"])
                mask_format_counter[gt_info["mode"]] += 1

                if gt_info["unique_values"] is not None:
                    gt_value_patterns[str(gt_info["unique_values"])] += 1
                else:
                    gt_value_patterns[f">{gt_info['n_unique']}_unique_values"] += 1

                if image_size != gt_size:
                    class_size_mismatch += 1
                    total_size_mismatch += 1
                    detailed_issues.append({
                        "class": class_name,
                        "stem": stem,
                        "type": "size_mismatch",
                        "image_size": list(image_size),
                        "gt_size": list(gt_size),
                        "image": str(image_path),
                        "gt": str(gt_path),
                    })

                if gt_info["is_empty"]:
                    class_empty += 1
                    total_empty_masks += 1
                    detailed_issues.append({
                        "class": class_name,
                        "stem": stem,
                        "type": "empty_mask",
                        "gt": str(gt_path),
                    })

                if gt_info["is_full"]:
                    class_full += 1
                    total_full_masks += 1
                    detailed_issues.append({
                        "class": class_name,
                        "stem": stem,
                        "type": "full_mask",
                        "gt": str(gt_path),
                    })

                if not gt_info["is_binary_like"]:
                    class_nonbinary += 1
                    total_nonbinary_masks += 1
                    detailed_issues.append({
                        "class": class_name,
                        "stem": stem,
                        "type": "nonbinary_mask",
                        "values": gt_info["unique_values"],
                        "n_unique": gt_info["n_unique"],
                        "gt": str(gt_path),
                    })

                if gt_info["rgb_channels_equal"] is False:
                    total_rgb_non_equal_masks += 1
                    detailed_issues.append({
                        "class": class_name,
                        "stem": stem,
                        "type": "rgb_channels_not_equal",
                        "gt": str(gt_path),
                    })

                if gt_info["foreground_fraction"] is not None:
                    class_fg_fractions.append(gt_info["foreground_fraction"])

        if missing_gt:
            for stem in missing_gt:
                detailed_issues.append({
                    "class": class_name,
                    "stem": stem,
                    "type": "missing_gt",
                    "image": str(image_map[stem]),
                })

        if missing_image:
            for stem in missing_image:
                detailed_issues.append({
                    "class": class_name,
                    "stem": stem,
                    "type": "missing_image",
                    "gt": str(gt_map[stem]),
                })

        mean_fg = (
            float(np.mean(class_fg_fractions))
            if class_fg_fractions else None
        )

        row = {
            "class": class_name,
            "n_images": len(image_files),
            "n_gt": len(gt_files),
            "n_pairs": len(paired),
            "missing_gt": len(missing_gt),
            "missing_image": len(missing_image),
            "size_mismatch": class_size_mismatch,
            "corrupt_images": class_corrupt_images,
            "corrupt_gt": class_corrupt_gts,
            "empty_masks": class_empty,
            "full_masks": class_full,
            "nonbinary_masks": class_nonbinary,
            "mean_fg_fraction": mean_fg,
        }
        per_class.append(row)

        fg_text = (
            f"{mean_fg:.3f}" if mean_fg is not None else "-"
        )

        print(
            f"{class_name:<14} "
            f"img={len(image_files):>4} "
            f"gt={len(gt_files):>4} "
            f"pairs={len(paired):>4} "
            f"missGT={len(missing_gt):>3} "
            f"missIMG={len(missing_image):>3} "
            f"dim={class_size_mismatch:>3} "
            f"empty={class_empty:>3} "
            f"nonbin={class_nonbinary:>3} "
            f"FGmean={fg_text:>6}"
        )

    duplicate_images_across_classes = {
        stem: classes
        for stem, classes in global_image_stems.items()
        if len(classes) > 1
    }
    duplicate_gts_across_classes = {
        stem: classes
        for stem, classes in global_gt_stems.items()
        if len(classes) > 1
    }

    unique_image_stems = len(global_image_stems)
    unique_gt_stems = len(global_gt_stems)

    if total_images != args.expected_total:
        warnings.append(
            f"Total de imagens por pasta de classe = {total_images}, "
            f"mas o esperado informado foi {args.expected_total}."
        )

    if total_images != total_gts:
        errors.append(
            f"Total de imagens ({total_images}) diferente do total de GTs ({total_gts})."
        )

    if total_missing_gt:
        errors.append(f"Há {total_missing_gt} imagens sem GT correspondente.")
    if total_missing_image:
        errors.append(f"Há {total_missing_image} GTs sem imagem correspondente.")
    if total_corrupt_images:
        errors.append(f"Há {total_corrupt_images} imagens corrompidas/ilegíveis.")
    if total_corrupt_gts:
        errors.append(f"Há {total_corrupt_gts} GTs corrompidos/ilegíveis.")
    if total_size_mismatch:
        errors.append(f"Há {total_size_mismatch} pares com dimensões incompatíveis.")
    if total_empty_masks:
        warnings.append(f"Há {total_empty_masks} máscaras completamente vazias.")
    if total_full_masks:
        warnings.append(f"Há {total_full_masks} máscaras 100% foreground.")
    if total_nonbinary_masks:
        warnings.append(
            f"Há {total_nonbinary_masks} máscaras que não usam apenas "
            "{0,1}, {0,255} ou {0,1,255}."
        )
    if total_rgb_non_equal_masks:
        warnings.append(
            f"Há {total_rgb_non_equal_masks} GTs RGB cujos canais não são idênticos."
        )

    summary = {
        "images_root": str(images_root),
        "gt_root": str(gt_root),
        "expected_total": args.expected_total,
        "n_expected_classes": len(EXPECTED_CLASSES),
        "n_image_classes": len(image_class_names),
        "n_gt_classes": len(gt_class_names),
        "image_classes": sorted(image_class_names),
        "gt_classes": sorted(gt_class_names),
        "missing_image_classes": missing_image_classes,
        "missing_gt_classes": missing_gt_classes,
        "extra_image_classes": extra_image_classes,
        "extra_gt_classes": extra_gt_classes,
        "total_images_class_memberships": total_images,
        "total_gt_class_memberships": total_gts,
        "total_pairs": total_pairs,
        "unique_image_stems_global": unique_image_stems,
        "unique_gt_stems_global": unique_gt_stems,
        "stems_repeated_across_image_classes": len(duplicate_images_across_classes),
        "stems_repeated_across_gt_classes": len(duplicate_gts_across_classes),
        "missing_gt": total_missing_gt,
        "missing_image": total_missing_image,
        "size_mismatch": total_size_mismatch,
        "corrupt_images": total_corrupt_images,
        "corrupt_gts": total_corrupt_gts,
        "empty_masks": total_empty_masks,
        "full_masks": total_full_masks,
        "nonbinary_masks": total_nonbinary_masks,
        "rgb_channels_not_equal": total_rgb_non_equal_masks,
        "mask_modes": dict(mask_format_counter),
        "gt_value_patterns": dict(gt_value_patterns),
        "per_class": per_class,
        "duplicate_image_stems_across_classes": duplicate_images_across_classes,
        "duplicate_gt_stems_across_classes": duplicate_gts_across_classes,
        "errors": errors,
        "warnings": warnings,
    }

    output.mkdir(parents=True, exist_ok=True)

    json_path = output / "pascal_prepared_audit.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = output / "pascal_prepared_per_class.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(per_class[0].keys()) if per_class else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class)

    issues_path = output / "pascal_prepared_issues.json"
    issues_path.write_text(
        json.dumps(detailed_issues, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 96)
    print("RESUMO GLOBAL")
    print("=" * 96)
    print(f"Classes em images/             : {len(image_class_names)}")
    print(f"Classes em GT/                 : {len(gt_class_names)}")
    print(f"Total imagens (por classe)     : {total_images}")
    print(f"Total GTs (por classe)         : {total_gts}")
    print(f"Pares imagem ↔ GT              : {total_pairs}")
    print(f"Stems únicos de imagem         : {unique_image_stems}")
    print(f"Stems repetidos entre classes  : {len(duplicate_images_across_classes)}")
    print(f"Imagens sem GT                 : {total_missing_gt}")
    print(f"GTs sem imagem                 : {total_missing_image}")
    print(f"Dimensões incompatíveis        : {total_size_mismatch}")
    print(f"Imagens corrompidas            : {total_corrupt_images}")
    print(f"GTs corrompidos                : {total_corrupt_gts}")
    print(f"Máscaras vazias                : {total_empty_masks}")
    print(f"Máscaras totalmente FG         : {total_full_masks}")
    print(f"Máscaras não-binárias          : {total_nonbinary_masks}")
    print(f"GT RGB com canais diferentes   : {total_rgb_non_equal_masks}")
    print()

    if errors:
        print("[ERROS]")
        for item in errors:
            print(f"  - {item}")
        print()

    if warnings:
        print("[AVISOS]")
        for item in warnings:
            print(f"  - {item}")
        print()

    if not errors:
        print("[OK] Nenhum erro estrutural crítico detectado.")
    else:
        print("[FALHA] Foram detectados erros estruturais críticos.")

    if (
        not errors
        and total_images == args.expected_total
        and len(image_class_names) == 20
        and len(gt_class_names) == 20
    ):
        print(
            "[OK] A estrutura é compatível com um benchmark "
            f"PASCAL CoSeg de 20 classes e {args.expected_total} itens."
        )
    elif not errors:
        print(
            "[ATENÇÃO] A estrutura é internamente consistente, mas a cardinalidade "
            "não coincide com a configuração esperada informada."
        )

    print()
    print(f"[Salvo] {json_path}")
    print(f"[Salvo] {csv_path}")
    print(f"[Salvo] {issues_path}")


if __name__ == "__main__":
    main()
