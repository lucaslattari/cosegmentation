#!/usr/bin/env python3
"""
analyze_stage_iou_pipeline.py
=============================

Diagnóstico de IoU por etapa do pipeline de cosegmentação:

    E2  -> E2B -> E3 -> E4 -> E5

O script NÃO reexecuta nenhuma etapa. Ele lê os artefatos existentes:

    E2   : correspondence.npz
    E2B  : correspondence_bg.npz
    E3   : cosaliency.npz
    E4   : refined_masks.npz
    E5   : segmented_e5/*_mask.png
           (fallback: tenta chaves conhecidas em iterative_masks.npz)

Para E2/E2B:
    usa fg_scores_2d (ou fg_scores + patch_grid), threshold 0.5,
    reconstrói a máscara removendo padding keep_ar e redimensiona para o GT.

Para E3:
    usa fg_mask_pixel.

Para E4:
    usa refined_masks.

Para E5:
    prefere as máscaras exportadas em segmented_e5/.

Saídas:
    stage_iou_summary.csv
    stage_iou_per_image.csv
    stage_iou_results.json

Exemplo:
    python ablation/analyze_stage_iou_pipeline.py \
        --features-root features/pascal_coseg_bg_recurrence_l0p50 \
        --gt-index datasets/PASCAL/gt_index.json \
        --classes person boat bottle bus \
        --output ablation_results/stage_iou_baseline

Atenção:
    Este é um diagnóstico de propagação de erro entre etapas.
    E2/E2B são avaliadas a partir de scores coarse em patches com threshold fixo,
    enquanto E3/E4/E5 já são máscaras em pixel. Portanto, os valores são úteis
    para localizar gargalos, não para substituir a métrica final publicada.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluate import (  # noqa: E402
    build_gt_mapping,
    get_gt_index_multi,
    load_gt_index,
    load_mask,
)

STAGES = ["E2", "E2B", "E3", "E4", "E5"]
TRANSITIONS = [
    ("E2", "E2B"),
    ("E2B", "E3"),
    ("E3", "E4"),
    ("E4", "E5"),
]


def safe_name(name: str) -> str:
    return "".join(
        ch if (ch.isalnum() or ch in "_-") else "_"
        for ch in name
    ).strip("_")


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def scalar_str(x: Any, default: str = "") -> str:
    try:
        arr = np.asarray(x)
        if arr.size == 1:
            return str(arr.item())
    except Exception:
        pass
    return default


def ensure_hw(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Converte máscara 2D para bool e ajusta por nearest-neighbor se necessário."""
    arr = np.asarray(mask)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Máscara deveria ser 2D, obtido shape={arr.shape}")

    arr_bool = arr > 0.5
    h, w = target_hw
    if arr_bool.shape == (h, w):
        return arr_bool

    img = Image.fromarray(arr_bool.astype(np.uint8) * 255, mode="L")
    img = img.resize((w, h), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8) > 128


def patch_mask_to_gt(
    patch_binary: np.ndarray,
    gt_hw: tuple[int, int],
    resize_row: np.ndarray | None,
) -> np.ndarray:
    """
    Reconstrói máscara patch-space no domínio real da imagem.

    keep_ar:
        grid de patches
          -> canvas E1
          -> remove padding
          -> resize para GT

    Sem resize_info:
        grid -> resize direto para GT
    """
    h_gt, w_gt = gt_hw
    patch_img = Image.fromarray(
        patch_binary.astype(np.uint8) * 255,
        mode="L",
    )

    if resize_row is None:
        out = patch_img.resize((w_gt, h_gt), Image.NEAREST)
        return np.asarray(out, dtype=np.uint8) > 128

    vals = [int(round(float(x))) for x in np.asarray(resize_row).tolist()]
    if len(vals) < 6:
        out = patch_img.resize((w_gt, h_gt), Image.NEAREST)
        return np.asarray(out, dtype=np.uint8) > 128

    resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = vals[:6]

    if min(resized_h, resized_w, canvas_h, canvas_w) <= 0:
        out = patch_img.resize((w_gt, h_gt), Image.NEAREST)
        return np.asarray(out, dtype=np.uint8) > 128

    canvas = patch_img.resize((canvas_w, canvas_h), Image.NEAREST)

    left = max(0, pad_left)
    top = max(0, pad_top)
    right = min(canvas_w, pad_left + resized_w)
    bottom = min(canvas_h, pad_top + resized_h)

    if right <= left or bottom <= top:
        out = patch_img.resize((w_gt, h_gt), Image.NEAREST)
        return np.asarray(out, dtype=np.uint8) > 128

    content = canvas.crop((left, top, right, bottom))
    out = content.resize((w_gt, h_gt), Image.NEAREST)
    return np.asarray(out, dtype=np.uint8) > 128


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return inter / union if union else 0.0


def accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(pred.astype(bool) == gt.astype(bool)))


def resolve_gt_path(path_like: str | Path, gt_index_path: Path) -> Path:
    p = Path(path_like)
    if p.is_absolute() and p.exists():
        return p

    candidates = [
        gt_index_path.parent / p,
        PROJECT_ROOT / p,
        p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return p


def load_features(class_dir: Path) -> dict[str, Any]:
    path = class_dir / "features.npz"
    if not path.exists():
        raise FileNotFoundError(f"features.npz ausente: {path}")

    d = load_npz_dict(path)
    required = ["image_paths", "image_sizes"]
    missing = [k for k in required if k not in d]
    if missing:
        raise KeyError(f"{path} sem chaves obrigatórias: {missing}")

    image_paths = [str(x) for x in d["image_paths"].tolist()]
    image_sizes = np.asarray(d["image_sizes"])
    resize_info = np.asarray(d["resize_info"]) if "resize_info" in d else None

    return {
        "path": path,
        "image_paths": image_paths,
        "image_sizes": image_sizes,
        "resize_info": resize_info,
    }


def stage_e2_masks(
    npz_path: Path,
    n_images: int,
    threshold: float,
    gt_shapes: list[tuple[int, int]],
    resize_info: np.ndarray | None,
) -> list[np.ndarray] | None:
    if not npz_path.exists():
        return None

    d = load_npz_dict(npz_path)

    if "fg_scores_2d" in d:
        scores = np.asarray(d["fg_scores_2d"], dtype=np.float32)
    elif "fg_scores" in d and "patch_grid" in d:
        fg = np.asarray(d["fg_scores"], dtype=np.float32)
        h_p, w_p = [int(x) for x in np.asarray(d["patch_grid"]).tolist()]
        scores = fg.reshape(len(fg), h_p, w_p)
    else:
        raise KeyError(
            f"{npz_path} não contém fg_scores_2d nem fg_scores+patch_grid."
        )

    if len(scores) != n_images:
        raise ValueError(
            f"{npz_path}: N={len(scores)} diferente de features N={n_images}"
        )

    masks = []
    for i in range(n_images):
        raw_patch = scores[i] >= threshold
        resize_row = resize_info[i] if resize_info is not None else None
        masks.append(
            patch_mask_to_gt(
                raw_patch,
                gt_shapes[i],
                resize_row,
            )
        )
    return masks


def stage_e3_masks(
    npz_path: Path,
    n_images: int,
    gt_shapes: list[tuple[int, int]],
    image_sizes: np.ndarray,
) -> list[np.ndarray] | None:
    if not npz_path.exists():
        return None

    d = load_npz_dict(npz_path)
    if "fg_mask_pixel" not in d:
        raise KeyError(f"{npz_path} sem fg_mask_pixel.")

    arr = np.asarray(d["fg_mask_pixel"])
    if len(arr) != n_images:
        raise ValueError(
            f"{npz_path}: N={len(arr)} diferente de features N={n_images}"
        )

    masks = []
    for i in range(n_images):
        h_i, w_i = [int(x) for x in image_sizes[i][:2]]
        m = arr[i, :h_i, :w_i]
        masks.append(ensure_hw(m, gt_shapes[i]))
    return masks


def stage_e4_masks(
    npz_path: Path,
    n_images: int,
    gt_shapes: list[tuple[int, int]],
    image_sizes: np.ndarray,
) -> list[np.ndarray] | None:
    if not npz_path.exists():
        return None

    d = load_npz_dict(npz_path)
    if "refined_masks" not in d:
        raise KeyError(f"{npz_path} sem refined_masks.")

    arr = np.asarray(d["refined_masks"])
    if len(arr) != n_images:
        raise ValueError(
            f"{npz_path}: N={len(arr)} diferente de features N={n_images}"
        )

    masks = []
    for i in range(n_images):
        h_i, w_i = [int(x) for x in image_sizes[i][:2]]
        m = arr[i, :h_i, :w_i]
        masks.append(ensure_hw(m, gt_shapes[i]))
    return masks


def find_exported_mask(seg_dir: Path, image_path: str) -> Path | None:
    stem = Path(image_path).stem

    exact = seg_dir / f"{stem}_mask.png"
    if exact.exists():
        return exact

    candidates = sorted(seg_dir.glob(f"{stem}*_mask.png"))
    if len(candidates) == 1:
        return candidates[0]

    return None


def stage_e5_from_export(
    seg_dir: Path,
    image_paths: list[str],
    gt_shapes: list[tuple[int, int]],
) -> list[np.ndarray] | None:
    if not seg_dir.exists():
        return None

    masks: list[np.ndarray] = []
    missing: list[str] = []

    for i, image_path in enumerate(image_paths):
        path = find_exported_mask(seg_dir, image_path)
        if path is None:
            missing.append(Path(image_path).name)
            masks.append(None)  # type: ignore[arg-type]
            continue

        m = load_mask(path)
        masks.append(ensure_hw(m, gt_shapes[i]))

    if missing:
        print(
            f"    [AVISO] E5 export: {len(missing)} máscara(s) ausente(s). "
            "Tentando fallback iterative_masks.npz."
        )
        return None

    return masks


def stage_e5_from_npz(
    npz_path: Path,
    n_images: int,
    gt_shapes: list[tuple[int, int]],
    image_sizes: np.ndarray,
) -> list[np.ndarray] | None:
    if not npz_path.exists():
        return None

    d = load_npz_dict(npz_path)

    candidate_keys = [
        "final_masks",
        "iterative_masks",
        "masks",
        "refined_masks",
    ]

    key = next((k for k in candidate_keys if k in d), None)
    if key is None:
        print(
            f"    [AVISO] {npz_path.name}: nenhuma chave de máscara conhecida. "
            f"Chaves: {sorted(d.keys())}"
        )
        return None

    arr = np.asarray(d[key])
    if len(arr) != n_images:
        print(
            f"    [AVISO] {npz_path.name}:{key} N={len(arr)} "
            f"diferente de features N={n_images}"
        )
        return None

    masks = []
    for i in range(n_images):
        h_i, w_i = [int(x) for x in image_sizes[i][:2]]
        m = arr[i, :h_i, :w_i]
        masks.append(ensure_hw(m, gt_shapes[i]))
    return masks


def mean_or_none(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def fmt(v: float | None, width: int = 8) -> str:
    return f"{v:>{width}.4f}" if v is not None else f"{'-':>{width}}"


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def dominant_drop(stage_means: dict[str, float | None]) -> tuple[str, float] | tuple[None, None]:
    candidates: list[tuple[str, float]] = []
    for a, b in TRANSITIONS:
        d = delta(stage_means.get(a), stage_means.get(b))
        if d is not None:
            candidates.append((f"{a}->{b}", d))

    if not candidates:
        return None, None

    return min(candidates, key=lambda x: x[1])


def evaluate_class(
    features_root: Path,
    gt_index: dict,
    gt_index_path: Path,
    class_name: str,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    class_dir = features_root / safe_name(class_name)
    feat = load_features(class_dir)

    image_paths = feat["image_paths"]
    image_sizes = feat["image_sizes"]
    resize_info = feat["resize_info"]
    n_images = len(image_paths)

    gt_multi = get_gt_index_multi(gt_index)
    mapping, not_found, ambiguous = build_gt_mapping(
        image_paths,
        gt_multi,
        collection_hint=class_name,
    )

    if not_found:
        raise RuntimeError(
            f"{class_name}: {len(not_found)} imagem(ns) sem GT."
        )
    if ambiguous:
        raise RuntimeError(
            f"{class_name}: {len(ambiguous)} imagem(ns) com GT ambíguo."
        )

    gt_paths_by_image: dict[str, Path] = {}
    for img_path, gt_path in mapping.items():
        gt_paths_by_image[str(img_path)] = resolve_gt_path(
            gt_path,
            gt_index_path,
        )

    gts: list[np.ndarray] = []
    gt_shapes: list[tuple[int, int]] = []

    for img_path in image_paths:
        gt_path = gt_paths_by_image.get(str(img_path))
        if gt_path is None:
            raise RuntimeError(f"GT não mapeado para: {img_path}")
        gt = load_mask(gt_path)
        gts.append(gt.astype(bool))
        gt_shapes.append(gt.shape)

    stage_masks: dict[str, list[np.ndarray] | None] = {}

    stage_masks["E2"] = stage_e2_masks(
        class_dir / "correspondence.npz",
        n_images,
        threshold,
        gt_shapes,
        resize_info,
    )

    stage_masks["E2B"] = stage_e2_masks(
        class_dir / "correspondence_bg.npz",
        n_images,
        threshold,
        gt_shapes,
        resize_info,
    )

    stage_masks["E3"] = stage_e3_masks(
        class_dir / "cosaliency.npz",
        n_images,
        gt_shapes,
        image_sizes,
    )

    stage_masks["E4"] = stage_e4_masks(
        class_dir / "refined_masks.npz",
        n_images,
        gt_shapes,
        image_sizes,
    )

    e5 = stage_e5_from_export(
        class_dir / "segmented_e5",
        image_paths,
        gt_shapes,
    )
    if e5 is None:
        e5 = stage_e5_from_npz(
            class_dir / "iterative_masks.npz",
            n_images,
            gt_shapes,
            image_sizes,
        )
    stage_masks["E5"] = e5

    rows: list[dict[str, Any]] = []

    for i, img_path in enumerate(image_paths):
        row: dict[str, Any] = {
            "class": class_name,
            "index": i,
            "image_name": Path(img_path).name,
            "image_path": str(img_path),
            "gt_path": str(gt_paths_by_image[str(img_path)]),
        }

        for stage in STAGES:
            masks = stage_masks[stage]
            if masks is None:
                row[f"iou_{stage.lower()}"] = None
                row[f"accuracy_{stage.lower()}"] = None
            else:
                pred = masks[i]
                row[f"iou_{stage.lower()}"] = iou(pred, gts[i])
                row[f"accuracy_{stage.lower()}"] = accuracy(pred, gts[i])

        for a, b in TRANSITIONS:
            va = row.get(f"iou_{a.lower()}")
            vb = row.get(f"iou_{b.lower()}")
            row[f"delta_{a.lower()}_{b.lower()}"] = (
                float(vb) - float(va)
                if va is not None and vb is not None
                else None
            )

        valid_deltas = [
            (f"{a}->{b}", row.get(f"delta_{a.lower()}_{b.lower()}"))
            for a, b in TRANSITIONS
            if row.get(f"delta_{a.lower()}_{b.lower()}") is not None
        ]
        if valid_deltas:
            worst = min(valid_deltas, key=lambda x: x[1])
            row["largest_drop_transition"] = worst[0]
            row["largest_drop_value"] = worst[1]
        else:
            row["largest_drop_transition"] = ""
            row["largest_drop_value"] = None

        rows.append(row)

    stage_means: dict[str, float | None] = {}
    stage_acc_means: dict[str, float | None] = {}

    for stage in STAGES:
        stage_means[stage] = mean_or_none(
            [r[f"iou_{stage.lower()}"] for r in rows]
        )
        stage_acc_means[stage] = mean_or_none(
            [r[f"accuracy_{stage.lower()}"] for r in rows]
        )

    drop_name, drop_value = dominant_drop(stage_means)

    summary: dict[str, Any] = {
        "class": class_name,
        "n_images": n_images,
        "threshold_e2_e2b": threshold,
        "features_dir": str(class_dir),
    }

    for stage in STAGES:
        summary[f"iou_{stage.lower()}"] = stage_means[stage]
        summary[f"accuracy_{stage.lower()}"] = stage_acc_means[stage]

    for a, b in TRANSITIONS:
        summary[f"delta_{a.lower()}_{b.lower()}"] = delta(
            stage_means[a],
            stage_means[b],
        )

    summary["largest_drop_transition"] = drop_name
    summary["largest_drop_value"] = drop_value

    # Metadados úteis da E2.
    corr_path = class_dir / "correspondence.npz"
    if corr_path.exists():
        corr = load_npz_dict(corr_path)
        summary["e2_n_pairs"] = (
            int(len(corr["top_pairs"]))
            if "top_pairs" in corr
            else None
        )
        summary["e2_pair_selection_mode"] = (
            scalar_str(corr["pair_selection_mode"], "global_top_k")
            if "pair_selection_mode" in corr
            else "global_top_k"
        )
        summary["e2_k_per_image"] = (
            int(np.asarray(corr["k_per_image"]).item())
            if "k_per_image" in corr
            else None
        )
    else:
        summary["e2_n_pairs"] = None
        summary["e2_pair_selection_mode"] = None
        summary["e2_k_per_image"] = None

    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def macro_summary(summaries: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for stage in STAGES:
        out[stage] = mean_or_none(
            [s.get(f"iou_{stage.lower()}") for s in summaries]
        )
    return out


def print_class_summary(summary: dict[str, Any]) -> None:
    means = {
        stage: summary.get(f"iou_{stage.lower()}")
        for stage in STAGES
    }

    print(f"\nClasse: {summary['class']}  (N={summary['n_images']})")
    print(
        f"  E2 pairs: {summary.get('e2_n_pairs')} | "
        f"mode={summary.get('e2_pair_selection_mode')} | "
        f"kpi={summary.get('e2_k_per_image')}"
    )
    print()
    print("  Etapa        IoU médio      Δ vs anterior")
    print("  ------------------------------------------")

    prev = None
    for stage in STAGES:
        value = means[stage]
        d = delta(prev, value) if prev is not None else None
        d_text = f"{d:+.4f}" if d is not None else "-"
        value_text = f"{value:.4f}" if value is not None else "-"
        print(f"  {stage:<5}        {value_text:>8}        {d_text:>9}")
        if value is not None:
            prev = value

    print(
        f"\n  Maior queda: "
        f"{summary.get('largest_drop_transition') or '-'} "
        f"({summary.get('largest_drop_value'):+.4f})"
        if summary.get("largest_drop_value") is not None
        else "\n  Maior queda: -"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analisa IoU por etapa E2->E2B->E3->E4->E5."
    )
    p.add_argument("--features-root", required=True, type=Path)
    p.add_argument("--gt-index", required=True, type=Path)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold usado somente para fg_scores de E2/E2B.",
    )
    args = p.parse_args()

    features_root = args.features_root.resolve()
    gt_index_path = args.gt_index.resolve()
    output = args.output.resolve()

    if not features_root.exists():
        raise SystemExit(
            f"[ERRO] features-root inexistente: {features_root}"
        )
    if not gt_index_path.exists():
        raise SystemExit(
            f"[ERRO] gt-index inexistente: {gt_index_path}"
        )
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("[ERRO] --threshold deve estar em [0,1].")

    gt_index = load_gt_index(gt_index_path)

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    print("=" * 105)
    print("DIAGNÓSTICO DE IoU POR ETAPA — E2 -> E2B -> E3 -> E4 -> E5")
    print("=" * 105)
    print(f"Features root : {features_root}")
    print(f"GT index      : {gt_index_path}")
    print(f"Threshold E2  : {args.threshold:.3f}")
    print(f"Classes       : {', '.join(args.classes)}")

    for class_name in args.classes:
        print("\n" + "-" * 105)
        try:
            summary, rows = evaluate_class(
                features_root,
                gt_index,
                gt_index_path,
                class_name,
                args.threshold,
            )
        except Exception as exc:
            print(f"[ERRO] {class_name}: {exc}")
            continue

        summaries.append(summary)
        all_rows.extend(rows)
        print_class_summary(summary)

    output.mkdir(parents=True, exist_ok=True)

    summary_csv = output / "stage_iou_summary.csv"
    per_image_csv = output / "stage_iou_per_image.csv"
    json_path = output / "stage_iou_results.json"

    write_csv(summary_csv, summaries)
    write_csv(per_image_csv, all_rows)

    macro = macro_summary(summaries)

    json_path.write_text(
        json.dumps(
            {
                "features_root": str(features_root),
                "gt_index": str(gt_index_path),
                "threshold_e2_e2b": args.threshold,
                "classes": summaries,
                "macro_iou": macro,
                "per_image": all_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 105)
    print("RESUMO POR CLASSE")
    print("=" * 105)
    print(
        f"{'Classe':<16}"
        f"{'E2':>9}"
        f"{'E2B':>9}"
        f"{'E3':>9}"
        f"{'E4':>9}"
        f"{'E5':>9}"
        f"{'Maior queda':>18}"
        f"{'Δ':>10}"
    )
    print("-" * 105)

    for s in summaries:
        print(
            f"{s['class']:<16}"
            f"{fmt(s.get('iou_e2'), 9)}"
            f"{fmt(s.get('iou_e2b'), 9)}"
            f"{fmt(s.get('iou_e3'), 9)}"
            f"{fmt(s.get('iou_e4'), 9)}"
            f"{fmt(s.get('iou_e5'), 9)}"
            f"{str(s.get('largest_drop_transition') or '-'):>18}"
            f"{fmt(s.get('largest_drop_value'), 10)}"
        )

    if summaries:
        print("-" * 105)
        print(
            f"{'MACRO':<16}"
            f"{fmt(macro.get('E2'), 9)}"
            f"{fmt(macro.get('E2B'), 9)}"
            f"{fmt(macro.get('E3'), 9)}"
            f"{fmt(macro.get('E4'), 9)}"
            f"{fmt(macro.get('E5'), 9)}"
        )

    print(f"\n[Salvo] {summary_csv}")
    print(f"[Salvo] {per_image_csv}")
    print(f"[Salvo] {json_path}")


if __name__ == "__main__":
    main()
