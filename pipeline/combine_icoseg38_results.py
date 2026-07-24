#!/usr/bin/env python3
"""
Combina resultados do protocolo semântico legado (29 coleções) com os 15 grupos
reprocessados no protocolo iCoseg-38.

Estratégia:
  - descarta do resultado semântico as 6 coleções que eram fusões:
      Agra Taj Mahal, Gymnastics, Kite, Pandas, Skating, Stonehenge
  - mantém as 23 coleções semânticas que já correspondem 1:1 a grupos originais
  - adiciona os 15 resultados reprocessados individualmente
  - valida 38 grupos e 643 imagens
  - reporta macro por grupo, média por imagem e global por pixel

Uso:
  python combine_icoseg38_results.py \
      --semantic-results results/icoseg_full_bg_recurrence_l0p50 \
      --affected-results results/icoseg38_bg_recurrence_l0p50 \
      --output results/icoseg38_combined_bg_recurrence_l0p50
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

MERGED_SEMANTIC_COLLECTIONS = {
    "Agra Taj Mahal",
    "Gymnastics",
    "Kite",
    "Pandas",
    "Skating",
    "Stonehenge",
}

EXPECTED_AFFECTED_COLLECTIONS = {
    "Stonehenge-Stongehenge-rjt208",
    "Stonehenge-Salisbury Cathedral and Stonehenge-Sou'wester",
    "Agra Taj Mahal-Taj Mahal-LeszekZadlo",
    "Agra Taj Mahal-Inde du Nord 2004-Mhln",
    "Pandas-Tai-Land 6708-TAIwiffic",
    "Pandas-National Zoo, Washington DC-smata2",
    "Kite-Brighton kite Festival-jadepike4",
    "Kite-kitekid",
    "Kite-Margate Kite Festival 08-only lines",
    "Kite-Colt Park 6-1-08-rreed3",
    "Gymnastics-1",
    "Gymnastics-2",
    "Gymnastics-3",
    "Skating-Skating-Rich Moffitt",
    "Skating-ISU Grand Prix 2007-albertosepe",
}

METRICS = ("accuracy", "precision", "recall", "f1", "iou")


def load_result_dirs(root: Path) -> list[dict]:
    if not root.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {root}")

    results = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("aggregate") or not data.get("per_image"):
            continue
        data["_metrics_path"] = str(metrics_path)
        results.append(data)
    return results


def collection_name(result: dict) -> str:
    return str(result.get("collection", "")).strip()


def aggregate_row(result: dict, source: str) -> dict:
    agg = result["aggregate"]
    row = {
        "collection": collection_name(result),
        "source": source,
        "n_images": int(agg["n_images"]),
        "n_flipped": int(agg.get("n_flipped", 0)),
        "metrics_path": result["_metrics_path"],
    }
    for metric in METRICS:
        row[metric] = float(agg[metric]["mean"])
    return row


def counts_from_results(results: list[dict]) -> tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for result in results:
        for item in result["per_image"]:
            tp += int(item["tp"])
            tn += int(item["tn"])
            fp += int(item["fp"])
            fn += int(item["fn"])
    return tp, tn, fp, fn


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_pixels": total,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--semantic-results", required=True, type=Path)
    p.add_argument("--affected-results", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()

    semantic_all = load_result_dirs(args.semantic_results)
    affected_all = load_result_dirs(args.affected_results)

    semantic_kept = [
        r for r in semantic_all
        if collection_name(r) not in MERGED_SEMANTIC_COLLECTIONS
    ]
    affected = [
        r for r in affected_all
        if collection_name(r) in EXPECTED_AFFECTED_COLLECTIONS
    ]

    kept_names = {collection_name(r) for r in semantic_kept}
    affected_names = {collection_name(r) for r in affected}

    missing_affected = EXPECTED_AFFECTED_COLLECTIONS - affected_names
    unexpected_affected = affected_names - EXPECTED_AFFECTED_COLLECTIONS

    errors = []
    if len(semantic_kept) != 23:
        errors.append(
            f"Esperados 23 grupos reaproveitados do protocolo semântico; "
            f"encontrados {len(semantic_kept)}."
        )
    if len(affected) != 15:
        errors.append(
            f"Esperados 15 grupos reprocessados; encontrados {len(affected)}."
        )
    if missing_affected:
        errors.append(
            "Grupos reprocessados ausentes: " + ", ".join(sorted(missing_affected))
        )
    if unexpected_affected:
        errors.append(
            "Grupos reprocessados inesperados: " + ", ".join(sorted(unexpected_affected))
        )
    if kept_names & affected_names:
        errors.append(
            "Há nomes duplicados entre os 23 reaproveitados e os 15 novos: "
            + ", ".join(sorted(kept_names & affected_names))
        )

    combined = semantic_kept + affected
    total_images = sum(int(r["aggregate"]["n_images"]) for r in combined)

    if len(combined) != 38:
        errors.append(f"Total esperado: 38 grupos; encontrado: {len(combined)}.")
    if total_images != 643:
        errors.append(f"Total esperado: 643 imagens; encontrado: {total_images}.")

    if errors:
        print("\n[ERRO] Combinação inválida:")
        for e in errors:
            print("  -", e)
        raise SystemExit(1)

    rows = (
        [aggregate_row(r, "semantic_reused") for r in semantic_kept]
        + [aggregate_row(r, "icoseg38_reprocessed") for r in affected]
    )
    rows.sort(key=lambda x: x["collection"].lower())

    macro = {
        metric: mean(row[metric] for row in rows)
        for metric in METRICS
    }

    image_mean = {}
    for metric in METRICS:
        image_mean[metric] = sum(
            row["n_images"] * row[metric] for row in rows
        ) / total_images

    tp, tn, fp, fn = counts_from_results(combined)
    global_pixel = metrics_from_counts(tp, tn, fp, fn)

    summary = {
        "protocol": "icoseg38_combined",
        "n_groups": 38,
        "n_images": 643,
        "n_reused_semantic_groups": 23,
        "n_reprocessed_groups": 15,
        "excluded_semantic_merged_collections": sorted(MERGED_SEMANTIC_COLLECTIONS),
        "macro_by_group": macro,
        "image_mean": image_mean,
        "global_pixel": global_pixel,
        "groups": rows,
    }

    args.output.mkdir(parents=True, exist_ok=True)

    json_path = args.output / "icoseg38_combined_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path = args.output / "icoseg38_combined_groups.csv"
    fieldnames = [
        "collection", "source", "n_images", "n_flipped",
        "accuracy", "precision", "recall", "f1", "iou", "metrics_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 72)
    print("iCoseg-38 combinado")
    print("=" * 72)
    print(f"Grupos                    : {len(combined)}")
    print(f"Imagens                   : {total_images}")
    print(f"Reaproveitados (1:1)      : {len(semantic_kept)}")
    print(f"Reprocessados             : {len(affected)}")
    print()
    print("MACRO POR GRUPO (comparação principal)")
    print(f"  Accuracy / P : {macro['accuracy']:.6f}")
    print(f"  IoU / J      : {macro['iou']:.6f}")
    print(f"  F1           : {macro['f1']:.6f}")
    print()
    print("MÉDIA POR IMAGEM")
    print(f"  Accuracy     : {image_mean['accuracy']:.6f}")
    print(f"  IoU          : {image_mean['iou']:.6f}")
    print()
    print("GLOBAL POR PIXEL")
    print(f"  Accuracy     : {global_pixel['accuracy']:.6f}")
    print(f"  IoU          : {global_pixel['iou']:.6f}")
    print()
    print(f"[Salvo] {json_path}")
    print(f"[Salvo] {csv_path}")


if __name__ == "__main__":
    main()
