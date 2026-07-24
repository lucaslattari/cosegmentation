#!/usr/bin/env python3
"""
=============================================================================
analyze_step5_iterative.py — Auditoria da Etapa 5 (refinamento iterativo)
=============================================================================

Lê o mask_history salvo em iterative_masks.npz (uma máscara por iteração,
índice 0 = estado inicial da Etapa 4) e calcula o IoU real contra GT em
CADA iteração — o que permite descobrir, a partir de uma única execução
com max_iter alto, qual seria o resultado para qualquer combinação de
max_iter/converge_thresh menor, sem precisar rodar o SAM de novo.

Requer que step5_iterative.py tenha sido rodado com save_mask_history=True
(o default atual). Arquivos antigos sem mask_history são detectados e
recusados com uma mensagem clara.

Uso:
  python3 analyze_step5_iterative.py \
    --iterative features/Kite/iterative_masks.npz \
    --gt_index gt_index.json \
    --output features/Kite/step5_quality

Saída:
  step5_quality.json  — curva completa de gt_iou por iteração + melhor iteração
  step5_quality_report.md
  viz_iou_curve.png   (se matplotlib disponível e --no_plots não usado)
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def _path_variants(path_str: str):
    import os
    variants = set()
    raw = str(path_str).replace("\\", "/")
    variants.add(raw)
    variants.add(raw.lstrip("./"))
    p = Path(path_str)
    try:
        variants.add(p.as_posix())
    except Exception:
        pass
    try:
        variants.add(os.path.relpath(p).replace("\\", "/"))
    except Exception:
        pass
    for marker in ["icoseg/images/", "images/"]:
        idx = raw.find(marker)
        if idx >= 0:
            variants.add(raw[idx:])
    return {v for v in variants if v}


def load_pixel_gt_masks(image_paths: List[str], image_sizes: np.ndarray,
                        gt_index: Dict[str, Any], fg_threshold: float = 0.5
                        ) -> List[Optional[np.ndarray]]:
    mappings = gt_index.get("mappings", {})
    variant_lookup: Dict[str, str] = {}
    basename_to_gt: Dict[str, List[str]] = {}
    for img_p, gt_p in mappings.items():
        for v in _path_variants(img_p):
            variant_lookup[v] = str(gt_p)
        base = Path(img_p).name
        basename_to_gt.setdefault(base, []).append(str(gt_p))
    unique_basename = {b: (sorted(set(g))[0] if len(set(g)) == 1 else None)
                       for b, g in basename_to_gt.items()}

    masks = []
    for i, img_path in enumerate(image_paths):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        gt_path = None
        for v in _path_variants(img_path):
            if v in variant_lookup:
                gt_path = Path(variant_lookup[v])
                break
        if gt_path is None:
            s = unique_basename.get(Path(img_path).name)
            gt_path = Path(s) if s else None
        if gt_path is None or not gt_path.exists():
            masks.append(None)
            continue
        try:
            arr = np.asarray(
                Image.open(gt_path).convert("L").resize((W_i, H_i), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            masks.append(arr >= fg_threshold)
        except Exception:
            masks.append(None)
    return masks


def load_iterative(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"iterative_masks.npz não encontrado: {path}")
    data = np.load(path, allow_pickle=True)
    if "mask_history" not in data.files:
        raise ValueError(
            f"{path} não tem mask_history — foi gerado com uma versão antiga de "
            "step5_iterative.py (sem save_mask_history). Rode a Etapa 5 de novo "
            "com o script atualizado para poder analisar por iteração."
        )
    return {
        "mask_history": data["mask_history"],   # (n_iters+1, N, H, W)
        "image_paths":  [str(x) for x in data["image_paths"].tolist()],
        "image_sizes":  data["image_sizes"],
        "n_iters_run":  int(data["n_iters_run"]),
        "converged":    bool(data["converged"]),
    }


def compute_iou_curve(mask_history: np.ndarray, image_paths: List[str],
                      image_sizes: np.ndarray,
                      pixel_gt_masks: List[Optional[np.ndarray]]) -> Dict[str, Any]:
    """
    Para cada iteração (eixo 0 de mask_history), calcula o IoU médio contra
    GT entre as imagens com GT disponível. Retorna a curva completa e a
    melhor iteração — equivalente a ter testado qualquer valor menor de
    max_iter/converge_thresh sem rodar o SAM de novo.
    """
    n_steps = mask_history.shape[0]
    curve = []
    for it in range(n_steps):
        masks = mask_history[it]
        ious = []
        for i, gt in enumerate(pixel_gt_masks):
            if gt is None:
                continue
            H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
            pred = masks[i, :H_i, :W_i] > 0.5
            if gt.any() or (~gt).any():
                ious.append(_iou(pred, gt.astype(bool)))
        curve.append(float(np.mean(ious)) if ious else None)

    valid = [(it, v) for it, v in enumerate(curve) if v is not None]
    if not valid:
        return {"gt_iou_curve": curve, "best_iter": None, "best_gt_iou": None,
                "baseline_gt_iou": None, "gain_over_baseline": None}

    best_iter, best_gt_iou = max(valid, key=lambda x: x[1])
    baseline = curve[0]

    return {
        "gt_iou_curve":        curve,
        "best_iter":           best_iter,
        "best_gt_iou":         best_gt_iou,
        "baseline_gt_iou":     baseline,
        "gain_over_baseline":  (best_gt_iou - baseline) if baseline is not None else None,
        "n_iters_available":   n_steps - 1,
    }


def plot_curve(curve: List[Optional[float]], best_iter: Optional[int],
              out_path: Path) -> None:
    if plt is None:
        return
    xs = [i for i, v in enumerate(curve) if v is not None]
    ys = [v for v in curve if v is not None]
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="steelblue", lw=2, ms=6)
    if best_iter is not None:
        ax.axvline(best_iter, color="green", ls="--", lw=1.2,
                   label=f"melhor iteração = {best_iter}")
    ax.set_xlabel("Iteração (0 = Etapa 4)")
    ax.set_ylabel("IoU médio vs GT")
    ax.set_title("Etapa 5 — IoU real por iteração")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_report(output_dir: Path, iterative_path: Path, result: Dict[str, Any],
                 n_images: int) -> Path:
    report_path = output_dir / "step5_quality_report.md"
    curve = result["gt_iou_curve"]
    lines = [
        "# Relatório de qualidade — Etapa 5 (Refinamento Iterativo)",
        "",
        f"Arquivo analisado: `{iterative_path}`",
        "",
        "## Resumo",
        "",
        f"- Imagens: **{n_images}**",
        f"- Iterações disponíveis no histórico: **{result.get('n_iters_available')}**",
        f"- Baseline (Etapa 4, iteração 0): **{result.get('baseline_gt_iou')}**",
        f"- Melhor iteração: **{result.get('best_iter')}**  "
        f"(gt_iou = {result.get('best_gt_iou')})",
        f"- Ganho sobre a Etapa 4: **{result.get('gain_over_baseline')}**",
        "",
        "## Curva completa (gt_iou por iteração)",
        "",
        "| Iteração | gt_iou |",
        "|---:|---:|",
    ]
    for i, v in enumerate(curve):
        vs = f"{v:.4f}" if v is not None else "n/a"
        marker = "  ← melhor" if i == result.get("best_iter") else ""
        lines.append(f"| {i} | {vs}{marker} |")
    lines += [
        "",
        "## Observações",
        "",
        "- Esta curva foi extraída de uma ÚNICA execução do step5_iterative.py "
        "com max_iter alto e mask_history salvo — equivale a ter testado "
        "qualquer combinação de max_iter/converge_thresh menor sem rodar o "
        "SAM novamente.",
        "- Se `melhor iteração` for igual ao teto de iterações rodado, "
        "considere rodar de novo com max_iter ainda maior — pode não ter "
        "convergido de fato.",
        "- Esta análise não cobre n_fg_pts/n_bg_pts, que mudam o resultado "
        "de cada iteração e por isso precisam de execuções separadas "
        "(ver run_step5_iterative_sweep.sh).",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analisa iterative_masks.npz da Etapa 5 (curva de IoU por iteração).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iterative", required=True, type=Path,
                   help="Caminho para iterative_masks.npz da Etapa 5.")
    p.add_argument("--gt_index", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--fg_threshold", type=float, default=0.5)
    p.add_argument("--no_plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output or (args.iterative.parent / "step5_quality")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Carregando {args.iterative}")
    data = load_iterative(args.iterative)
    n_images = len(data["image_paths"])
    print(f"      {n_images} imagens, {data['mask_history'].shape[0]} passos no histórico "
          f"(0..{data['n_iters_run']})")

    if args.gt_index is None or not args.gt_index.exists():
        print("[ERRO] --gt_index é obrigatório para esta análise (sem GT não há o que medir).")
        sys.exit(1)

    print("[2/3] Carregando GT e calculando IoU por iteração...")
    gt_index = json.loads(args.gt_index.read_text(encoding="utf-8"))
    pixel_gt = load_pixel_gt_masks(data["image_paths"], data["image_sizes"],
                                   gt_index, args.fg_threshold)
    n_matched = sum(1 for m in pixel_gt if m is not None)
    print(f"      GT encontrado para {n_matched}/{n_images} imagens")

    result = compute_iou_curve(data["mask_history"], data["image_paths"],
                               data["image_sizes"], pixel_gt)

    print("\n" + "=" * 60)
    print("  CURVA DE IoU POR ITERAÇÃO")
    print("=" * 60)
    for i, v in enumerate(result["gt_iou_curve"]):
        marker = "  <- melhor" if i == result["best_iter"] else ""
        vs = f"{v:.4f}" if v is not None else "n/a"
        print(f"  iter {i}: {vs}{marker}")
    print(f"\n  Baseline (E4): {result['baseline_gt_iou']}")
    print(f"  Melhor iteração: {result['best_iter']} (gt_iou={result['best_gt_iou']})")
    print(f"  Ganho: {result['gain_over_baseline']}")
    print("=" * 60)

    print("\n[3/3] Salvando relatório...")
    json_path = output_dir / "step5_quality.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_plots:
        plot_curve(result["gt_iou_curve"], result["best_iter"],
                  output_dir / "viz_iou_curve.png")

    report_path = write_report(output_dir, args.iterative, result, n_images)
    print(f"[Salvo] {json_path}")
    print(f"[Salvo] {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
