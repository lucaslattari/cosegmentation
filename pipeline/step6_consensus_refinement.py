#!/usr/bin/env python3
"""
=============================================================================
step6_consensus_refinement.py — Refinamento por consenso cross-image
=============================================================================

Reaproveita o resultado já calculado das Etapas 1 (features.npz) e 5
(iterative_masks.npz) — não chama o SAM de novo, não recalcula nada caro.

Ideia (Estratégia 1 da investigação de "maioria certa, minoria errada"):
para cada imagem i, constrói um protótipo de features FG usando só as
OUTRAS imagens da coleção (leave-one-out, baseado na própria máscara já
calculada pela Etapa 5 — não usa GT, funciona em produção). Mede o quanto
os patches marcados como FG na imagem i concordam com esse protótipo. Se a
concordância for MUITO abaixo do normal para aquela coleção (z-score
relativo às outras imagens da MESMA classe, não um limiar absoluto), a
imagem é sinalizada como outlier e sua máscara é recalculada usando a
similaridade ao protótipo em vez do resultado original.

IMPORTANTE — por que z-score relativo, não limiar absoluto: um limiar fixo
trataria igual uma classe onde a maioria vai bem e uma minoria erra (ex.:
Kite) e uma classe com falha sistêmica em quase todas as imagens (ex.:
Liverpool, Kendo). Usando o desvio da PRÓPRIA distribuição da classe, o
mecanismo naturalmente fica "quieto" (não sinaliza quase nada) em classes
onde não existe uma maioria clara de consenso — que é exatamente onde essa
estratégia NÃO deveria mexer (ver conversa/ablação anterior sobre o risco
de reforçar um consenso majoritário errado, tipo o caso do Liverpool).

Uso:
  python3 step6_consensus_refinement.py \
    --features features/icoseg_full/Kite/features.npz \
    --iterative features/icoseg_full/Kite/iterative_masks.npz \
    --output features/icoseg_full/Kite \
    --export

Saída: refined_masks_e6.npz (mesmo formato de final_masks que E4/E5, pronto
pra avaliação) + relatório com quais imagens foram sinalizadas/corrigidas.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Utilidades reaproveitadas (mesma convenção usada em toda a Etapa 1-5)
# -----------------------------------------------------------------------------

def compute_content_mask(resize_info: "np.ndarray | None",
                         patch_grid: Tuple[int, int]) -> "np.ndarray | None":
    if resize_info is None:
        return None
    H_p, W_p = patch_grid
    N = resize_info.shape[0]
    canvas_h, canvas_w = int(resize_info[0, 4]), int(resize_info[0, 5])
    patch_h = canvas_h / H_p
    patch_w = canvas_w / W_p
    row_starts = np.arange(H_p) * patch_h
    col_starts = np.arange(W_p) * patch_w
    row_ends = row_starts + patch_h
    col_ends = col_starts + patch_w
    masks = np.zeros((N, H_p, W_p), dtype=bool)
    for i in range(N):
        resized_h, resized_w, pad_top, pad_left, _, _ = [int(x) for x in resize_info[i]]
        content_row = (row_ends > pad_top) & (row_starts < pad_top + resized_h)
        content_col = (col_ends > pad_left) & (col_starts < pad_left + resized_w)
        masks[i] = content_row[:, None] & content_col[None, :]
    return masks.reshape(N, -1)


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def _path_variants(path_str: str):
    import os
    variants = set()
    raw = str(path_str).replace("\\", "/")
    variants.add(raw); variants.add(raw.lstrip("./"))
    p = Path(path_str)
    try: variants.add(p.as_posix())
    except Exception: pass
    try: variants.add(os.path.relpath(p).replace("\\", "/"))
    except Exception: pass
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
                gt_path = Path(variant_lookup[v]); break
        if gt_path is None:
            s = unique_basename.get(Path(img_path).name)
            gt_path = Path(s) if s else None
        if gt_path is None or not gt_path.exists():
            masks.append(None); continue
        try:
            arr = np.asarray(
                Image.open(gt_path).convert("L").resize((W_i, H_i), Image.BILINEAR),
                dtype=np.float32,
            ) / 255.0
            masks.append(arr >= fg_threshold)
        except Exception:
            masks.append(None)
    return masks


# -----------------------------------------------------------------------------
# Núcleo: protótipo leave-one-out + sinalização + correção
# -----------------------------------------------------------------------------

def downsample_pixel_masks_to_patch(final_masks: np.ndarray, image_sizes: np.ndarray,
                                    patch_grid: Tuple[int, int]) -> np.ndarray:
    """(N, H, W) pixel -> (N, P) patch-grid, threshold 0.5 (voto majoritário do patch)."""
    H_p, W_p = patch_grid
    N = final_masks.shape[0]
    out = np.zeros((N, H_p * W_p), dtype=bool)
    for i in range(N):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        m = final_masks[i, :H_i, :W_i]
        small = np.asarray(
            Image.fromarray((m > 0.5).astype(np.uint8) * 255).resize((W_p, H_p), Image.BILINEAR)
        )
        out[i] = (small > 127).reshape(-1)
    return out


def upsample_patch_score_to_pixel(score_patch: np.ndarray, patch_grid: Tuple[int, int],
                                  H_i: int, W_i: int) -> np.ndarray:
    """(P,) [0,1] score -> (H_i, W_i) float32 [0,1], via resize direto (sem padding — a
    máscara de pixel da E5 já está no tamanho real da foto, não do canvas com padding)."""
    H_p, W_p = patch_grid
    grid = (score_patch.reshape(H_p, W_p) * 255).astype(np.uint8)
    big = np.asarray(Image.fromarray(grid).resize((W_i, H_i), Image.BILINEAR), dtype=np.float32) / 255.0
    return big


def compute_leave_one_out_agreement(
    patch_features: np.ndarray,     # (N, P, D)
    patch_labels_fg: np.ndarray,    # (N, P) bool — pseudo-label da E5 (downsampled)
    content_mask: "np.ndarray | None",   # (N, P) bool
) -> Dict[str, np.ndarray]:
    """
    Para cada imagem i, protótipo FG = média das features FG de TODAS as
    outras imagens (leave-one-out). Calculado de forma eficiente via soma
    global menos contribuição da própria imagem — O(N), não O(N²).

    Retorna, por imagem:
      agreement_score : (N,)  margem de similaridade FG-vs-BG contra o protótipo LOO
      similarity_map   : (N, P) similaridade de cada patch ao protótipo daquela imagem
    """
    N, P, D = patch_features.shape
    flat_feat = patch_features.reshape(N * P, D)
    flat_fg = patch_labels_fg.reshape(N * P)
    flat_content = content_mask.reshape(N * P) if content_mask is not None else np.ones(N * P, dtype=bool)

    valid_fg = flat_fg & flat_content
    global_sum = flat_feat[valid_fg].sum(axis=0) if valid_fg.any() else np.zeros(D, dtype=np.float32)
    global_count = int(valid_fg.sum())

    agreement_score = np.zeros(N, dtype=np.float32)
    similarity_map = np.zeros((N, P), dtype=np.float32)

    for i in range(N):
        img_valid = content_mask[i] if content_mask is not None else np.ones(P, dtype=bool)
        img_fg = patch_labels_fg[i] & img_valid
        img_feat = patch_features[i]  # (P, D)

        own_sum = img_feat[img_fg].sum(axis=0) if img_fg.any() else np.zeros(D, dtype=np.float32)
        own_count = int(img_fg.sum())

        loo_sum = global_sum - own_sum
        loo_count = global_count - own_count
        if loo_count <= 0:
            # sem base de comparação (classe com N muito pequeno) — neutro
            agreement_score[i] = np.nan
            continue
        prototype = loo_sum / loo_count
        proto_norm = prototype / (np.linalg.norm(prototype) + 1e-8)

        feat_norm = img_feat / (np.linalg.norm(img_feat, axis=1, keepdims=True) + 1e-8)
        sim = feat_norm @ proto_norm  # (P,)
        similarity_map[i] = sim

        fg_sim = sim[img_fg]
        bg_mask = img_valid & (~patch_labels_fg[i])
        bg_sim = sim[bg_mask]
        if len(fg_sim) > 0 and len(bg_sim) > 0:
            agreement_score[i] = float(fg_sim.mean() - bg_sim.mean())
        else:
            agreement_score[i] = np.nan

    return {"agreement_score": agreement_score, "similarity_map": similarity_map}


def flag_outliers_zscore(agreement_score: np.ndarray, zscore_thresh: float) -> np.ndarray:
    """
    Sinaliza como outlier quem está muito ABAIXO da distribuição da própria
    classe (não um limiar absoluto — ver docstring do módulo). NaN nunca é
    sinalizado (dado insuficiente pra decidir).
    """
    valid = ~np.isnan(agreement_score)
    flagged = np.zeros(len(agreement_score), dtype=bool)
    if valid.sum() < 3:
        return flagged  # poucas imagens pra ter uma distribuição confiável
    mean = agreement_score[valid].mean()
    std = agreement_score[valid].std()
    if std < 1e-6:
        return flagged  # toda a classe concorda igual — nada pra corrigir
    z = np.full(len(agreement_score), np.nan)
    z[valid] = (agreement_score[valid] - mean) / std
    flagged[valid] = z[valid] < zscore_thresh
    return flagged


def apply_correction(
    final_masks: np.ndarray,        # (N, H, W) pixel, original da E5
    similarity_map: np.ndarray,     # (N, P)
    flagged: np.ndarray,            # (N,)
    patch_grid: Tuple[int, int],
    image_sizes: np.ndarray,
    content_mask: "np.ndarray | None",
    fg_threshold: float,
    correction_weight: float,
) -> np.ndarray:
    """Só mexe nas imagens sinalizadas. As outras saem idênticas à E5."""
    N, H, W = final_masks.shape
    H_p, W_p = patch_grid
    out = final_masks.copy()

    for i in range(N):
        if not flagged[i]:
            continue
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])

        sim = similarity_map[i].copy()
        if content_mask is not None:
            sim[~content_mask[i]] = sim.min() - 1  # padding nunca vira FG
        # normaliza min-max dentro da própria imagem pra virar um score [0,1] comparável
        s_min, s_max = sim.min(), sim.max()
        score_norm = (sim - s_min) / (s_max - s_min + 1e-8)

        corrected_pixel = upsample_patch_score_to_pixel(score_norm, patch_grid, H_i, W_i)
        original_pixel = out[i, :H_i, :W_i]

        blended = correction_weight * corrected_pixel + (1 - correction_weight) * original_pixel
        out[i, :H_i, :W_i] = (blended > fg_threshold).astype(final_masks.dtype)

    return out


# -----------------------------------------------------------------------------
# Exportação de PNGs (mesmo padrão de E4/E5)
# -----------------------------------------------------------------------------

def export_segmented_images(final_masks: np.ndarray, image_paths: List[str],
                            image_sizes: np.ndarray, output_dir: Path) -> None:
    out = output_dir / "segmented_e6"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[Export] Salvando {len(image_paths)} imagens segmentadas em {out}/")
    for i, img_path in enumerate(image_paths):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        mask = final_masks[i, :H_i, :W_i] > 0.5
        stem = Path(img_path).stem
        try:
            img = Image.open(img_path).convert("RGB").resize((W_i, H_i))
        except Exception:
            img = Image.new("RGB", (W_i, H_i))
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{stem}_mask.png")
        arr = np.asarray(img).copy()
        overlay = arr.copy()
        overlay[mask] = (overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        Image.fromarray(overlay).save(out / f"{stem}_overlay.jpg", quality=90)
        cutout = np.zeros((H_i, W_i, 4), dtype=np.uint8)
        cutout[..., :3] = arr
        cutout[..., 3] = (mask * 255).astype(np.uint8)
        Image.fromarray(cutout, mode="RGBA").save(out / f"{stem}_cutout.png")
    print(f"[Export] Concluído → {out}/  ({len(image_paths) * 3} arquivos)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Etapa 6 — refinamento por consenso leave-one-out sobre E1+E5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features",  required=True, type=Path, help="features.npz da Etapa 1.")
    p.add_argument("--iterative", required=True, type=Path, help="iterative_masks.npz da Etapa 5.")
    p.add_argument("--output",    required=True, type=Path)
    p.add_argument("--agreement_zscore_thresh", type=float, default=-1.5,
                   help="Sinaliza imagens com agreement_score abaixo de "
                        "mean - |thresh|*std DENTRO da própria classe (default: -1.5).")
    p.add_argument("--correction_weight", type=float, default=1.0,
                   help="Peso da correção nas imagens sinalizadas "
                        "[0=mantém original, 1=substitui totalmente] (default: 1.0).")
    p.add_argument("--fg_threshold", type=float, default=0.5)
    p.add_argument("--gt_index", type=Path, default=None,
                   help="OPCIONAL — só para diagnóstico (mostra se as correções ajudaram "
                        "ou pioraram). Não influencia a decisão de correção.")
    p.add_argument("--export", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Carregando features (E1) e máscaras finais (E5)...")
    feat = np.load(args.features, allow_pickle=True)
    it = np.load(args.iterative, allow_pickle=True)

    patch_features = feat["patch_features"]
    patch_grid = tuple(int(x) for x in feat["patch_grid"])
    resize_info = feat["resize_info"] if "resize_info" in feat.files else None

    final_masks = it["final_masks"]
    image_paths = [str(x) for x in it["image_paths"].tolist()]
    image_sizes = it["image_sizes"]
    N = len(image_paths)

    content_mask = compute_content_mask(resize_info, patch_grid)
    print(f"      {N} imagens, grade de patches {patch_grid}")

    print(f"[2/4] Calculando protótipo leave-one-out e concordância por imagem...")
    patch_labels_fg = downsample_pixel_masks_to_patch(final_masks, image_sizes, patch_grid)
    agree = compute_leave_one_out_agreement(patch_features, patch_labels_fg, content_mask)
    agreement_score = agree["agreement_score"]

    flagged = flag_outliers_zscore(agreement_score, args.agreement_zscore_thresh)
    n_flagged = int(flagged.sum())
    print(f"      {n_flagged}/{N} imagens sinalizadas como outlier "
          f"(z < {args.agreement_zscore_thresh})")

    for i in range(N):
        marker = " <- SINALIZADA" if flagged[i] else ""
        score_s = f"{agreement_score[i]:.4f}" if not np.isnan(agreement_score[i]) else "n/a"
        print(f"      {Path(image_paths[i]).stem[:40]:40s} agreement={score_s}{marker}")

    print(f"[3/4] Aplicando correção (peso={args.correction_weight})...")
    final_masks_e6 = apply_correction(
        final_masks, agree["similarity_map"], flagged, patch_grid, image_sizes,
        content_mask, args.fg_threshold, args.correction_weight,
    )

    if args.gt_index is not None and args.gt_index.exists() and n_flagged > 0:
        print(f"\n[Diagnóstico opcional — usa GT só pra mostrar, não influenciou a decisão]")
        gt_index = json.loads(args.gt_index.read_text(encoding="utf-8"))
        pixel_gt = load_pixel_gt_masks(image_paths, image_sizes, gt_index, args.fg_threshold)
        for i in range(N):
            if not flagged[i] or pixel_gt[i] is None:
                continue
            H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
            iou_antes = _iou(final_masks[i, :H_i, :W_i] > 0.5, pixel_gt[i])
            iou_depois = _iou(final_masks_e6[i, :H_i, :W_i] > 0.5, pixel_gt[i])
            veredito = "MELHOROU" if iou_depois > iou_antes + 0.01 else \
                      ("PIOROU" if iou_depois < iou_antes - 0.01 else "neutro")
            print(f"      {Path(image_paths[i]).stem[:40]:40s} "
                  f"iou_antes={iou_antes:.4f}  iou_depois={iou_depois:.4f}  {veredito}")

    print(f"[4/4] Salvando refined_masks_e6.npz...")
    np.savez_compressed(
        args.output / "refined_masks_e6.npz",
        final_masks=final_masks_e6,
        agreement_score=agreement_score,
        flagged=flagged,
        agreement_zscore_thresh=np.array(args.agreement_zscore_thresh),
        correction_weight=np.array(args.correction_weight),
        image_paths=image_paths,
        image_sizes=image_sizes,
    )
    print(f"[Salvo] {args.output / 'refined_masks_e6.npz'}")

    if args.export:
        export_segmented_images(final_masks_e6, image_paths, image_sizes, args.output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
