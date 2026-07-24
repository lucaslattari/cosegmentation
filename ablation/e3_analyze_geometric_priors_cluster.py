#!/usr/bin/env python3
"""
=============================================================================
analyze_geometric_priors.py — Priors geométricos alternativos para
mapeamento cluster -> FG/BG (Etapa 3)
=============================================================================

Motivação: em classes onde o fundo é mais consistente entre as imagens do
que o objeto de interesse (ex.: Liverpool — grama de campo de futebol
sempre presente, sempre parecida, enquanto os jogadores mudam de pose,
posição e contagem), tanto a saliência CLS (produção) quanto o prior
espacial de centralidade/compacidade já testado (spatial_based, alpha da
Etapa 3) podem falhar, porque nenhum dos dois foi desenhado especificamente
pra reconhecer "isto aqui claramente NÃO é o objeto".

Este script testa três priors geométricos novos, calculados por cluster
global da Etapa 3 (cluster_labels em cosaliency.npz), cada um capturando
uma intuição diferente e complementar:

  1. border_prior     — clusters que tocam a borda real da foto (não do
                         canvas com padding) tendem a ser fundo. Score alto
                         = LONGE da borda = mais parecido com objeto.

  2. extent_prior      — clusters cujo bounding box cobre uma fração muito
                         grande da foto tendem a ser fundo (grama, céu,
                         parede). Score alto = bounding box pequeno = mais
                         parecido com objeto.

  3. position_prior     — clusters cuja posição/centróide varia MUITO POUCO
                         entre as imagens da coleção são suspeitos de ser
                         fundo estático (o "assunto" da foto normalmente
                         muda de pose/posição entre fotos; o cenário não).
                         Score alto = posição instável entre imagens = mais
                         parecido com objeto.

IMPORTANTE (lição já aprendida com spatial_based): nenhum desses priors é
testado como substituto automático do que já está em produção. O script
testa cada um isoladamente e em combinação, e SÓ reporta os números — a
decisão de promover algum deles a produção exige que ele generalize bem
nas 6 classes de calibração (não piore nenhuma, principalmente Kite, que
já sabemos ser sensível a qualquer prior espacial ingênuo) e ajude nas
classes problemáticas (Liverpool, Red Sox Players).

Uso (uma classe):
  python3 analyze_geometric_priors.py \
    --cosaliency features/icoseg16/Liverpool/cosaliency.npz \
    --features   features/icoseg16/Liverpool/features.npz \
    --gt_index   gt_index.json \
    --output     features/icoseg16/Liverpool/geometric_priors

Uso (todas as classes de uma pasta, agregado):
  python3 analyze_geometric_priors.py \
    --features_root features/icoseg16 \
    --gt_index gt_index.json \
    --output geometric_priors_all

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
# Utilidades reaproveitadas (mesma lógica validada em analyze_step3_cosaliency.py)
# -----------------------------------------------------------------------------

def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)


def compute_content_mask(resize_info: "np.ndarray | None",
                         patch_grid: Tuple[int, int]) -> "np.ndarray | None":
    """Idêntica à versão usada em step2/step3/analyze_step1/2/3 — ver lá para docs."""
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


def load_patch_gt_masks(image_paths: List[str], patch_grid: Tuple[int, int],
                        gt_index: Optional[Dict[str, Any]], fg_threshold: float = 0.5,
                        resize_info: Optional[np.ndarray] = None
                        ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    if gt_index is None:
        return None, {"available": False, "matched": 0, "missing": len(image_paths)}
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

    H_p, W_p = patch_grid
    masks, missing = [], []
    for idx, img_path in enumerate(image_paths):
        gt_path = None
        for v in _path_variants(img_path):
            if v in variant_lookup:
                gt_path = Path(variant_lookup[v]); break
        if gt_path is None:
            s = unique_basename.get(Path(img_path).name)
            gt_path = Path(s) if s else None
        if gt_path is None or not gt_path.exists():
            masks.append(np.zeros((H_p, W_p), dtype=bool)); missing.append(img_path); continue
        try:
            gt_img = Image.open(gt_path).convert("L")
            if resize_info is not None:
                resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = \
                    [int(x) for x in resize_info[idx]]
                gt_resized = gt_img.resize((resized_w, resized_h), Image.BILINEAR)
                canvas = Image.new("L", (canvas_w, canvas_h), 0)
                canvas.paste(gt_resized, (pad_left, pad_top))
                mask_small = canvas.resize((W_p, H_p), Image.BILINEAR)
            else:
                mask_small = gt_img.resize((W_p, H_p), Image.BILINEAR)
            arr = np.asarray(mask_small, dtype=np.float32) / 255.0
            masks.append(arr >= fg_threshold)
        except Exception:
            masks.append(np.zeros((H_p, W_p), dtype=bool)); missing.append(img_path)

    patch_gt = np.stack([m.reshape(-1) for m in masks], axis=0)
    matched = len(image_paths) - len(missing)
    info = {"available": matched > 0, "matched": matched, "missing": len(missing)}
    return (patch_gt if matched > 0 else None), info


# -----------------------------------------------------------------------------
# Os três priors geométricos
# -----------------------------------------------------------------------------

def compute_geometric_priors(
    cluster_labels: np.ndarray,   # (N, P) — já com padding, se houver
    patch_grid: Tuple[int, int],
    n_clusters: int,
    content_mask: "np.ndarray | None" = None,   # (N, P) bool
    min_presence_patches: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Calcula border_prior, extent_prior e position_prior por cluster global
    (arrays de shape (n_clusters,), cada valor em [0,1], onde 1 = "mais
    parecido com objeto" segundo aquele critério).

    Todos os três são calculados só sobre patches de conteúdo real (exclui
    padding) — a "borda" usada em border_prior é a borda da FOTO real
    dentro do canvas, não a borda do canvas com padding, senão qualquer
    imagem com bastante padding teria clusters "tocando a borda" só por
    causa do padding, não do conteúdo.

    min_presence_patches é uma contagem ABSOLUTA de patches (não uma
    fração da imagem) para considerar o centróide de um cluster válido
    numa imagem. Uma fração fixa penalizaria sistematicamente objetos
    pequenos (ex.: Kite) — jogando position_prior deles pro valor neutro
    (0.5) em vez do valor real, o oposto do que queremos, já que objetos
    pequenos são justamente onde esse prior mais importa.
    """
    N, P = cluster_labels.shape
    H_p, W_p = patch_grid

    row_idx = np.repeat(np.arange(H_p), W_p)
    col_idx = np.tile(np.arange(W_p), H_p)

    # Para cada imagem, descobre os limites reais do conteúdo (não do canvas)
    content_bounds = []   # (row_min, row_max, col_min, col_max) por imagem
    for i in range(N):
        if content_mask is not None:
            valid = content_mask[i]
            if not valid.any():
                valid = np.ones(P, dtype=bool)
        else:
            valid = np.ones(P, dtype=bool)
        rows_valid = row_idx[valid]
        cols_valid = col_idx[valid]
        content_bounds.append((rows_valid.min(), rows_valid.max(),
                               cols_valid.min(), cols_valid.max()))

    border_frac  = np.zeros(n_clusters, dtype=np.float64)
    extent_frac  = np.zeros(n_clusters, dtype=np.float64)
    n_border_measurements = np.zeros(n_clusters, dtype=np.int64)
    n_extent_measurements = np.zeros(n_clusters, dtype=np.int64)

    # Para position_prior: centróide normalizado do cluster por imagem
    centroids_per_cluster: Dict[int, List[Tuple[float, float]]] = {c: [] for c in range(n_clusters)}

    for i in range(N):
        r0, r1, c0, c1 = content_bounds[i]
        h_content = max(r1 - r0, 1)
        w_content = max(c1 - c0, 1)

        if content_mask is not None:
            valid = content_mask[i]
        else:
            valid = np.ones(P, dtype=bool)

        labels_i = cluster_labels[i]
        for c in range(n_clusters):
            mask_c = valid & (labels_i == c)
            n_c = int(mask_c.sum())
            if n_c == 0:
                continue

            rows_c = row_idx[mask_c]
            cols_c = col_idx[mask_c]

            # --- border: fração dos patches do cluster na borda real da foto
            on_border = ((rows_c == r0) | (rows_c == r1) |
                        (cols_c == c0) | (cols_c == c1))
            border_frac[c] += float(on_border.mean())
            n_border_measurements[c] += 1

            # --- extent: fração da área da foto (bbox do cluster / área do conteúdo)
            bbox_h = (rows_c.max() - rows_c.min() + 1)
            bbox_w = (cols_c.max() - cols_c.min() + 1)
            bbox_frac = (bbox_h * bbox_w) / (h_content * w_content)
            extent_frac[c] += float(min(bbox_frac, 1.0))
            n_extent_measurements[c] += 1

            # --- centróide normalizado (só conta se o cluster tem presença
            #     não-trivial nessa imagem, evita ruído de clusters
            #     minúsculos/acidentais)
            if n_c >= min_presence_patches:
                cy = float((rows_c.mean() - r0) / h_content)
                cx = float((cols_c.mean() - c0) / w_content)
                centroids_per_cluster[c].append((cy, cx))

    border_frac = np.divide(border_frac, np.maximum(n_border_measurements, 1))
    extent_frac = np.divide(extent_frac, np.maximum(n_extent_measurements, 1))

    # position_prior: desvio padrão dos centróides entre as imagens onde o
    # cluster aparece de forma não-trivial. Poucos pontos (<2) => indefinido,
    # cai no meio do caminho (0.5) em vez de extremo, pra não distorcer.
    position_std = np.full(n_clusters, np.nan)
    for c in range(n_clusters):
        pts = centroids_per_cluster[c]
        if len(pts) >= 2:
            ys = [p[0] for p in pts]
            xs = [p[1] for p in pts]
            position_std[c] = (float(np.std(ys)) + float(np.std(xs))) / 2.0

    # Normaliza: 0.289 = desvio padrão máximo de U[0,1] (mesma referência já
    # usada em compute_spatial_saliency da Etapa 3, mantém escala consistente).
    MAX_STD = 0.289
    position_prior = np.where(
        np.isnan(position_std), 0.5,
        np.clip(position_std / MAX_STD, 0.0, 1.0)
    )

    border_prior = 1.0 - border_frac       # longe da borda = mais objeto
    extent_prior = 1.0 - extent_frac       # bbox pequeno   = mais objeto
    # position_prior já está na convenção certa (alta variação = mais objeto)

    return {
        "border_prior": border_prior.astype(np.float32),
        "extent_prior": extent_prior.astype(np.float32),
        "position_prior": position_prior.astype(np.float32),
    }


# -----------------------------------------------------------------------------
# Avaliação: qual critério de decisão FG melhor bate com o GT
# -----------------------------------------------------------------------------

def evaluate_candidates(
    cluster_labels: np.ndarray,          # (N, P)
    cluster_scores_e2: np.ndarray,       # (n_clusters,) — score bruto E2 (produção)
    priors: Dict[str, np.ndarray],       # cada um (n_clusters,)
    patch_gt: np.ndarray,                # (N, P)
    content_mask: "np.ndarray | None",
    fg_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Para cada candidato de score (E2 sozinho, cada prior geométrico sozinho,
    a combinação dos três priors geométricos, e E2+combinação 50/50),
    decide cluster_is_fg = score > threshold, monta a máscara de patches
    resultante (padding sempre forçado a 0/BG) e calcula o IoU real contra
    GT. Retorna um dict {nome_candidato: patch_iou}.
    """
    n_clusters = len(cluster_scores_e2)
    combined_geo = np.cbrt(
        np.clip(priors["border_prior"], 1e-6, 1.0) *
        np.clip(priors["extent_prior"], 1e-6, 1.0) *
        np.clip(priors["position_prior"], 1e-6, 1.0)
    )
    border = priors["border_prior"]

    candidates = {
        "e2_only (produção)":         cluster_scores_e2,
        "border_only":                priors["border_prior"],
        "extent_only":                priors["extent_prior"],
        "position_only":              priors["position_prior"],
        "geo_combined":               combined_geo,
        "e2_+_geo_combined (50/50)":  0.5 * cluster_scores_e2 + 0.5 * combined_geo,
        "e2_+_border (70/30)":        0.7 * cluster_scores_e2 + 0.3 * border,
        "e2_+_border (50/50)":        0.5 * cluster_scores_e2 + 0.5 * border,
        "e2_+_border (30/70)":        0.3 * cluster_scores_e2 + 0.7 * border,
    }

    N, P = cluster_labels.shape
    labels_flat = cluster_labels.reshape(-1)
    gt = patch_gt.astype(bool)
    content_flat = content_mask.reshape(-1) if content_mask is not None else None

    def iou_for_subset(subset_idx: np.ndarray) -> float:
        pred_flat = np.isin(labels_flat, subset_idx)
        if content_flat is not None:
            pred_flat = pred_flat & content_flat
        pred = pred_flat.reshape(N, P)
        ious = []
        for i in range(N):
            if gt[i].any() or (~gt[i]).any():
                ious.append(_iou(pred[i], gt[i]))
        return float(np.mean(ious)) if ious else None

    results = {}
    results_best_rank_cutoff = {}
    results_best_rank_k = {}
    for name, scores in candidates.items():
        # 1) threshold fixo em 0.5 (o que já era reportado)
        cluster_is_fg = scores > fg_threshold
        results[name] = iou_for_subset(np.flatnonzero(cluster_is_fg))

        # 2) melhor corte possível ao longo do ranking desse score — separa
        #    "o sinal existe, o limiar 0.5 é que está errado" de "não existe
        #    sinal nenhum nessa ordenação". Testa top-1, top-2, ..., top-K
        #    clusters por esse score (K=0..n_clusters) e fica com o melhor.
        order = np.argsort(-scores)  # do maior pro menor score
        best_iou = None
        best_k = 0
        for k in range(0, n_clusters + 1):
            subset = order[:k]
            iou_k = iou_for_subset(subset)
            if iou_k is not None and (best_iou is None or iou_k > best_iou):
                best_iou = iou_k
                best_k = k
        results_best_rank_cutoff[name] = best_iou
        results_best_rank_k[name] = best_k

    return {
        "threshold_0.5": results,
        "best_rank_cutoff": results_best_rank_cutoff,
        "best_rank_k": results_best_rank_k,
    }


# Frações fixas testadas por evaluate_fixed_fraction — mesmo valor pra
# todas as classes, nunca ajustado por classe ou usando GT na decisão.
FIXED_FRACTIONS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def evaluate_fixed_fraction(
    cluster_labels: np.ndarray,
    cluster_scores_e2: np.ndarray,
    priors: Dict[str, np.ndarray],
    patch_gt: np.ndarray,
    content_mask: "np.ndarray | None",
) -> Dict[str, Dict[str, float]]:
    """
    Testa uma regra de decisão que NÃO usa GT: em vez de threshold sobre o
    valor do score (que muda de escala entre classes — foi isso que quebrou
    o limiar de 0.5), pega sempre os top-K clusters por score, onde
    K = ceil(fração * n_clusters), com a MESMA fração fixa em toda classe.
    O GT só entra depois, pra medir o IoU resultante — nunca pra escolher K.
    """
    n_clusters = len(cluster_scores_e2)
    border = priors["border_prior"]

    candidates = {
        "e2_only (produção)":  cluster_scores_e2,
        "border_only":         border,
        "e2_+_border (30/70)": 0.3 * cluster_scores_e2 + 0.7 * border,
        "e2_+_border (50/50)": 0.5 * cluster_scores_e2 + 0.5 * border,
    }

    N, P = cluster_labels.shape
    labels_flat = cluster_labels.reshape(-1)
    gt = patch_gt.astype(bool)
    content_flat = content_mask.reshape(-1) if content_mask is not None else None

    def iou_for_subset(subset_idx: np.ndarray) -> "float | None":
        pred_flat = np.isin(labels_flat, subset_idx)
        if content_flat is not None:
            pred_flat = pred_flat & content_flat
        pred = pred_flat.reshape(N, P)
        ious = []
        for i in range(N):
            if gt[i].any() or (~gt[i]).any():
                ious.append(_iou(pred[i], gt[i]))
        return float(np.mean(ious)) if ious else None

    out: Dict[str, Dict[str, float]] = {name: {} for name in candidates}
    for name, scores in candidates.items():
        order = np.argsort(-scores)
        for frac in FIXED_FRACTIONS:
            k = max(1, int(np.ceil(frac * n_clusters)))
            subset = order[:k]
            out[name][f"{frac:.2f}"] = iou_for_subset(subset)

    return out


# -----------------------------------------------------------------------------
# Processamento de uma classe
# -----------------------------------------------------------------------------

def process_class(cosaliency_path: Path, features_path: Path,
                  gt_index: Dict[str, Any], fg_threshold: float = 0.5
                  ) -> Optional[Dict[str, Any]]:
    if not cosaliency_path.exists() or not features_path.exists():
        return None

    cos = np.load(cosaliency_path, allow_pickle=True)
    feat = np.load(features_path, allow_pickle=True)

    cluster_labels = cos["cluster_labels"]
    n_clusters = int(cos["n_clusters_used"])
    image_paths = [str(x) for x in cos["image_paths"].tolist()]
    patch_grid = tuple(int(x) for x in feat["patch_grid"])
    resize_info = feat["resize_info"] if "resize_info" in feat.files else None

    if "cluster_scores_e2" not in cos.files:
        print("      [Aviso] cosaliency.npz sem cluster_scores_e2 (versão antiga) — pulando.")
        return None
    cluster_scores_e2 = cos["cluster_scores_e2"]

    content_mask = compute_content_mask(resize_info, patch_grid)

    patch_gt, gt_info = load_patch_gt_masks(image_paths, patch_grid, gt_index,
                                            fg_threshold, resize_info=resize_info)
    if patch_gt is None:
        print("      [Aviso] GT não encontrado para nenhuma imagem — pulando.")
        return None

    priors = compute_geometric_priors(cluster_labels, patch_grid, n_clusters, content_mask)
    results = evaluate_candidates(cluster_labels, cluster_scores_e2, priors,
                                  patch_gt, content_mask, fg_threshold)
    fixed_fraction = evaluate_fixed_fraction(cluster_labels, cluster_scores_e2, priors,
                                             patch_gt, content_mask)

    return {
        "n_images": len(image_paths),
        "n_clusters": n_clusters,
        "gt_matched": gt_info.get("matched"),
        "results": results,
        "fixed_fraction": fixed_fraction,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Testa priors geométricos (borda/extensão/estabilidade posicional) "
                    "como alternativa/complemento ao score da Etapa 2 no mapeamento FG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cosaliency", type=Path, default=None,
                   help="cosaliency.npz de uma única classe.")
    p.add_argument("--features", type=Path, default=None,
                   help="features.npz correspondente (para patch_grid/resize_info).")
    p.add_argument("--features_root", type=Path, default=None,
                   help="Alternativa a --cosaliency/--features: pasta com "
                        "<classe>/cosaliency.npz e <classe>/features.npz para "
                        "TODAS as classes (roda em lote, agrega no final).")
    p.add_argument("--gt_index", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("geometric_priors_output"))
    p.add_argument("--fg_threshold", type=float, default=0.5)
    return p.parse_args()


def print_class_results(name_class: str, out: Dict[str, Any]) -> None:
    thr = out["results"]["threshold_0.5"]
    best = out["results"]["best_rank_cutoff"]
    best_k = out["results"]["best_rank_k"]
    n_clusters = out["n_clusters"]
    print(f"  {'candidato':28s} {'limiar=0.5':>11s} {'melhor_corte':>13s} {'k*':>5s}")
    for cand in thr:
        v_thr = thr[cand]
        v_best = best[cand]
        k = best_k[cand]
        v_thr_s = f"{v_thr:.4f}" if v_thr is not None else "n/a"
        v_best_s = f"{v_best:.4f}" if v_best is not None else "n/a"
        print(f"  {cand:28s} {v_thr_s:>11s} {v_best_s:>13s} {k:>3d}/{n_clusters}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gt_index = json.loads(args.gt_index.read_text(encoding="utf-8"))

    all_results: Dict[str, Dict[str, Any]] = {}

    if args.features_root is not None:
        class_dirs = sorted(d for d in args.features_root.iterdir() if d.is_dir())
        for cls_dir in class_dirs:
            cos_path = cls_dir / "cosaliency.npz"
            feat_path = cls_dir / "features.npz"
            if not cos_path.exists() or not feat_path.exists():
                continue
            print(f"\n=== {cls_dir.name} ===")
            out = process_class(cos_path, feat_path, gt_index, args.fg_threshold)
            if out is None:
                continue
            print_class_results(cls_dir.name, out)
            all_results[cls_dir.name] = out
    else:
        if args.cosaliency is None or args.features is None:
            print("[ERRO] Informe --cosaliency + --features, ou --features_root.")
            sys.exit(1)
        out = process_class(args.cosaliency, args.features, gt_index, args.fg_threshold)
        if out is None:
            print("[ERRO] Não foi possível processar.")
            sys.exit(1)
        print_class_results(args.cosaliency.parent.name, out)
        all_results[args.cosaliency.parent.name] = out

    json_path = args.output / "geometric_priors_results.json"
    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(all_results) > 1:
        print("\n" + "=" * 100)
        print("  RESUMO — média e pior caso por candidato (limiar=0.5 vs melhor corte possível no ranking)")
        print("=" * 100)
        candidate_names = list(next(iter(all_results.values()))["results"]["threshold_0.5"].keys())
        print(f"  {'candidato':28s} {'media_thr':>10s} {'pior_thr':>10s}   {'media_best':>10s} {'pior_best':>10s}")
        for name in candidate_names:
            thr_vals = [r["results"]["threshold_0.5"][name] for r in all_results.values()
                       if r["results"]["threshold_0.5"].get(name) is not None]
            best_vals = [r["results"]["best_rank_cutoff"][name] for r in all_results.values()
                        if r["results"]["best_rank_cutoff"].get(name) is not None]
            if not thr_vals or not best_vals:
                continue
            print(f"  {name:28s} {np.mean(thr_vals):>10.4f} {min(thr_vals):>10.4f}   "
                  f"{np.mean(best_vals):>10.4f} {min(best_vals):>10.4f}")
        print()
        print("  'media_best' alto e 'media_thr' baixo pra um candidato = o sinal existe,")
        print("  mas o limiar 0.5 fixo está descartando ele. Se os dois ficarem baixos juntos,")
        print("  não há sinal útil nesse candidato, independente de como ele é usado.")

        print("\n" + "=" * 100)
        print("  FRAÇÃO FIXA DO RANKING (regra livre de GT na decisão — GT só mede o resultado)")
        print("=" * 100)
        ff_candidate_names = list(next(iter(all_results.values()))["fixed_fraction"].keys())
        print(f"  {'candidato':22s} " + " ".join(f"{f:>8s}" for f in
              [f'{x:.2f}' for x in FIXED_FRACTIONS]))
        best_overall = None
        for name in ff_candidate_names:
            row = []
            for frac in FIXED_FRACTIONS:
                fkey = f"{frac:.2f}"
                vals = [r["fixed_fraction"][name].get(fkey) for r in all_results.values()
                       if r["fixed_fraction"][name].get(fkey) is not None]
                if not vals:
                    row.append(None); continue
                mean_v = float(np.mean(vals))
                min_v = float(min(vals))
                row.append((mean_v, min_v))
                if best_overall is None or mean_v > best_overall[2]:
                    best_overall = (name, fkey, mean_v, min_v)
            print(f"  {name:22s} " + " ".join(
                f"{v[0]:>8.3f}" if v is not None else f"{'n/a':>8s}" for v in row))
        if best_overall:
            print(f"\n  Melhor combinação por MÉDIA: {best_overall[0]} @ fração={best_overall[1]}"
                  f"  (média={best_overall[2]:.4f}, pior_caso={best_overall[3]:.4f})")
        print()
        print("  Comparar essa tabela com e2_only (produção) na mesma fração, ou com o e2_only")
        print("  original (threshold_0.5, tabela acima) — essa é a comparação que importa: uma")
        print("  regra sem GT na decisão bate a produção atual?")

    print(f"\n[Salvo] {json_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
