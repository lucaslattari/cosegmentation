#!/usr/bin/env python3
"""
=============================================================================
analyze_step1_features.py — Auditoria da Etapa 1 de um pipeline de cosegmentação
=============================================================================

Lê o features.npz gerado pela Etapa 1 e calcula métricas para responder:

  1. As features globais (CLS) parecem coesas na coleção?
  2. As features locais (patches) geram correspondências consistentes?
  3. Existem imagens outliers que podem atrapalhar as etapas seguintes?
  4. Se houver GT, as features locais separam foreground/background?
  5. Se houver GT, qual é o limite superior aproximado de clustering em patches?

Funciona com ou sem ground truth.

Uso mínimo:
  python analyze_step1_features.py --features features/Pandas/features.npz

Com ground truth:
  python analyze_step1_features.py \
      --features features/Pandas/features.npz \
      --gt_index gt_index.json \
      --output features/Pandas/feature_quality

Saídas:
  feature_quality.json
  feature_quality_report.md
  cls_similarity_heatmap.png
  local_similarity_heatmap.png
  cycle_top1_heatmap.png
  cycle_reconstruction_heatmap.png
  cls_outlier_scores.png
  cluster_distribution.png
  fg_bg_score_hist.png                 (se GT disponível)
  oracle_cluster_preview.png           (se GT disponível)

Observação:
  As interpretações são heurísticas. Elas servem para diagnosticar gargalos,
  não como prova formal de qualidade das features.

Mascaramento de padding (--resize keep_ar):
  Quando a Etapa 1 rodou com --resize keep_ar, a grade de patches cobre um
  canvas quadrado que inclui uma faixa de padding (cor média do ImageNet,
  ~zero após normalização) ao redor da imagem real. Patches inteiramente
  nessa faixa têm entrada quase constante entre imagens diferentes, o que
  gera afinidade cross-image artificialmente alta e pode: (a) inflar
  métricas não supervisionadas de correspondência/clustering local, e
  (b) inflar métricas supervisionadas com GT ao adicionar "negativos
  fáceis" triviais de separar. Por padrão este script identifica e exclui
  esses patches das métricas (mesma lógica usada em step2/step3/step4).
  Use --no_padding_mask para desativar (comportamento antigo).
=============================================================================
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
except Exception:
    MiniBatchKMeans = None
    roc_auc_score = None
    f1_score = None
    balanced_accuracy_score = None
    GroupKFold = None
    LogisticRegression = None


# -----------------------------------------------------------------------------
# Utilidades gerais
# -----------------------------------------------------------------------------

@dataclass
class Assessment:
    metric: str
    value: Optional[float]
    verdict: str
    indication: str
    suggested_test: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def offdiag_values(m: np.ndarray) -> np.ndarray:
    if m.shape[0] <= 1:
        return np.array([], dtype=np.float32)
    mask = ~np.eye(m.shape[0], dtype=bool)
    return m[mask]


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def compact_float(x: Optional[float], ndigits: int = 4) -> str:
    x = safe_float(x)
    if x is None:
        return "n/a"
    return f"{x:.{ndigits}f}"


def path_variants(path_str: str) -> set[str]:
    """Gera variantes robustas para casar caminhos do features.npz com gt_index."""
    variants: set[str] = set()
    if path_str is None:
        return variants

    raw = str(path_str).replace("\\", "/")
    variants.add(raw)
    variants.add(raw.lstrip("./"))

    p = Path(path_str)
    try:
        variants.add(p.as_posix())
    except Exception:
        pass
    try:
        variants.add(p.resolve().as_posix())
    except Exception:
        pass
    try:
        variants.add(os.path.relpath(p).replace("\\", "/"))
    except Exception:
        pass

    # Também tenta remover prefixos comuns de execução.
    for marker in ["icoseg/images/", "images/"]:
        idx = raw.find(marker)
        if idx >= 0:
            variants.add(raw[idx:])

    return {v for v in variants if v}


# -----------------------------------------------------------------------------
# Mascaramento de padding (portado de step2_correspondence.py / step3_cosaliency.py)
# -----------------------------------------------------------------------------


def compute_content_mask(resize_info: "np.ndarray | None",
                         patch_grid: Tuple[int, int]) -> "np.ndarray | None":
    """
    A partir de resize_info (só existe quando a Etapa 1 rodou com
    --resize keep_ar), calcula, por imagem, quais patches da grade caem
    dentro da imagem real e quais caem inteira ou parcialmente na faixa de
    padding.

    Um patch é considerado "conteúdo" se sua área tem QUALQUER interseção
    com a região da imagem real dentro do canvas — mesmo critério usado em
    step2_correspondence.py e step3_cosaliency.py para excluir patches de
    padding do grafo de afinidade / NCut / clustering.

    Retorna None se resize_info for None (Etapa 1 rodou em modo 'square',
    sem padding — não há nada para mascarar).

    Retorna: (N, P) bool — True = patch de conteúdo real, False = padding.
    """
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


def sample_from_pool(pool: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    """Amostra até max_n índices de um pool arbitrário de índices (não necessariamente 0..n)."""
    if len(pool) <= max_n:
        return np.sort(pool)
    return np.sort(rng.choice(pool, size=max_n, replace=False))


# -----------------------------------------------------------------------------
# Interpretações heurísticas
# -----------------------------------------------------------------------------


def assess_high_good(metric: str, value: Optional[float],
                     strong: float, moderate: float,
                     strong_msg: str, moderate_msg: str, weak_msg: str,
                     suggested_test: str) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível", "Não foi possível calcular esta métrica.", suggested_test)
    if v >= strong:
        return Assessment(metric, v, "bom", strong_msg, suggested_test)
    if v >= moderate:
        return Assessment(metric, v, "intermediário", moderate_msg, suggested_test)
    return Assessment(metric, v, "fraco", weak_msg, suggested_test)


def assess_low_good(metric: str, value: Optional[float],
                    good: float, moderate: float,
                    good_msg: str, moderate_msg: str, bad_msg: str,
                    suggested_test: str) -> Assessment:
    v = safe_float(value)
    if v is None:
        return Assessment(metric, None, "indisponível", "Não foi possível calcular esta métrica.", suggested_test)
    if v <= good:
        return Assessment(metric, v, "bom", good_msg, suggested_test)
    if v <= moderate:
        return Assessment(metric, v, "intermediário", moderate_msg, suggested_test)
    return Assessment(metric, v, "alerta", bad_msg, suggested_test)


def assessment_to_dict(a: Assessment) -> Dict[str, Any]:
    return {
        "metric": a.metric,
        "value": a.value,
        "verdict": a.verdict,
        "indication": a.indication,
        "suggested_test": a.suggested_test,
    }


# -----------------------------------------------------------------------------
# Carregamento
# -----------------------------------------------------------------------------


def load_features(features_path: Path) -> Dict[str, Any]:
    if not features_path.exists():
        raise FileNotFoundError(f"features.npz não encontrado: {features_path}")
    data = np.load(features_path, allow_pickle=True)

    required = ["patch_features", "cls_features", "patch_grid", "image_paths", "image_sizes"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"Arquivo inválido. Campos ausentes em features.npz: {missing}")

    patch_features = np.asarray(data["patch_features"], dtype=np.float32)
    cls_features = np.asarray(data["cls_features"], dtype=np.float32)
    patch_grid = tuple(int(x) for x in np.asarray(data["patch_grid"]).tolist())
    image_paths = [str(x) for x in data["image_paths"].tolist()]
    image_sizes = np.asarray(data["image_sizes"])
    # resize_info só existe se a Etapa 1 rodou com --resize keep_ar.
    # Colunas: resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w
    resize_info = np.asarray(data["resize_info"]) if "resize_info" in data.files else None

    if patch_features.ndim != 3:
        raise ValueError(f"patch_features deveria ser (N, P, D), obtido {patch_features.shape}")
    if cls_features.ndim != 2:
        raise ValueError(f"cls_features deveria ser (N, D), obtido {cls_features.shape}")
    if patch_features.shape[0] != cls_features.shape[0]:
        raise ValueError("patch_features e cls_features têm números diferentes de imagens.")
    if patch_grid[0] * patch_grid[1] != patch_features.shape[1]:
        raise ValueError(
            f"patch_grid={patch_grid} não bate com P={patch_features.shape[1]}. "
            "O arquivo features.npz pode estar inconsistente."
        )

    return {
        "patch_features": patch_features,
        "cls_features": cls_features,
        "patch_grid": patch_grid,
        "image_paths": image_paths,
        "image_sizes": image_sizes,
        "resize_info": resize_info,
    }


def load_gt_index(gt_index_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if gt_index_path is None:
        return None
    if not gt_index_path.exists():
        print(f"[Aviso] gt_index.json não encontrado: {gt_index_path}. Rodando sem GT.")
        return None
    with gt_index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_gt_lookup(gt_index: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Optional[str]]]:
    """Retorna lookup por variantes de caminho e lookup por basename único."""
    mappings = gt_index.get("mappings", {})
    variant_lookup: Dict[str, str] = {}
    basename_to_gt: Dict[str, List[str]] = {}

    for img_path, gt_path in mappings.items():
        for v in path_variants(img_path):
            variant_lookup[v] = str(gt_path)
        base = Path(img_path).name
        basename_to_gt.setdefault(base, []).append(str(gt_path))

    unique_basename_lookup: Dict[str, Optional[str]] = {}
    for base, gts in basename_to_gt.items():
        unique = sorted(set(gts))
        unique_basename_lookup[base] = unique[0] if len(unique) == 1 else None

    return variant_lookup, unique_basename_lookup


def find_gt_for_image(image_path: str,
                      variant_lookup: Dict[str, str],
                      basename_lookup: Dict[str, Optional[str]]) -> Optional[Path]:
    for v in path_variants(image_path):
        if v in variant_lookup:
            return Path(variant_lookup[v])
    by_base = basename_lookup.get(Path(image_path).name)
    if by_base:
        return Path(by_base)
    return None


def load_patch_gt_masks(image_paths: List[str],
                        patch_grid: Tuple[int, int],
                        gt_index: Optional[Dict[str, Any]],
                        fg_threshold: float = 0.5,
                        resize_info: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Carrega máscaras GT e reduz para a grade de patches.

    Se resize_info for fornecido (Etapa 1 rodada com --resize keep_ar), o GT
    é primeiro redimensionado para o tamanho da imagem DENTRO do canvas (sem
    padding), colado num canvas do mesmo tamanho usado como entrada do DINOv2
    e só então reduzido para a grade de patches. Sem isso, a grade de patches
    (que cobre o canvas com padding) fica desalinhada com o GT (que cobre só
    a imagem original), inflando FN/FP perto das bordas em imagens não-quadradas.
    """
    if gt_index is None:
        return None, {
            "available": False,
            "matched": 0,
            "missing": len(image_paths),
            "missing_images": image_paths,
        }

    variant_lookup, basename_lookup = build_gt_lookup(gt_index)
    H_p, W_p = patch_grid
    masks: List[np.ndarray] = []
    missing: List[str] = []

    for idx, img_path in enumerate(image_paths):
        gt_path = find_gt_for_image(img_path, variant_lookup, basename_lookup)
        if gt_path is None or not gt_path.exists():
            masks.append(np.zeros((H_p, W_p), dtype=bool))
            missing.append(img_path)
            continue

        try:
            gt_img = Image.open(gt_path).convert("L")

            if resize_info is not None:
                resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = \
                    [int(x) for x in resize_info[idx]]

                # Redimensiona o GT para o tamanho da imagem sem padding...
                gt_resized = gt_img.resize((resized_w, resized_h), Image.BILINEAR)

                # ...e cola num canvas preto (background) do mesmo tamanho
                # usado como entrada do DINOv2, para alinhar com a grade de patches.
                canvas = Image.new("L", (canvas_w, canvas_h), 0)
                canvas.paste(gt_resized, (pad_left, pad_top))

                mask_small = canvas.resize((W_p, H_p), Image.BILINEAR)
            else:
                # BILINEAR aproxima a fração foreground dentro do patch ao reduzir.
                mask_small = gt_img.resize((W_p, H_p), Image.BILINEAR)

            arr = np.asarray(mask_small, dtype=np.float32) / 255.0
            masks.append(arr >= fg_threshold)
        except Exception as e:
            print(f"[Aviso] Falha lendo GT {gt_path}: {e}")
            masks.append(np.zeros((H_p, W_p), dtype=bool))
            missing.append(img_path)

    patch_gt = np.stack([m.reshape(-1) for m in masks], axis=0)
    matched = len(image_paths) - len(missing)
    info = {
        "available": matched > 0,
        "matched": matched,
        "missing": len(missing),
        "missing_images": missing[:50],
        "fg_patch_fraction": float(patch_gt.mean()) if matched > 0 else None,
    }

    if matched == 0:
        return None, info
    return patch_gt, info


# -----------------------------------------------------------------------------
# Métricas sem GT
# -----------------------------------------------------------------------------


def compute_cls_metrics(cls_features: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    cls_n = normalize_rows(cls_features)
    sim = cls_n @ cls_n.T
    np.fill_diagonal(sim, 1.0)
    off = offdiag_values(sim)

    if len(off) == 0:
        outlier_scores = np.zeros((cls_features.shape[0],), dtype=np.float32)
        mean_sim_per_image = np.ones((cls_features.shape[0],), dtype=np.float32)
    else:
        tmp = sim.copy()
        np.fill_diagonal(tmp, np.nan)
        mean_sim_per_image = np.nanmean(tmp, axis=1)
        outlier_scores = 1.0 - mean_sim_per_image

    metrics = {
        "cls_mean_similarity": safe_float(np.mean(off)) if len(off) else None,
        "cls_std_similarity": safe_float(np.std(off)) if len(off) else None,
        "cls_min_similarity": safe_float(np.min(off)) if len(off) else None,
        "cls_max_similarity": safe_float(np.max(off)) if len(off) else None,
        "cls_outlier_score_mean": safe_float(np.mean(outlier_scores)),
        "cls_outlier_score_max": safe_float(np.max(outlier_scores)),
    }
    return metrics, sim, outlier_scores


def sample_indices(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def compute_local_pair_metrics(patch_features: np.ndarray,
                               content_mask: "np.ndarray | None" = None,
                               max_images: int = 50,
                               max_patches: int = 128,
                               seed: int = 42) -> Tuple[
                                   Dict[str, Any],
                                   np.ndarray,
                                   np.ndarray,
                                   np.ndarray,
                                   np.ndarray,
                               ]:
    """
    Calcula similaridade local aproximada entre imagens e cycle consistency.

    Para cada par A/B:
      - amostra até max_patches por imagem (apenas entre patches de
        conteúdo real, se content_mask for fornecida — patches de padding
        têm entrada quase constante e inflam artificialmente a similaridade
        cross-image entre imagens diferentes);
      - calcula S = A @ B.T;
      - local_nn = média das melhores similaridades A->B e B->A;
      - cycle_top1 = fração de patches que retornam ao mesmo patch amostrado no ciclo A->B->A.

    Não assume alinhamento espacial entre imagens.

    Retorna:
      - metrics: métricas agregadas;
      - local_sim: similaridade local NN por par de imagens;
      - cycle: taxa de retorno exato Top-1 por par;
      - recon_sim: similaridade entre o patch original e o patch reconstruído;
      - idx_images: índices globais das imagens usadas nas matrizes.
    """
    rng = np.random.default_rng(seed)
    N, P, D = patch_features.shape
    idx_images = sample_indices(N, max_images, rng)
    pf = normalize_rows(patch_features[idx_images])
    M = len(idx_images)

    patch_samples = []
    n_content_used = 0
    for local_i, global_i in enumerate(idx_images):
        if content_mask is not None:
            pool = np.flatnonzero(content_mask[global_i])
            if pool.size == 0:
                pool = np.arange(P)  # salvaguarda: imagem sem conteúdo detectado
            else:
                n_content_used += 1
        else:
            pool = np.arange(P)
        patch_samples.append(sample_from_pool(pool, max_patches, rng))

    local_sim = np.eye(M, dtype=np.float32)
    cycle = np.eye(M, dtype=np.float32)
    recon_sim = np.eye(M, dtype=np.float32)

    for a in range(M):
        A = pf[a, patch_samples[a], :]
        for b in range(a + 1, M):
            B = pf[b, patch_samples[b], :]
            S = A @ B.T

            # A -> B e B -> A
            nn_ab = np.argmax(S, axis=1)
            nn_ba = np.argmax(S, axis=0)
            score_ab = float(np.mean(np.max(S, axis=1)))
            score_ba = float(np.mean(np.max(S, axis=0)))
            local_score = 0.5 * (score_ab + score_ba)

            # Ciclo A -> B -> A no subconjunto amostrado.
            recon_a = nn_ba[nn_ab]
            cyc_ab = float(np.mean(recon_a == np.arange(len(A))))
            rec_sim_ab = float(np.mean(np.sum(A * A[recon_a], axis=1)))

            # Ciclo B -> A -> B.
            recon_b = nn_ab[nn_ba]
            cyc_ba = float(np.mean(recon_b == np.arange(len(B))))
            rec_sim_ba = float(np.mean(np.sum(B * B[recon_b], axis=1)))

            cycle_score = 0.5 * (cyc_ab + cyc_ba)
            recon_score = 0.5 * (rec_sim_ab + rec_sim_ba)

            local_sim[a, b] = local_sim[b, a] = local_score
            cycle[a, b] = cycle[b, a] = cycle_score
            recon_sim[a, b] = recon_sim[b, a] = recon_score

    off_local = offdiag_values(local_sim)
    off_cycle = offdiag_values(cycle)
    off_recon = offdiag_values(recon_sim)

    metrics = {
        "local_pair_images_used": int(M),
        "local_pair_patch_sample": int(min(P, max_patches)),
        "local_pair_padding_masked": bool(content_mask is not None),
        "local_pair_images_with_content_mask_applied": int(n_content_used) if content_mask is not None else None,
        "local_nn_similarity_mean": safe_float(np.mean(off_local)) if len(off_local) else None,
        "local_nn_similarity_std": safe_float(np.std(off_local)) if len(off_local) else None,
        "local_nn_similarity_min": safe_float(np.min(off_local)) if len(off_local) else None,
        "cycle_top1_rate_mean": safe_float(np.mean(off_cycle)) if len(off_cycle) else None,
        "cycle_reconstruction_similarity_mean": safe_float(np.mean(off_recon)) if len(off_recon) else None,
    }
    return metrics, local_sim, cycle, recon_sim, idx_images


def compute_unsupervised_cluster_metrics(patch_features: np.ndarray,
                                         content_mask: "np.ndarray | None" = None,
                                         n_clusters: int = 5,
                                         max_cluster_patches: int = 50000,
                                         seed: int = 42) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[Any]]:
    """
    Clustering não supervisionado dos patches.

    Se content_mask for fornecida, patches de padding são excluídos do
    treino e da predição do K-means inteiramente (não apenas re-rotulados
    depois) — do contrário, coleções com muito padding podem gastar parte
    do orçamento de n_clusters em "clusters de padding" sem qualquer valor
    semântico, inflando cluster_recurrence_mean artificialmente. Patches de
    padding recebem rótulo -1 no array de saída (mantém shape (N, P) para
    compatibilidade com o restante do script) e são automaticamente
    tratados como "fora de qualquer cluster FG" nas métricas com GT
    (oracle_cluster_iou), o que é o comportamento correto já que padding é
    sempre background no GT alinhado.
    """
    if MiniBatchKMeans is None:
        return {"cluster_error": "scikit-learn não disponível."}, None, None

    rng = np.random.default_rng(seed)
    N, P, D = patch_features.shape
    X = normalize_rows(patch_features.reshape(-1, D))
    total = X.shape[0]

    if content_mask is not None:
        valid_idx = np.flatnonzero(content_mask.reshape(-1))
        if valid_idx.size == 0:
            valid_idx = np.arange(total)  # salvaguarda
    else:
        valid_idx = np.arange(total)

    train_sel = sample_indices(len(valid_idx), max_cluster_patches, rng)
    train_pool = valid_idx[train_sel]

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=4096,
        n_init="auto",
        max_iter=200,
    )
    kmeans.fit(X[train_pool])

    labels_valid = np.empty((len(valid_idx),), dtype=np.int32)
    chunk = 50000
    for start in range(0, len(valid_idx), chunk):
        end = min(start + chunk, len(valid_idx))
        labels_valid[start:end] = kmeans.predict(X[valid_idx[start:end]])

    labels_flat = np.full(total, -1, dtype=np.int32)
    labels_flat[valid_idx] = labels_valid
    labels_img = labels_flat.reshape(N, P)

    # Distribuição por imagem — só entre patches de conteúdo real.
    hists = np.zeros((N, n_clusters), dtype=np.float32)
    for i in range(N):
        li = labels_img[i]
        li_valid = li[li >= 0]
        if li_valid.size == 0:
            continue
        counts = np.bincount(li_valid, minlength=n_clusters).astype(np.float32)
        hists[i] = counts / max(1.0, counts.sum())

    # Entropia normalizada: alta significa que muitos clusters aparecem por imagem.
    eps = 1e-12
    entropy = -np.sum(hists * np.log(hists + eps), axis=1) / math.log(n_clusters)

    # Recorrência: fração de imagens em que cada cluster aparece com pelo menos 2% dos patches.
    recurrence_per_cluster = np.mean(hists >= 0.02, axis=0)
    common_clusters = int(np.sum(recurrence_per_cluster >= 0.75))

    metrics = {
        "cluster_n_clusters": int(n_clusters),
        "cluster_train_patches": int(len(train_pool)),
        "cluster_padding_excluded_patches": int(total - len(valid_idx)) if content_mask is not None else 0,
        "cluster_entropy_mean": safe_float(np.mean(entropy)),
        "cluster_entropy_std": safe_float(np.std(entropy)),
        "cluster_recurrence_mean": safe_float(np.mean(recurrence_per_cluster)),
        "cluster_common_clusters_75pct": common_clusters,
        "cluster_distribution_mean": hists.mean(axis=0).tolist(),
        "cluster_recurrence_per_cluster": recurrence_per_cluster.tolist(),
    }
    return metrics, labels_img, kmeans


# -----------------------------------------------------------------------------
# Métricas com GT
# -----------------------------------------------------------------------------


def random_pair_similarity(A: np.ndarray, B: np.ndarray, n_pairs: int, rng: np.random.Generator) -> Optional[float]:
    if len(A) == 0 or len(B) == 0:
        return None
    ia = rng.integers(0, len(A), size=n_pairs)
    ib = rng.integers(0, len(B), size=n_pairs)
    return safe_float(np.mean(np.sum(A[ia] * B[ib], axis=1)))


def compute_fg_bg_similarity_metrics(patch_features: np.ndarray,
                                     patch_gt: np.ndarray,
                                     content_mask: "np.ndarray | None" = None,
                                     max_samples: int = 20000,
                                     seed: int = 42) -> Dict[str, Any]:
    """
    Se content_mask for fornecida, patches de padding são excluídos por
    completo (não contam nem como FG nem como BG). Mantê-los como
    "background fácil" infla fg_bg_similarity_margin artificialmente, já
    que padding tem entrada quase constante e é trivialmente separável do
    foreground — o que não diz nada sobre a qualidade da separação
    FG/background real que interessa às etapas seguintes.
    """
    rng = np.random.default_rng(seed)
    N, P, D = patch_features.shape
    X_all = normalize_rows(patch_features.reshape(-1, D))
    y_all = patch_gt.reshape(-1).astype(bool)

    if content_mask is not None:
        valid = content_mask.reshape(-1)
        X = X_all[valid]
        y = y_all[valid]
    else:
        X = X_all
        y = y_all

    fg_idx_all = np.flatnonzero(y)
    bg_idx_all = np.flatnonzero(~y)

    if len(fg_idx_all) == 0 or len(bg_idx_all) == 0:
        return {"fg_bg_error": "GT não contém foreground e background suficientes (após excluir padding)."}

    fg_idx = fg_idx_all if len(fg_idx_all) <= max_samples else rng.choice(fg_idx_all, size=max_samples, replace=False)
    bg_idx = bg_idx_all if len(bg_idx_all) <= max_samples else rng.choice(bg_idx_all, size=max_samples, replace=False)

    FG = X[fg_idx]
    BG = X[bg_idx]

    n_pairs = min(max_samples, 50000)
    fg_fg = random_pair_similarity(FG, FG, n_pairs, rng)
    bg_bg = random_pair_similarity(BG, BG, n_pairs, rng)
    fg_bg = random_pair_similarity(FG, BG, n_pairs, rng)
    margin = None if fg_fg is None or fg_bg is None else fg_fg - fg_bg

    return {
        "gt_fg_patch_count": int(len(fg_idx_all)),
        "gt_bg_patch_count": int(len(bg_idx_all)),
        "gt_fg_patch_fraction": safe_float(np.mean(y)),
        "fg_bg_padding_excluded": bool(content_mask is not None),
        "fg_fg_similarity_mean": fg_fg,
        "bg_bg_similarity_mean": bg_bg,
        "fg_bg_similarity_mean": fg_bg,
        "fg_bg_similarity_margin": safe_float(margin),
    }


def compute_leave_one_image_prototype_auc(patch_features: np.ndarray,
                                          patch_gt: np.ndarray,
                                          content_mask: "np.ndarray | None" = None) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Se content_mask for fornecida, patches de padding são excluídos tanto
    do treino do protótipo (centróides FG/BG) quanto da avaliação —
    sem isso, o protótipo de BG fica parcialmente "contaminado" por vetores
    quase constantes de padding, e a AUC de avaliação fica artificialmente
    alta ao incluir esses mesmos patches triviais como exemplos de teste.
    """
    if roc_auc_score is None:
        return {"prototype_auc_error": "scikit-learn não disponível."}, None, None

    N, P, D = patch_features.shape
    X = normalize_rows(patch_features)
    Y = patch_gt.astype(bool)

    all_scores: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for i in range(N):
        train_mask = np.ones(N, dtype=bool)
        train_mask[i] = False
        X_train_imgs = X[train_mask]
        Y_train_imgs = Y[train_mask]

        if content_mask is not None:
            cm_train = content_mask[train_mask]
            X_train = X_train_imgs.reshape(-1, D)[cm_train.reshape(-1)]
            y_train = Y_train_imgs.reshape(-1)[cm_train.reshape(-1)]
        else:
            X_train = X_train_imgs.reshape(-1, D)
            y_train = Y_train_imgs.reshape(-1)

        if y_train.sum() == 0 or (~y_train).sum() == 0:
            continue

        if content_mask is not None:
            valid_i = content_mask[i]
            if valid_i.sum() == 0:
                valid_i = np.ones(P, dtype=bool)
        else:
            valid_i = np.ones(P, dtype=bool)

        Yi_valid = Y[i][valid_i]
        if Yi_valid.sum() == 0 or (~Yi_valid).sum() == 0:
            # Esta imagem não contribui para AUC, mas ainda pode ter score.
            continue

        fg_centroid = X_train[y_train].mean(axis=0)
        bg_centroid = X_train[~y_train].mean(axis=0)
        fg_centroid = normalize_rows(fg_centroid[None, :])[0]
        bg_centroid = normalize_rows(bg_centroid[None, :])[0]

        Xi = X[i][valid_i]
        score = Xi @ fg_centroid - Xi @ bg_centroid
        all_scores.append(score.astype(np.float32))
        all_y.append(Yi_valid.astype(np.uint8))

    if not all_scores:
        return {"prototype_auc_error": "Não havia imagens suficientes com FG/BG para validação leave-one-image-out (após excluir padding)."}, None, None

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_y)

    if len(np.unique(labels)) < 2:
        return {"prototype_auc_error": "Rótulos insuficientes para AUC."}, scores, labels

    auc = float(roc_auc_score(labels, scores))

    # Escolhe limiar 0 como regra simples: mais parecido com protótipo FG do que BG.
    pred = scores >= 0
    f1 = float(f1_score(labels, pred)) if f1_score is not None else None
    bal_acc = float(balanced_accuracy_score(labels, pred)) if balanced_accuracy_score is not None else None

    return {
        "prototype_padding_excluded": bool(content_mask is not None),
        "prototype_leave_one_image_auc": auc,
        "prototype_threshold0_f1": f1,
        "prototype_threshold0_balanced_accuracy": bal_acc,
    }, scores, labels


def compute_linear_probe_cv(patch_features: np.ndarray,
                            patch_gt: np.ndarray,
                            content_mask: "np.ndarray | None" = None,
                            max_samples: int = 15000,
                            seed: int = 42) -> Dict[str, Any]:
    """
    Linear probe com validação por grupos de imagem.

    Serve como teste forte: se uma regressão logística linear não separa FG/BG,
    provavelmente as features locais não estão muito boas para cosegmentação.

    Se content_mask for fornecida, patches de padding são excluídos antes da
    amostragem balanceada, pelo mesmo motivo das demais métricas com GT:
    são negativos triviais que não representam a dificuldade real de
    separar FG do fundo.
    """
    if LogisticRegression is None or GroupKFold is None or roc_auc_score is None:
        return {"linear_probe_error": "scikit-learn incompleto ou indisponível."}

    rng = np.random.default_rng(seed)
    N, P, D = patch_features.shape
    X_all = normalize_rows(patch_features.reshape(-1, D))
    y_all = patch_gt.reshape(-1).astype(np.uint8)
    groups_all = np.repeat(np.arange(N), P)

    if content_mask is not None:
        valid = content_mask.reshape(-1)
        X = X_all[valid]
        y = y_all[valid]
        groups = groups_all[valid]
    else:
        X = X_all
        y = y_all
        groups = groups_all

    fg_idx = np.flatnonzero(y == 1)
    bg_idx = np.flatnonzero(y == 0)
    if len(fg_idx) == 0 or len(bg_idx) == 0:
        return {"linear_probe_error": "GT sem FG/BG suficientes (após excluir padding)."}

    # Amostragem balanceada para manter o teste rápido.
    half = max_samples // 2
    fg_s = fg_idx if len(fg_idx) <= half else rng.choice(fg_idx, size=half, replace=False)
    bg_s = bg_idx if len(bg_idx) <= half else rng.choice(bg_idx, size=half, replace=False)
    idx = np.concatenate([fg_s, bg_s])
    rng.shuffle(idx)

    Xs = X[idx]
    ys = y[idx]
    gs = groups[idx]

    unique_groups = np.unique(gs)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        return {"linear_probe_error": "Menos de duas imagens com amostras suficientes."}

    aucs: List[float] = []
    f1s: List[float] = []
    bals: List[float] = []

    cv = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in cv.split(Xs, ys, gs):
        if len(np.unique(ys[train_idx])) < 2 or len(np.unique(ys[test_idx])) < 2:
            continue
        clf = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        clf.fit(Xs[train_idx], ys[train_idx])
        scores = clf.decision_function(Xs[test_idx])
        pred = clf.predict(Xs[test_idx])
        aucs.append(float(roc_auc_score(ys[test_idx], scores)))
        if f1_score is not None:
            f1s.append(float(f1_score(ys[test_idx], pred)))
        if balanced_accuracy_score is not None:
            bals.append(float(balanced_accuracy_score(ys[test_idx], pred)))

    if not aucs:
        return {"linear_probe_error": "Nenhum fold válido para linear probe."}

    return {
        "linear_probe_padding_excluded": bool(content_mask is not None),
        "linear_probe_samples": int(len(idx)),
        "linear_probe_group_folds": int(n_splits),
        "linear_probe_auc_mean": safe_float(np.mean(aucs)),
        "linear_probe_auc_std": safe_float(np.std(aucs)),
        "linear_probe_f1_mean": safe_float(np.mean(f1s)) if f1s else None,
        "linear_probe_balanced_accuracy_mean": safe_float(np.mean(bals)) if bals else None,
    }


def iou_bool(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def compute_oracle_cluster_iou(labels_img: Optional[np.ndarray],
                               patch_gt: np.ndarray,
                               n_clusters: int) -> Dict[str, Any]:
    """
    labels_img pode conter -1 para patches de padding (quando content_mask
    foi usada em compute_unsupervised_cluster_metrics). Como os subsets
    testados aqui são sempre subconjuntos de {0, ..., n_clusters-1},
    np.isin(labels_flat, subset) nunca inclui os patches de padding —
    eles são automaticamente tratados como "não FG", o que é correto pois
    o GT alinhado também os marca como background.
    """
    if labels_img is None:
        return {"oracle_cluster_error": "Clusters não disponíveis."}

    N, P = patch_gt.shape
    labels_flat = labels_img.reshape(-1)
    gt_flat = patch_gt.reshape(-1).astype(bool)

    # Melhor cluster único global.
    single_ious = []
    for k in range(n_clusters):
        pred = labels_flat == k
        single_ious.append(iou_bool(pred, gt_flat))
    best_single_k = int(np.argmax(single_ious))
    best_single_iou = float(single_ious[best_single_k])

    # Melhor subconjunto global de clusters, viável para K pequeno.
    best_subset: Tuple[int, ...] = (best_single_k,)
    best_subset_iou = best_single_iou
    if n_clusters <= 12:
        for r in range(1, n_clusters + 1):
            for subset in itertools.combinations(range(n_clusters), r):
                pred = np.isin(labels_flat, subset)
                val = iou_bool(pred, gt_flat)
                if val > best_subset_iou:
                    best_subset_iou = val
                    best_subset = subset

    # Oracle por imagem: upper bound mais permissivo.
    imagewise_best_single = []
    imagewise_best_subset = []
    for i in range(N):
        yi = patch_gt[i].astype(bool)
        li = labels_img[i]
        best_i = 0.0
        for k in range(n_clusters):
            best_i = max(best_i, iou_bool(li == k, yi))
        imagewise_best_single.append(best_i)

        best_s = best_i
        if n_clusters <= 8:
            for r in range(1, n_clusters + 1):
                for subset in itertools.combinations(range(n_clusters), r):
                    best_s = max(best_s, iou_bool(np.isin(li, subset), yi))
        imagewise_best_subset.append(best_s)

    return {
        "oracle_best_single_cluster": best_single_k,
        "oracle_best_single_cluster_iou": best_single_iou,
        "oracle_best_global_subset": list(best_subset),
        "oracle_best_global_subset_iou": float(best_subset_iou),
        "oracle_imagewise_best_single_iou_mean": safe_float(np.mean(imagewise_best_single)),
        "oracle_imagewise_best_single_iou_std": safe_float(np.std(imagewise_best_single)),
        "oracle_imagewise_best_subset_iou_mean": safe_float(np.mean(imagewise_best_subset)),
        "oracle_imagewise_best_subset_iou_std": safe_float(np.std(imagewise_best_subset)),
    }


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_heatmap(matrix: np.ndarray,
                 out_path: Path,
                 title: str,
                 xlabel: str = "Imagem",
                 ylabel: str = "Imagem") -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", vmin=float(np.nanmin(matrix)), vmax=float(np.nanmax(matrix)))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_outliers(outlier_scores: np.ndarray,
                  image_paths: List[str],
                  out_path: Path) -> None:
    if plt is None:
        return
    order = np.argsort(outlier_scores)[::-1]
    top = order[:min(30, len(order))]
    labels = [Path(image_paths[i]).stem[:24] for i in top]
    values = outlier_scores[top]

    fig_h = max(5, 0.28 * len(top))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(top))
    ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Outlier score = 1 - similaridade CLS média")
    ax.set_title("Imagens mais discrepantes por CLS")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_cluster_distribution(cluster_distribution_mean: List[float],
                              recurrence: List[float],
                              out_path: Path) -> None:
    if plt is None:
        return
    k = len(cluster_distribution_mean)
    x = np.arange(k)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.38
    ax.bar(x - width / 2, cluster_distribution_mean, width, label="fração média de patches")
    ax.bar(x + width / 2, recurrence, width, label="recorrência nas imagens")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(k)])
    ax.set_xlabel("Cluster")
    ax.set_ylim(0, 1)
    ax.set_title("Distribuição e recorrência dos clusters locais")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_fg_bg_hist(scores: Optional[np.ndarray],
                    labels: Optional[np.ndarray],
                    out_path: Path) -> None:
    if plt is None or scores is None or labels is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(scores[labels == 0], bins=60, alpha=0.65, label="background", density=True)
    ax.hist(scores[labels == 1], bins=60, alpha=0.65, label="foreground", density=True)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title("Score por protótipo leave-one-image-out")
    ax.set_xlabel("sim(feature, protótipo FG) - sim(feature, protótipo BG)")
    ax.set_ylabel("densidade")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_oracle_preview(labels_img: Optional[np.ndarray],
                        patch_gt: Optional[np.ndarray],
                        oracle_metrics: Dict[str, Any],
                        patch_grid: Tuple[int, int],
                        out_path: Path,
                        max_images: int = 8) -> None:
    if plt is None or labels_img is None or patch_gt is None:
        return
    subset = oracle_metrics.get("oracle_best_global_subset")
    if subset is None:
        return
    subset = set(int(x) for x in subset)
    H_p, W_p = patch_grid
    N = min(max_images, labels_img.shape[0])

    fig, axes = plt.subplots(N, 3, figsize=(8, 2.3 * N))
    if N == 1:
        axes = np.array([axes])

    for i in range(N):
        cluster_map = labels_img[i].reshape(H_p, W_p)
        pred = np.isin(cluster_map, list(subset))
        gt = patch_gt[i].reshape(H_p, W_p)

        axes[i, 0].imshow(cluster_map, interpolation="nearest")
        axes[i, 0].set_title(f"clusters img {i}")
        axes[i, 1].imshow(pred, interpolation="nearest", vmin=0, vmax=1)
        axes[i, 1].set_title("oracle subset")
        axes[i, 2].imshow(gt, interpolation="nearest", vmin=0, vmax=1)
        axes[i, 2].set_title("GT patch")
        for j in range(3):
            axes[i, j].axis("off")

    fig.suptitle("Oracle clustering em nível de patch", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Relatório
# -----------------------------------------------------------------------------


def build_assessments(metrics: Dict[str, Any], has_gt: bool) -> List[Assessment]:
    assessments: List[Assessment] = []

    assessments.append(assess_high_good(
        "cls_mean_similarity", metrics.get("cls_mean_similarity"),
        strong=0.70, moderate=0.50,
        strong_msg="Os CLS tokens estão coesos. A coleção parece semanticamente consistente.",
        moderate_msg="A coesão global é razoável, mas pode haver variação de pose, escala, fundo ou outliers.",
        weak_msg="A coesão global está baixa. Pode haver imagens fora do conceito comum ou forte domínio do fundo.",
        suggested_test="Inspecione cls_similarity_heatmap.png e cls_outlier_scores.png; remova temporariamente os maiores outliers e rode a Etapa 2/3 novamente."
    ))

    assessments.append(assess_low_good(
        "cls_outlier_score_max", metrics.get("cls_outlier_score_max"),
        good=0.30, moderate=0.50,
        good_msg="Não há outlier global muito severo segundo o CLS.",
        moderate_msg="Há pelo menos uma imagem relativamente discrepante. Pode atrapalhar clustering/correspondência.",
        bad_msg="Há imagem fortemente discrepante. Ela pode ser gargalo mesmo se a média da coleção parecer aceitável.",
        suggested_test="Veja as imagens no topo de cls_outlier_scores.png e teste o pipeline com e sem elas."
    ))

    assessments.append(assess_high_good(
        "local_nn_similarity_mean", metrics.get("local_nn_similarity_mean"),
        strong=0.65, moderate=0.45,
        strong_msg="Os patches encontram vizinhos locais relativamente parecidos em outras imagens.",
        moderate_msg="As correspondências locais existem, mas podem estar frágeis ou misturadas com fundo.",
        weak_msg="As correspondências locais estão fracas. A Etapa 2 tende a receber sinais ruidosos.",
        suggested_test="Compare com outros --img_size e modelos DINOv2; se CLS for bom e local_nn for ruim, teste camadas intermediárias ou média das últimas camadas."
    ))

    assessments.append(assess_high_good(
        "cycle_top1_rate_mean", metrics.get("cycle_top1_rate_mean"),
        strong=0.20, moderate=0.08,
        strong_msg="As correspondências patch-a-patch apresentam estabilidade razoável em ciclo.",
        moderate_msg="Há alguma estabilidade, mas as correspondências podem ser ambíguas.",
        weak_msg="A consistência cíclica está baixa. Muitos patches não têm correspondência recíproca clara.",
        suggested_test="Verifique cycle_top1_heatmap.png; aumente top_k na Etapa 2 ou teste features mais locais se poucos pares tiverem retorno recíproco claro."
    ))

    assessments.append(assess_high_good(
        "cycle_reconstruction_similarity_mean", metrics.get("cycle_reconstruction_similarity_mean"),
        strong=0.85, moderate=0.70,
        strong_msg="Mesmo quando o ciclo não volta ao mesmo patch exato, ele retorna para patches feature-wise semelhantes.",
        moderate_msg="O ciclo retorna para patches moderadamente semelhantes; pode haver ambiguidade semântica.",
        weak_msg="O ciclo retorna para patches pouco semelhantes. Isso sugere instabilidade das features locais.",
        suggested_test="Verifique cycle_reconstruction_heatmap.png para identificar os pares mais ambíguos; teste resolução maior se o objeto comum é pequeno."
    ))

    assessments.append(assess_high_good(
        "cluster_recurrence_mean", metrics.get("cluster_recurrence_mean"),
        strong=0.70, moderate=0.45,
        strong_msg="Os clusters locais aparecem de forma recorrente entre as imagens.",
        moderate_msg="Há recorrência parcial dos clusters; alguns podem estar associados ao objeto comum ou a fundos frequentes.",
        weak_msg="Os clusters locais parecem pouco recorrentes entre imagens. Pode faltar estrutura comum nas features.",
        suggested_test="Compare cluster_distribution.png com visualizações PCA; com GT, use oracle_cluster_iou para confirmar se algum cluster corresponde ao objeto."
    ))

    # Com GT.
    if has_gt:
        assessments.append(assess_high_good(
            "fg_bg_similarity_margin", metrics.get("fg_bg_similarity_margin"),
            strong=0.08, moderate=0.03,
            strong_msg="Patches de foreground são claramente mais parecidos entre si do que com background.",
            moderate_msg="Existe separação FG/BG, mas ela é modesta.",
            weak_msg="Foreground e background estão pouco separados no espaço de features.",
            suggested_test="Teste vitb14/vitl14, maior resolução, ou features de camadas intermediárias; veja se a margem melhora antes de rodar SAM."
        ))

        assessments.append(assess_high_good(
            "prototype_leave_one_image_auc", metrics.get("prototype_leave_one_image_auc"),
            strong=0.85, moderate=0.70,
            strong_msg="As features locais generalizam bem para separar FG/BG em imagens não vistas.",
            moderate_msg="As features têm informação útil, mas a separação ainda pode falhar em casos difíceis.",
            weak_msg="A separação FG/BG por protótipos é fraca. Isso aponta para gargalo na Etapa 1 ou no pré-processamento.",
            suggested_test="Compare esse AUC entre configurações da Etapa 1; escolha as melhores antes de rodar E2–E5."
        ))

        assessments.append(assess_high_good(
            "linear_probe_auc_mean", metrics.get("linear_probe_auc_mean"),
            strong=0.85, moderate=0.70,
            strong_msg="Um classificador linear separa bem FG/BG. As features contêm sinal útil para segmentação.",
            moderate_msg="O classificador linear separa parcialmente FG/BG. As etapas seguintes precisarão ser robustas.",
            weak_msg="Nem um classificador linear simples separa bem FG/BG. Forte suspeita de gargalo na Etapa 1.",
            suggested_test="Se o linear probe for baixo, priorize mudar features/preprocessamento antes de ajustar SAM."
        ))

        assessments.append(assess_high_good(
            "oracle_best_global_subset_iou", metrics.get("oracle_best_global_subset_iou"),
            strong=0.55, moderate=0.35,
            strong_msg="Existe um subconjunto de clusters que aproxima bem o objeto comum. A informação está presente nas features.",
            moderate_msg="O clustering contém algum sinal do objeto, mas ainda misturado com fundo ou partes do objeto.",
            weak_msg="Nem o melhor subconjunto global de clusters representa bem o objeto. Isso sugere limitação das features ou do número de clusters.",
            suggested_test="Se oracle for alto e E3 for baixo, o gargalo está na seleção do cluster/saliência; se oracle for baixo, volte à Etapa 1."
        ))

        assessments.append(assess_high_good(
            "oracle_imagewise_best_subset_iou_mean", metrics.get("oracle_imagewise_best_subset_iou_mean"),
            strong=0.65, moderate=0.45,
            strong_msg="Por imagem, há clusters capazes de aproximar bem o objeto. O problema pode ser tornar essa escolha consistente entre imagens.",
            moderate_msg="Algumas imagens têm clusters bons, outras não. Outliers ou variação intra-classe podem ser importantes.",
            weak_msg="Mesmo o oracle por imagem é fraco. A Etapa 1 provavelmente não está gerando features locais adequadas.",
            suggested_test="Veja oracle_cluster_preview.png; se algumas imagens falham, compare com cls_outlier_scores.png."
        ))

    return assessments


def infer_bottleneck(metrics: Dict[str, Any], has_gt: bool) -> str:
    cls = metrics.get("cls_mean_similarity")
    local = metrics.get("local_nn_similarity_mean")
    proto = metrics.get("prototype_leave_one_image_auc")
    oracle = metrics.get("oracle_best_global_subset_iou")

    cls_v = safe_float(cls)
    local_v = safe_float(local)
    proto_v = safe_float(proto)
    oracle_v = safe_float(oracle)

    if has_gt:
        if proto_v is not None and proto_v < 0.70 and oracle_v is not None and oracle_v < 0.35:
            return (
                "Diagnóstico provável: Etapa 1 é gargalo. As features locais não separam bem foreground/background "
                "e o oracle de clustering também é baixo. Priorize testar outro modelo, resolução, camada ou pré-processamento."
            )
        if proto_v is not None and proto_v >= 0.85 and oracle_v is not None and oracle_v >= 0.55:
            return (
                "Diagnóstico provável: Etapa 1 não é o principal gargalo. As features contêm sinal suficiente; "
                "se E3/E4/E5 forem ruins, investigue correspondência, clustering, seleção de cluster e prompts do SAM."
            )
        if oracle_v is not None and oracle_v >= 0.55:
            return (
                "Diagnóstico provável: as features têm informação útil, mas a extração/seleção não supervisionada pode estar falhando. "
                "Compare o oracle com a saída real da Etapa 3."
            )
        return (
            "Diagnóstico provável: resultado inconclusivo ou intermediário. Use este script para comparar ablações da Etapa 1 "
            "e depois rode E2–E5 apenas nas melhores configurações."
        )

    # Sem GT.
    if cls_v is not None and local_v is not None:
        if cls_v >= 0.70 and local_v < 0.45:
            return (
                "Diagnóstico sem GT: a coleção parece coesa globalmente, mas as correspondências locais são fracas. "
                "Suspeita de gargalo nas features locais; teste camadas intermediárias, resolução e pré-processamento."
            )
        if cls_v < 0.50 and local_v < 0.45:
            return (
                "Diagnóstico sem GT: tanto a coesão global quanto as correspondências locais estão fracas. "
                "Pode haver outliers, coleção heterogênea ou features inadequadas."
            )
        if cls_v >= 0.70 and local_v >= 0.65:
            return (
                "Diagnóstico sem GT: os sinais globais e locais são bons. Se o pipeline falhar depois, investigue E2/E3/E4 antes de trocar o backbone."
            )
    return (
        "Diagnóstico sem GT: use os heatmaps e outliers como triagem. Para concluir sobre gargalo, rode também com gt_index.json."
    )


def write_report(output_dir: Path,
                 features_path: Path,
                 metrics: Dict[str, Any],
                 assessments: List[Assessment],
                 diagnosis: str,
                 gt_info: Dict[str, Any],
                 padding_info: Dict[str, Any]) -> Path:
    report_path = output_dir / "feature_quality_report.md"

    lines: List[str] = []
    lines.append("# Relatório de qualidade das features — Etapa 1")
    lines.append("")
    lines.append(f"Arquivo analisado: `{features_path}`")
    lines.append("")
    lines.append("## Resumo")
    lines.append("")
    lines.append(f"- Imagens: **{metrics.get('n_images')}**")
    lines.append(f"- Patches por imagem: **{metrics.get('n_patches')}**")
    lines.append(f"- Dimensão da feature: **{metrics.get('feature_dim')}**")
    lines.append(f"- Grade: **{metrics.get('patch_grid')}**")
    lines.append(f"- GT encontrado: **{gt_info.get('matched', 0)} / {metrics.get('n_images')}**")
    if padding_info.get("available"):
        lines.append(
            f"- Padding (--resize keep_ar): **{padding_info['fraction_padding']:.1%} dos patches**, "
            f"mascarado nas métricas: **{'sim' if padding_info.get('masked') else 'não (--no_padding_mask)'}**"
        )
    else:
        lines.append("- Padding: sem `resize_info` no `.npz` (modo `square` da Etapa 1) — nada a mascarar")
    lines.append("")
    lines.append("## Diagnóstico")
    lines.append("")
    lines.append(diagnosis)
    lines.append("")
    lines.append("## Métricas interpretadas")
    lines.append("")
    lines.append("| Métrica | Valor | Veredito | Indicação | Teste sugerido |")
    lines.append("|---|---:|---|---|---|")
    for a in assessments:
        lines.append(
            f"| `{a.metric}` | {compact_float(a.value)} | {a.verdict} | "
            f"{a.indication} | {a.suggested_test} |"
        )
    lines.append("")
    lines.append("## Arquivos gerados")
    lines.append("")
    for name in [
        "feature_quality.json",
        "cls_similarity_heatmap.png",
        "local_similarity_heatmap.png",
        "cycle_top1_heatmap.png",
        "cycle_reconstruction_heatmap.png",
        "cls_outlier_scores.png",
        "cluster_distribution.png",
        "fg_bg_score_hist.png",
        "oracle_cluster_preview.png",
    ]:
        p = output_dir / name
        if p.exists():
            lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Observações")
    lines.append("")
    lines.append("- Os limiares usados no veredito são heurísticos e devem ser comparados entre coleções/configurações.")
    lines.append("- O melhor uso deste relatório é em ablação: modelo, resolução, camada, pré-processamento e normalização.")
    lines.append("- Se o oracle de clustering for alto, mas a Etapa 3 for ruim, o problema provavelmente está depois da Etapa 1.")
    lines.append("- Se o linear probe e o oracle forem baixos, a Etapa 1 é forte candidata a gargalo.")
    if padding_info.get("available") and padding_info.get("masked"):
        lines.append(
            "- Patches de padding (modo `--resize keep_ar` na Etapa 1) foram excluídos de todas as métricas "
            "não supervisionadas e supervisionadas: incluí-los infla artificialmente a correspondência local, "
            "a recorrência de clusters e a separação FG/BG (padding é um 'negativo fácil' quase constante). "
            "Use `--no_padding_mask` para reproduzir o comportamento antigo, se necessário para comparação."
        )
    elif padding_info.get("available") and not padding_info.get("masked"):
        lines.append(
            "- `--no_padding_mask` estava ativo: as métricas acima incluem patches de padding sem exclusão, "
            "o que tende a inflar artificialmente correspondência local, recorrência de cluster e separação FG/BG."
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def print_console_summary(metrics: Dict[str, Any], assessments: List[Assessment], diagnosis: str,
                          padding_info: Dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("  AUDITORIA DA ETAPA 1 — QUALIDADE DAS FEATURES")
    print("=" * 72)
    print(f"  Imagens        : {metrics.get('n_images')}")
    print(f"  Grade patches  : {metrics.get('patch_grid')} = {metrics.get('n_patches')} patches/img")
    print(f"  Dim feature    : {metrics.get('feature_dim')}")
    if padding_info.get("available"):
        estado = "mascarado" if padding_info.get("masked") else "NÃO mascarado (--no_padding_mask)"
        print(f"  Padding        : {padding_info['fraction_padding']:.1%} dos patches — {estado}")
    print("-" * 72)

    for a in assessments:
        print(f"\n[{a.verdict.upper()}] {a.metric}: {compact_float(a.value)}")
        print(f"  Indicação     : {a.indication}")
        print(f"  Teste sugerido: {a.suggested_test}")

    print("\n" + "-" * 72)
    print(diagnosis)
    print("=" * 72)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analisa a qualidade das features geradas pela Etapa 1 (features.npz).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features", required=True, type=Path,
                   help="Caminho para features.npz gerado pela Etapa 1.")
    p.add_argument("--gt_index", type=Path, default=None,
                   help="Caminho para gt_index.json. Opcional.")
    p.add_argument("--output", type=Path, default=None,
                   help="Diretório de saída. Default: <dir_features>/feature_quality.")
    p.add_argument("--n_clusters", type=int, default=5,
                   help="Número de clusters para diagnóstico/oracle local.")
    p.add_argument("--max_pair_images", type=int, default=50,
                   help="Máximo de imagens usadas nas métricas pairwise locais.")
    p.add_argument("--max_pair_patches", type=int, default=128,
                   help="Máximo de patches amostrados por imagem nas métricas pairwise locais.")
    p.add_argument("--max_cluster_patches", type=int, default=50000,
                   help="Máximo de patches usados para treinar MiniBatchKMeans.")
    p.add_argument("--max_fg_bg_samples", type=int, default=20000,
                   help="Máximo de patches FG/BG usados em métricas de similaridade com GT.")
    p.add_argument("--max_linear_probe_samples", type=int, default=15000,
                   help="Máximo de patches usados no linear probe com GT.")
    p.add_argument("--fg_threshold", type=float, default=0.5,
                   help="Limiar para converter máscara GT reduzida à grade de patches em foreground/background.")
    p.add_argument("--seed", type=int, default=42,
                   help="Semente aleatória.")
    p.add_argument("--no_plots", action="store_true",
                   help="Não gerar imagens de diagnóstico.")
    p.add_argument("--no_padding_mask", action="store_true",
                   help="Desativa o mascaramento de patches de padding (modo --resize keep_ar da Etapa 1) "
                        "em todas as métricas. Por padrão, o mascaramento é ativado sempre que resize_info "
                        "estiver disponível no features.npz (mesmo comportamento de step2/step3/step4).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features_path = args.features
    output_dir = args.output or (features_path.parent / "feature_quality")
    ensure_dir(output_dir)

    print(f"[1/7] Carregando features: {features_path}")
    feat = load_features(features_path)
    patch_features = feat["patch_features"]
    cls_features = feat["cls_features"]
    patch_grid = feat["patch_grid"]
    image_paths = feat["image_paths"]
    resize_info = feat.get("resize_info")

    N, P, D = patch_features.shape
    metrics: Dict[str, Any] = {
        "features_path": str(features_path),
        "n_images": int(N),
        "n_patches": int(P),
        "feature_dim": int(D),
        "patch_grid": list(patch_grid),
    }

    content_mask = None
    padding_info: Dict[str, Any] = {"available": False, "masked": False, "fraction_padding": 0.0}
    if not args.no_padding_mask:
        content_mask = compute_content_mask(resize_info, patch_grid)
    if resize_info is not None:
        frac_padding = 1.0 - float((content_mask if content_mask is not None
                                    else compute_content_mask(resize_info, patch_grid)).mean())
        padding_info = {
            "available": True,
            "masked": content_mask is not None,
            "fraction_padding": frac_padding,
        }
        if content_mask is not None and frac_padding > 1e-6:
            print(f"      resize_info encontrado em features.npz (modo keep_ar) — "
                  f"{frac_padding:.1%} dos patches são padding e serão excluídos das métricas.")
        elif content_mask is None:
            print(f"      resize_info encontrado, mas --no_padding_mask ativo — "
                  f"{frac_padding:.1%} dos patches de padding NÃO serão excluídos.")
    metrics["padding_info"] = padding_info

    print("[2/7] Calculando coesão global por CLS...")
    cls_metrics, cls_sim, cls_outliers = compute_cls_metrics(cls_features)
    metrics.update(cls_metrics)

    print("[3/7] Calculando correspondência local aproximada entre imagens...")
    local_metrics, local_sim, cycle_top1, cycle_reconstruction, local_img_idx = compute_local_pair_metrics(
        patch_features,
        content_mask=content_mask,
        max_images=args.max_pair_images,
        max_patches=args.max_pair_patches,
        seed=args.seed,
    )
    metrics.update(local_metrics)

    print("[4/7] Rodando clustering local não supervisionado...")
    cluster_metrics, labels_img, _ = compute_unsupervised_cluster_metrics(
        patch_features,
        content_mask=content_mask,
        n_clusters=args.n_clusters,
        max_cluster_patches=args.max_cluster_patches,
        seed=args.seed,
    )
    metrics.update(cluster_metrics)

    print("[5/7] Procurando ground truth...")
    gt_index = load_gt_index(args.gt_index)
    if resize_info is not None:
        print("      resize_info encontrado em features.npz (modo keep_ar) — "
              "GT será alinhado ao padding antes de reduzir para a grade de patches.")
    patch_gt, gt_info = load_patch_gt_masks(
        image_paths,
        patch_grid,
        gt_index,
        fg_threshold=args.fg_threshold,
        resize_info=resize_info,
    )
    metrics["gt_info"] = gt_info
    has_gt = patch_gt is not None and gt_info.get("matched", 0) > 0

    if has_gt:
        print(f"      GT encontrado para {gt_info['matched']} / {N} imagens.")
        print("[6/7] Calculando testes com GT: FG/BG, protótipos, linear probe e oracle clustering...")
        metrics.update(compute_fg_bg_similarity_metrics(
            patch_features,
            patch_gt,
            content_mask=content_mask,
            max_samples=args.max_fg_bg_samples,
            seed=args.seed,
        ))
        proto_metrics, proto_scores, proto_labels = compute_leave_one_image_prototype_auc(
            patch_features,
            patch_gt,
            content_mask=content_mask,
        )
        metrics.update(proto_metrics)
        metrics.update(compute_linear_probe_cv(
            patch_features,
            patch_gt,
            content_mask=content_mask,
            max_samples=args.max_linear_probe_samples,
            seed=args.seed,
        ))
        oracle_metrics = compute_oracle_cluster_iou(labels_img, patch_gt, args.n_clusters)
        metrics.update(oracle_metrics)
    else:
        print("      GT indisponível ou não casou com as imagens. Seguindo apenas com métricas sem GT.")
        proto_scores, proto_labels = None, None

    print("[7/7] Gerando relatório e visualizações...")
    assessments = build_assessments(metrics, has_gt=has_gt)
    diagnosis = infer_bottleneck(metrics, has_gt=has_gt)

    # Salva JSON antes dos plots.
    json_path = output_dir / "feature_quality.json"
    json_payload = {
        "metrics": metrics,
        "assessments": [assessment_to_dict(a) for a in assessments],
        "diagnosis": diagnosis,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_plots:
        plot_heatmap(cls_sim, output_dir / "cls_similarity_heatmap.png", "Similaridade coseno entre CLS tokens")
        plot_heatmap(
            local_sim,
            output_dir / "local_similarity_heatmap.png",
            "Similaridade local NN entre imagens",
        )
        plot_heatmap(
            cycle_top1,
            output_dir / "cycle_top1_heatmap.png",
            "Consistência cíclica Top-1 entre imagens",
        )
        plot_heatmap(
            cycle_reconstruction,
            output_dir / "cycle_reconstruction_heatmap.png",
            "Similaridade da reconstrução cíclica entre imagens",
        )
        plot_outliers(cls_outliers, image_paths, output_dir / "cls_outlier_scores.png")

        if "cluster_distribution_mean" in metrics and "cluster_recurrence_per_cluster" in metrics:
            plot_cluster_distribution(
                metrics["cluster_distribution_mean"],
                metrics["cluster_recurrence_per_cluster"],
                output_dir / "cluster_distribution.png",
            )
        if has_gt:
            plot_fg_bg_hist(proto_scores, proto_labels, output_dir / "fg_bg_score_hist.png")
            plot_oracle_preview(
                labels_img,
                patch_gt,
                metrics,
                patch_grid,
                output_dir / "oracle_cluster_preview.png",
                max_images=8,
            )

    report_path = write_report(
        output_dir,
        features_path,
        metrics,
        assessments,
        diagnosis,
        gt_info,
        padding_info,
    )

    print_console_summary(metrics, assessments, diagnosis, padding_info)
    print(f"\n[Salvo] JSON     : {json_path}")
    print(f"[Salvo] Relatório: {report_path}")
    print(f"[Salvo] Figuras  : {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
