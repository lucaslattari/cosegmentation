#!/usr/bin/env python3
"""
=============================================================================
diagnose_e2_ncut_optimal_threshold.py — Diagnóstico: threshold do NCut
minimizando o custo real do corte, comparado à mediana (produção) e ao
oracle de GT (Etapa 2 / Cross-Image Correspondence)
=============================================================================

Contexto:
  diagnose_e2_threshold_bias.py já estabeleceu, em 374 pares / 8 classes,
  uma correlação forte (Pearson pooled r=0.960, within-class médio
  r=0.809, robusta a teste de exclusão) entre o desbalanceamento real de
  um par (|frac_gt - 0.5|) e o custo de usar a mediana em vez do melhor
  threshold possível sobre o mesmo vetor de Fiedler (acc_oracle -
  acc_median). Esse resultado usa GT como régua diagnóstica, não como
  entrada de decisão — mas não propõe, por si só, nenhuma estratégia
  utilizável em produção (usar GT para escolher o threshold violaria o
  princípio de que GT só entra em avaliação, nunca em decisão).

  Este script testa a estratégia B discutida: em vez de usar a mediana
  como proxy de corte, buscar o ponto de corte do vetor de Fiedler que
  MINIMIZA o critério original de Normalized Cuts (Shi & Malik, 2000):

      Ncut(A,B) = cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V)

  para todos os P-1 pontos de corte possíveis na ordenação do vetor de
  Fiedler — sem nenhum uso de GT. Isso é o critério que o próprio NCut
  foi desenhado para otimizar; a mediana é sempre um proxy barato, não a
  solução do problema que o método propõe resolver.

  Para cada par, calcula-se três variantes de partição sobre o MESMO
  vetor de Fiedler (mesmo grafo, mesma orientação FG/BG de produção):
    - acc_median         : threshold por mediana (comportamento atual)
    - acc_ncut_optimal    : threshold que minimiza o custo Ncut(A,B) —
                            candidato a mudança de produção, SEM GT
    - acc_oracle          : melhor threshold possível vs. GT (teto,
                            diagnóstico apenas, nunca decisão)

  Também se reporta o CUSTO NCut (não a acurácia) do split por mediana
  vs. do split ótimo — essa comparação não depende de GT em nenhum
  momento e serve como evidência independente de que a mediana é
  subótima segundo o próprio critério do método, não só segundo o GT.

  `recovered_frac = (acc_ncut_optimal - acc_median) / (acc_oracle - acc_median)`
  mede, por par, que fração do gap identificado no diagnóstico anterior
  esta estratégia (sem GT) recupera. 0.0 = não recupera nada; 1.0 =
  recupera o teto inteiro.

Consistência com o restante do projeto:
  Importa build_pair_affinity, normalized_cut, orient_partition (para a
  baseline de produção inalterada), compute_cls_saliency,
  load_correspondence e _spatial_score_for_image_side de
  step2_correspondence.py — SEM alterar esse arquivo. A decisão de
  orientação (cls/spatial) é reimplementada localmente apenas para
  aceitar uma partição pré-computada em vez de recalculá-la via mediana
  internamente (orient_partition_cls/spatial não expõem esse parâmetro
  hoje); a lógica de scoring em si (médias de saliência, score espacial)
  não é duplicada, só reaproveitada com uma partição diferente de entrada.

  Importa load_gt_index, load_patch_gt_masks de e1_analyze_features.py
  (GT usado só como régua para acc_oracle e para medir acc_median/
  acc_ncut_optimal — nunca para decidir o corte).

Uso:
  python diagnose_e2_ncut_optimal_threshold.py \
      --features       features/.../Brown_Bear/features.npz \
      --correspondence features/.../Brown_Bear/correspondence.npz \
      --gt_index gt_index.json \
      --temperature 0.05 \
      --output diagnostics/e2_ncut_optimal_threshold

Saídas:
  e2_ncut_optimal_threshold.csv        — uma linha por par
  e2_ncut_optimal_threshold_report.md  — comparação agregada das 3 variantes
  scatter_recovered_frac_vs_imbalance.png
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None

# ---------------------------------------------------------------------------
# Reutiliza a lógica já validada e calibrada do pipeline em vez de duplicá-la.
# step2_correspondence.py NÃO é alterado.
# ---------------------------------------------------------------------------
try:
    from step2_correspondence import (
        load_step1,
        compute_content_mask,
        build_pair_affinity,
        normalized_cut,
        orient_partition,
        compute_cls_saliency,
        load_correspondence,
        _spatial_score_for_image_side,   # scoring espacial, independente de threshold
    )
except ImportError as e:
    print(
        "[ERRO] Não foi possível importar step2_correspondence.py.\n"
        "       Este script precisa estar no MESMO diretório desse arquivo\n"
        "       (ou step2_correspondence.py precisa estar no PYTHONPATH).\n"
        f"       Erro original: {e}"
    )
    sys.exit(1)

try:
    from e1_analyze_features import load_gt_index, load_patch_gt_masks
except ImportError as e:
    print(
        "[ERRO] Não foi possível importar e1_analyze_features.py.\n"
        "       Mesma exigência de diretório/PYTHONPATH do import acima.\n"
        f"       Erro original: {e}"
    )
    sys.exit(1)


# -----------------------------------------------------------------------------
# Threshold que minimiza o custo real do NCut (SEM GT) — O(P^2) por par,
# mesma ordem de grandeza de custo que já se paga para construir W.
# -----------------------------------------------------------------------------

def compute_ncut_cost_curve(
    W: np.ndarray, fiedler: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Busca, entre todos os P-1 pontos de corte da ordenação do vetor de
    Fiedler, o que minimiza Ncut(A,B) = cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V),
    calculado sobre a MESMA matriz de afinidade W usada para construir o
    Laplaciano (já com degree-reweighting aplicado, se degree_alpha > 0).

    Retorna a "curva de custo" completa (um valor de Ncut por ponto de corte
    possível), para que o CALL desta função (cara: O(P^2)) seja feito uma
    única vez por par, mesmo quando depois se quer testar múltiplas faixas
    de busca (irrestrita e restrita) sobre o MESMO grafo/Fiedler vector.
    """
    n = len(fiedler)
    order = np.argsort(fiedler)
    W_perm = W[order][:, order]

    d = W.sum(axis=1)          # grau de cada nó no MESMO grafo usado no NCut
    d_perm = d[order]

    prefix_d = np.concatenate([[0.0], np.cumsum(d_perm)])
    total_d = float(prefix_d[-1])

    # Soma acumulada 2D (integral image) do bloco W_perm[0:k, 0:k]
    S = np.cumsum(np.cumsum(W_perm, axis=0), axis=1)
    internal = np.concatenate([[0.0], np.diag(S)])   # internal[k] = sum W_perm[:k,:k]

    ks = np.arange(1, n)  # exclui k=0 e k=n (partições triviais/vazias)
    assocA = prefix_d[ks]
    assocB = total_d - assocA
    cutAB = assocA - internal[ks]

    with np.errstate(divide="ignore", invalid="ignore"):
        ncut_vals = cutAB / assocA + cutAB / assocB
    ncut_vals = np.where(np.isfinite(ncut_vals), ncut_vals, np.inf)

    # Custo Ncut do split por MEDIANA, para comparação — sem GT.
    k_median = int(np.sum(fiedler <= np.median(fiedler)))
    if 1 <= k_median <= n - 1:
        ncut_cost_median = float(ncut_vals[k_median - 1])
    else:
        ncut_cost_median = np.nan

    return order, ks, ncut_vals, ncut_cost_median


def select_ncut_partition(
    order: np.ndarray, ks: np.ndarray, ncut_vals: np.ndarray, n: int,
    fiedler: np.ndarray, k_min: Optional[int] = None, k_max: Optional[int] = None,
) -> Tuple[np.ndarray, float, int]:
    """
    Seleciona o k* que minimiza ncut_vals dentro de uma faixa opcional
    [k_min, k_max] (em nº de patches do lado "baixo" na ordenação do
    Fiedler). Sem faixa (default), é a busca irrestrita original.

    k_min/k_max permitem restringir a busca a proporções "razoáveis"
    (ex.: k_min=0.1*n, k_max=0.9*n) para evitar cortes degenerados
    (um lado quase vazio) que minimizam Ncut(A,B) por um motivo
    estrutural do grafo sem corresponder à fronteira FG/BG real —
    padrão observado empiricamente em diagnose_e2_ncut_optimal_threshold
    (Kite-Colt_Park, Stonehenge-Salisbury).

    Retorna: side_pos (P,) booleano em ordem ORIGINAL, custo do corte
    escolhido, e k_star (tamanho do lado "baixo" escolhido, para
    diagnóstico de quão desproporcional o corte ficou).
    """
    lo = 1 if k_min is None else max(1, int(k_min))
    hi = (n - 1) if k_max is None else min(n - 1, int(k_max))

    mask = (ks >= lo) & (ks <= hi)
    if not mask.any() or not np.isfinite(ncut_vals[mask]).any():
        # Faixa vazia ou grafo degenerado dentro da faixa — cai para a
        # mediana como fallback seguro, sem nunca deixar o par sem partição.
        return _partition_from_fiedler_local(fiedler), np.nan, int(np.sum(fiedler <= np.median(fiedler)))

    vals_in_range = np.where(mask, ncut_vals, np.inf)
    best_pos = int(np.argmin(vals_in_range))
    k_star = int(ks[best_pos])
    cost = float(ncut_vals[best_pos])

    side_pos_sorted = np.zeros(n, dtype=bool)
    side_pos_sorted[:k_star] = True
    side_pos = np.zeros(n, dtype=bool)
    side_pos[order] = side_pos_sorted

    return side_pos, cost, k_star


def _partition_from_fiedler_local(fiedler: np.ndarray) -> np.ndarray:
    """Fallback idêntico a _partition_from_fiedler de step2_correspondence.py,
    usado só no caso degenerado tratado acima (não duplica lógica de produção
    em uso normal — normalized_cut/build_pair_affinity continuam sendo os
    importados)."""
    threshold = np.median(fiedler)
    return fiedler > threshold


# -----------------------------------------------------------------------------
# Orientação FG/BG a partir de uma partição JÁ CALCULADA (não recomputa via
# mediana) — reaproveita o scoring espacial de produção, duplica só a
# comparação de médias/soma que orient_partition_cls/spatial fazem inline.
# -----------------------------------------------------------------------------

def orient_given_side_cls(side_pos: np.ndarray, saliency_i: np.ndarray,
                          saliency_j: np.ndarray) -> Tuple[np.ndarray, float, float]:
    saliency_all = np.concatenate([saliency_i, saliency_j])
    sal_pos = saliency_all[side_pos].mean() if side_pos.any() else 0.0
    sal_neg = saliency_all[~side_pos].mean() if (~side_pos).any() else 0.0
    margin = abs(sal_pos - sal_neg)
    margin_rel = margin / (abs(sal_pos) + abs(sal_neg) + 1e-9)
    mask = side_pos if sal_pos >= sal_neg else ~side_pos
    return mask, float(margin), float(margin_rel)


def orient_given_side_spatial(side_pos: np.ndarray, n_patches_i: int,
                              idx_i: np.ndarray, idx_j: np.ndarray,
                              patch_grid: tuple) -> Tuple[np.ndarray, float, float]:
    pos_i = side_pos[:n_patches_i]
    pos_j = side_pos[n_patches_i:]
    score_pos_i = _spatial_score_for_image_side(pos_i, idx_i, patch_grid)
    score_pos_j = _spatial_score_for_image_side(pos_j, idx_j, patch_grid)
    score_neg_i = _spatial_score_for_image_side(~pos_i, idx_i, patch_grid)
    score_neg_j = _spatial_score_for_image_side(~pos_j, idx_j, patch_grid)
    score_pos = 0.5 * (score_pos_i + score_pos_j)
    score_neg = 0.5 * (score_neg_i + score_neg_j)
    margin = abs(score_pos - score_neg)
    margin_rel = margin / (abs(score_pos) + abs(score_neg) + 1e-9)
    mask = side_pos if score_pos >= score_neg else ~side_pos
    return mask, float(margin), float(margin_rel)


def orient_given_side(side_pos: np.ndarray, n_patches_i: int,
                      saliency_i: np.ndarray, saliency_j: np.ndarray, *,
                      method: str, idx_i: np.ndarray, idx_j: np.ndarray,
                      patch_grid: tuple) -> Tuple[np.ndarray, float, float]:
    """Retorna (máscara orientada, margem absoluta, margem relativa). A margem
    mede o quão mais forte é a evidência (saliência CLS ou score espacial) do
    lado escolhido como FG em relação ao lado descartado — margem baixa
    sinaliza um corte "ambíguo" (nenhum dos dois lados parece claramente mais
    objeto que o outro), candidato a NÃO confiar no corte do NCut e cair para
    a mediana. Nunca usa GT — só a mesma evidência que a orientação de
    produção já calcula."""
    method = str(method).lower()
    if method == "cls":
        return orient_given_side_cls(side_pos, saliency_i, saliency_j)
    if method == "spatial":
        return orient_given_side_spatial(side_pos, n_patches_i, idx_i, idx_j, patch_grid)
    raise ValueError(f"Método de orientação desconhecido: {method!r}. Use 'cls' ou 'spatial'.")


# -----------------------------------------------------------------------------
# Oracle de GT (mesmo critério de diagnose_e2_threshold_bias.py) — teto
# diagnóstico, nunca usado para decisão.
# -----------------------------------------------------------------------------

def oracle_threshold_accuracy(fiedler: np.ndarray, gt: np.ndarray) -> float:
    m = len(fiedler)
    if m == 0:
        return 0.0
    order = np.argsort(fiedler)
    gt_sorted = gt[order].astype(np.int64)
    prefix = np.concatenate([[0], np.cumsum(gt_sorted)])
    total_gt = int(prefix[-1])
    ks = np.arange(0, m + 1)
    correct_low_fg = 2 * prefix - ks + m - total_gt
    correct_low_bg = ks - 2 * prefix + total_gt
    best_correct = max(int(correct_low_fg.max()), int(correct_low_bg.max()))
    return best_correct / m


# -----------------------------------------------------------------------------
# Diagnóstico por par
# -----------------------------------------------------------------------------

def diagnose_pair(
    i: int, j: int, sim_score: float,
    patch_features: np.ndarray,
    cls_saliency: np.ndarray,
    patch_gt: np.ndarray,
    content_mask: "np.ndarray | None",
    patch_grid: tuple,
    temperature: float,
    degree_alpha: float,
    orientation: str,
    min_frac: float,
    max_frac: float,
) -> Optional[Dict[str, Any]]:
    P = patch_features.shape[1]

    if content_mask is not None:
        idx_i = np.flatnonzero(content_mask[i])
        idx_j = np.flatnonzero(content_mask[j])
        if idx_i.size == 0 or idx_j.size == 0:
            idx_i = np.arange(P)
            idx_j = np.arange(P)
    else:
        idx_i = np.arange(P)
        idx_j = np.arange(P)

    feat_i_c = patch_features[i][idx_i]
    feat_j_c = patch_features[j][idx_j]
    sal_i_c  = cls_saliency[i][idx_i]
    sal_j_c  = cls_saliency[j][idx_j]
    Pi_c     = idx_i.size
    Pj_c     = idx_j.size
    if Pi_c == 0 or Pj_c == 0:
        return None

    n = Pi_c + Pj_c

    # --- Recompute idêntico ao passo 2-3 de step2_correspondence.py ---
    W = build_pair_affinity(feat_i_c, feat_j_c, temperature=temperature, degree_alpha=degree_alpha)
    eigvecs = normalized_cut(W, n_cuts=1)
    fiedler = eigvecs[:, 1]

    gt_i_c = patch_gt[i][idx_i].astype(bool)
    gt_j_c = patch_gt[j][idx_j].astype(bool)
    gt_combined = np.concatenate([gt_i_c, gt_j_c])
    if gt_combined.sum() == 0 or gt_combined.sum() == gt_combined.size:
        return None

    # --- (a) Produção: mediana + orientação real, inalterado ---
    fg_mask_median = orient_partition(
        fiedler, Pi_c, sal_i_c, sal_j_c,
        method=orientation, idx_i=idx_i, idx_j=idx_j, patch_grid=patch_grid,
    )
    acc_median = float((fg_mask_median == gt_combined).mean())

    # Margem de confiança da mediana: recomputa a MESMA side_pos que
    # orient_partition usou internamente (via _partition_from_fiedler_local,
    # idêntica à lógica de produção), só para extrair a margem de
    # score — orient_partition (produção) não expõe esse valor.
    side_pos_median = _partition_from_fiedler_local(fiedler)
    _, margin_median, margin_median_rel = orient_given_side(
        side_pos_median, Pi_c, sal_i_c, sal_j_c,
        method=orientation, idx_i=idx_i, idx_j=idx_j, patch_grid=patch_grid,
    )

    # --- Curva de custo Ncut computada UMA vez, reaproveitada nas duas buscas ---
    order, ks, ncut_vals, ncut_cost_median = compute_ncut_cost_curve(W, fiedler)

    # --- (b) Candidato 1: threshold ótimo IRRESTRITO, SEM GT ---
    side_pos_opt, ncut_cost_opt, k_star_opt = select_ncut_partition(
        order, ks, ncut_vals, n, fiedler, k_min=None, k_max=None,
    )
    fg_mask_opt, margin_unconstrained, margin_unconstrained_rel = orient_given_side(
        side_pos_opt, Pi_c, sal_i_c, sal_j_c,
        method=orientation, idx_i=idx_i, idx_j=idx_j, patch_grid=patch_grid,
    )
    acc_ncut_optimal = float((fg_mask_opt == gt_combined).mean())
    predicted_fg_frac_unconstrained = float(fg_mask_opt.mean())

    # --- (c) Candidato 2: threshold ótimo RESTRITO a [min_frac, max_frac] de n,
    #         SEM GT — evita cortes degenerados (um lado quase vazio) que
    #         minimizam custo Ncut por motivo estrutural do grafo, sem
    #         corresponder à fronteira FG/BG real (Kite-Colt_Park,
    #         Stonehenge-Salisbury na rodada anterior).
    k_min = int(np.ceil(min_frac * n))
    k_max = int(np.floor(max_frac * n))
    side_pos_constrained, ncut_cost_constrained, k_star_constrained = select_ncut_partition(
        order, ks, ncut_vals, n, fiedler, k_min=k_min, k_max=k_max,
    )
    fg_mask_constrained, margin_constrained, margin_constrained_rel = orient_given_side(
        side_pos_constrained, Pi_c, sal_i_c, sal_j_c,
        method=orientation, idx_i=idx_i, idx_j=idx_j, patch_grid=patch_grid,
    )
    acc_ncut_constrained = float((fg_mask_constrained == gt_combined).mean())
    predicted_fg_frac_constrained = float(fg_mask_constrained.mean())

    # --- (d) Oracle: teto diagnóstico, GT só como régua ---
    acc_oracle = oracle_threshold_accuracy(fiedler, gt_combined)

    frac_gt_pair = float(gt_combined.mean())
    gap_median = acc_oracle - acc_median
    gap_ncut_optimal = acc_oracle - acc_ncut_optimal
    gap_ncut_constrained = acc_oracle - acc_ncut_constrained
    recovered_frac = (acc_ncut_optimal - acc_median) / gap_median if gap_median > 1e-9 else np.nan
    recovered_frac_constrained = (acc_ncut_constrained - acc_median) / gap_median if gap_median > 1e-9 else np.nan

    return {
        "class_i": i, "class_j": j,
        "cls_sim": float(sim_score),
        "n_patches_pair": n,
        "frac_gt_pair": frac_gt_pair,
        "imbalance": abs(frac_gt_pair - 0.5),
        "acc_median": acc_median,
        "acc_ncut_optimal": acc_ncut_optimal,
        "acc_ncut_constrained": acc_ncut_constrained,
        "acc_oracle": acc_oracle,
        "gap_median": gap_median,
        "gap_ncut_optimal": gap_ncut_optimal,
        "gap_ncut_constrained": gap_ncut_constrained,
        "recovered_frac": recovered_frac,
        "recovered_frac_constrained": recovered_frac_constrained,
        "ncut_cost_median": ncut_cost_median,
        "ncut_cost_optimal": ncut_cost_opt,
        "ncut_cost_constrained": ncut_cost_constrained,
        # Diagnóstico direto da hipótese de corte degenerado: quanto mais longe
        # de frac_gt_pair, mais o corte "ótimo" divergiu da fronteira real.
        "predicted_fg_frac_unconstrained": predicted_fg_frac_unconstrained,
        "predicted_fg_frac_constrained": predicted_fg_frac_constrained,
        "k_star_unconstrained": k_star_opt,
        "k_star_constrained": k_star_constrained,
        # Margem de confiança da orientação (evidência de saliência/espacial a
        # favor do lado escolhido como FG) — sinal GT-free candidato a decidir
        # quando confiar no corte do NCut em vez de cair para a mediana.
        "margin_median": margin_median,
        "margin_median_rel": margin_median_rel,
        "margin_unconstrained": margin_unconstrained,
        "margin_unconstrained_rel": margin_unconstrained_rel,
        "margin_constrained": margin_constrained,
        "margin_constrained_rel": margin_constrained_rel,
    }


# -----------------------------------------------------------------------------
# Pipeline por classe
# -----------------------------------------------------------------------------

def process_class(
    features_path: Path,
    correspondence_path: Path,
    gt_index: Dict[str, Any],
    temperature: float,
    fg_threshold: float,
    limit_pairs: Optional[int],
    min_frac: float,
    max_frac: float,
) -> List[Dict[str, Any]]:
    class_name = features_path.parent.name
    print(f"\n[Classe] {class_name}")

    feat = load_step1(features_path)
    patch_features = feat["patch_features"]
    cls_features   = feat["cls_features"]
    patch_grid     = feat["patch_grid"]
    image_paths    = feat["image_paths"]
    resize_info    = feat.get("resize_info")

    content_mask = compute_content_mask(resize_info, patch_grid)

    corr = load_correspondence(correspondence_path)
    top_pairs    = [(int(r[0]), int(r[1]), float(r[2])) for r in corr["top_pairs"]]
    orientation  = corr["orientation"]
    degree_alpha = corr["degree_alpha"]
    print(f"  {len(top_pairs)} pares  |  orientation={orientation}  degree_alpha={degree_alpha}  "
          f"|  busca restrita: [{min_frac:.0%}, {max_frac:.0%}] do grafo  "
          f"(temperature usada no recompute: {temperature} — confira contra a que gerou o correspondence.npz)")

    patch_gt, gt_info = load_patch_gt_masks(
        image_paths, patch_grid, gt_index,
        fg_threshold=fg_threshold, resize_info=resize_info,
    )
    if patch_gt is None or gt_info.get("matched", 0) == 0:
        print(f"  [Aviso] Sem GT casado para {class_name} — pulando classe.")
        return []
    print(f"  GT casado: {gt_info['matched']}/{len(image_paths)} imagens.")

    cls_saliency = compute_cls_saliency(patch_features, cls_features)
    pairs_to_run = top_pairs[:limit_pairs] if limit_pairs else top_pairs

    rows = []
    for rank, (i, j, sim) in enumerate(pairs_to_run):
        result = diagnose_pair(
            i, j, sim, patch_features, cls_saliency, patch_gt,
            content_mask, patch_grid, temperature, degree_alpha, orientation,
            min_frac, max_frac,
        )
        if result is None:
            continue
        result["class"] = class_name
        result["pair_rank"] = rank
        rows.append(result)
        rec_str = f"{result['recovered_frac']:.2f}" if not np.isnan(result["recovered_frac"]) else "N/A"
        rec_c_str = f"{result['recovered_frac_constrained']:.2f}" if not np.isnan(result["recovered_frac_constrained"]) else "N/A"
        print(f"  [{rank+1:>2}/{len(pairs_to_run)}] par ({i},{j})  "
              f"frac_gt={result['frac_gt_pair']:.3f}  "
              f"acc_median={result['acc_median']:.3f}  "
              f"acc_ncut_opt={result['acc_ncut_optimal']:.3f} (margin={result['margin_unconstrained_rel']:.2f}, rec={rec_str})  "
              f"acc_ncut_constr={result['acc_ncut_constrained']:.3f} (rec={rec_c_str})  "
              f"acc_oracle={result['acc_oracle']:.3f}")

    return rows


# -----------------------------------------------------------------------------
# Correlação, plots, relatório
# -----------------------------------------------------------------------------

def correlate(x: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    if len(x) < 3 or pearsonr is None:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


def scatter_recovered_vs_imbalance(rows: List[Dict[str, Any]], out_path: Path,
                                   min_gap_for_recovered_frac: float = 0.02):
    if plt is None:
        return
    # Mesmo filtro do relatório: pares com gap_median~0 produzem razões
    # explosivas que dominariam a escala do eixo Y sem acrescentar informação.
    valid = [r for r in rows if not np.isnan(r["recovered_frac"])
            and r["gap_median"] >= min_gap_for_recovered_frac]
    if not valid:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    classes = sorted(set(r["class"] for r in valid))
    cmap = plt.get_cmap("tab10")

    for ax, rec_col, title in [
        (axes[0], "recovered_frac", "Busca IRRESTRITA"),
        (axes[1], "recovered_frac_constrained", "Busca RESTRITA"),
    ]:
        for ci, c in enumerate(classes):
            rc = [r for r in valid if r["class"] == c and not np.isnan(r[rec_col])]
            x = [r["imbalance"] for r in rc]
            y = [r[rec_col] for r in rc]
            ax.scatter(x, y, label=c, alpha=0.7, color=cmap(ci % 10), s=28)
        ax.axhline(1.0, color="green", lw=1, ls="--", alpha=0.6)
        ax.axhline(0.0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("|frac_gt_pair - 0.5|")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("recovered_frac")
    axes[1].legend(fontsize=6, loc="best", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Fração do gap recuperada: irrestrito vs. restrito", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _variant_summary(rows: List[Dict[str, Any]], acc_col: str, gap_col: str, rec_col: str,
                     min_gap: float) -> Dict[str, Any]:
    """Resumo robusto (agregado + filtrado) para UMA variante (irrestrita ou restrita),
    evitando duplicar a lógica de filtro/agregação para cada uma."""
    mean_acc_median = np.mean([r["acc_median"] for r in rows])
    mean_acc_variant = np.mean([r[acc_col] for r in rows])
    mean_acc_oracle = np.mean([r["acc_oracle"] for r in rows])
    mean_gap_median = mean_acc_oracle - mean_acc_median
    aggregate = (mean_acc_variant - mean_acc_median) / mean_gap_median if mean_gap_median > 1e-9 else float("nan")

    filtered = [r for r in rows if r["gap_median"] >= min_gap]
    valid_rec = [r[rec_col] for r in filtered if not np.isnan(r[rec_col])]

    # Pares "catastróficos": a variante piora a acurácia em relação à mediana
    # por uma margem que não é ruído (> 0.1, mesmo limiar usado na investigação manual).
    n_catastrophic = sum(1 for r in rows if (r["acc_median"] - r[acc_col]) > 0.1)

    return {
        "mean_acc_variant": mean_acc_variant,
        "aggregate_recovered_frac": aggregate,
        "filtered_mean": float(np.mean(valid_rec)) if valid_rec else float("nan"),
        "filtered_median": float(np.median(valid_rec)) if valid_rec else float("nan"),
        "n_filtered": len(valid_rec),
        "n_catastrophic": n_catastrophic,
        "pct_catastrophic": 100.0 * n_catastrophic / len(rows) if rows else 0.0,
    }


def write_report(rows: List[Dict[str, Any]], output_dir: Path,
                 min_gap_for_recovered_frac: float = 0.02,
                 min_frac: float = 0.10, max_frac: float = 0.90) -> Path:
    classes = sorted(set(r["class"] for r in rows))

    summ_opt = _variant_summary(rows, "acc_ncut_optimal", "gap_ncut_optimal", "recovered_frac",
                                min_gap_for_recovered_frac)
    summ_con = _variant_summary(rows, "acc_ncut_constrained", "gap_ncut_constrained", "recovered_frac_constrained",
                                min_gap_for_recovered_frac)
    mean_acc_median = np.mean([r["acc_median"] for r in rows])
    mean_acc_oracle = np.mean([r["acc_oracle"] for r in rows])

    lines = []
    lines.append("# Diagnóstico: threshold ótimo do NCut — irrestrito vs. restrito vs. mediana vs. oracle (Etapa 2)\n")
    lines.append(f"Classes analisadas ({len(classes)}): " + ", ".join(classes) + "\n")
    lines.append(f"Total de pares válidos: {len(rows)}\n")
    lines.append(f"Busca restrita: cortes considerados só entre {min_frac:.0%} e {max_frac:.0%} do grafo "
                 f"(evita cortes degenerados — um lado quase vazio).\n")

    lines.append("\n## Resumo agregado (médias pooled) — as 3 variantes\n")
    lines.append("| Métrica | Mediana (produção) | NCut ótimo IRRESTRITO | NCut ótimo RESTRITO | Oracle (teto) |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| acurácia média vs. GT | {mean_acc_median:.3f} | "
        f"{summ_opt['mean_acc_variant']:.3f} | {summ_con['mean_acc_variant']:.3f} | {mean_acc_oracle:.3f} |"
    )
    lines.append(
        f"\n**recovered_frac AGREGADO — irrestrito: {summ_opt['aggregate_recovered_frac']:.3f}  |  "
        f"restrito: {summ_con['aggregate_recovered_frac']:.3f}**\n"
    )
    lines.append(
        f"\nPares CATASTRÓFICOS (queda de acurácia > 0.10 vs. mediana) — "
        f"irrestrito: {summ_opt['n_catastrophic']} ({summ_opt['pct_catastrophic']:.1f}%)  |  "
        f"restrito: {summ_con['n_catastrophic']} ({summ_con['pct_catastrophic']:.1f}%)\n"
    )
    lines.append(
        f"\nrecovered_frac médio por-par filtrado (gap_median≥{min_gap_for_recovered_frac}) — "
        f"irrestrito: {summ_opt['filtered_mean']:.3f} (mediana {summ_opt['filtered_median']:.3f}, N={summ_opt['n_filtered']})  |  "
        f"restrito: {summ_con['filtered_mean']:.3f} (mediana {summ_con['filtered_median']:.3f}, N={summ_con['n_filtered']})\n"
    )

    ncut_costs = [(r["ncut_cost_median"], r["ncut_cost_optimal"]) for r in rows
                 if not np.isnan(r["ncut_cost_median"]) and not np.isnan(r["ncut_cost_optimal"])]
    if ncut_costs:
        med_costs = np.array([c[0] for c in ncut_costs])
        opt_costs = np.array([c[1] for c in ncut_costs])
        pct_higher = 100.0 * np.mean((med_costs - opt_costs) / np.maximum(opt_costs, 1e-9))
        lines.append(
            f"\n## Custo Ncut(A,B) — comparação SEM GT (evidência independente, só irrestrito)\n\n"
            f"Custo médio (mediana): {med_costs.mean():.4f}  |  "
            f"Custo médio (ótimo irrestrito): {opt_costs.mean():.4f}  |  "
            f"mediana é, em média, **{pct_higher:.1f}% mais cara** que o ótimo irrestrito "
            f"(nenhum uso de GT nesta comparação).\n"
        )

    # --- Margem de confiança como preditor GT-free de quando confiar no NCut ---
    margin = np.array([r["margin_unconstrained_rel"] for r in rows])
    acc_diff = np.array([r["acc_ncut_optimal"] - r["acc_median"] for r in rows])  # >0: NCut ajuda; <0: atrapalha
    corr_margin = correlate(margin, acc_diff)

    lines.append(
        f"\n## Margem de confiança da orientação como preditor de quando confiar no NCut\n\n"
        f"`margin_unconstrained_rel` mede o quão mais forte é a evidência (saliência/score "
        f"espacial) do lado escolhido como FG em relação ao lado descartado, no corte "
        f"IRRESTRITO — sem nenhum uso de GT. Testamos se essa margem prediz "
        f"`acc_ncut_optimal - acc_median` (se o NCut ajuda [+] ou atrapalha [-] naquele par).\n\n"
        f"Correlação pooled: Pearson r={corr_margin['pearson_r']:.3f} (p={corr_margin['pearson_p']:.4f})  "
        f"Spearman ρ={corr_margin['spearman_r']:.3f}\n\n"
        f"- r positivo e significativo: margem alta → NCut mais confiável — a margem é um "
        f"sinal GT-free utilizável para decidir, POR PAR, quando usar o corte do NCut e quando "
        f"cair para a mediana (regra global, sem depender de classe).\n"
        f"- r fraco/não-significativo: a margem sozinha não basta como preditor — precisaria "
        f"de outro sinal, ou uma combinação de sinais.\n"
    )

    # --- Simulação do critério híbrido: usa NCut irrestrito só quando margin >= limiar,
    #     senão cai para a mediana. Varre percentis da margem (regra GLOBAL, não por classe).
    lines.append("\n### Simulação: estratégia híbrida (NCut se margem ≥ limiar, senão mediana)\n\n")
    lines.append("| Limiar (percentil da margem) | valor do limiar | acc média (híbrido) | "
                 "catastróficos (híbrido) | pior classe (acc média) |")
    lines.append("|---|---|---|---|---|")
    classes_list = sorted(set(r["class"] for r in rows))
    for pct in [0, 10, 25, 50, 75, 90]:
        thresh = float(np.percentile(margin, pct)) if pct > 0 else float(margin.min()) - 1e-9
        hybrid_acc = np.array([
            r["acc_ncut_optimal"] if r["margin_unconstrained_rel"] >= thresh else r["acc_median"]
            for r in rows
        ])
        n_catastrophic_hybrid = sum(
            1 for r in rows
            if (r["acc_median"] - (r["acc_ncut_optimal"] if r["margin_unconstrained_rel"] >= thresh else r["acc_median"])) > 0.1
        )
        per_class_mean = {
            c: np.mean([
                r["acc_ncut_optimal"] if r["margin_unconstrained_rel"] >= thresh else r["acc_median"]
                for r in rows if r["class"] == c
            ])
            for c in classes_list
        }
        worst_class = min(per_class_mean, key=per_class_mean.get)
        lines.append(
            f"| p{pct} | {thresh:.3f} | {hybrid_acc.mean():.3f} | {n_catastrophic_hybrid} | "
            f"{worst_class[:35]} ({per_class_mean[worst_class]:.3f}) |"
        )
    lines.append(
        "\nCompare esta tabela com a linha 'Mediana' (baseline) e 'NCut ótimo IRRESTRITO' "
        "(sem filtro de margem, equivalente ao limiar p0) do resumo agregado no topo. Um "
        "limiar intermediário que mantenha acurácia média alta E reduza pares catastróficos "
        "E não piore a pior classe é candidato a critério de decisão para produção — sempre "
        "escolhido pelo critério dual mean+minimax, nunca por classe individual.\n"
    )

    # --- Teste direto da hipótese de corte degenerado ---
    x_frac_gt = np.array([r["frac_gt_pair"] for r in rows])
    x_pred_unc = np.array([r["predicted_fg_frac_unconstrained"] for r in rows])
    x_pred_con = np.array([r["predicted_fg_frac_constrained"] for r in rows])
    x_pred_med = np.full(len(rows), 0.5)  # mediana força ~50/50 por construção

    err_unc = np.abs(x_pred_unc - x_frac_gt)
    err_con = np.abs(x_pred_con - x_frac_gt)
    err_med = np.abs(x_pred_med - x_frac_gt)

    lines.append(
        f"\n## Teste da hipótese de corte degenerado: |fração prevista de FG - frac_gt_pair real|\n\n"
        f"Quanto MENOR, mais a proporção do corte escolhido bate com a proporção real de FG do par "
        f"(sem que a estratégia tenha visto o GT — é só diagnóstico).\n\n"
        f"| Estratégia | erro médio |erro mediano | erro máximo |\n"
        f"|---|---|---|---|\n"
        f"| Mediana (força ~50%) | {err_med.mean():.3f} | {np.median(err_med):.3f} | {err_med.max():.3f} |\n"
        f"| NCut ótimo IRRESTRITO | {err_unc.mean():.3f} | {np.median(err_unc):.3f} | {err_unc.max():.3f} |\n"
        f"| NCut ótimo RESTRITO | {err_con.mean():.3f} | {np.median(err_con):.3f} | {err_con.max():.3f} |\n\n"
        f"Se o erro do irrestrito for bem maior que o do restrito, confirma que a busca sem "
        f"limite estava escolhendo cortes desproporcionais (degenerados) numa fração relevante "
        f"dos pares, e que restringir a faixa ajuda a evitar isso.\n"
    )

    lines.append(f"\n## Por classe (recovered_frac considera pares com gap_median ≥ {min_gap_for_recovered_frac})\n")
    lines.append("| Classe | N pares | acc_median | acc_ncut_opt (irrestr.) | acc_ncut (restr.) | acc_oracle | "
                 "rec. agregado (irrestr.) | rec. agregado (restr.) | catastróficos (irrestr./restr.) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in classes:
        rc = [r for r in rows if r["class"] == c]
        s_opt = _variant_summary(rc, "acc_ncut_optimal", "gap_ncut_optimal", "recovered_frac", min_gap_for_recovered_frac)
        s_con = _variant_summary(rc, "acc_ncut_constrained", "gap_ncut_constrained", "recovered_frac_constrained", min_gap_for_recovered_frac)
        m_med = np.mean([r["acc_median"] for r in rc])
        m_ora = np.mean([r["acc_oracle"] for r in rc])
        lines.append(
            f"| {c} | {len(rc)} | {m_med:.3f} | {s_opt['mean_acc_variant']:.3f} | "
            f"{s_con['mean_acc_variant']:.3f} | {m_ora:.3f} | "
            f"{s_opt['aggregate_recovered_frac']:.3f} | {s_con['aggregate_recovered_frac']:.3f} | "
            f"{s_opt['n_catastrophic']}/{s_con['n_catastrophic']} |"
        )

    lines.append("\n## Como interpretar\n")
    lines.append(
        "- **Compare a variante RESTRITA com a IRRESTRITA, não só cada uma contra a mediana.** "
        "O objetivo desta rodada é ver se limitar a busca a uma faixa razoável de proporção "
        "(evitando cortes que isolam um grupo quase vazio) recupera parte do que a busca "
        "irrestrita perdeu, sem reintroduzir o viés de balanceamento forçado da mediana.\n"
        "- Se a RESTRITA tiver `recovered_frac` agregado MAIOR e MENOS pares catastróficos que "
        "a IRRESTRITA, é evidência de que cortes degenerados eram de fato o problema — a faixa "
        "de busca é uma correção válida e generalizável (aplicada a TODO par pela mesma regra, "
        "sem depender de classe).\n"
        "- Se ainda houver pares catastróficos na RESTRITA, especialmente concentrados nas "
        "mesmas classes de antes, o problema pode ser mais profundo que só o tamanho do corte "
        "— vale investigar se a fronteira ótima do NCut nesses pares está genuinamente alinhada "
        "com outra coisa que não FG/BG (ex.: iluminação, textura de fundo recorrente).\n"
        "- **Evite decisões por-classe** (ex.: 'usar restrito só em Kite'): isso violaria o "
        "princípio de generalização do projeto. Se a variante restrita ainda for heterogênea "
        "entre classes, o próximo passo é procurar um sinal POR-PAR (não por-classe) que "
        "prediga quando confiar em cada estratégia — não um mapeamento fixo de classe para "
        "estratégia.\n"
        "- **A tabela de simulação híbrida é o resultado mais acionável desta rodada.** Se "
        "algum limiar de margem produzir acurácia média competitiva, poucos pares "
        "catastróficos, E sem degradar a pior classe em relação à mediana pura, isso vira "
        "candidato direto a mudança de produção — calibrado via ablação (como `degree_alpha`), "
        "não decidido por par no momento da inferência.\n"
    )

    report_path = output_dir / "e2_ncut_optimal_threshold_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compara threshold por mediana vs. threshold ótimo do NCut (sem GT) vs. oracle (Etapa 2).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--features", type=Path, nargs="+", required=True)
    p.add_argument("--correspondence", type=Path, nargs="+", required=True)
    p.add_argument("--gt_index", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=0.05,
                   help="Temperatura usada para gerar o correspondence.npz sendo analisado "
                        "(NÃO persistida no npz — default: 0.05, valor de produção).")
    p.add_argument("--fg_threshold", type=float, default=0.5)
    p.add_argument("--limit_pairs", type=int, default=None,
                   help="Limita nº de pares por classe (smoke test).")
    p.add_argument("--min_gap_for_recovered_frac", type=float, default=0.02,
                   help="gap_median mínimo para um par entrar nas médias/correlações de "
                        "recovered_frac (evita divisão por gap≈0). Default: 0.02.")
    p.add_argument("--min_frac", type=float, default=0.10,
                   help="Fração mínima do grafo que a busca RESTRITA pode isolar como um dos "
                        "lados (evita cortes degenerados). Default: 0.10.")
    p.add_argument("--max_frac", type=float, default=0.90,
                   help="Fração máxima do grafo que a busca RESTRITA pode isolar como um dos "
                        "lados. Default: 0.90.")
    p.add_argument("--output", type=Path, default=Path("diagnostics/e2_ncut_optimal_threshold"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if len(args.features) != len(args.correspondence):
        print("[ERRO] --features e --correspondence precisam ter o mesmo número de caminhos, na mesma ordem.")
        sys.exit(1)

    print(f"[Setup] {len(args.features)} classe(s)  |  gt_index={args.gt_index}  |  temperature={args.temperature}")
    gt_index = load_gt_index(args.gt_index)
    if gt_index is None:
        print("[ERRO] gt_index.json não encontrado ou inválido.")
        sys.exit(1)

    all_rows: List[Dict[str, Any]] = []
    for feat_path, corr_path in zip(args.features, args.correspondence):
        if not feat_path.exists() or not corr_path.exists():
            print(f"[Aviso] Caminho ausente: {feat_path} ou {corr_path} — pulando.")
            continue
        rows = process_class(feat_path, corr_path, gt_index, args.temperature,
                             args.fg_threshold, args.limit_pairs, args.min_frac, args.max_frac)
        all_rows.extend(rows)

    if not all_rows:
        print("[ERRO] Nenhum par processado com sucesso.")
        sys.exit(1)

    import csv
    csv_path = args.output / "e2_ncut_optimal_threshold.csv"
    fieldnames = list(all_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    scatter_recovered_vs_imbalance(all_rows, args.output / "scatter_recovered_frac_vs_imbalance.png",
                                   args.min_gap_for_recovered_frac)
    report_path = write_report(all_rows, args.output, args.min_gap_for_recovered_frac,
                               args.min_frac, args.max_frac)

    summ_opt = _variant_summary(all_rows, "acc_ncut_optimal", "gap_ncut_optimal", "recovered_frac",
                                args.min_gap_for_recovered_frac)
    summ_con = _variant_summary(all_rows, "acc_ncut_constrained", "gap_ncut_constrained", "recovered_frac_constrained",
                                args.min_gap_for_recovered_frac)
    mean_acc_median = np.mean([r["acc_median"] for r in all_rows])
    mean_acc_oracle = np.mean([r["acc_oracle"] for r in all_rows])
    print("\n" + "=" * 60)
    print("  RESULTADO")
    print("=" * 60)
    print(f"  acc_median médio       : {mean_acc_median:.3f}")
    print(f"  acc_ncut_opt (irrestr.): {summ_opt['mean_acc_variant']:.3f}   "
          f"recovered agregado: {summ_opt['aggregate_recovered_frac']:.3f}   "
          f"catastróficos: {summ_opt['n_catastrophic']} ({summ_opt['pct_catastrophic']:.1f}%)")
    print(f"  acc_ncut (restrito)    : {summ_con['mean_acc_variant']:.3f}   "
          f"recovered agregado: {summ_con['aggregate_recovered_frac']:.3f}   "
          f"catastróficos: {summ_con['n_catastrophic']} ({summ_con['pct_catastrophic']:.1f}%)")
    print(f"  acc_oracle médio       : {mean_acc_oracle:.3f}")
    print(f"\n  [Salvo] CSV      : {csv_path}")
    print(f"  [Salvo] Relatório: {report_path}")
    print(f"  [Salvo] Figura   : {args.output / 'scatter_recovered_frac_vs_imbalance.png'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
