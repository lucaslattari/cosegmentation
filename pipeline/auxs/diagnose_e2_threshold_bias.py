#!/usr/bin/env python3
"""
=============================================================================
diagnose_e2_threshold_bias.py — Diagnóstico: custo do threshold por mediana
no Fiedler vector da Etapa 2 (Cross-Image Correspondence)
=============================================================================

Contexto:
  A partição binária de cada par (i, j) na Etapa 2 é obtida via threshold
  do vetor de Fiedler pela MEDIANA (_partition_from_fiedler em
  step2_correspondence.py), o que força um split exatamente 50/50 do
  número de patches, independente da proporção real de foreground no par.

  Este script testa a hipótese: pares cuja fração real de FG (segundo o
  GT) se desvia muito de 50% sofrem uma penalidade sistemática por esse
  threshold forçado — e portanto objetos pequenos (ou muito grandes) são
  estruturalmente prejudicados já na Etapa 2, antes de qualquer coisa a
  jusante (E2B, E3, E4, E5) entrar em jogo.

  Para isolar esse efeito SÓ na Etapa 2 (sem misturar erro de estágios
  posteriores), o script recomputa, para cada par já selecionado em
  produção (top_pairs salvos em correspondence.npz), o grafo de afinidade
  e o vetor de Fiedler — usando as MESMAS funções de step2_correspondence.py
  (build_pair_affinity, normalized_cut, orient_partition), garantindo que
  o grafo testado é bit-a-bit o mesmo que roda em produção. O GT é usado
  SÓ como régua diagnóstica (nunca como entrada de decisão), no mesmo
  espírito dos scores oracle já existentes em evaluate.py e em
  e1_analyze_object_size_vs_oracle.py.

  Para cada par, calcula-se:
    - frac_gt_pair   : fração real de patches FG no par (via GT), na
                        mesma base de patches (conteúdo válido) que entra
                        no grafo NCut do par.
    - acc_median      : acurácia da partição de PRODUÇÃO (threshold por
                        mediana + orientação real do pipeline) vs. GT.
    - acc_oracle      : melhor acurácia possível entre TODOS os pontos de
                        corte do vetor de Fiedler (busca O(P log P)),
                        considerando as duas orientações possíveis —
                        i.e., o teto que um threshold "perfeito" atingiria
                        sobre o MESMO vetor de Fiedler.
    - gap             : acc_oracle - acc_median. Isola o custo específico
                        de usar a mediana em vez do corte ótimo.

  A correlação relevante é entre `gap` e `|frac_gt_pair - 0.5|` (desvio do
  balanceamento forçado) — não entre tamanho do objeto e IoU final, para
  não confundir o efeito do threshold do E2 com erros de estágios
  posteriores.

  IMPORTANTE (mesmo princípio dos outros diagnósticos do projeto): isto
  decide SE vale a pena investir numa mudança de threshold, não prescreve
  qual estratégia adotar. Uma correlação pooled forte e consistente
  within-class é o que importa; um efeito isolado numa única classe
  (ex.: Kite, que falha de forma severa e constante em TODOS os valores
  de degree_alpha já testados) não deve virar mudança de pipeline.

Consistência com o restante do projeto:
  Este script IMPORTA diretamente de step2_correspondence.py
  (load_step1, compute_content_mask, build_pair_affinity, normalized_cut,
  orient_partition, compute_cls_saliency, load_correspondence) e de
  e1_analyze_features.py (load_gt_index, load_patch_gt_masks) — nenhuma
  lógica de grafo, NCut, orientação ou alinhamento de GT é duplicada.
  Ambos os módulos precisam estar no mesmo diretório ou no PYTHONPATH.

  ATENÇÃO: correspondence.npz não persiste a temperatura usada na Etapa 2
  (não é salva no schema atual). Passe --temperature igual ao valor usado
  para gerar o correspondence.npz sendo analisado (produção = 0.05); um
  valor incorreto aqui invalida a fidelidade do recompute.

Uso:
  python diagnose_e2_threshold_bias.py \
      --features       features/Speedskating/features.npz \
                        features/Kite-Brighton/features.npz \
                        features/Windmill/features.npz \
      --correspondence features/Speedskating/correspondence.npz \
                        features/Kite-Brighton/correspondence.npz \
                        features/Windmill/correspondence.npz \
      --gt_index gt_index.json \
      --temperature 0.05 \
      --output diagnostics/e2_threshold_bias

Saídas:
  e2_threshold_bias.csv        — uma linha por par (todas as classes)
  e2_threshold_bias_report.md  — correlações (pooled, within-class,
                                  exclusão de classes) + interpretação
  scatter_gap_vs_imbalance.png — |frac_gt - 0.5| vs. gap (acc_oracle - acc_median)
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# Busca do threshold oracle sobre o vetor de Fiedler (O(P log P))
# -----------------------------------------------------------------------------

def oracle_threshold_accuracy(fiedler: np.ndarray, gt: np.ndarray) -> float:
    """
    Melhor acurácia possível particionando `fiedler` em dois grupos
    CONTÍGUOS na ordenação por valor (qualquer threshold é, por definição,
    um corte contíguo nessa ordenação), testando as duas orientações
    (grupo-baixo=FG vs. grupo-alto=FG) e todos os P+1 pontos de corte.

    Isto é o teto de acurácia atingível por QUALQUER threshold escalar
    sobre este vetor de Fiedler específico — não é um oracle "de cluster"
    como em analyze_object_size_vs_oracle.py, é um oracle "de corte 1D".

    fiedler : (P,) valores contínuos do 2º autovetor
    gt      : (P,) booleano, GT de foreground na mesma ordem/base de patches

    Retorna: acc_oracle em [0, 1]
    """
    m = len(fiedler)
    if m == 0:
        return 0.0

    order = np.argsort(fiedler)
    gt_sorted = gt[order].astype(np.int64)
    prefix = np.concatenate([[0], np.cumsum(gt_sorted)])   # prefix[k] = nº de GT=True nos k primeiros (ordenados)
    total_gt = int(prefix[-1])

    ks = np.arange(0, m + 1)
    # Corte em k: grupo "baixo" = primeiros k (ordenados), grupo "alto" = restante.
    # Orientação A: baixo=FG, alto=BG
    correct_low_fg = 2 * prefix - ks + m - total_gt
    # Orientação B: baixo=BG, alto=FG
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
) -> Optional[Dict[str, Any]]:
    """Recomputa o grafo/Fiedler de UM par e mede gap oracle-vs-mediana."""
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

    # --- Recompute idêntico ao passo 2-3 de step2_correspondence.py ---
    W = build_pair_affinity(feat_i_c, feat_j_c, temperature=temperature, degree_alpha=degree_alpha)
    eigvecs = normalized_cut(W, n_cuts=1)
    fiedler = eigvecs[:, 1]

    # --- GT combinado na MESMA ordem/base (idx_i depois idx_j) ---
    gt_i_c = patch_gt[i][idx_i].astype(bool)
    gt_j_c = patch_gt[j][idx_j].astype(bool)
    gt_combined = np.concatenate([gt_i_c, gt_j_c])

    if gt_combined.sum() == 0 or gt_combined.sum() == gt_combined.size:
        # Par sem sinal de FG/BG no GT (tudo FG ou tudo BG) — não informativo
        # para este diagnóstico (acc_oracle seria trivialmente 100%).
        return None

    # --- acc_median: partição de PRODUÇÃO real (mediana + orientação) ---
    fg_mask_production = orient_partition(
        fiedler, Pi_c, sal_i_c, sal_j_c,
        method=orientation, idx_i=idx_i, idx_j=idx_j, patch_grid=patch_grid,
    )
    acc_median = float((fg_mask_production == gt_combined).mean())

    # --- acc_oracle: melhor threshold possível sobre o MESMO Fiedler vector ---
    acc_oracle = oracle_threshold_accuracy(fiedler, gt_combined)

    frac_gt_pair = float(gt_combined.mean())

    return {
        "class_i": i, "class_j": j,
        "cls_sim": float(sim_score),
        "n_patches_pair": int(Pi_c + Pj_c),
        "frac_gt_pair": frac_gt_pair,
        "imbalance": abs(frac_gt_pair - 0.5),
        "acc_median": acc_median,
        "acc_oracle": acc_oracle,
        "gap": acc_oracle - acc_median,
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
) -> List[Dict[str, Any]]:
    class_name = features_path.parent.name
    print(f"\n[Classe] {class_name}")
    print(f"  features       : {features_path}")
    print(f"  correspondence : {correspondence_path}")

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
          f"(temperature usada no recompute: {temperature} — NÃO persistida no npz, confira!)")

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
        )
        if result is None:
            continue
        result["class"] = class_name
        result["pair_rank"] = rank
        rows.append(result)
        print(f"  [{rank+1:>2}/{len(pairs_to_run)}] par ({i},{j})  "
              f"frac_gt={result['frac_gt_pair']:.3f}  "
              f"acc_median={result['acc_median']:.3f}  acc_oracle={result['acc_oracle']:.3f}  "
              f"gap={result['gap']:+.3f}")

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


def partial_correlate_controlling_cls_sim(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Correlação parcial entre imbalance e gap, controlando por cls_sim do
    par (pares mais parecidos podem ter correspondência mais fácil
    independentemente do desbalanceamento) — via resíduos de regressão
    linear simples (OLS de 1 variável de controle).
    """
    if len(rows) < 4 or pearsonr is None:
        return {"pearson_r": None, "pearson_p": None}

    x = np.array([r["imbalance"] for r in rows])
    y = np.array([r["gap"] for r in rows])
    z = np.array([r["cls_sim"] for r in rows])

    def residualize(v, z):
        A = np.vstack([z, np.ones_like(z)]).T
        coef, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ coef

    x_res = residualize(x, z)
    y_res = residualize(y, z)
    pr, pp = pearsonr(x_res, y_res)
    return {"pearson_r": float(pr), "pearson_p": float(pp)}


def within_class_correlation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Correlação calculada separadamente dentro de cada classe (evita
    falácia ecológica), depois resumida como média simples entre classes
    com pelo menos 5 pares válidos."""
    classes = sorted(set(r["class"] for r in rows))
    per_class = {}
    valid_rs = []
    for c in classes:
        rc = [r for r in rows if r["class"] == c]
        if len(rc) < 5:
            continue
        x = np.array([r["imbalance"] for r in rc])
        y = np.array([r["gap"] for r in rc])
        corr = correlate(x, y)
        per_class[c] = corr
        if corr["pearson_r"] is not None:
            valid_rs.append(corr["pearson_r"])
    mean_r = float(np.mean(valid_rs)) if valid_rs else None
    return {"per_class": per_class, "mean_pearson_r": mean_r, "n_classes_used": len(valid_rs)}


def scatter_gap_vs_imbalance(rows: List[Dict[str, Any]], out_path: Path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    classes = sorted(set(r["class"] for r in rows))
    cmap = plt.get_cmap("tab10")
    for ci, c in enumerate(classes):
        rc = [r for r in rows if r["class"] == c]
        x = [r["imbalance"] for r in rc]
        y = [r["gap"] for r in rc]
        ax.scatter(x, y, label=c, alpha=0.75, color=cmap(ci % 10), s=35)
    x_all = np.array([r["imbalance"] for r in rows])
    y_all = np.array([r["gap"] for r in rows])
    if len(x_all) >= 2:
        coef = np.polyfit(x_all, y_all, 1)
        xs = np.linspace(x_all.min(), x_all.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), "k--", alpha=0.6, linewidth=1.5, label="ajuste linear (pooled)")
    ax.axhline(0.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("|frac_gt_pair - 0.5|  (desvio do balanceamento forçado)")
    ax.set_ylabel("gap = acc_oracle - acc_median")
    ax.set_title("Custo do threshold por mediana vs. desbalanceamento do par (Etapa 2)", fontsize=10)
    ax.legend(fontsize=6, loc="best", ncol=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_report(rows: List[Dict[str, Any]], output_dir: Path,
                 corr_pooled, corr_partial, corr_within, excluded_classes: List[str]) -> Path:
    classes = sorted(set(r["class"] for r in rows))
    lines = []
    lines.append("# Diagnóstico: custo do threshold por mediana (Etapa 2)\n")
    lines.append(f"Classes analisadas ({len(classes)}): " + ", ".join(classes) + "\n")
    lines.append(f"Total de pares válidos: {len(rows)}\n")

    lines.append("\n## Correlação pooled: |frac_gt - 0.5| × gap (acc_oracle - acc_median)\n")
    lines.append("| Método | r | p-valor |")
    lines.append("|---|---|---|")
    lines.append(f"| Pearson | {corr_pooled['pearson_r']:.3f} | {corr_pooled['pearson_p']:.4f} |")
    lines.append(f"| Spearman | {corr_pooled['spearman_r']:.3f} | {corr_pooled['spearman_p']:.4f} |")
    if corr_partial["pearson_r"] is not None:
        lines.append(f"| Pearson parcial (controlando cls_sim) | {corr_partial['pearson_r']:.3f} | {corr_partial['pearson_p']:.4f} |")

    lines.append(f"\n## Correlação within-class (média simples entre {corr_within['n_classes_used']} classes com N≥5 pares)\n")
    lines.append(f"Pearson r médio (within-class): "
                 f"{corr_within['mean_pearson_r']:.3f}" if corr_within["mean_pearson_r"] is not None else "N/A (poucas classes com N≥5)")
    lines.append("\n| Classe | N pares | Pearson r | p-valor |")
    lines.append("|---|---|---|---|")
    for c, corr in corr_within["per_class"].items():
        n_c = len([r for r in rows if r["class"] == c])
        r_str = f"{corr['pearson_r']:.3f}" if corr["pearson_r"] is not None else "N/A"
        p_str = f"{corr['pearson_p']:.4f}" if corr["pearson_p"] is not None else "N/A"
        lines.append(f"| {c} | {n_c} | {r_str} | {p_str} |")

    if excluded_classes:
        rows_excl = [r for r in rows if r["class"] not in excluded_classes]
        x = np.array([r["imbalance"] for r in rows_excl])
        y = np.array([r["gap"] for r in rows_excl])
        corr_excl = correlate(x, y)
        lines.append(f"\n## Teste de exclusão (removendo: {', '.join(excluded_classes)})\n")
        lines.append(f"N pares restantes: {len(rows_excl)}")
        lines.append(f"Pearson r: {corr_excl['pearson_r']:.3f}  (p={corr_excl['pearson_p']:.4f})" if corr_excl["pearson_r"] is not None else "N/A")

    lines.append("\n## Por classe (médias)\n")
    lines.append("| Classe | N pares | imbalance médio | acc_median médio | acc_oracle médio | gap médio |")
    lines.append("|---|---|---|---|---|---|")
    for c in classes:
        rc = [r for r in rows if r["class"] == c]
        lines.append(
            f"| {c} | {len(rc)} | "
            f"{np.mean([r['imbalance'] for r in rc]):.3f} | "
            f"{np.mean([r['acc_median'] for r in rc]):.3f} | "
            f"{np.mean([r['acc_oracle'] for r in rc]):.3f} | "
            f"{np.mean([r['gap'] for r in rc]):+.3f} |"
        )

    lines.append("\n## Como interpretar\n")
    lines.append(
        "- Pearson r positivo e significativo (p < 0.05), consistente pooled E "
        "within-class, entre **desbalanceamento do par** e **gap oracle-vs-mediana** "
        "é evidência de que o threshold por mediana penaliza sistematicamente pares "
        "com objeto pequeno (ou muito grande) — candidato a mudança global no "
        "critério de threshold da Etapa 2 (ex.: busca do corte ótimo do NCut, "
        "zero-crossing, ou outra estratégia data-driven).\n"
        "- Se a correlação pooled for forte mas sumir/inverter no within-class, "
        "é sinal de confounding por classe (ex.: classes com objeto pequeno também "
        "tendo outro problema não relacionado a threshold) — não generalizar.\n"
        "- Se o teste de exclusão mudar drasticamente o resultado, o efeito é "
        "dirigido por outlier(s), não generalizado — documentar como negative "
        "result específico daquela(s) classe(s), não como padrão do pipeline.\n"
    )

    report_path = output_dir / "e2_threshold_bias_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnóstico: custo do threshold por mediana no Fiedler vector da Etapa 2.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--features", type=Path, nargs="+", required=True,
                   help="Um ou mais caminhos para features.npz (um por classe, mesma ordem de --correspondence).")
    p.add_argument("--correspondence", type=Path, nargs="+", required=True,
                   help="Um ou mais caminhos para correspondence.npz (mesma ordem de --features).")
    p.add_argument("--gt_index", type=Path, required=True,
                   help="Caminho para gt_index.json (compartilhado entre classes).")
    p.add_argument("--temperature", type=float, default=0.05,
                   help="Temperatura usada para gerar o correspondence.npz sendo analisado "
                        "(NÃO persistida no npz — default: 0.05, valor de produção).")
    p.add_argument("--fg_threshold", type=float, default=0.5,
                   help="Threshold de binarização do GT por patch (mesmo default dos outros diagnósticos).")
    p.add_argument("--limit_pairs", type=int, default=None,
                   help="Limita nº de pares processados por classe (smoke test rápido).")
    p.add_argument("--exclude_classes", type=str, nargs="*", default=[],
                   help="Nomes de classe (pasta) a excluir do teste de exclusão no relatório, ex: Kite")
    p.add_argument("--output", type=Path, default=Path("diagnostics/e2_threshold_bias"),
                   help="Diretório de saída.")
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
                             args.fg_threshold, args.limit_pairs)
        all_rows.extend(rows)

    if not all_rows:
        print("[ERRO] Nenhum par processado com sucesso. Verifique os caminhos e o GT.")
        sys.exit(1)

    # CSV
    import csv
    csv_path = args.output / "e2_threshold_bias.csv"
    fieldnames = list(all_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    x = np.array([r["imbalance"] for r in all_rows])
    y = np.array([r["gap"] for r in all_rows])
    corr_pooled = correlate(x, y)
    corr_partial = partial_correlate_controlling_cls_sim(all_rows)
    corr_within = within_class_correlation(all_rows)

    scatter_gap_vs_imbalance(all_rows, args.output / "scatter_gap_vs_imbalance.png")
    report_path = write_report(all_rows, args.output, corr_pooled, corr_partial,
                               corr_within, args.exclude_classes)

    print("\n" + "=" * 60)
    print("  RESULTADO")
    print("=" * 60)
    print(f"  Pooled  : Pearson r={corr_pooled['pearson_r']:.3f} (p={corr_pooled['pearson_p']:.4f})  "
          f"Spearman ρ={corr_pooled['spearman_r']:.3f}")
    if corr_partial["pearson_r"] is not None:
        print(f"  Parcial (controlando cls_sim): r={corr_partial['pearson_r']:.3f} (p={corr_partial['pearson_p']:.4f})")
    if corr_within["mean_pearson_r"] is not None:
        print(f"  Within-class (média entre {corr_within['n_classes_used']} classes): r={corr_within['mean_pearson_r']:.3f}")
    print(f"\n  [Salvo] CSV      : {csv_path}")
    print(f"  [Salvo] Relatório: {report_path}")
    print(f"  [Salvo] Figura   : {args.output / 'scatter_gap_vs_imbalance.png'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
