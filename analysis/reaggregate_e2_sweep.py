#!/usr/bin/env python3
"""
reaggregate_e2_sweep.py

Corrige a agregação do e2_sweep_correspondence.sh sem precisar rerodar
nenhuma execução: lê o metrics.tsv já gerado (dado bruto por execução
está correto), remove o prefixo de classe de `step1_stem` (bug de origem:
`step1_stem="$(basename "$step1_npz" .npz)"` inclui "<Classe>__" porque é
assim que os .npz do E1 são nomeados), e refaz exatamente a mesma lógica
de agregação de make_aggregate() em e2_sweep_correspondence.sh — agora
agrupando de verdade por (config, top_k, temperature) entre as 14 classes.

best_by_class.tsv já gerado pelo sweep NÃO precisa de correção (ele já
agrupa por class_name corretamente); este script não o toca.

Uso:
    python3 reaggregate_e2_sweep.py \
        --metrics ablation_results/e2_sweep_selected/metrics.tsv \
        --output_dir ablation_results/e2_sweep_selected_fixed
"""

import argparse
import csv
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


def safe(s):
    try:
        v = float(s)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    except Exception:
        return None


def strip_class_prefix(step1_stem: str, class_name: str) -> str:
    """
    step1_stem vem como '<safe_class>__model-...'. Removemos o prefixo
    usando o próprio class_name da linha (normalizado com underscore no
    lugar de espaço, igual ao sanitize() do bash) em vez de adivinhar
    onde o nome termina — mais robusto a nomes de classe com underscore
    interno.
    """
    safe_class = class_name.replace(" ", "_")
    prefix = f"{safe_class}__"
    if step1_stem.startswith(prefix):
        return step1_stem[len(prefix):]
    # Fallback: tenta achar o primeiro "__model-" e corta ali.
    marker = "__model-"
    idx = step1_stem.find(marker)
    if idx != -1:
        return step1_stem[idx + 2:]  # mantém "model-..."
    return step1_stem  # não deveria acontecer; mantém como está e avisa depois


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.metrics.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    rows = [r for r in rows if safe(r.get("proxy_score")) is not None]
    print(f"[OK] {len(rows)} linhas com proxy_score válido lidas de {args.metrics}")

    all_classes = sorted(set(r["class_name"] for r in rows))
    total_cls = len(all_classes)
    print(f"[OK] {total_cls} classes encontradas: {all_classes}")

    n_fallback = 0
    for r in rows:
        original = r["step1_stem"]
        fixed = strip_class_prefix(original, r["class_name"])
        if fixed == original:
            n_fallback += 1
        r["step1_config"] = fixed  # nova coluna, sem prefixo de classe

    if n_fallback:
        print(f"[Aviso] {n_fallback} linha(s) não tiveram o prefixo removido "
              f"(nem pelo class_name nem pelo marcador '__model-') — confira "
              f"manualmente antes de confiar na agregação.")

    n_distinct_before = len(set(r["step1_stem"] for r in rows))
    n_distinct_after = len(set(r["step1_config"] for r in rows))
    print(f"[Diagnóstico] Configs distintas ANTES da correção (com prefixo de "
          f"classe, bug original): {n_distinct_before}")
    print(f"[Diagnóstico] Configs distintas DEPOIS da correção (sem prefixo, "
          f"deveria ser ~len(top_k)*len(temperature)): {n_distinct_after}")

    # --- Agregação corrigida: por (step1_config, top_k, temperature) --------
    groups = defaultdict(list)
    for r in rows:
        key = (r["step1_config"], r["top_k"], r["temperature"])
        groups[key].append(r)

    def mean(xs):
        return sum(xs) / len(xs) if xs else math.nan

    def stdev(xs):
        return stats.pstdev(xs) if len(xs) > 1 else 0.0

    agg_rows = []
    for (config, tk, temp), gr in groups.items():
        classes = sorted({r["class_name"] for r in gr})
        proxies = [float(r["proxy_score"]) for r in gr]
        polars = [float(r["polar_fraction_decided"]) for r in gr if safe(r.get("polar_fraction_decided"))]
        agrees = [float(r["pair_agreement_mean"]) for r in gr if safe(r.get("pair_agreement_mean"))]
        stabs = [float(r["stability_loo_mae_mean"]) for r in gr if safe(r.get("stability_loo_mae_mean"))]
        corrs = [float(r["cls_corr_mean"]) for r in gr if safe(r.get("cls_corr_mean"))]
        ious = [float(r["gt_best_iou"]) for r in gr if safe(r.get("gt_best_iou"))]
        aucs = [float(r["gt_auc_roc"]) for r in gr if safe(r.get("gt_auc_roc"))]
        orients = [float(r["orientation_accuracy"]) for r in gr if safe(r.get("orientation_accuracy"))]
        orient_improves = [
            float(r["orientation_improvement_over_baseline"]) for r in gr
            if safe(r.get("orientation_improvement_over_baseline"))
        ]
        n_alt_wins = sum(
            1 for r in gr
            if r.get("orientation_best_heuristic") not in (None, "", "n/a", "cls_saliency")
            and safe(r.get("orientation_improvement_over_baseline")) is not None
            and safe(r.get("orientation_improvement_over_baseline")) > 0
        )

        agg_rows.append({
            "step1_config": config,
            "top_k": tk,
            "temperature": temp,
            "classes_done": len(classes),
            "classes_total": total_cls,
            "coverage": len(classes) / total_cls if total_cls else 0.0,
            "classes_list": ";".join(classes),
            "mean_proxy": mean(proxies),
            "std_proxy": stdev(proxies),
            "mean_polar_decided": mean(polars),
            "mean_pair_agreement": mean(agrees),
            "mean_stability_loo_mae": mean(stabs),
            "mean_cls_corr": mean(corrs),
            "mean_gt_best_iou": mean(ious) if ious else math.nan,
            "mean_gt_auc_roc": mean(aucs) if aucs else math.nan,
            "mean_orientation_acc": mean(orients) if orients else math.nan,
            "mean_orientation_improve": mean(orient_improves) if orient_improves else math.nan,
            "n_alt_wins": n_alt_wins,
        })

    # Só configs com cobertura completa (14/14) entram no ranking final,
    # mesmo critério de honestidade que já aplicamos no sweep de E1.
    full_coverage = [r for r in agg_rows if r["classes_done"] == total_cls]
    if not full_coverage:
        print(f"\n[ALERTA] NENHUMA config tem cobertura completa ({total_cls}/{total_cls}) "
              f"mesmo após a correção. Isso seria muito estranho — provavelmente indica "
              f"que alguma execução falhou silenciosamente para alguma classe. Usando "
              f"todas as configs disponíveis mesmo assim, mas INVESTIGUE antes de confiar "
              f"no resultado.")
        ranking_pool = agg_rows
    else:
        ranking_pool = full_coverage
        print(f"\n[OK] {len(full_coverage)} configs com cobertura completa "
              f"({total_cls}/{total_cls}) — ranking baseado só nelas.")

    ranking_pool_sorted = sorted(ranking_pool, key=lambda r: r["mean_proxy"], reverse=True)
    top_n = ranking_pool_sorted[:args.top_n]

    agg_fields = [
        "step1_config", "top_k", "temperature", "classes_done", "classes_total",
        "coverage", "classes_list", "mean_proxy", "std_proxy", "mean_polar_decided",
        "mean_pair_agreement", "mean_stability_loo_mae", "mean_cls_corr",
        "mean_gt_best_iou", "mean_gt_auc_roc", "mean_orientation_acc",
        "mean_orientation_improve", "n_alt_wins",
    ]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}" if not (math.isnan(v) or math.isinf(v)) else "nan"
        return str(v)

    agg_tsv = args.output_dir / "aggregate_by_config_FIXED.tsv"
    top10_tsv = args.output_dir / "top10_aggregate_FIXED.tsv"
    top10_md = args.output_dir / "top10_aggregate_FIXED.md"

    all_sorted = sorted(agg_rows, key=lambda r: (r["classes_done"], r["mean_proxy"]), reverse=True)
    for path, data in [(agg_tsv, all_sorted), (top10_tsv, top_n)]:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, delimiter="\t", fieldnames=agg_fields)
            w.writeheader()
            for r in data:
                w.writerow({k: fmt(r.get(k, "")) for k in agg_fields})

    with top10_md.open("w", encoding="utf-8") as f:
        f.write("# Top 10 agregado — configurações da Etapa 2 (CORRIGIDO)\n\n")
        f.write("> Ranking pela média de `proxy_score` entre classes. "
                "Reagregado após correção do bug de prefixo de classe em "
                "`step1_stem` (ver nota no script `reaggregate_e2_sweep.py`).\n\n")
        f.write(f"Classes avaliadas: **{total_cls}**. "
                f"Ranking restrito a configs com cobertura {total_cls}/{total_cls}.\n\n")
        f.write("| # | top_k | temp | classes | mean_proxy | std_proxy | "
                "mean_gt_best_iou | mean_orientation_acc | n_alt_wins |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for i, r in enumerate(top_n, 1):
            f.write(
                f"| {i} | {r['top_k']} | {r['temperature']} | "
                f"{r['classes_done']}/{r['classes_total']} | "
                f"{fmt(r['mean_proxy'])} | {fmt(r['std_proxy'])} | "
                f"{fmt(r['mean_gt_best_iou'])} | {fmt(r['mean_orientation_acc'])} | "
                f"{r['n_alt_wins']} |\n"
            )

    print(f"\n[Salvo] {agg_tsv}")
    print(f"[Salvo] {top10_tsv}")
    print(f"[Salvo] {top10_md}")
    print("\n[OK] best_by_class.tsv do sweep original NÃO precisa de correção — "
          "já estava agrupando por classe corretamente.")


if __name__ == "__main__":
    main()
