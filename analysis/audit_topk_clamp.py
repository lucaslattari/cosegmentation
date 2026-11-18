"""
Audita, para toda classe em features/icoseg38_topk250/<Classe>/correspondence.npz,
se o clamp `top_k = min(E2_TOP_K, N*(N-1)//2)` da E2 foi ativado -- ou seja,
se a classe teve MENOS pares possiveis do que o K=250 configurado, tornando a
"selecao de subconjunto" da E2 um no-op (todos os pares foram usados).

Uso:
    python audit_topk_clamp.py /caminho/para/features/icoseg38_topk250
"""

import sys
from pathlib import Path

import numpy as np

K_CONFIG = 250  # E2_TOP_K de producao, run_pipeline.sh


def audit_class(corr_path: Path) -> dict:
    corr = np.load(corr_path, allow_pickle=True)
    N = int(corr["image_paths"].shape[0])
    n_pairs_used = int(corr["top_pairs"].shape[0])
    max_possible_pairs = N * (N - 1) // 2
    clamped = max_possible_pairs < K_CONFIG
    subset_fraction = (
        n_pairs_used / max_possible_pairs if max_possible_pairs > 0 else float("nan")
    )
    return {
        "classe": corr_path.parent.name,
        "N": N,
        "max_pares": max_possible_pairs,
        "pares_usados": n_pairs_used,
        "clamped": clamped,
        "fracao_grafo": subset_fraction,
    }


def main():
    if len(sys.argv) != 2:
        print(f"Uso: python {sys.argv[0]} <pasta_icoseg38_topk250>")
        sys.exit(1)

    root = Path(sys.argv[1])
    corr_files = sorted(root.glob("*/correspondence.npz"))

    if not corr_files:
        print(f"[ERRO] Nenhum correspondence.npz encontrado em {root}/*/")
        sys.exit(1)

    rows = [audit_class(f) for f in corr_files]

    print(f"{'Classe':<22} {'N':>4} {'Max pares':>10} {'Usados':>8} {'Clamped?':>9} {'% grafo':>8}")
    print("-" * 66)
    for r in rows:
        print(
            f"{r['classe']:<22} {r['N']:>4} {r['max_pares']:>10} "
            f"{r['pares_usados']:>8} {'SIM' if r['clamped'] else 'nao':>9} "
            f"{r['fracao_grafo']:>7.1%}"
        )

    n_total = len(rows)
    n_clamped = sum(r["clamped"] for r in rows)
    print("-" * 66)
    print(
        f"Resumo: {n_clamped}/{n_total} classes ({n_clamped/n_total:.1%}) "
        f"tiveram o clamp ativado -- ou seja, TODOS os pares foram usados "
        f"em vez de uma subselecao real de K={K_CONFIG}."
    )

    ns = [r["N"] for r in rows]
    print(
        f"\nDistribuicao de N: min={min(ns)}  mediana={sorted(ns)[len(ns)//2]}  "
        f"max={max(ns)}  media={sum(ns)/len(ns):.1f}"
    )
    print(f"(clamp so deixa de ser ativado quando N >= 23, ou seja C(N,2) > {K_CONFIG})")


if __name__ == "__main__":
    main()
