#!/usr/bin/env python3
"""
check_pair_capacity.py

Verifica, para cada classe, quantas imagens (N) e quantos pares possíveis
(N*(N-1)/2) existem — para saber se top_k=100 (ou valores maiores que
vamos testar) já está "usando todos os pares disponíveis" em alguma
classe pequena, o que tornaria a comparação de top_k parcialmente vazia
para essa classe específica.

Uso:
    python3 check_pair_capacity.py \
        --npz_root ablation_results/e1_npz_consolidated \
        --config_filter "vits14__img-518__resize-keep_ar__layer-last__feat-l2" \
        --top_k_values 20 40 60 80 100 120 150 200 250
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_root", required=True, type=Path)
    ap.add_argument("--config_filter", required=True)
    ap.add_argument("--top_k_values", nargs="+", type=int,
                     default=[20, 40, 60, 80, 100, 120, 150, 200, 250])
    args = ap.parse_args()

    rows = []
    for cls_dir in sorted(args.npz_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        matches = [p for p in cls_dir.iterdir()
                   if p.suffix == ".npz" and args.config_filter in p.name]
        if not matches:
            print(f"[Aviso] Nenhum .npz casando com o filtro em {cls_dir.name}")
            continue
        npz_path = matches[0]
        data = np.load(npz_path, allow_pickle=True)
        image_paths = data["image_paths"] if "image_paths" in data.files else None
        n = len(image_paths) if image_paths is not None else None
        if n is None:
            print(f"[Aviso] Não consegui ler image_paths em {npz_path.name}")
            continue
        max_pairs = n * (n - 1) // 2
        rows.append((cls_dir.name, n, max_pairs))

    rows.sort(key=lambda r: r[2])

    print(f"\n{'Classe':<22}{'N imagens':>10}{'Pares máx.':>12}   Top_k que já esgota os pares disponíveis")
    print("-" * 100)
    for name, n, max_pairs in rows:
        saturated_at = [k for k in sorted(args.top_k_values) if k >= max_pairs]
        flag = f"top_k >= {min(saturated_at)}" if saturated_at else "nenhum (nunca satura no grid testado)"
        marker = "  <<< ATENÇÃO" if saturated_at and min(saturated_at) <= 100 else ""
        print(f"{name:<22}{n:>10}{max_pairs:>12}   {flag}{marker}")

    print(f"\nValores de top_k testados/planejados: {sorted(args.top_k_values)}")


if __name__ == "__main__":
    main()
