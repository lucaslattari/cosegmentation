#!/usr/bin/env python3
"""
E2 experimental — seleção de pares por k vizinhos por imagem.

Objetivo:
  Substituir o Top-K GLOBAL da Etapa 2 pela união dos k vizinhos CLS mais
  similares de CADA imagem.

Exemplo:
  N=115, k=4
  - cada imagem escolhe seus 4 vizinhos mais similares;
  - pares duplicados (i escolhe j e j escolhe i) são unificados;
  - toda imagem participa de pelo menos k escolhas;
  - o restante da Etapa 2 (NCut, orientação, degree normalization, padding)
    é exatamente o implementado em pipeline/step2_correspondence.py.

O arquivo produzido é compatível com E2B/E3:
  correspondence.npz

Uso:
  python ablation/step2_correspondence_per_image.py \
      --features features/pascal_coseg_bg_recurrence_l0p50/person/features.npz \
      --output features/pascal_coseg_bg_recurrence_l0p50_kpi4/person \
      --k-per-image 4 \
      --temperature 0.05 \
      --orientation spatial \
      --degree-alpha 0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Permite importar pipeline.* quando executado a partir de ablation/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pipeline.step2_correspondence as base  # noqa: E402


def select_pairs_per_image(
    cls_sim: np.ndarray,
    k_per_image: int,
) -> list[tuple[int, int, float]]:
    """
    Cada imagem escolhe seus k vizinhos CLS mais similares.

    A união é não-direcionada:
      escolha i->j ou j->i gera o mesmo par (min(i,j), max(i,j)).

    Isso garante cobertura por construção, ao contrário de um Top-K global
    que pode concentrar todos os pares em um subconjunto de imagens.
    """
    N = int(cls_sim.shape[0])
    if N < 2:
        return []

    k = max(1, min(int(k_per_image), N - 1))
    selected: dict[tuple[int, int], float] = {}

    for i in range(N):
        # Diagonal foi zerada na função original; removemos i explicitamente.
        candidates = [
            (j, float(cls_sim[i, j]))
            for j in range(N)
            if j != i
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        for j, score in candidates[:k]:
            a, b = (i, j) if i < j else (j, i)
            # A matriz é simétrica, mas max() torna o código robusto a
            # pequenas assimetrias numéricas.
            selected[(a, b)] = max(selected.get((a, b), -np.inf), score)

    pairs = [(i, j, s) for (i, j), s in selected.items()]
    pairs.sort(key=lambda x: x[2], reverse=True)

    degree = np.zeros(N, dtype=np.int32)
    for i, j, _ in pairs:
        degree[i] += 1
        degree[j] += 1

    print("\n[Pares] Seleção per-image kNN por similaridade CLS")
    print(f"  Imagens                    : {N}")
    print(f"  k por imagem               : {k}")
    print(f"  Pares únicos processados   : {len(pairs)}")
    print(f"  Grau mínimo/médio/máximo   : "
          f"{degree.min()} / {degree.mean():.2f} / {degree.max()}")
    print(f"  Imagens sem nenhum par     : {int(np.sum(degree == 0))}")
    print(f"  Cobertura                  : {float(np.mean(degree > 0)):.1%}")

    return pairs


def save_correspondence(
    result: dict,
    output_dir: Path,
    image_paths: list[str],
    k_per_image: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "correspondence.npz"

    np.savez_compressed(
        out,
        fg_scores=result["fg_scores"],
        fg_scores_2d=result["fg_scores_2d"],
        cls_sim_matrix=result["cls_sim_matrix"],
        top_pairs=result["top_pairs"],
        pair_votes=result["pair_votes"],
        cls_saliency=result["cls_saliency"],
        patch_grid=result["patch_grid"],
        image_paths=image_paths,
        orientation=np.array(result.get("orientation", "cls")),
        degree_alpha=np.array(
            result.get("degree_alpha", 0.0),
            dtype=np.float32,
        ),
        pair_selection_mode=np.array("per_image_knn"),
        k_per_image=np.array(int(k_per_image), dtype=np.int32),
    )

    size_mb = out.stat().st_size / 1024 ** 2
    print(f"\n[Salvo] {out} ({size_mb:.1f} MB)")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E2 com seleção de k vizinhos por imagem."
    )
    p.add_argument("--features", "-f", required=True)
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--k-per-image", type=int, required=True)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument(
        "--orientation",
        choices=["cls", "spatial"],
        default="spatial",
    )
    p.add_argument("--degree-alpha", type=float, default=0.10)
    p.add_argument("--no-padding-mask", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--viz-max", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.k_per_image < 1:
        raise SystemExit("[ERRO] --k-per-image deve ser >= 1")

    features_path = Path(args.features).resolve()
    output_dir = Path(args.output).resolve()

    feats = base.load_step1(features_path)
    patch_features = feats["patch_features"]
    cls_features = feats["cls_features"]
    patch_grid = feats["patch_grid"]
    image_paths = feats["image_paths"]
    resize_info = feats.get("resize_info")
    N = len(image_paths)

    k_effective = min(args.k_per_image, max(1, N - 1))

    content_mask = None
    if not args.no_padding_mask:
        content_mask = base.compute_content_mask(resize_info, patch_grid)

    print("\n" + "=" * 70)
    print("  Coseg — E2 experimental: pares kNN por imagem")
    print("=" * 70)
    print(f"  Features       : {features_path}")
    print(f"  Imagens        : {N}")
    print(f"  k por imagem   : {k_effective}")
    print(f"  Temperatura    : {args.temperature}")
    print(f"  Orientação     : {args.orientation}")
    print(f"  Degree alpha   : {args.degree_alpha}")
    print(f"  Saída          : {output_dir}")

    # run_correspondence() chama select_top_pairs() internamente.
    # Substituímos SOMENTE essa função durante esta execução.
    original_selector = base.select_top_pairs

    def experimental_selector(
        cls_sim: np.ndarray,
        _unused_top_k: int,
    ) -> list[tuple[int, int, float]]:
        return select_pairs_per_image(cls_sim, k_effective)

    base.select_top_pairs = experimental_selector
    try:
        result = base.run_correspondence(
            patch_features=patch_features,
            cls_features=cls_features,
            patch_grid=patch_grid,
            # O valor é ignorado pelo selector experimental, mas precisa
            # satisfazer a assinatura da função original.
            top_k=k_effective,
            temperature=args.temperature,
            content_mask=content_mask,
            orientation=args.orientation,
            degree_alpha=args.degree_alpha,
        )
    finally:
        base.select_top_pairs = original_selector

    save_correspondence(result, output_dir, image_paths, k_effective)
    base.print_summary(result, image_paths)

    if args.visualize:
        base.visualize_correspondence(
            result,
            image_paths,
            output_dir,
            max_images=args.viz_max,
        )

    print("\n[OK] E2 experimental concluída.")
    print(f"     {output_dir / 'correspondence.npz'}")


if __name__ == "__main__":
    main()
