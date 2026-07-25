#!/usr/bin/env python3
"""
=============================================================================
ablation_e1_resolution_multiscale.py — Ablação: resolução única maior vs.
fusão multi-escala na Etapa 1
=============================================================================

Contexto:
  A análise de object_size_vs_oracle.py encontrou uma correlação forte e
  robusta (r≈0.73-0.76, within-class, sobrevive a controle de fragmentação
  e à remoção do caso mais extremo) entre tamanho do objeto no GT e oracle
  IoU por imagem. Isso é evidência de que a grade de patches da Etapa 1
  (37×37 com img_size=518) é grosseira demais para objetos pequenos.

  Este script testa DUAS intervenções candidatas, comparando contra o
  baseline atual:
    1. baseline_518        : configuração atual (img_size=518, keep_ar, l2)
    2. highres_<N>          : mesma extração, só que com img_size maior
                              (grid mais fino, mesmo backbone)
    3. multiscale_fusion    : concatena o token de cada patch fino (alta
                              resolução, detalhe local) com o token do
                              patch grosseiro correspondente (baixa
                              resolução, contexto mais amplo), interpolado
                              via bilinear para a grade fina. Reaproveita
                              as DUAS extrações já feitas para baseline e
                              highres — não roda um terceiro forward pass
                              pelo DINOv2, só a fusão é custo extra.

  IMPORTANTE — mesmo princípio científico do resto do pipeline: a decisão
  de adotar uma dessas configs como novo default GLOBAL só se sustenta se
  ela melhorar as classes de objeto pequeno SEM prejudicar sensivelmente
  as classes onde o baseline já funciona bem. Por isso o script roda tanto
  nas classes problemáticas quanto em classes de controle (objeto grande,
  IoU já alto) — se a melhora vier às custas de piorar o controle, isso é
  sinal de trade-off, não de vitória incondicional, e precisa ser
  documentado explicitamente (mesmo padrão já usado para o trade-off do
  Panda na Etapa 3).

Custo computacional:
  Atenção em ViT escala ~quadraticamente com o nº de patches. Dobrar o
  img_size ~quadruplica o nº de patches e ~multiplica por 16 o custo de
  atenção por imagem. O default aqui (728, ~2704 patches vs. 1369 do
  baseline) já cerca de quadruplica o custo — ajuste --img_size_highres
  com cautela, principalmente sem GPU.

Reutilização:
  Importa diretamente de step1_feature_extraction.py (load_dinov2,
  extract_features, postprocess_features, resolve_collection, find_index)
  e de analyze_step1_features.py (load_gt_index, load_patch_gt_masks,
  compute_content_mask, compute_unsupervised_cluster_metrics,
  compute_oracle_cluster_iou) para garantir que a métrica de oracle IoU
  usada aqui é EXATAMENTE a mesma já reportada na auditoria oficial da
  Etapa 1 (oracle_imagewise_best_subset_iou_mean).

Uso:
  python ablation_e1_resolution_multiscale.py \
      --classes_bad \
          "Speedskating-Worldcup_Final_2008-Guido_Jansen" \
          "Liverpool_FC_Players-Liverpool_FC-phes999" \
          "Red_Sox_Players_-_Red_Sox_Yankees_Game-JB_Imaging" \
          "Woman_Soccer_Players-Soccer-Dustin_Hall-red" \
          "Woman_Soccer_Players-Soccer-Dustin_Hall-white" \
      --classes_control \
          "Ferrari_599_GTB_Fiorano-599-auTomoTiVe_FreAk_" \
          "Cheetah-National_Zoo_Washington_DC-smata2" \
      --gt_index gt_index.json \
      --output ablation_results/e1_resolution_multiscale

Saídas:
  ablation_results.csv        — uma linha por (classe, config)
  ablation_report.md          — tabela comparativa + deltas + custo + interpretação
  (opcional, com --save_features) features.npz por (classe, config)
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import warnings
warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None

try:
    from sklearn.preprocessing import normalize as sk_normalize
except Exception:
    sk_normalize = None

# -----------------------------------------------------------------------------
# Imports dos módulos existentes do projeto
# -----------------------------------------------------------------------------
#
# Estrutura esperada:
#
# projeto/
# ├── pipeline/
# │   └── step1_feature_extraction.py
# └── ablation/
#     ├── analyze_step1_features.py
#     └── ablation_e1_resolution_multiscale.py
#
# Este arquivo está em ablation/. Portanto:
#   SCRIPT_DIR   = projeto/ablation
#   PROJECT_ROOT = projeto
#   PIPELINE_DIR = projeto/pipeline
#
# Adicionamos pipeline/ ao sys.path para permitir importar
# step1_feature_extraction diretamente.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

try:
    # pipeline/step1_feature_extraction.py
    from step1_feature_extraction import (
        load_dinov2,
        extract_features,
        postprocess_features,
        resolve_collection,
    )

    # ablation/analyze_step1_features.py
    # Como este script também está em ablation/, o Python já encontra
    # este módulo no diretório do próprio script.
    from e1_analyze_features  import (
        load_gt_index,
        load_patch_gt_masks,
        compute_content_mask,
        compute_unsupervised_cluster_metrics,
        compute_oracle_cluster_iou,
    )

except ImportError as e:
    print(
        "[ERRO] Não foi possível importar os módulos necessários.\n"
        f"       Pipeline esperado em: {PIPELINE_DIR}\n"
        f"       Ablation esperado em: {SCRIPT_DIR}\n"
        f"       Erro original: {e}"
    )
    sys.exit(1)
    print(
        "[ERRO] Não foi possível importar step1_feature_extraction.py e/ou\n"
        "       analyze_step1_features.py. Este script precisa estar no MESMO\n"
        "       diretório desses dois arquivos (ou eles precisam estar no\n"
        "       PYTHONPATH), para reusar exatamente a mesma lógica de\n"
        "       extração, GT e oracle já usada no resto do pipeline.\n"
        f"       Erro original: {e}"
    )
    sys.exit(1)


# -----------------------------------------------------------------------------
# Fusão multi-escala
# -----------------------------------------------------------------------------

def fuse_multiscale(feat_lo: Dict[str, Any], feat_hi: Dict[str, Any]) -> Dict[str, Any]:
    """
    Funde features de duas escalas (baixa/lo = contexto, alta/hi = detalhe)
    na grade de patches da escala ALTA.

    Para cada patch da grade fina (hi), concatena:
      - o token do próprio patch fino (detalhe local)
      - o token do patch grosseiro (lo) interpolado via bilinear até a
        grade fina (contexto de vizinhança mais ampla)

    Resultado tem dimensão de feature = D_lo + D_hi (tipicamente 2D, já
    que lo e hi vêm do mesmo backbone). L2-normaliza o vetor concatenado
    de cada patch, mesma convenção usada em --feature_mode l2 no resto do
    pipeline.

    resize_info / patch_grid / image_paths de saída seguem a escala ALTA,
    já que é essa grade que define a saída fundida.
    """
    if torch is None:
        raise RuntimeError("torch não disponível — necessário para interpolação bilinear na fusão.")

    pf_lo = feat_lo["patch_features"]  # (N, P_lo, D)
    pf_hi = feat_hi["patch_features"]  # (N, P_hi, D)
    H_lo, W_lo = feat_lo["patch_grid"]
    H_hi, W_hi = feat_hi["patch_grid"]
    N, _, D_lo = pf_lo.shape
    D_hi = pf_hi.shape[-1]

    if feat_lo["image_paths"] != feat_hi["image_paths"]:
        raise ValueError(
            "As duas extrações não estão na mesma ordem/conjunto de imagens — "
            "não é seguro fundir. Rode as duas extrações a partir da mesma lista de image_paths."
        )

    # (N, H_lo, W_lo, D) -> (N, D, H_lo, W_lo) para F.interpolate
    lo_grid = pf_lo.reshape(N, H_lo, W_lo, D_lo).transpose(0, 3, 1, 2)
    t = torch.from_numpy(lo_grid).float()
    t_up = F.interpolate(t, size=(H_hi, W_hi), mode="bilinear", align_corners=False)
    lo_upsampled = t_up.permute(0, 2, 3, 1).numpy().reshape(N, H_hi * W_hi, D_lo)

    fused = np.concatenate([lo_upsampled, pf_hi], axis=-1)  # (N, P_hi, D_lo+D_hi)
    fused_flat = sk_normalize(
        fused.reshape(-1, D_lo + D_hi), axis=1
    ).reshape(N, H_hi * W_hi, D_lo + D_hi)

    cls_fused = sk_normalize(
        np.concatenate([feat_lo["cls_features"], feat_hi["cls_features"]], axis=-1),
        axis=1,
    )

    return {
        "patch_features": fused_flat.astype(np.float32),
        "cls_features": cls_fused.astype(np.float32),
        "patch_grid": (H_hi, W_hi),
        "image_paths": feat_hi["image_paths"],
        "image_sizes": feat_hi["image_sizes"],
        "resize_info": feat_hi.get("resize_info"),
    }


# -----------------------------------------------------------------------------
# Avaliação (oracle IoU) de uma config já extraída
# -----------------------------------------------------------------------------

def evaluate_config(
    features: Dict[str, Any],
    gt_index: Dict[str, Any],
    n_clusters: int,
    fg_threshold: float,
    seed: int,
    max_cluster_patches: int,
    no_padding_mask: bool,
) -> Optional[Dict[str, Any]]:
    patch_features = features["patch_features"]
    patch_grid = features["patch_grid"]
    image_paths = features["image_paths"]
    resize_info = features.get("resize_info")

    content_mask = None
    if not no_padding_mask:
        content_mask = compute_content_mask(resize_info, patch_grid)

    patch_gt, gt_info = load_patch_gt_masks(
        image_paths, patch_grid, gt_index,
        fg_threshold=fg_threshold, resize_info=resize_info,
    )
    if patch_gt is None or gt_info.get("matched", 0) == 0:
        print("    [Aviso] Sem GT casado — pulando avaliação desta config.")
        return None

    _, labels_img, _ = compute_unsupervised_cluster_metrics(
        patch_features, content_mask=content_mask,
        n_clusters=n_clusters, max_cluster_patches=max_cluster_patches, seed=seed,
    )
    if labels_img is None:
        print("    [Aviso] Clustering falhou — pulando avaliação desta config.")
        return None

    oracle = compute_oracle_cluster_iou(labels_img, patch_gt, n_clusters)
    return {
        "n_images_matched": gt_info["matched"],
        "n_patches": int(patch_features.shape[1]),
        "feature_dim": int(patch_features.shape[-1]),
        "oracle_iou_mean": oracle.get("oracle_imagewise_best_subset_iou_mean"),
        "oracle_iou_std": oracle.get("oracle_imagewise_best_subset_iou_std"),
        "oracle_global_subset_iou": oracle.get("oracle_best_global_subset_iou"),
    }


# -----------------------------------------------------------------------------
# Pipeline por classe
# -----------------------------------------------------------------------------

def run_class(
    class_name: str,
    group: str,
    model,
    gt_index: Dict[str, Any],
    img_size_baseline: int,
    img_size_highres: int,
    n_clusters: int,
    fg_threshold: float,
    seed: int,
    max_cluster_patches: int,
    no_padding_mask: bool,
    save_features_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    print(f"\n[Classe] {class_name}  (grupo={group})")
    image_paths = resolve_collection(class_name)
    print(f"  {len(image_paths)} imagens.")

    print(f"  Extraindo baseline (img_size={img_size_baseline})...")
    t0 = time.time()
    raw_lo = extract_features(model, image_paths, img_size=img_size_baseline, resize_mode="keep_ar")
    feat_lo = postprocess_features(raw_lo, "l2")
    t_lo = time.time() - t0
    print(f"    {t_lo:.1f}s")

    print(f"  Extraindo alta resolução (img_size={img_size_highres})...")
    t0 = time.time()
    raw_hi = extract_features(model, image_paths, img_size=img_size_highres, resize_mode="keep_ar")
    feat_hi = postprocess_features(raw_hi, "l2")
    t_hi = time.time() - t0
    print(f"    {t_hi:.1f}s")

    print("  Fundindo multi-escala (reaproveita as duas extrações acima)...")
    t0 = time.time()
    feat_multi = fuse_multiscale(feat_lo, feat_hi)
    t_fuse = time.time() - t0
    print(f"    {t_fuse:.1f}s (custo extra, sem novo forward pass)")

    configs = {
        f"baseline_{img_size_baseline}": (feat_lo, t_lo),
        f"highres_{img_size_highres}": (feat_hi, t_hi),
        "multiscale_fusion": (feat_multi, t_lo + t_hi + t_fuse),
    }

    rows = []
    for cfg_name, (feat, elapsed) in configs.items():
        print(f"  Avaliando config '{cfg_name}'...")
        result = evaluate_config(
            feat, gt_index, n_clusters, fg_threshold, seed, max_cluster_patches, no_padding_mask,
        )
        if result is None:
            continue
        row = {
            "class": class_name,
            "group": group,
            "config": cfg_name,
            "extraction_seconds": round(elapsed, 2),
            **result,
        }
        rows.append(row)
        print(f"    oracle_iou_mean={result['oracle_iou_mean']:.4f}  "
              f"n_patches={result['n_patches']}  tempo={elapsed:.1f}s")

        if save_features_dir is not None:
            out_dir = save_features_dir / class_name / cfg_name
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_dir / "features.npz",
                **{k: v for k, v in feat.items() if k != "image_paths"},
                image_paths=feat["image_paths"],
            )

    return rows


# -----------------------------------------------------------------------------
# Relatório
# -----------------------------------------------------------------------------

def write_report(rows: List[Dict[str, Any]], output_dir: Path, img_size_baseline: int, img_size_highres: int) -> Path:
    classes = sorted(set(r["class"] for r in rows))
    configs = [f"baseline_{img_size_baseline}", f"highres_{img_size_highres}", "multiscale_fusion"]

    lines = []
    lines.append("# Ablação Etapa 1 — resolução maior vs. fusão multi-escala\n")
    lines.append(f"Configs comparadas: {', '.join(configs)}\n")

    lines.append("\n## Oracle IoU por classe e config\n")
    lines.append("| Classe | Grupo | " + " | ".join(configs) + " | Δ highres | Δ multiscale |")
    lines.append("|---|---|" + "---|" * len(configs) + "---|---|")
    for c in classes:
        rc = {r["config"]: r for r in rows if r["class"] == c}
        group = next((r["group"] for r in rows if r["class"] == c), "?")
        base_val = rc.get(configs[0], {}).get("oracle_iou_mean")
        vals = []
        for cfg in configs:
            v = rc.get(cfg, {}).get("oracle_iou_mean")
            vals.append(f"{v:.4f}" if v is not None else "n/a")
        d_hi = (rc.get(configs[1], {}).get("oracle_iou_mean") - base_val
                if base_val is not None and rc.get(configs[1], {}).get("oracle_iou_mean") is not None else None)
        d_multi = (rc.get(configs[2], {}).get("oracle_iou_mean") - base_val
                   if base_val is not None and rc.get(configs[2], {}).get("oracle_iou_mean") is not None else None)
        d_hi_s = f"{d_hi:+.4f}" if d_hi is not None else "n/a"
        d_multi_s = f"{d_multi:+.4f}" if d_multi is not None else "n/a"
        lines.append(f"| {c} | {group} | " + " | ".join(vals) + f" | {d_hi_s} | {d_multi_s} |")

    lines.append("\n## Resumo por grupo (média dos deltas vs. baseline)\n")
    lines.append("| Grupo | Δ médio highres | Δ médio multiscale | N classes |")
    lines.append("|---|---|---|---|")
    for group in ["bad", "control"]:
        deltas_hi, deltas_multi = [], []
        for c in classes:
            rc = {r["config"]: r for r in rows if r["class"] == c and r["group"] == group}
            if not rc:
                continue
            base_val = rc.get(configs[0], {}).get("oracle_iou_mean")
            if base_val is None:
                continue
            v_hi = rc.get(configs[1], {}).get("oracle_iou_mean")
            v_multi = rc.get(configs[2], {}).get("oracle_iou_mean")
            if v_hi is not None:
                deltas_hi.append(v_hi - base_val)
            if v_multi is not None:
                deltas_multi.append(v_multi - base_val)
        n_g = len(set(r["class"] for r in rows if r["group"] == group))
        mh = f"{np.mean(deltas_hi):+.4f}" if deltas_hi else "n/a"
        mm = f"{np.mean(deltas_multi):+.4f}" if deltas_multi else "n/a"
        lines.append(f"| {group} | {mh} | {mm} | {n_g} |")

    lines.append("\n## Custo computacional (tempo de extração, segundos)\n")
    lines.append("| Classe | " + " | ".join(configs) + " |")
    lines.append("|---|" + "---|" * len(configs))
    for c in classes:
        rc = {r["config"]: r for r in rows if r["class"] == c}
        vals = [f"{rc.get(cfg, {}).get('extraction_seconds', 'n/a')}" for cfg in configs]
        lines.append(f"| {c} | " + " | ".join(vals) + " |")

    lines.append("\n## Como interpretar\n")
    lines.append(
        "- **Critério de adoção como default global**: delta médio positivo e "
        "relevante no grupo `bad`, com delta médio no grupo `control` próximo "
        "de zero ou levemente positivo. Se o `control` piorar de forma "
        "considerável, isso é um trade-off — documente explicitamente (mesmo "
        "padrão já usado para o trade-off do Panda na Etapa 3) em vez de "
        "adotar a mudança sem ressalva.\n"
        "- **highres vs. multiscale**: se `highres` sozinho já captura a maior "
        "parte do ganho, prefira essa opção — é mais simples e mais barata "
        "que a fusão. Só vale a complexidade extra da fusão multi-escala se "
        "ela superar `highres` por margem que justifique o custo/complexidade "
        "adicional.\n"
        "- **Custo**: números de `extraction_seconds` são por classe nesta "
        "máquina/config específica — útil para comparar a ordem de grandeza "
        "entre configs, não para extrapolar tempo total do dataset sem medir "
        "em mais classes.\n"
        "- Esta ablação é exploratória (poucas classes). Se uma config vencer "
        "aqui, o próximo passo é validar no sweep formal com todas as classes "
        "de validação antes de trocar o default do pipeline, seguindo o mesmo "
        "processo já usado nas Etapas 2 e 3.\n"
    )

    report_path = output_dir / "ablation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablação Etapa 1: resolução maior vs. fusão multi-escala, com classes problemáticas e de controle.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--classes_bad", nargs="+", required=True,
                   help="Classes onde o oracle IoU atual é baixo (grupo a melhorar).")
    p.add_argument("--classes_control", nargs="+", default=[],
                   help="Classes onde o oracle IoU atual já é alto (grupo de controle negativo).")
    p.add_argument("--gt_index", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("ablation_results/e1_resolution_multiscale"))
    p.add_argument("--model", default="dinov2_vits14",
                   choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    p.add_argument("--img_size_baseline", type=int, default=518)
    p.add_argument("--img_size_highres", type=int, default=728,
                   help="~2704 patches vs. 1369 do baseline (~4x custo de atenção). "
                        "Aumente com cautela — custo escala ~quadraticamente com nº de patches.")
    p.add_argument("--n_clusters", type=int, default=5,
                   help="Mesmo valor default da auditoria oficial da Etapa 1.")
    p.add_argument("--fg_threshold", type=float, default=0.5)
    p.add_argument("--max_cluster_patches", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_padding_mask", action="store_true")
    p.add_argument("--save_features", action="store_true",
                   help="Salva features.npz de cada (classe, config) em <output>/<classe>/<config>/.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    save_dir = args.output if args.save_features else None

    print(f"[Setup] {len(args.classes_bad)} classe(s) 'bad', {len(args.classes_control)} classe(s) 'control'")
    gt_index = load_gt_index(args.gt_index)
    if gt_index is None:
        print("[ERRO] gt_index.json não encontrado ou inválido.")
        sys.exit(1)

    print(f"[Modelo] Carregando {args.model}...")
    model = load_dinov2(args.model)

    all_rows: List[Dict[str, Any]] = []
    for class_name in args.classes_bad:
        all_rows.extend(run_class(
            class_name, "bad", model, gt_index,
            args.img_size_baseline, args.img_size_highres,
            args.n_clusters, args.fg_threshold, args.seed,
            args.max_cluster_patches, args.no_padding_mask, save_dir,
        ))
    for class_name in args.classes_control:
        all_rows.extend(run_class(
            class_name, "control", model, gt_index,
            args.img_size_baseline, args.img_size_highres,
            args.n_clusters, args.fg_threshold, args.seed,
            args.max_cluster_patches, args.no_padding_mask, save_dir,
        ))

    if not all_rows:
        print("[ERRO] Nenhuma classe/config avaliada com sucesso.")
        sys.exit(1)

    import csv
    csv_path = args.output / "ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    report_path = write_report(all_rows, args.output, args.img_size_baseline, args.img_size_highres)

    print("\n" + "=" * 60)
    print("  CONCLUÍDO")
    print("=" * 60)
    print(f"  [Salvo] CSV      : {csv_path}")
    print(f"  [Salvo] Relatório: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
