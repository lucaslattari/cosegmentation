#!/usr/bin/env python3
"""
=============================================================================
analyze_object_size_vs_oracle.py — Diagnóstico: tamanho/fragmentação do
objeto vs. IoU oracle por imagem (Etapa 1)
=============================================================================

Contexto:
  A auditoria da Etapa 1 relata "IoU oracle por imagem" agregado (média/std)
  em feature_quality.json, mas não expõe o valor por imagem nem cruza esse
  valor com o tamanho do objeto no GT. Este script existe para testar uma
  hipótese específica levantada ao investigar a classe Liverpool: será que
  o gargalo da Etapa 1 (oracle IoU baixo mesmo com acesso perfeito ao GT
  para escolher clusters) está correlacionado com objetos pequenos e/ou
  fragmentados em múltiplas instâncias (multidões, times, grupos)?

  Se a correlação for forte e aparecer de forma consistente em várias
  classes independentes, é evidência de um gargalo ESTRUTURAL da Etapa 1
  (resolução de patch grosseira demais para o tamanho do objeto) — candidato
  a mudança global (ex: maior resolução, extração multi-escala). Se não
  aparecer, ou só aparecer numa única classe, a hipótese não generaliza e
  não deve virar mudança de pipeline.

  IMPORTANTE (mesmo princípio do resto do pipeline): este script serve para
  DECIDIR SE vale a pena investigar uma mudança global, não para justificar
  overrides por classe. Uma correlação forte pooled entre várias classes é
  o que importa; um efeito isolado numa classe não é.

Consistência com analyze_step1_features.py:
  Este script IMPORTA diretamente load_features, compute_content_mask,
  load_gt_index, load_patch_gt_masks, compute_unsupervised_cluster_metrics
  e iou_bool do seu módulo analyze_step1_features.py (precisa estar no
  mesmo diretório ou no PYTHONPATH). Isso garante que o "oracle IoU por
  imagem" calculado aqui usa exatamente o mesmo K-means, mesmo tratamento
  de padding (--no_padding_mask) e mesmo alinhamento de GT que já são
  usados na auditoria oficial da Etapa 1 — evitando que os dois scripts
  divirjam silenciosamente ao longo do tempo.

Uso:
  python analyze_object_size_vs_oracle.py \
      --features features/Speedskating/features.npz \
                 features/Liverpool_FC_Players/features.npz \
                 features/Red_Sox_Players/features.npz \
                 features/Woman_Soccer_Players-red/features.npz \
                 features/Woman_Soccer_Players-white/features.npz \
      --gt_index gt_index.json \
      --output diagnostics/object_size_vs_oracle

  Classes recomendadas para rodar primeiro (Grupo A — oracle IoU baixo já
  na Etapa 1, ver auditoria de all_classes.json):
    Speedskating-Worldcup_Final_2008-Guido_Jansen  (oracle ≈ 0.019)
    Liverpool_FC_Players-Liverpool_FC-phes999      (oracle ≈ 0.138)
    Red_Sox_Players_-_Red_Sox_Yankees_Game-JB_Imaging (oracle ≈ 0.063)
    Woman_Soccer_Players-Soccer-Dustin_Hall-red    (oracle ≈ 0.245)
    Woman_Soccer_Players-Soccer-Dustin_Hall-white  (oracle ≈ 0.403)

  NÃO rode isso esperando explicar os casos do Grupo B (ex: Kendo, onde o
  oracle já é alto mas o resultado final do pipeline é baixo) — lá o
  gargalo está depois da Etapa 1, essa análise não vai mostrar nada útil.

Saídas:
  object_size_vs_oracle.csv        — uma linha por imagem, todas as classes
  object_size_vs_oracle_report.md  — correlações (pooled e por classe) + interpretação
  scatter_size_vs_oracle.png       — tamanho (fração FG) vs oracle IoU
  scatter_fragmentation_vs_oracle.png — nº de componentes vs oracle IoU
=============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings_silenced = True
import warnings
warnings.filterwarnings("ignore")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from scipy import ndimage
    from scipy.stats import pearsonr, spearmanr
except Exception:
    ndimage = None
    pearsonr = None
    spearmanr = None

# ---------------------------------------------------------------------------
# Reutiliza a lógica já validada de analyze_step1_features.py em vez de
# duplicá-la. Ajuste o import abaixo se o módulo estiver em outro caminho.
# ---------------------------------------------------------------------------
try:
    from e1_analyze_features import (
        load_features,
        compute_content_mask,
        load_gt_index,
        load_patch_gt_masks,
        compute_unsupervised_cluster_metrics,
        iou_bool,
    )
except ImportError as e:
    print(
        "[ERRO] Não foi possível importar analyze_step1_features.py.\n"
        "       Este script precisa estar no MESMO diretório desse arquivo\n"
        "       (ou analyze_step1_features.py precisa estar no PYTHONPATH),\n"
        "       para reusar exatamente a mesma lógica de oracle/GT/padding.\n"
        f"       Erro original: {e}"
    )
    sys.exit(1)


# -----------------------------------------------------------------------------
# Métricas por imagem: tamanho e fragmentação do objeto no grid de patches
# -----------------------------------------------------------------------------

def object_size_and_fragmentation(
    patch_gt_img: np.ndarray,
    content_mask_img: Optional[np.ndarray],
    patch_grid: Tuple[int, int],
) -> Dict[str, Any]:
    """
    A partir da máscara GT já reduzida à grade de patches (1D, tamanho P)
    de UMA imagem, calcula:
      - n_content_patches : patches de conteúdo real (exclui padding)
      - n_fg_patches       : patches de conteúdo marcados como foreground
      - object_size_frac   : n_fg_patches / n_content_patches
      - n_components       : nº de componentes conexos do FG na grade 2D
                              (4-conectividade) — proxy de fragmentação
                              (multidão/objetos separados vs. blob único)
      - mean_component_size: n_fg_patches / n_components (tamanho médio
                              de cada instância, em patches)
    """
    H_p, W_p = patch_grid
    gt_2d = patch_gt_img.reshape(H_p, W_p).astype(bool)

    if content_mask_img is not None:
        content_2d = content_mask_img.reshape(H_p, W_p).astype(bool)
    else:
        content_2d = np.ones((H_p, W_p), dtype=bool)

    n_content = int(content_2d.sum())
    # FG só conta dentro de patches de conteúdo real (padding nunca é FG
    # depois do alinhamento em load_patch_gt_masks, mas reforça por segurança).
    fg_2d = gt_2d & content_2d
    n_fg = int(fg_2d.sum())

    if n_content == 0:
        return {
            "n_content_patches": 0, "n_fg_patches": 0,
            "object_size_frac": None, "n_components": 0,
            "mean_component_size": None,
        }

    object_size_frac = n_fg / n_content

    if n_fg == 0 or ndimage is None:
        n_components = 0
        mean_component_size = 0.0
    else:
        labeled, n_components = ndimage.label(
            fg_2d, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )
        mean_component_size = n_fg / max(1, n_components)

    return {
        "n_content_patches": n_content,
        "n_fg_patches": n_fg,
        "object_size_frac": object_size_frac,
        "n_components": int(n_components),
        "mean_component_size": mean_component_size,
    }


def oracle_iou_per_image(
    labels_img: np.ndarray,
    patch_gt: np.ndarray,
    n_clusters: int,
) -> List[float]:
    """
    Mesma lógica de 'oracle por imagem' de compute_oracle_cluster_iou em
    analyze_step1_features.py, mas retornando a lista por imagem em vez de
    só a média/desvio agregados — é essa lista que queremos correlacionar
    com tamanho/fragmentação do objeto.
    """
    import itertools

    N, P = patch_gt.shape
    out = []
    for i in range(N):
        yi = patch_gt[i].astype(bool)
        li = labels_img[i]
        best = 0.0
        for k in range(n_clusters):
            best = max(best, iou_bool(li == k, yi))
        if n_clusters <= 8:
            for r in range(1, n_clusters + 1):
                for subset in itertools.combinations(range(n_clusters), r):
                    best = max(best, iou_bool(np.isin(li, subset), yi))
        out.append(best)
    return out


# -----------------------------------------------------------------------------
# Pipeline por classe
# -----------------------------------------------------------------------------

def process_class(
    features_path: Path,
    gt_index: Dict[str, Any],
    n_clusters: int,
    fg_threshold: float,
    no_padding_mask: bool,
    max_cluster_patches: int,
    seed: int,
) -> List[Dict[str, Any]]:
    class_name = features_path.parent.name
    print(f"\n[Classe] {class_name}  ({features_path})")

    feat = load_features(features_path)
    patch_features = feat["patch_features"]
    patch_grid = feat["patch_grid"]
    image_paths = feat["image_paths"]
    resize_info = feat.get("resize_info")

    content_mask = None
    if not no_padding_mask:
        content_mask = compute_content_mask(resize_info, patch_grid)

    patch_gt, gt_info = load_patch_gt_masks(
        image_paths, patch_grid, gt_index,
        fg_threshold=fg_threshold, resize_info=resize_info,
    )
    if patch_gt is None or gt_info.get("matched", 0) == 0:
        print(f"  [Aviso] Sem GT casado para {class_name} — pulando classe.")
        return []
    print(f"  GT casado: {gt_info['matched']}/{len(image_paths)} imagens.")

    _, labels_img, _ = compute_unsupervised_cluster_metrics(
        patch_features, content_mask=content_mask,
        n_clusters=n_clusters, max_cluster_patches=max_cluster_patches,
        seed=seed,
    )
    if labels_img is None:
        print(f"  [Aviso] Clustering falhou para {class_name} — pulando classe.")
        return []

    oracle_ious = oracle_iou_per_image(labels_img, patch_gt, n_clusters)

    rows = []
    for i, img_path in enumerate(image_paths):
        cm_i = content_mask[i] if content_mask is not None else None
        size_info = object_size_and_fragmentation(patch_gt[i], cm_i, patch_grid)
        rows.append({
            "class": class_name,
            "image_path": img_path,
            "oracle_iou": oracle_ious[i],
            **size_info,
        })
    return rows


# -----------------------------------------------------------------------------
# Agregação, correlação, plots, relatório
# -----------------------------------------------------------------------------

def correlate(x: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    if len(x) < 3 or pearsonr is None:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
    }


def scatter_with_fit(x, y, classes, xlabel, ylabel, title, out_path: Path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    uniq = sorted(set(classes))
    cmap = plt.get_cmap("tab10")
    for ci, c in enumerate(uniq):
        sel = [j for j, cc in enumerate(classes) if cc == c]
        ax.scatter(np.array(x)[sel], np.array(y)[sel], label=c, alpha=0.75,
                   color=cmap(ci % 10), s=40)
    if len(x) >= 2:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, np.polyval(coef, xs), "k--", alpha=0.6, linewidth=1.5,
                label="ajuste linear (pooled)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_report(rows: List[Dict[str, Any]], output_dir: Path,
                 corr_size: Dict, corr_frag: Dict, corr_comp_size: Dict) -> Path:
    classes = sorted(set(r["class"] for r in rows))
    lines = []
    lines.append("# Diagnóstico: tamanho/fragmentação do objeto vs. oracle IoU (Etapa 1)\n")
    lines.append(f"Classes analisadas ({len(classes)}): " + ", ".join(classes) + "\n")
    lines.append(f"Total de imagens: {len(rows)}\n")

    lines.append("\n## Correlação pooled (todas as classes juntas)\n")
    lines.append("| Par de variáveis | Pearson r | p-valor | Spearman ρ | p-valor |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| tamanho do objeto (frac. patches FG) × oracle IoU | "
        f"{corr_size['pearson_r']:.3f} | {corr_size['pearson_p']:.4f} | "
        f"{corr_size['spearman_r']:.3f} | {corr_size['spearman_p']:.4f} |"
    )
    lines.append(
        f"| nº de componentes (fragmentação) × oracle IoU | "
        f"{corr_frag['pearson_r']:.3f} | {corr_frag['pearson_p']:.4f} | "
        f"{corr_frag['spearman_r']:.3f} | {corr_frag['spearman_p']:.4f} |"
    )
    lines.append(
        f"| tamanho médio por componente × oracle IoU | "
        f"{corr_comp_size['pearson_r']:.3f} | {corr_comp_size['pearson_p']:.4f} | "
        f"{corr_comp_size['spearman_r']:.3f} | {corr_comp_size['spearman_p']:.4f} |"
    )

    lines.append("\n## Por classe (mediana)\n")
    lines.append("| Classe | N | tamanho médio (frac.) | nº compon. médio | oracle IoU médio |")
    lines.append("|---|---|---|---|---|")
    for c in classes:
        rc = [r for r in rows if r["class"] == c]
        sizes = [r["object_size_frac"] for r in rc if r["object_size_frac"] is not None]
        comps = [r["n_components"] for r in rc]
        ious = [r["oracle_iou"] for r in rc]
        lines.append(
            f"| {c} | {len(rc)} | {np.mean(sizes):.4f} | {np.mean(comps):.2f} | {np.mean(ious):.4f} |"
        )

    lines.append("\n## Como interpretar\n")
    lines.append(
        "- Pearson r negativo e significativo (p < 0.05) entre **tamanho do objeto** "
        "e **oracle IoU** é evidência de que objetos pequenos sofrem mais com a "
        "resolução de patch da Etapa 1 — candidato a mudança global (ex: maior "
        "resolução, extração multi-escala).\n"
        "- Pearson r negativo e significativo entre **nº de componentes** e "
        "**oracle IoU**, mesmo controlando por tamanho total, reforça a hipótese "
        "de que fragmentação (multidão/múltiplas instâncias), não só tamanho "
        "absoluto, é o fator relevante — isso mudaria a direção da intervenção "
        "(ex: agregação espacial mais fina em vez de só aumentar resolução global).\n"
        "- Se as correlações forem fracas (|r| < 0.3) ou não significativas, ou "
        "se só uma classe estiver puxando o efeito, NÃO trate isso como gargalo "
        "estrutural — evite generalizar a mudança para o pipeline inteiro.\n"
    )

    report_path = output_dir / "object_size_vs_oracle_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correlaciona tamanho/fragmentação do objeto no GT com oracle IoU por imagem (Etapa 1).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--features", type=Path, nargs="+", required=True,
                   help="Um ou mais caminhos para features.npz (um por classe).")
    p.add_argument("--gt_index", type=Path, required=True,
                   help="Caminho para gt_index.json (compartilhado entre classes).")
    p.add_argument("--output", type=Path, default=Path("diagnostics/object_size_vs_oracle"),
                   help="Diretório de saída.")
    p.add_argument("--n_clusters", type=int, default=5,
                   help="Mesmo valor usado na auditoria da Etapa 1 (default: 5).")
    p.add_argument("--fg_threshold", type=float, default=0.5)
    p.add_argument("--max_cluster_patches", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_padding_mask", action="store_true",
                   help="Desativa mascaramento de padding (mesmo flag da Etapa 1).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] {len(args.features)} classe(s), gt_index={args.gt_index}")
    gt_index = load_gt_index(args.gt_index)
    if gt_index is None:
        print("[ERRO] gt_index.json não encontrado ou inválido — sem GT não há oracle IoU calculável.")
        sys.exit(1)

    all_rows: List[Dict[str, Any]] = []
    for fp in args.features:
        if not fp.exists():
            print(f"[Aviso] {fp} não existe — pulando.")
            continue
        rows = process_class(
            fp, gt_index, args.n_clusters, args.fg_threshold,
            args.no_padding_mask, args.max_cluster_patches, args.seed,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("[ERRO] Nenhuma imagem processada com sucesso. Verifique os caminhos.")
        sys.exit(1)

    # Remove imagens sem GT válido (object_size_frac None ou content vazio)
    valid_rows = [r for r in all_rows if r["object_size_frac"] is not None]
    dropped = len(all_rows) - len(valid_rows)
    if dropped:
        print(f"[Aviso] {dropped} imagem(ns) sem patches de conteúdo válidos — excluídas da correlação.")

    sizes = np.array([r["object_size_frac"] for r in valid_rows])
    comps = np.array([r["n_components"] for r in valid_rows])
    comp_sizes = np.array([
        r["mean_component_size"] if r["mean_component_size"] is not None else 0.0
        for r in valid_rows
    ])
    ious = np.array([r["oracle_iou"] for r in valid_rows])
    classes = [r["class"] for r in valid_rows]

    corr_size = correlate(sizes, ious)
    corr_frag = correlate(comps.astype(float), ious)
    corr_comp_size = correlate(comp_sizes, ious)

    # CSV
    import csv
    csv_path = args.output / "object_size_vs_oracle.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Plots
    scatter_with_fit(
        sizes, ious, classes,
        xlabel="Tamanho do objeto (fração de patches de conteúdo que são FG)",
        ylabel="Oracle IoU por imagem",
        title="Tamanho do objeto × Oracle IoU (Etapa 1)",
        out_path=args.output / "scatter_size_vs_oracle.png",
    )
    scatter_with_fit(
        comps, ious, classes,
        xlabel="Nº de componentes conexos do objeto na grade de patches",
        ylabel="Oracle IoU por imagem",
        title="Fragmentação do objeto × Oracle IoU (Etapa 1)",
        out_path=args.output / "scatter_fragmentation_vs_oracle.png",
    )

    report_path = write_report(valid_rows, args.output, corr_size, corr_frag, corr_comp_size)

    print("\n" + "=" * 60)
    print("  RESULTADO")
    print("=" * 60)
    print(f"  tamanho×oracle_iou     : Pearson r={corr_size['pearson_r']:.3f} "
          f"(p={corr_size['pearson_p']:.4f})  Spearman ρ={corr_size['spearman_r']:.3f}")
    print(f"  fragmentação×oracle_iou: Pearson r={corr_frag['pearson_r']:.3f} "
          f"(p={corr_frag['pearson_p']:.4f})  Spearman ρ={corr_frag['spearman_r']:.3f}")
    print(f"  tam.componente×oracle_iou: Pearson r={corr_comp_size['pearson_r']:.3f} "
          f"(p={corr_comp_size['pearson_p']:.4f})  Spearman ρ={corr_comp_size['spearman_r']:.3f}")
    print(f"\n  [Salvo] CSV      : {csv_path}")
    print(f"  [Salvo] Relatório: {report_path}")
    print(f"  [Salvo] Figuras  : {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
