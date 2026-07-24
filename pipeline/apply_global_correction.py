#!/usr/bin/env python3
"""
=============================================================================
apply_global_correction.py — Correção global via scribble (Estratégia 1)
=============================================================================

Recebe até DOIS scribbles desenhados pelo usuário em UMA imagem de
referência — um marcando "isto aqui É objeto" (FG) e outro "isto aqui NUNCA
é objeto" (BG), independentes um do outro — constrói um protótipo de
feature para cada um, e aplica a correção a QUALQUER cluster global
(Etapa 3) cuja feature média seja parecida com um desses protótipos — em
TODAS as imagens da coleção, não só na de referência.

Os dois scribbles são processados como duas correções SEPARADAS, cada uma
com seu próprio protótipo. Isso importa: numa mesma imagem de referência,
você pode legitimamente marcar "isto é grama, nunca objeto" numa região E
"isto é o jogador, é objeto" em outra — cada marcação usa seu próprio
protótipo de similaridade, então nunca fazem sentido serem fundidas numa
máscara binária única com um "brush_type" global (isso faria toda a
marcação FG ser tratada como BG sempre que as duas coexistissem na mesma
submissão — bug real da versão anterior deste script, reportado pelo
usuário). Se um cluster passa do limiar de similaridade para os dois
protótipos ao mesmo tempo (conflito raro), vence o protótipo com maior
similaridade.

Isto NÃO é correção manual por imagem. É uma correção única, propagada por
similaridade de features — o usuário aponta o padrão problemático uma vez
("essa grama verde", "esse jogador"), e a correção vale pra coleção inteira.

Saída: cosaliency_corrected.npz com o MESMO formato de cosaliency.npz
(mesmos campos), com cluster_is_fg / fg_mask_patch / fg_mask_coarse
corrigidos — e, criticamente, fg_mask_pixel / seeds RECONSTRUÍDOS a partir
da correção (upsample_masks + extract_seeds do step3_cosaliency.py). São
fg_mask_pixel e seeds que step4_sam_refinement.py realmente consome pra
montar os prompts do SAM 2 (load_step3() nem lê fg_mask_patch/fg_mask_coarse/
cluster_is_fg) — sem essa reconstrução a correção fica sem nenhum efeito
no resultado final, mesmo reportando clusters corrigidos corretamente.
step4_sam_refinement.py consome esse arquivo diretamente, sem nenhuma
modificação, apontando --cosaliency para ele em vez do original.

Uso:
  python3 apply_global_correction.py \
    --features features.npz \
    --cosaliency cosaliency.npz \
    --reference_image "2664307423_e2b1ccf21f.jpg" \
    --scribble_mask_bg scribble_bg.png \
    --scribble_mask_fg scribble_fg.png \
    --output cosaliency_corrected.npz

  (pelo menos um dos dois --scribble_mask_* é obrigatório; pode passar só um)
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# Reaproveita exatamente a mesma lógica da Etapa 3 pra ir de fg_mask_patch
# (resolução de patch) até fg_mask_pixel + seeds (resolução de pixel) —
# incluindo o recorte do padding do modo keep_ar via resize_info. Isso
# garante que a correção produza os MESMOS campos, no MESMO formato, que
# step4_sam_refinement.py espera (ele lê fg_mask_pixel e seeds, não
# fg_mask_patch/fg_mask_coarse/cluster_is_fg diretamente).
from step3_cosaliency import upsample_masks, extract_seeds


def block_max_pool(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reduz arr (H,W) pra (out_h,out_w) por MAX-pooling (marca a célula de
    saída se QUALQUER pixel de entrada dentro dela for não-zero).

    Um resize BILINEAR clássico faz média de área — equivalente a exigir que
    mais de 50% da célula esteja coberta pra passar de um threshold de 127.
    Isso é adequado pra máscaras de região preenchida, mas quebra pra um
    scribble de traço fino: um traço de poucos pixels de espessura cruzando
    uma célula de ~14x14px nunca cobre 50% da área dela, então o traço inteiro
    desaparece no downsample e nenhum patch é marcado — mesmo com o traço
    claramente visível na imagem. Max-pooling resolve isso: basta o traço
    tocar a célula, não dominar sua área.
    """
    H, W = arr.shape
    row_edges = np.linspace(0, H, out_h + 1).round().astype(int)
    col_edges = np.linspace(0, W, out_w + 1).round().astype(int)
    out = np.zeros((out_h, out_w), dtype=arr.dtype)
    for i in range(out_h):
        r0, r1 = row_edges[i], max(row_edges[i + 1], row_edges[i] + 1)
        for j in range(out_w):
            c0, c1 = col_edges[j], max(col_edges[j + 1], col_edges[j] + 1)
            out[i, j] = arr[r0:r1, c0:c1].max()
    return out


def downsample_scribble_to_patch(
    scribble_mask: np.ndarray,
    patch_grid: Tuple[int, int],
    resize_info: np.ndarray | None = None,
) -> np.ndarray:
    """(H_orig, W_orig) 0/255 -> (P,) bool, marcado se o traço tocou o patch
    (max-pooling — ver block_max_pool).

    O scribble é desenhado em cima da imagem ORIGINAL (correct.html carrega
    image.original, sem padding). Mas a grade de patches cobre o canvas
    quadrado inteiro produzido pela Etapa 1 no modo --resize keep_ar (imagem
    redimensionada + faixas de padding). Resizar o scribble direto de
    (H_orig, W_orig) pra (H_p, W_p) ignora esse padding e desloca/comprime
    sistematicamente os patches marcados sempre que a imagem original não é
    quadrada — o mesmo bug de padding já corrigido nos analisadores da
    Etapa 1-3, reaparecendo aqui.

    Esta função inverte exatamente o que upsample_masks() faz em
    step3_cosaliency.py: reinsere o scribble na geometria com padding (resize
    pro tamanho pré-padding + colagem no offset pad_top/pad_left do canvas
    quadrado) antes de reamostrar pra resolução de patch — usando max-pooling
    nessa última etapa, não um resize de área que dilui traços finos.
    """
    H_p, W_p = patch_grid

    if resize_info is not None:
        resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = \
            [int(x) for x in resize_info]

        # 1) resize do scribble (coords da imagem original) pro tamanho que a
        #    imagem tinha DEPOIS do resize da Etapa 1, ANTES do padding —
        #    max-pooling aqui também, pra nunca diluir o traço em nenhum estágio.
        resized_scribble = block_max_pool(scribble_mask, resized_h, resized_w)

        # 2) cola no canvas quadrado com padding, no mesmo offset da Etapa 1
        #    (fora desse offset é padding — nunca corresponde a conteúdo real)
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        canvas[pad_top:pad_top + resized_h, pad_left:pad_left + resized_w] = resized_scribble

        # 3) só agora reamostra pra grade de patches — max-pooling, não média
        small = block_max_pool(canvas, H_p, W_p)
    else:
        # Modo "square" da Etapa 1 — sem padding, mas ainda assim o downsample
        # final pra grade de patches precisa ser max-pooling, não BILINEAR.
        small = block_max_pool(scribble_mask, H_p, W_p)

    return (small > 127).reshape(-1)


def find_reference_index(image_paths: list, reference_image: str) -> int:
    ref_name = Path(reference_image).name
    ref_stem = Path(reference_image).stem
    for i, p in enumerate(image_paths):
        name = Path(p).name
        if name == ref_name or Path(p).stem == ref_stem:
            return i
    raise ValueError(
        f"Imagem de referência '{reference_image}' não encontrada em image_paths "
        f"do features.npz (esperado um destes: {[Path(p).name for p in image_paths]})."
    )


def build_prototype(
    scribble_path: Path,
    label: str,
    patch_features: np.ndarray,
    patch_grid: Tuple[int, int],
    ref_idx: int,
    ref_resize_info,
    P: int,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """Lê um scribble, converte pra máscara de patch e monta o protótipo de
    feature (média normalizada dos patches marcados). Retorna
    (prototype ou None, marked_patches bool array, n_marked)."""
    scribble = np.asarray(Image.open(scribble_path).convert("L"))
    marked_patches = downsample_scribble_to_patch(scribble, patch_grid, ref_resize_info)
    n_marked = int(marked_patches.sum())
    if n_marked == 0:
        print(f"      [Aviso] scribble de {label} não marcou nenhum patch — ignorando essa marcação.")
        return None, marked_patches, 0
    print(f"      {label}: {n_marked}/{P} patches marcados na imagem de referência")
    ref_features = patch_features[ref_idx][marked_patches]
    prototype = ref_features.mean(axis=0)
    prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
    return prototype, marked_patches, n_marked


def save_debug_overlay(
    scribble_path: Path,
    marked_patches: np.ndarray | None,
    color: tuple[int, int, int],
    ref_img: np.ndarray,
    image_sizes: np.ndarray,
    resize_info,
    patch_grid: Tuple[int, int],
    ref_idx: int,
    out_path: Path,
    raw: bool,
) -> Path | None:
    """Salva um overlay de diagnóstico: raw=True mostra o scribble cru (sem
    conversão pra patch); raw=False mostra os patches marcados projetados de
    volta pro espaço de pixel (o que o algoritmo usa de fato)."""
    try:
        H_ref, W_ref = int(image_sizes[ref_idx, 0]), int(image_sizes[ref_idx, 1])
        if raw:
            scribble = np.asarray(Image.open(scribble_path).convert("L"))
            mask_pixel = np.array(
                Image.fromarray(scribble).resize((W_ref, H_ref), Image.NEAREST)
            ) > 127
            mask_pixel = mask_pixel.astype(np.float32)
        else:
            if marked_patches is None:
                return None
            ref_size = image_sizes[ref_idx:ref_idx + 1]
            ref_ri = resize_info[ref_idx:ref_idx + 1] if resize_info is not None else None
            marked_patch_grid = marked_patches.astype(np.float32).reshape(1, patch_grid[0], patch_grid[1])
            mask_pixel = upsample_masks(marked_patch_grid, ref_size, resize_info=ref_ri)[0]

        overlay = ref_img.astype(np.float32)
        tint = np.zeros_like(overlay)
        tint[:, :, 0] = color[0]; tint[:, :, 1] = color[1]; tint[:, :, 2] = color[2]
        alpha = mask_pixel[:, :, None] * 0.55
        overlay = (overlay * (1 - alpha) + tint * alpha).clip(0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(out_path)
        return out_path
    except Exception as exc:
        print(f"      [Aviso] Não foi possível gerar {out_path.name}: {exc}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--cosaliency", required=True, type=Path)
    ap.add_argument("--reference_image", required=True, type=str,
                    help="Nome (ou stem) do arquivo de imagem em que os scribbles foram desenhados.")
    ap.add_argument("--scribble_mask_fg", type=Path, default=None,
                    help="PNG marcando 'isto é objeto' (foreground). Opcional.")
    ap.add_argument("--scribble_mask_bg", type=Path, default=None,
                    help="PNG marcando 'isto nunca é objeto' (background). Opcional.")
    ap.add_argument("--similarity_threshold", type=float, default=0.6,
                    help="Similaridade de cosseno mínima entre a média de um cluster e o "
                        "protótipo do scribble para esse cluster ser forçado (default: 0.6).")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    if not args.scribble_mask_fg and not args.scribble_mask_bg:
        print("[ERRO] Forneça pelo menos --scribble_mask_fg ou --scribble_mask_bg.")
        sys.exit(1)

    print(f"[1/6] Carregando features e cosaliency...")
    feat = np.load(args.features, allow_pickle=True)
    cos = np.load(args.cosaliency, allow_pickle=True)

    patch_features = feat["patch_features"]  # (N, P, D)
    patch_grid = tuple(int(x) for x in feat["patch_grid"])
    image_paths = [str(x) for x in feat["image_paths"].tolist()]
    image_sizes = feat["image_sizes"]  # (N, 2) — necessário pra recriar fg_mask_pixel/seeds
    resize_info = feat["resize_info"] if "resize_info" in feat.files else None
    N, P, D = patch_features.shape

    ref_idx = find_reference_index(image_paths, args.reference_image)
    print(f"      imagem de referência: {Path(image_paths[ref_idx]).name} (índice {ref_idx})")
    ref_resize_info = resize_info[ref_idx] if resize_info is not None else None
    H_ref, W_ref = int(image_sizes[ref_idx, 0]), int(image_sizes[ref_idx, 1])
    ref_img = np.array(Image.open(image_paths[ref_idx]).convert("RGB").resize((W_ref, H_ref)))

    print(f"[2/6] Lendo os scribbles e construindo os protótipos de feature...")
    prototype_fg = marked_fg = None
    n_marked_fg = 0
    if args.scribble_mask_fg:
        prototype_fg, marked_fg, n_marked_fg = build_prototype(
            args.scribble_mask_fg, "FG", patch_features, patch_grid, ref_idx, ref_resize_info, P
        )

    prototype_bg = marked_bg = None
    n_marked_bg = 0
    if args.scribble_mask_bg:
        prototype_bg, marked_bg, n_marked_bg = build_prototype(
            args.scribble_mask_bg, "BG", patch_features, patch_grid, ref_idx, ref_resize_info, P
        )

    if prototype_fg is None and prototype_bg is None:
        print("[ERRO] Nenhum patch coberto por nenhum dos scribbles (traço vazio ou área muito pequena).")
        sys.exit(1)

    # Diagnóstico: overlay do traço cru (sem conversão pra patch) e do
    # resultado após quantização pra grade de patches, um por classe.
    # FG em verde-água, BG em rosa/magenta — as mesmas cores usadas em
    # correct.html, pra ficar consistente com o que o usuário desenhou.
    debug_dir = args.output.parent
    FG_COLOR = (70, 214, 173)
    BG_COLOR = (255, 90, 135)
    debug_images = {}
    if args.scribble_mask_fg:
        p = save_debug_overlay(args.scribble_mask_fg, None, FG_COLOR, ref_img, image_sizes,
                                resize_info, patch_grid, ref_idx, debug_dir / "scribble_fg_raw_debug.png", raw=True)
        debug_images["scribble_fg_raw_debug_image"] = p.name if p else None
        p = save_debug_overlay(args.scribble_mask_fg, marked_fg, FG_COLOR, ref_img, image_sizes,
                                resize_info, patch_grid, ref_idx, debug_dir / "scribble_fg_debug.png", raw=False)
        debug_images["scribble_fg_debug_image"] = p.name if p else None
    if args.scribble_mask_bg:
        p = save_debug_overlay(args.scribble_mask_bg, None, BG_COLOR, ref_img, image_sizes,
                                resize_info, patch_grid, ref_idx, debug_dir / "scribble_bg_raw_debug.png", raw=True)
        debug_images["scribble_bg_raw_debug_image"] = p.name if p else None
        p = save_debug_overlay(args.scribble_mask_bg, marked_bg, BG_COLOR, ref_img, image_sizes,
                                resize_info, patch_grid, ref_idx, debug_dir / "scribble_bg_debug.png", raw=False)
        debug_images["scribble_bg_debug_image"] = p.name if p else None
    print(f"      [Diagnóstico] imagens salvas: {[v for v in debug_images.values() if v]}")

    print(f"[3/6] Calculando a feature média de cada cluster global e comparando aos protótipos...")
    cluster_labels = cos["cluster_labels"]      # (N, P)
    cluster_is_fg = cos["cluster_is_fg"].copy()  # (n_clusters,) bool
    n_clusters = int(cos["n_clusters_used"])
    flat_labels = cluster_labels.reshape(-1)
    flat_features = patch_features.reshape(N * P, D)

    forced = []
    for c in range(n_clusters):
        mask_c = flat_labels == c
        if not mask_c.any():
            continue
        mean_feat = flat_features[mask_c].mean(axis=0)
        mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-8)
        sim_fg = float(np.dot(mean_feat, prototype_fg)) if prototype_fg is not None else -1.0
        sim_bg = float(np.dot(mean_feat, prototype_bg)) if prototype_bg is not None else -1.0

        # Se o cluster passa do limiar pros dois protótipos ao mesmo tempo
        # (raro), vence quem tiver maior similaridade — nunca fundimos os
        # dois numa decisão só baseada em qual checkbox estava marcado.
        source = None
        if sim_fg >= args.similarity_threshold and sim_fg >= sim_bg:
            new_val, source = True, "fg"
        elif sim_bg >= args.similarity_threshold:
            new_val, source = False, "bg"

        marker = ""
        if source is not None:
            sim = sim_fg if source == "fg" else sim_bg
            if cluster_is_fg[c] != new_val:
                forced.append({"cluster": c, "similarity": sim, "source": source,
                              "de": bool(cluster_is_fg[c]), "para": new_val})
            cluster_is_fg[c] = new_val
            marker = f"  <- FORÇADO ({source})"
        sim_str = f"sim_fg={sim_fg:.3f}" if prototype_fg is not None else "sim_fg=--"
        sim_str += f"  sim_bg={sim_bg:.3f}" if prototype_bg is not None else "  sim_bg=--"
        print(f"      cluster {c:>3d}  {sim_str}{marker}")

    if not forced:
        print(f"\n[Aviso] Nenhum cluster passou do limiar de similaridade ({args.similarity_threshold}). "
              f"Nada foi corrigido — considere baixar --similarity_threshold ou marcar uma área "
              f"mais representativa do padrão problemático.")
    else:
        print(f"\n[Correção] {len(forced)} cluster(s) alterado(s):")
        for f in forced:
            print(f"      cluster {f['cluster']} ({f['source']}): FG={f['de']} -> FG={f['para']}  "
                  f"(similaridade={f['similarity']:.3f})")

    print(f"[4/6] Reconstruindo fg_mask_patch / fg_mask_coarse com o cluster_is_fg corrigido...")
    H_p, W_p = patch_grid
    fg_mask_patch_flat = cluster_is_fg[flat_labels].astype(np.float32)
    fg_mask_patch = fg_mask_patch_flat.reshape(N, H_p, W_p)
    fg_mask_coarse = fg_mask_patch.copy()

    print(f"[5/6] Reconstruindo fg_mask_pixel / seeds (é isso que a Etapa 4 realmente lê)...")
    fg_mask_pixel = upsample_masks(fg_mask_patch, image_sizes, resize_info=resize_info)
    seeds_list = extract_seeds(fg_mask_pixel, image_sizes)
    seeds_arr = np.empty(len(seeds_list), dtype=object)
    for i, s in enumerate(seeds_list):
        seeds_arr[i] = s
    n_fg_pixels_changed = int((fg_mask_pixel != (cos["fg_mask_pixel"] > 0.5)).sum())
    print(f"      fg_mask_pixel: {n_fg_pixels_changed} pixels mudaram de FG/BG "
          f"em relação ao cosaliency.npz original")

    print(f"[6/6] Salvando {args.output}...")
    payload = {key: cos[key] for key in cos.files}
    payload["cluster_is_fg"] = cluster_is_fg
    payload["fg_mask_patch"] = fg_mask_patch
    payload["fg_mask_coarse"] = fg_mask_coarse
    payload["fg_mask_pixel"] = fg_mask_pixel
    payload["seeds"] = seeds_arr
    payload["correction_applied"] = np.array(True)
    payload["correction_n_fg_pixels_changed"] = np.array(n_fg_pixels_changed)
    payload["correction_reference_image"] = image_paths[ref_idx]
    payload["correction_n_clusters_forced"] = np.array(len(forced))
    payload["correction_similarity_threshold"] = np.array(args.similarity_threshold)

    np.savez_compressed(args.output, **payload)
    print(f"[Salvo] {args.output}  ({len(forced)} cluster(s) corrigido(s), afeta todas as {N} imagens)")

    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps({
        "reference_image": image_paths[ref_idx],
        "similarity_threshold": args.similarity_threshold,
        "n_marked_patches_fg": n_marked_fg,
        "n_marked_patches_bg": n_marked_bg,
        "clusters_forced": forced,
        "n_clusters_forced": len(forced),
        "n_fg_pixels_changed": n_fg_pixels_changed,
        **debug_images,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Salvo] {report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
