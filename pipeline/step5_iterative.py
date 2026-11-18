"""
=============================================================================
Cosegmentation Pipeline — Etapa 5: Refinamento Iterativo
=============================================================================

PAPEL DESTA ETAPA NO PIPELINE:
  Entrada : refined_masks.npz  (Etapa 4) — máscaras SAM iniciais
            features.npz       (Etapa 1) — image_paths, patch_features
  Saída   : iterative_masks.npz — máscaras convergidas + histórico

INSTALAÇÃO:
  pip install git+https://github.com/facebookresearch/sam2.git
  pip install scipy  (já requerido pelas etapas anteriores)

COMO RODAR:
  python step5_iterative.py \
    --features     features/Panda/features.npz \
    --refined      features/Panda/refined_masks.npz \
    --visualize --export

  # Controle do loop:
  python step5_iterative.py -f ... -r ... \
    --max_iter 5 --converge_thresh 0.97 --prompt_mode box_points_mask

O QUE ESTA ETAPA FAZ E POR QUE É DIFERENTE DA ETAPA 4:

  Contexto: este é o "Cosegmentation Refinement" descrito no paper original
  (Lattari et al., ICIP 2015), atualizado para o pipeline moderno.

  No paper original:
    iter 1: saliência = Cheng et al. → Global Clustering → Graph Cuts → mask_1
    iter 2: saliência = mask_1       → Global Clustering → Graph Cuts → mask_2
    (a máscara anterior substituía o mapa de saliência)

  Neste pipeline:
    iter 0: seeds coarse (E3) → SAM 2 → mask_0          [= resultado da E4]
    iter 1: seeds refinadas de mask_0 → SAM 2 → mask_1
    iter 2: seeds refinadas de mask_1 → SAM 2 → mask_2
    ...
    para quando mask_k ≈ mask_{k-1}  (convergência)

  O que muda a cada iteração:

  1. SEEDS PIXEL-PERFECT
     Na E4 as seeds (bbox, pontos FG/BG) vinham do clustering em patch space
     (resolução 37×37). Na E5 elas são derivadas diretamente da máscara SAM
     da iteração anterior, que é pixel-perfect.
     → bbox mais justa, pontos FG no interior seguro (erosão morfológica),
       pontos BG no exterior seguro (dilatação + inversão).

  2. MASK_INPUT PROGRESSIVO
     A low_res_mask (256×256 logits) gerada pelo SAM na iteração k é passada
     como mask_input na iteração k+1. O SAM foi treinado para aproveitar
     esse sinal — é o mecanismo nativo de refinamento do SAM 2.

  3. CRITÉRIO DE CONVERGÊNCIA AUTOMÁTICO
     A cada iteração calcula o IoU médio entre mask_k e mask_{k-1}.
     Se IoU > converge_thresh (default 0.95) → para.
     Evita iterações desnecessárias e detecta quando o pipeline estabilizou.

  4. TRACKING COMPLETO POR ITERAÇÃO
     Guarda métricas de cada iteração para análise de convergência:
     iou_sam, stability (IoU inter-iter), fg_ratio, fg_coherence.

SAÍDA (iterative_masks.npz):
  final_masks     : (N, H, W)        — máscaras da última iteração
  iou_scores      : (N,)             — IoU SAM da última iteração
  history         : list[dict]       — métricas por iteração
  n_iters_run     : int              — iterações efetivamente executadas
  converged       : bool             — parou por convergência ou max_iter
  image_paths     : lista de paths
  image_sizes     : (N, 2)

=============================================================================
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Interfaces de carga (Etapas 1 e 4)
# ---------------------------------------------------------------------------

def load_step1(path: Path) -> dict:
    if not path.exists():
        print(f"[ERRO] features.npz não encontrado: {path}")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    return {
        "patch_features": data["patch_features"],
        "patch_grid"    : tuple(int(x) for x in data["patch_grid"]),
        "image_paths"   : list(data["image_paths"]),
        "image_sizes"   : data["image_sizes"],
        # Necessário para projetar corretamente a máscara de volta ao grid de
        # patches em compute_fg_coherence (modo keep_ar) — mesmo padrão de
        # step4_sam_refinement.py. Ausente em features.npz mais antigos.
        "resize_info": (
            data["resize_info"]
            if "resize_info" in data.files
            else None
        ),
    }


def load_step4(path: Path) -> dict:
    if not path.exists():
        print(f"[ERRO] refined_masks.npz não encontrado: {path}")
        print("       Rode: python step4_sam_refinement.py --features ... --cosaliency ...")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    return {
        "refined_masks" : data["refined_masks"],    # (N, H, W)
        "iou_scores_sam": data["iou_scores_sam"],   # (N,)
        "low_res_masks" : data["low_res_masks"],    # (N, 256, 256)
        "image_paths"   : list(data["image_paths"]),
        "image_sizes"   : data["image_sizes"],
    }


# ---------------------------------------------------------------------------
# Carregamento do SAM 2
# ---------------------------------------------------------------------------

def load_sam2(model_id: str) -> object:
    print(f"\n[SAM 2] Carregando {model_id}...")
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("\n[ERRO] SAM 2 não instalado.")
        print("         pip install git+https://github.com/facebookresearch/sam2.git")
        sys.exit(1)
    try:
        predictor = SAM2ImagePredictor.from_pretrained(model_id)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM 2] Pronto ({device})")
    return predictor


# ---------------------------------------------------------------------------
# Derivação de seeds pixel-perfect a partir de máscara SAM
# ---------------------------------------------------------------------------

def mask_to_seeds(
    mask     : np.ndarray,   # (H, W) float32 {0,1}
    n_fg     : int = 5,
    n_bg     : int = 5,
    erosion_px  : int = 12,  # raio de erosão para pontos FG seguros
    dilation_px : int = 12,  # raio de dilatação para pontos BG seguros
) -> dict | None:
    """
    Deriva seeds pixel-perfect a partir de uma máscara SAM.

    Diferença fundamental em relação às seeds da Etapa 3:
    - Na E3: seeds vinham do clustering em patch space (37×37 pixels)
    - Na E5: seeds vêm diretamente da máscara SAM (resolução full)

    Estratégia:
    - FG points: erosão morfológica da máscara → pontos no interior seguro,
                  longe das bordas onde a máscara é menos confiável
    - BG points: dilatação + inversão → pontos no exterior seguro,
                  suficientemente afastados do objeto
    - bbox: caixa delimitadora exata dos pixels FG

    NOTA (correção): erosion_px e dilation_px agora geram cada um o seu
    próprio kernel circular. Antes, o kernel usado em binary_dilation()
    era construído só a partir de erosion_px -- dilation_px existia na
    assinatura mas nunca era lido, então passar um valor diferente pra
    ele não tinha efeito nenhum (silenciosamente). Com os defaults
    atuais (12 e 12) o comportamento numérico não muda; a diferença só
    aparece se algum dia os dois raios forem calibrados para valores
    distintos.

    Retorna None se a máscara estiver vazia.
    """
    mask_bool = mask > 0.5
    if not mask_bool.any():
        return None

    # Kernels morfológicos circulares (um para cada raio -- ver nota abaixo)
    def _circular_kernel(radius: int) -> np.ndarray:
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        return x**2 + y**2 <= radius**2

    erosion_kernel  = _circular_kernel(erosion_px)
    dilation_kernel = _circular_kernel(dilation_px)

    # FG seguro: interior da máscara (eroded)
    fg_safe = binary_erosion(mask_bool, structure=erosion_kernel)
    if not fg_safe.any():
        fg_safe = mask_bool  # fallback: sem erosão

    # BG seguro: exterior expandido (dilated + inverted)
    bg_safe = ~binary_dilation(mask_bool, structure=dilation_kernel)

    def _sample(pts_arr: np.ndarray, n: int) -> list:
        if not len(pts_arr):
            return []
        idx = np.linspace(0, len(pts_arr) - 1, min(n, len(pts_arr)), dtype=int)
        return pts_arr[idx].tolist()   # lista de [row, col]

    fg_pts  = np.argwhere(fg_safe)
    bg_pts  = np.argwhere(bg_safe)
    fg_all  = np.argwhere(mask_bool)
    r0, c0  = fg_all.min(axis=0)
    r1, c1  = fg_all.max(axis=0)

    return {
        "bbox"      : [int(c0), int(r0), int(c1), int(r1)],
        "fg_points" : _sample(fg_pts, n_fg),
        "bg_points" : _sample(bg_pts, n_bg),
        "fg_ratio"  : float(mask_bool.mean()),
    }


# ---------------------------------------------------------------------------
# Critério de convergência
# ---------------------------------------------------------------------------

def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """
    IoU entre duas máscaras binárias.
    Usado como métrica de estabilidade entre iterações consecutivas:
    IoU(mask_{k-1}, mask_k) → 1.0 quando convergiu.
    """
    b1    = m1 > 0.5
    b2    = m2 > 0.5
    inter = (b1 & b2).sum()
    union = (b1 | b2).sum()
    return float(inter / union) if union > 0 else 1.0


def check_convergence(
    prev_masks : np.ndarray,   # (N, H, W)
    curr_masks : np.ndarray,   # (N, H, W)
    threshold  : float,
) -> tuple[bool, float]:
    """
    Retorna (convergiu, stability_mean).
    stability_mean = IoU médio entre máscaras consecutivas sobre toda a coleção.
    """
    stabilities   = [mask_iou(prev_masks[i], curr_masks[i])
                     for i in range(len(prev_masks))]
    stability_mean = float(np.mean(stabilities))
    return stability_mean >= threshold, stability_mean


# ---------------------------------------------------------------------------
# Métricas de qualidade
# ---------------------------------------------------------------------------
#
# NOTA (correção): compute_fg_coherence NÃO é mais definida localmente.
# A versão antiga deste arquivo comparava patches individuais (não um
# protótipo por imagem), agrupava TODOS os patches de TODAS as imagens
# antes de calcular similaridade (sem excluir pares da mesma imagem — só
# a diagonal, patch contra ele mesmo, era excluída) e ignorava
# image_sizes/resize_info ao projetar a máscara de volta ao grid de
# patches — apesar do docstring afirmar "Igual à E4". Não era: a
# construção real da E4 é um protótipo por imagem (média L2-normalizada
# dos patches FG, normalizada de novo), comparado só entre pares de
# imagens DIFERENTES. Como fg_coherence é um diagnóstico sem ground
# truth, sem nenhum papel na decisão de convergência (isso é
# check_convergence/mask_iou, abaixo) ou na seleção de configuração nos
# sweeps de ablação (que usam IoU contra GT), esse bug nunca afetou
# nenhum resultado numérico reportado no artigo — só o valor de
# fg_coherence em si. Corrigido importando a função de verdade de
# step4_sam_refinement.py (ver run_iterative), o que também garante que
# o delta fg_coherence_e4→e5 seja uma comparação legítima, calculada com
# a mesma fórmula dos dois lados.


# ---------------------------------------------------------------------------
# Pipeline iterativo
# ---------------------------------------------------------------------------

def run_iterative(
    predictor      : object,
    image_paths    : list[str],
    image_sizes    : np.ndarray,
    init_masks     : np.ndarray,    # (N, H, W) — resultado da E4
    init_iou       : np.ndarray,    # (N,)
    init_low_res   : np.ndarray,    # (N, 256, 256)
    patch_features : np.ndarray,    # (N, P, D)
    patch_grid     : tuple,
    resize_info    : "np.ndarray | None" = None,  # (N, 6), modo keep_ar — ver load_step1
    prompt_mode    : str   = "box_points_mask",
    max_iter       : int   = 4,
    converge_thresh: float = 0.95,
    n_fg_pts       : int   = 5,
    n_bg_pts       : int   = 5,
    save_mask_history: bool = True,
) -> dict:
    """
    Loop de refinamento iterativo.

    A cada iteração:
      1. Deriva seeds pixel-perfect da máscara atual
      2. Monta prompts SAM (com mask_input da low_res anterior)
      3. Infere nova máscara com SAM 2
      4. Verifica convergência
      5. Registra métricas

    Se save_mask_history=True (default), guarda uma cópia da máscara de
    CADA iteração (não só a final) em mask_history, shape
    (n_iters_run+1, N, H, W) — iteração 0 = estado inicial (E4). Isso
    permite, numa única execução com max_iter alto, avaliar o IoU contra
    GT em cada passo intermediário depois (via analisador externo), sem
    precisar re-rodar o SAM para cada valor de max_iter/converge_thresh
    testado — o gargalo de custo aqui é a inferência do SAM, não o
    armazenamento.
    """
    # Importa da E4 para reutilizar: conversão de prompts, inferência SAM 2,
    # e o cálculo de fg_coherence (protótipo por imagem — ver nota acima de
    # compute_fg_coherence sobre por que a versão local foi removida).
    try:
        from step4_sam_refinement import (
            seeds_to_sam_prompts, predict_one, compute_fg_coherence,
        )
    except ImportError:
        print("[ERRO] step4_sam_refinement.py não encontrado no mesmo diretório.")
        sys.exit(1)

    N = len(image_paths)

    # Estado corrente (começa com o resultado da E4 = iteração 0)
    curr_masks   = init_masks.copy()
    curr_iou     = init_iou.copy()
    curr_low_res = init_low_res.copy()

    mask_history = [curr_masks.copy()] if save_mask_history else None

    # Histórico de métricas: iteração 0 = estado inicial da E4
    coh_0 = compute_fg_coherence(
        patch_features, curr_masks, patch_grid, image_sizes, resize_info
    )
    history = [{
        "iter"       : 0,
        "iou_mean"   : float(curr_iou.mean()),
        "iou_std"    : float(curr_iou.std()),
        "stability"  : 1.0,     # referência: máscara inicial vs ela mesma
        "fg_ratio"   : float(curr_masks.mean() * 100),
        "fg_coherence": coh_0,
        "converged"  : False,
    }]

    print(f"\n[E5] Estado inicial (Etapa 4):")
    print(f"     IoU_SAM={curr_iou.mean():.3f}  "
          f"FG%={curr_masks.mean()*100:.1f}%  "
          f"coherence={coh_0:.4f}")
    print(f"\n[E5] Iniciando loop iterativo (max={max_iter}, thresh={converge_thresh})...")

    n_iters_run = 0
    converged   = False

    for it in range(1, max_iter + 1):
        prev_masks = curr_masks.copy()
        new_masks   = np.zeros_like(curr_masks)
        new_ious    = np.zeros(N, dtype=np.float32)
        new_low_res = np.zeros((N, 256, 256), dtype=np.float32)

        print(f"\n  ── Iteração {it}/{max_iter} " + "─" * 40)

        for i, img_path in enumerate(image_paths):
            H_i = int(image_sizes[i, 0])
            W_i = int(image_sizes[i, 1])

            # Seeds pixel-perfect da máscara atual
            seeds = mask_to_seeds(
                curr_masks[i, :H_i, :W_i],
                n_fg=n_fg_pts, n_bg=n_bg_pts
            )

            if seeds is None:
                # Máscara vazia — mantém e avisa
                new_masks[i] = curr_masks[i]
                new_ious[i]  = curr_iou[i]
                print(f"  [{i+1:>3}/{N}] {Path(img_path).name:35s}  "
                      f"[AVISO: máscara vazia — mantida]")
                continue

            # Prompts: seeds refinadas + low_res da iteração anterior
            prompts = seeds_to_sam_prompts(
                seeds, curr_masks[i, :H_i, :W_i], prompt_mode
            )
            # Injeta low_res da iteração anterior como mask_input
            # (mesmo se prompt_mode não for box_points_mask — aproveita
            #  o canal nativo de refinamento do SAM 2)
            if "mask_input" not in prompts:
                prompts["mask_input"] = curr_low_res[i][np.newaxis, :, :]

            # Carrega imagem
            image_rgb = np.array(Image.open(img_path).convert("RGB"))

            try:
                best_mask, best_iou, low_res = predict_one(
                    predictor, image_rgb, prompts, n_iter=1
                )
            except Exception as e:
                print(f"  [{i+1:>3}/{N}] AVISO: {e} — mantendo máscara anterior")
                new_masks[i]  = curr_masks[i]
                new_ious[i]   = curr_iou[i]
                continue

            new_masks[i, :H_i, :W_i] = best_mask.astype(np.float32)
            new_ious[i]               = best_iou
            new_low_res[i]            = low_res

            delta_pct = (new_masks[i].mean() - curr_masks[i].mean()) * 100
            print(f"  [{i+1:>3}/{N}] {Path(img_path).name:35s}  "
                  f"IoU={best_iou:.3f}  "
                  f"FG%={new_masks[i].mean()*100:.0f}%  "
                  f"Δ={delta_pct:+.1f}%")

        # Atualiza estado
        curr_masks   = new_masks
        curr_iou     = new_ious
        curr_low_res = new_low_res
        n_iters_run  = it

        if save_mask_history:
            mask_history.append(curr_masks.copy())

        # Convergência
        converged, stability = check_convergence(
            prev_masks, curr_masks, converge_thresh
        )
        coh = compute_fg_coherence(
            patch_features, curr_masks, patch_grid, image_sizes, resize_info
        )

        entry = {
            "iter"        : it,
            "iou_mean"    : float(curr_iou.mean()),
            "iou_std"     : float(curr_iou.std()),
            "stability"   : stability,
            "fg_ratio"    : float(curr_masks.mean() * 100),
            "fg_coherence": coh,
            "converged"   : converged,
        }
        history.append(entry)

        print(f"\n  → stability={stability:.4f}  "
              f"IoU={entry['iou_mean']:.3f}  "
              f"FG%={entry['fg_ratio']:.1f}%  "
              f"coherence={coh:.4f}", end="")

        if converged:
            print(f"  ✓ CONVERGIU (≥ {converge_thresh})")
            break
        else:
            print()

    if not converged:
        print(f"\n[E5] Atingido max_iter={max_iter} sem convergência total.")

    # Ganho total E4→E5
    iou_gain = history[-1]["iou_mean"] - history[0]["iou_mean"]
    coh_gain = history[-1]["fg_coherence"] - history[0]["fg_coherence"]
    print(f"\n[E5] Ganho E4→E5: IoU {iou_gain:+.3f}  coherence {coh_gain:+.4f}")

    return {
        "final_masks"  : curr_masks,        # (N, H, W)
        "iou_scores"   : curr_iou,          # (N,)
        "low_res_masks": curr_low_res,      # (N, 256, 256)
        "history"      : history,
        "n_iters_run"  : n_iters_run,
        "converged"    : converged,
        "iou_gain"     : iou_gain,          # ganho total E4→E5, IoU SAM
        "coh_gain"     : coh_gain,          # ganho total E4→E5, fg_coherence
        "mask_history" : np.stack(mask_history, axis=0) if save_mask_history else None,
        # shape (n_iters_run+1, N, H, W) — índice 0 = estado inicial (E4)
    }


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def save_iterative(
    result      : dict,
    output_dir  : Path,
    image_paths : list[str],
    image_sizes : np.ndarray,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "iterative_masks.npz"

    # Serializa histórico como object array
    history_arr = np.empty(len(result["history"]), dtype=object)
    for i, h in enumerate(result["history"]):
        history_arr[i] = h

    npz_kwargs = dict(
        final_masks    = result["final_masks"],
        iou_scores     = result["iou_scores"],
        low_res_masks  = result["low_res_masks"],
        history        = history_arr,
        n_iters_run    = np.array(result["n_iters_run"]),
        converged      = np.array(result["converged"]),
        iou_gain       = np.array(result["iou_gain"]),
        coh_gain       = np.array(result["coh_gain"]),
        image_paths    = image_paths,
        image_sizes    = image_sizes,
    )
    if result.get("mask_history") is not None:
        npz_kwargs["mask_history"] = result["mask_history"]

    np.savez_compressed(out, **npz_kwargs)
    print(f"\n[Salvo] {out}  ({out.stat().st_size/1024**2:.1f} MB)")
    return out


def load_iterative(path: Path) -> dict:
    """Interface de carregamento para análise pós-pipeline."""
    data = np.load(path, allow_pickle=True)
    history = list(data["history"])

    # iou_gain/coh_gain: presentes em arquivos gerados a partir desta
    # correção; para arquivos mais antigos, recalcula a partir do
    # history (iteração final − iteração 0), que sempre esteve presente.
    if "iou_gain" in data.files:
        iou_gain = float(data["iou_gain"])
        coh_gain = float(data["coh_gain"])
    else:
        iou_gain = float(history[-1]["iou_mean"] - history[0]["iou_mean"])
        coh_gain = float(history[-1]["fg_coherence"] - history[0]["fg_coherence"])

    return {
        "final_masks"  : data["final_masks"],
        "iou_scores"   : data["iou_scores"],
        "low_res_masks": data["low_res_masks"],
        "history"      : history,
        "n_iters_run"  : int(data["n_iters_run"]),
        "converged"    : bool(data["converged"]),
        "iou_gain"     : iou_gain,
        "coh_gain"     : coh_gain,
        "image_paths"  : list(data["image_paths"]),
        "image_sizes"  : data["image_sizes"],
        # mask_history só existe em arquivos gerados com save_mask_history=True
        # (o default a partir de agora); None para arquivos antigos.
        "mask_history" : data["mask_history"] if "mask_history" in data.files else None,
    }


# ---------------------------------------------------------------------------
# Sumário
# ---------------------------------------------------------------------------

def print_summary(result: dict, image_paths: list[str]):
    masks   = result["final_masks"]
    ious    = result["iou_scores"]
    history = result["history"]
    N       = len(image_paths)

    print("\n" + "=" * 68)
    print("  SUMÁRIO ETAPA 5")
    print("=" * 68)
    print(f"  Iterações rodadas : {result['n_iters_run']}  "
          f"({'convergiu' if result['converged'] else 'atingiu max_iter'})")
    print()
    # Tabela de convergência
    print(f"  {'ITER':>5}  {'IoU_SAM':>8}  {'STABILITY':>10}  {'FG%':>6}  {'COHERENCE':>10}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*6}  {'─'*10}")
    for h in history:
        conv_mark = " ✓" if h["converged"] else ""
        stab_str  = f"{h['stability']:.4f}" if h["stability"] is not None else "  —   "
        print(f"  {h['iter']:>5}  {h['iou_mean']:>8.3f}  "
              f"{stab_str:>10}  {h['fg_ratio']:>5.1f}%  "
              f"{h['fg_coherence']:>10.4f}{conv_mark}")
    print()
    # Por imagem
    print(f"  {'IMG':>4}  {'IoU':>6}  {'FG%':>6}  NOME")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*35}")
    for i in range(N):
        pct  = masks[i].mean() * 100
        bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        name = Path(image_paths[i]).name[:35]
        print(f"  {i:>4}  {ious[i]:>6.3f}  {pct:>5.1f}%  {bar}  {name}")
    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# Exportação de imagens individuais
# ---------------------------------------------------------------------------

def export_segmented_images(
    result      : dict,
    image_paths : list[str],
    image_sizes : np.ndarray,
    output_dir  : Path,
):
    """
    Salva cada imagem segmentada individualmente em output_dir/segmented/.

    Três arquivos por imagem:
      <nome>_overlay.jpg   — imagem + máscara verde semitransparente
      <nome>_mask.png      — máscara binária pura (branco=FG, preto=BG)
      <nome>_cutout.png    — objeto recortado com fundo transparente (RGBA)
    """
    seg_dir = output_dir / "segmented_e5"
    seg_dir.mkdir(parents=True, exist_ok=True)

    masks = result["final_masks"]
    ious  = result["iou_scores"]
    N     = len(image_paths)

    print(f"\n[Export] Salvando {N} imagens segmentadas em {seg_dir}/")

    for i, img_path in enumerate(image_paths):
        H_i = int(image_sizes[i, 0])
        W_i = int(image_sizes[i, 1])

        stem    = Path(img_path).stem
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        mask    = masks[i, :H_i, :W_i]
        mask_u8 = (mask * 255).astype(np.uint8)

        # 1. Overlay
        overlay = img_arr.astype(np.float32)
        green   = np.zeros_like(img_arr, dtype=np.float32)
        green[:, :, 1] = 255.0
        alpha   = mask[:, :, None] * 0.5
        overlay = (overlay * (1 - alpha) + green * alpha).clip(0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(seg_dir / f"{stem}_overlay.jpg", quality=92)

        # 2. Máscara binária
        Image.fromarray(mask_u8).save(seg_dir / f"{stem}_mask.png")

        # 3. Cutout RGBA
        rgba            = np.zeros((H_i, W_i, 4), dtype=np.uint8)
        rgba[:, :, :3]  = img_arr
        rgba[:, :,  3]  = mask_u8
        Image.fromarray(rgba, mode="RGBA").save(seg_dir / f"{stem}_cutout.png")

        print(f"  [{i+1:>3}/{N}] {stem:35s}  IoU={ious[i]:.3f}  FG%={mask.mean()*100:.0f}%")

    print(f"[Export] Concluído → {seg_dir}/  ({N*3} arquivos)")
    return seg_dir


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def visualize_iterative(
    result         : dict,
    init_masks     : np.ndarray,   # (N, H, W) — E4 para comparação
    image_paths    : list[str],
    image_sizes    : np.ndarray,
    output_dir     : Path,
    max_images     : int = 8,
):
    """
    Quatro visualizações:

    1. Curvas de convergência
       IoU SAM, stability e fg_coherence por iteração.
       Permite ver em quantas iterações o pipeline estabilizou
       e se houve ganho real de qualidade.

    2. Comparação E4 → E5 final (por imagem)
       Diferença entre a máscara inicial (E4) e a máscara convergida (E5).
       Regiões verdes = FG novo adicionado pelo refinamento.
       Regiões vermelhas = FG removido.

    3. Overlay final sobre imagem original
       Resultado final do pipeline completo.

    4. Grade de máscaras finais
       Todas as imagens lado a lado para avaliação de consistência.
    """
    final   = result["final_masks"]   # (N, H, W)
    ious    = result["iou_scores"]
    history = result["history"]
    N_vis   = min(len(image_paths), max_images)

    # ── Fig 1: Curvas de convergência ────────────────────────────────────────
    iters   = [h["iter"]         for h in history]
    iou_m   = [h["iou_mean"]     for h in history]
    stabs   = [h["stability"]    for h in history]
    cohs    = [h["fg_coherence"] for h in history]
    fg_rs   = [h["fg_ratio"]     for h in history]

    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))
    n_iters = result["n_iters_run"]
    conv_str = "convergiu" if result["converged"] else f"max_iter={n_iters}"
    fig1.suptitle(
        f"Etapa 5 — Convergência  ({conv_str} em {n_iters} iterações)",
        fontsize=11
    )

    axes1[0].plot(iters, iou_m, "o-", color="#2ecc71", lw=2, ms=8, label="IoU SAM")
    axes1[0].fill_between(iters,
                          [v - h["iou_std"] for h, v in zip(history, iou_m)],
                          [v + h["iou_std"] for h, v in zip(history, iou_m)],
                          alpha=0.2, color="#2ecc71")
    axes1[0].set_xlabel("Iteração"); axes1[0].set_ylabel("IoU SAM (médio)")
    axes1[0].set_title("IoU SAM por iteração"); axes1[0].grid(True, alpha=0.3)
    axes1[0].set_xticks(iters)

    axes1[1].plot(iters, stabs, "s--", color="#3498db", lw=2, ms=8, label="stability")
    axes1[1].axhline(0.95, color="gray", ls=":", lw=1, label="thresh=0.95")
    axes1[1].set_xlabel("Iteração"); axes1[1].set_ylabel("IoU(mask_k, mask_{k-1})")
    axes1[1].set_title("Estabilidade entre iterações"); axes1[1].grid(True, alpha=0.3)
    axes1[1].set_xticks(iters); axes1[1].legend(fontsize=8)
    axes1[1].set_ylim(0, 1.05)

    ax_r = axes1[2]
    ax_c = ax_r.twinx()
    ax_r.plot(iters, fg_rs,  "^-",  color="#e67e22", lw=2, ms=8, label="FG%")
    ax_c.plot(iters, cohs,   "D--", color="#9b59b6", lw=2, ms=8, label="coherence")
    ax_r.set_xlabel("Iteração"); ax_r.set_ylabel("FG% médio", color="#e67e22")
    ax_c.set_ylabel("FG coherence", color="#9b59b6")
    ax_r.set_title("FG ratio e coesão cross-image"); ax_r.grid(True, alpha=0.3)
    ax_r.set_xticks(iters)
    lines1, labs1 = ax_r.get_legend_handles_labels()
    lines2, labs2 = ax_c.get_legend_handles_labels()
    ax_r.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    fig1.tight_layout()
    out1 = output_dir / "viz_convergence.png"
    fig1.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig1)

    # ── Fig 2: Comparação E4 → E5 ────────────────────────────────────────────
    fig2, axes2 = plt.subplots(N_vis, 4, figsize=(16, 3.8 * N_vis))
    if N_vis == 1:
        axes2 = axes2[np.newaxis, :]
    fig2.suptitle(
        "E4 → E5: Comparação de máscaras\n"
        "Col1: original  Col2: E4 inicial  Col3: E5 final  Col4: diferença (verde=ganho, vermelho=perda)",
        fontsize=10, y=1.01
    )
    for i in range(N_vis):
        H_i = int(image_sizes[i, 0])
        W_i = int(image_sizes[i, 1])
        img = Image.open(image_paths[i]).convert("RGB")

        e4_i  = init_masks[i,  :H_i, :W_i]
        e5_i  = final[i,       :H_i, :W_i]

        # Diferença: +1=FG ganho, -1=FG perdido, 0=sem mudança
        diff = (e5_i > 0.5).astype(float) - (e4_i > 0.5).astype(float)

        axes2[i, 0].imshow(img); axes2[i, 0].axis("off")
        axes2[i, 0].set_title(Path(image_paths[i]).name[:22], fontsize=7)

        axes2[i, 1].imshow(e4_i, cmap="RdYlGn", vmin=0, vmax=1)
        axes2[i, 1].set_title(f"E4  FG%={e4_i.mean()*100:.0f}%", fontsize=7)
        axes2[i, 1].axis("off")

        axes2[i, 2].imshow(e5_i, cmap="RdYlGn", vmin=0, vmax=1)
        axes2[i, 2].set_title(f"E5  IoU={ious[i]:.3f}  FG%={e5_i.mean()*100:.0f}%", fontsize=7)
        axes2[i, 2].axis("off")

        axes2[i, 3].imshow(diff, cmap="RdYlGn", vmin=-1, vmax=1, interpolation="nearest")
        n_gain = (diff > 0).sum(); n_loss = (diff < 0).sum()
        axes2[i, 3].set_title(f"Δ: +{n_gain}px  -{n_loss}px", fontsize=7)
        axes2[i, 3].axis("off")

    fig2.tight_layout()
    out2 = output_dir / "viz_e4_vs_e5.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # ── Fig 3: Overlay final ─────────────────────────────────────────────────
    fig3, axes3 = plt.subplots(N_vis, 2, figsize=(10, 3.8 * N_vis))
    if N_vis == 1:
        axes3 = axes3[np.newaxis, :]
    fig3.suptitle(
        "Cosegmentação Final — Etapa 5 (pipeline completo)\n"
        "Verde = foreground co-segmentado",
        fontsize=11, y=1.01
    )
    for i in range(N_vis):
        H_i = int(image_sizes[i, 0])
        W_i = int(image_sizes[i, 1])
        img_arr = np.array(Image.open(image_paths[i]).convert("RGB"))
        mask_i  = final[i, :H_i, :W_i]

        axes3[i, 0].imshow(img_arr); axes3[i, 0].axis("off")
        axes3[i, 0].set_title(Path(image_paths[i]).name[:28], fontsize=7)

        overlay = img_arr.astype(np.float32)
        green   = np.zeros_like(img_arr, dtype=np.float32); green[:, :, 1] = 255
        alpha   = mask_i[:, :, None] * 0.5
        overlay = (overlay*(1-alpha) + green*alpha).clip(0,255).astype(np.uint8)
        axes3[i, 1].imshow(overlay); axes3[i, 1].axis("off")
        axes3[i, 1].set_title(f"FG={mask_i.mean()*100:.0f}%  IoU={ious[i]:.3f}", fontsize=7)

    fig3.tight_layout()
    out3 = output_dir / "viz_overlay_final.png"
    fig3.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig3)

    # ── Fig 4: Grade de máscaras finais ──────────────────────────────────────
    n_cols  = 4
    n_rows  = (N_vis + n_cols - 1) // n_cols
    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows))
    axes4_flat  = np.array(axes4).flatten()
    fig4.suptitle(
        "Grade de máscaras finais — E5\n"
        "Consistência visual = boa cosegmentação",
        fontsize=10
    )
    for idx in range(len(axes4_flat)):
        ax = axes4_flat[idx]
        if idx >= N_vis:
            ax.axis("off"); continue
        H_i = int(image_sizes[idx, 0]); W_i = int(image_sizes[idx, 1])
        img_arr = np.array(Image.open(image_paths[idx]).convert("RGB"))
        mask_i  = final[idx, :H_i, :W_i]
        overlay = img_arr.astype(np.float32)
        green   = np.zeros_like(img_arr, dtype=np.float32); green[:,:,1]=255
        alpha   = mask_i[:,:,None] * 0.45
        overlay = (overlay*(1-alpha)+green*alpha).clip(0,255).astype(np.uint8)
        ax.imshow(overlay); ax.axis("off")
        ax.set_title(f"[{idx}] IoU={ious[idx]:.2f}  FG={mask_i.mean()*100:.0f}%", fontsize=7)

    fig4.tight_layout()
    out4 = output_dir / "viz_mask_grid.png"
    fig4.savefig(out4, dpi=120, bbox_inches="tight")
    plt.close(fig4)

    outs = [out1, out2, out3, out4]
    print("\n[Viz]")
    for o in outs:
        if o.exists():
            print(f"  {o.name}  ({o.stat().st_size // 1024} KB)")
    return outs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Coseg Etapa 5 — refinamento iterativo com SAM 2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--features",    "-f", required=True,
                   help="features.npz da Etapa 1")
    p.add_argument("--refined",     "-r", required=True,
                   help="refined_masks.npz da Etapa 4")
    p.add_argument("--output",      "-o", default=None,
                   help="Pasta de saída (default: mesma do features.npz)")
    p.add_argument("--model",       default="facebook/sam2.1-hiera-small",
                   choices=[
                       "facebook/sam2.1-hiera-tiny",
                       "facebook/sam2.1-hiera-small",
                       "facebook/sam2.1-hiera-base-plus",
                       "facebook/sam2.1-hiera-large",
                       "facebook/sam2-hiera-tiny",
                       "facebook/sam2-hiera-small",
                       "facebook/sam2-hiera-base-plus",
                       "facebook/sam2-hiera-large",
                   ])
    p.add_argument("--prompt_mode", default="box_points_mask",
                   choices=["box_only", "box_points", "box_points_mask"],
                   help="Estratégia de prompting (default: box_points_mask)")
    p.add_argument("--max_iter",         type=int,   default=4,
                   help="Máximo de iterações (default: 4)")
    p.add_argument("--converge_thresh",  type=float, default=0.95,
                   help="IoU inter-iteração para parada (default: 0.95)")
    p.add_argument("--n_fg_pts",         type=int,   default=5,
                   help="Pontos FG por imagem (default: 5)")
    p.add_argument("--n_bg_pts",         type=int,   default=5,
                   help="Pontos BG por imagem (default: 5)")
    p.add_argument("--no_mask_history",  action="store_true",
                   help="Não salva a máscara de cada iteração em mask_history "
                        "(só a final). Por padrão salva todas — permite avaliar "
                        "IoU contra GT em cada passo de uma única execução, sem "
                        "precisar rodar de novo para cada max_iter/converge_thresh "
                        "testado. Desative só se o tamanho do .npz for um problema.")
    p.add_argument("--visualize",  action="store_true",
                   help="Gera visualizações (incluindo curvas de convergência)")
    p.add_argument("--viz_max",    type=int, default=8)
    p.add_argument("--export",     action="store_true",
                   help="Salva cada imagem segmentada individualmente em segmented/")
    return p.parse_args()


def main():
    args = parse_args()

    feat_path  = Path(args.features)
    ref_path   = Path(args.refined)
    output_dir = Path(args.output) if args.output else feat_path.parent

    print(f"\n{'='*68}")
    print(f"  Coseg — Etapa 5: Refinamento Iterativo")
    print(f"{'='*68}")

    step1 = load_step1(feat_path)
    step4 = load_step4(ref_path)

    assert step1["image_paths"] == step4["image_paths"], (
        "[ERRO] image_paths da E1 e E4 não coincidem."
    )

    image_paths    = step1["image_paths"]
    image_sizes    = step1["image_sizes"]
    patch_features = step1["patch_features"]
    patch_grid     = step1["patch_grid"]
    resize_info    = step1.get("resize_info")
    init_masks     = step4["refined_masks"]
    init_iou       = step4["iou_scores_sam"]
    init_low_res   = step4["low_res_masks"]

    N = len(image_paths)
    print(f"\n  Etapa 1   : {feat_path}")
    print(f"  Etapa 4   : {ref_path}")
    print(f"  Imagens   : {N}")
    print(f"  Modelo    : {args.model}")
    print(f"  Prompts   : {args.prompt_mode}")
    print(f"  Max iter  : {args.max_iter}")
    print(f"  Convergência: IoU ≥ {args.converge_thresh}")
    print(f"  Saída     : {output_dir}")

    predictor = load_sam2(args.model)

    result = run_iterative(
        predictor       = predictor,
        image_paths     = image_paths,
        image_sizes     = image_sizes,
        init_masks      = init_masks,
        init_iou        = init_iou,
        init_low_res    = init_low_res,
        patch_features  = patch_features,
        patch_grid      = patch_grid,
        resize_info     = resize_info,
        prompt_mode     = args.prompt_mode,
        max_iter        = args.max_iter,
        converge_thresh = args.converge_thresh,
        n_fg_pts        = args.n_fg_pts,
        n_bg_pts        = args.n_bg_pts,
        save_mask_history = not args.no_mask_history,
    )

    save_iterative(result, output_dir, image_paths, image_sizes)
    print_summary(result, image_paths)

    if args.export:
        export_segmented_images(result, image_paths, image_sizes, output_dir)

    if args.visualize:
        visualize_iterative(
            result, init_masks, image_paths, image_sizes,
            output_dir, max_images=args.viz_max
        )

    print(f"\n[Etapa 5 concluída]  →  {output_dir}/iterative_masks.npz")
    print(f"\n  Pipeline completo:")
    print(f"    E1 features.npz")
    print(f"    E2 correspondence.npz")
    print(f"    E3 cosaliency.npz")
    print(f"    E4 refined_masks.npz")
    print(f"    E5 iterative_masks.npz  ← resultado final")


if __name__ == "__main__":
    main()