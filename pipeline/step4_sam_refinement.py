"""
=============================================================================
Cosegmentation Pipeline — Etapa 4: Refinamento com SAM 2
=============================================================================

PAPEL DESTA ETAPA NO PIPELINE:
  Entrada : cosaliency.npz  (Etapa 3) — máscaras coarse + seeds
            features.npz    (Etapa 1) — image_paths + patch_features
  Saída   : refined_masks.npz — segmentações precisas em nível de pixel

INSTALAÇÃO DO SAM 2:
  pip install git+https://github.com/facebookresearch/sam2.git

  Na primeira execução os pesos são baixados automaticamente via HuggingFace.
  Modelos disponíveis (do mais leve ao mais preciso):
    facebook/sam2.1-hiera-tiny        (~38 MB,  mais rápido)
    facebook/sam2.1-hiera-small       (~46 MB,  recomendado)
    facebook/sam2.1-hiera-base-plus   (~80 MB)
    facebook/sam2.1-hiera-large       (~224 MB, mais preciso)

COMO RODAR:
  python step4_sam_refinement.py \
    --features      features/Panda/features.npz \
    --cosaliency    features/Panda/cosaliency.npz

  python step4_sam_refinement.py \
    --features      features/Panda/features.npz \
    --cosaliency    features/Panda/cosaliency.npz \
    --visualize

  # Modo de prompting (box_only | box_points | box_points_mask):
  python step4_sam_refinement.py -f ... -c ... \
    --prompt_mode box_points_mask --visualize

  # Modelo menor para máquinas sem GPU:
  python step4_sam_refinement.py -f ... -c ... \
    --model facebook/sam2.1-hiera-tiny

O QUE ESTA ETAPA FAZ:

  A Etapa 3 produziu máscaras em resolução de patch (H_p × W_p ~ 37×37).
  O SAM 2 opera em resolução de pixel full e é capaz de delinear bordas
  precisas de objetos a partir de prompts simples.

  Esta etapa usa os artefatos da Etapa 3 como prompts para o SAM 2:

  1. PREPARAÇÃO DOS PROMPTS
     Para cada imagem converte as seeds da Etapa 3 no formato SAM 2:
       • bbox        → box prompt (XYXY, pixels absolutos)
       • fg_points   → point prompts com label=1 (foreground)
       • bg_points   → point prompts com label=0 (background)
       • fg_mask_pixel → mask_input (resizeado para 256×256 em logits)

  2. INFERÊNCIA SAM 2
     SAM2ImagePredictor recebe os prompts e retorna 3 máscaras candidatas
     com scores de qualidade (IoU previsto). Selecionamos a de maior score.
     Para prompts ricos (box+points+mask), multimask_output=False é mais
     eficiente e tipicamente suficiente.

  3. REFINAMENTO ITERATIVO (opcional)
     A máscara gerada pode ser re-alimentada como mask_input em uma segunda
     rodada com os mesmos pontos. Isso é o "Cosegmentation Refinement" do
     paper original, agora via SAM 2.

  4. MÉTRICAS INTERNAS (sem ground truth)
     - iou_score_sam  : confiança do SAM na máscara gerada
     - fg_coherence   : sim. coseno entre protótipos FG das imagens
                         (análogo à métrica do paper — mede se o "co-objeto"
                         é semanticamente coeso na coleção inteira)
     - fg_ratio       : fração de pixels FG por imagem

MODOS DE PROMPTING:
  box_only         Apenas a bounding box da E3.
                   Mais robusto quando as seeds de pontos são ruidosas.

  box_points       Box + pontos FG/BG da E3 (default).
                   Bom balanço entre precisão e robustez.

  box_points_mask  Box + pontos + fg_mask_pixel como mask_input.
                   Melhor qualidade quando a E3 gerou boa máscara coarse.
                   Usa a máscara da E3 como prior de baixa resolução.

SAÍDA (refined_masks.npz):
  refined_masks   : (N, H_max, W_max)  — máscaras binárias finais
  iou_scores_sam  : (N,)               — confiança SAM por imagem
  fg_coherence    : float              — alias de fg_coherence_e4
  fg_coherence_e3 : float              — coesão por protótipos antes do SAM
  fg_coherence_e4 : float              — coesão por protótipos depois do SAM
  fg_coherence_delta : float           — diferença E4 − E3
  prompt_mode     : str                — modo usado
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
import matplotlib.patches as mpatches
from PIL import Image
from sklearn.preprocessing import normalize as sk_normalize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_MODEL      = "facebook/sam2.1-hiera-small"
MASK_INPUT_SIZE    = 256     # SAM 2 espera mask_input de 256×256
LOGIT_SCALE        = 10.0    # escala para converter [0,1] → logits


# ---------------------------------------------------------------------------
# Interfaces de carga (Etapas 1 e 3)
# ---------------------------------------------------------------------------

def load_step1(path: Path) -> dict:
    if not path.exists():
        print(f"[ERRO] features.npz não encontrado: {path}")
        print("       Rode: python step1_feature_extraction.py --collection <classe>")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    return {
        "patch_features": data["patch_features"],
        "patch_grid"    : tuple(int(x) for x in data["patch_grid"]),
        "image_paths"   : list(data["image_paths"]),
        "image_sizes"   : data["image_sizes"],
        "resize_info": (
            data["resize_info"]
            if "resize_info" in data.files
            else None
        ),
    }


def load_step3(path: Path) -> dict:
    if not path.exists():
        print(f"[ERRO] cosaliency.npz não encontrado: {path}")
        print("       Rode: python step3_cosaliency.py --features ... --correspondence ...")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    return {
        "fg_mask_pixel" : data["fg_mask_pixel"],    # (N, H, W)
        "seeds"         : list(data["seeds"]),       # list of dicts
        "image_paths"   : list(data["image_paths"]),
        "image_sizes"   : data["image_sizes"],
    }


# ---------------------------------------------------------------------------
# Carregamento do SAM 2
# ---------------------------------------------------------------------------

def load_sam2(model_id: str = DEFAULT_MODEL) -> object:
    """
    Carrega o SAM2ImagePredictor a partir do HuggingFace.
    Na primeira execução faz download dos pesos (~38-224 MB).

    Requer: pip install git+https://github.com/facebookresearch/sam2.git
    """
    print(f"\n[SAM 2] Carregando {model_id}...")
    print("        (primeira execução faz download — aguarde)")

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("\n[ERRO] SAM 2 não instalado.")
        print("       Instale com:")
        print("         pip install git+https://github.com/facebookresearch/sam2.git")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"        device: {device}")

    try:
        predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)
    except Exception as e:
        print(f"\n[ERRO] Falha ao carregar {model_id}: {e}")
        sys.exit(1)

    print(f"[SAM 2] Pronto ({device})")
    return predictor


# ---------------------------------------------------------------------------
# Conversão de prompts
# ---------------------------------------------------------------------------

def seeds_to_sam_prompts(
    seed       : dict,
    fg_mask    : np.ndarray,           # (H, W) float32 — da Etapa 3
    prompt_mode: str,
) -> dict:
    """
    Converte as seeds da Etapa 3 para o formato do SAM 2.

    Convenções importantes:
      Etapa 3 armazena pontos como [row, col]  (convenção NumPy)
      SAM 2 espera point_coords como [X, Y] = [col, row]  (convenção imagem)

    Retorna dict com chaves opcionais dependendo do prompt_mode:
      box          : np.ndarray (4,) em XYXY
      point_coords : np.ndarray (K, 2) em XY
      point_labels : np.ndarray (K,) — 1=FG, 0=BG
      mask_input   : np.ndarray (1, 256, 256) — logits da máscara coarse
    """
    prompts = {}

    # Bounding box
    if seed["bbox"]:
        prompts["box"] = np.array(seed["bbox"], dtype=np.float32)   # XYXY

    # Pontos FG e BG (converte row,col → X,Y)
    if prompt_mode in ("box_points", "box_points_mask"):
        pt_coords, pt_labels = [], []

        for pt in seed["fg_points"]:
            row, col = (pt[0], pt[1]) if isinstance(pt, (list, tuple)) else (pt[0], pt[1])
            pt_coords.append([col, row])    # X=col, Y=row
            pt_labels.append(1)

        for pt in seed["bg_points"]:
            row, col = (pt[0], pt[1]) if isinstance(pt, (list, tuple)) else (pt[0], pt[1])
            pt_coords.append([col, row])
            pt_labels.append(0)

        if pt_coords:
            prompts["point_coords"] = np.array(pt_coords, dtype=np.float32)
            prompts["point_labels"] = np.array(pt_labels, dtype=np.int32)

    # Máscara coarse como mask_input (logits 256×256)
    if prompt_mode == "box_points_mask" and fg_mask is not None:
        H, W   = fg_mask.shape
        mask_pil = Image.fromarray(fg_mask)
        mask_256 = np.array(mask_pil.resize(
            (MASK_INPUT_SIZE, MASK_INPUT_SIZE), Image.BILINEAR
        ), dtype=np.float32)
        # Converte probabilidades [0,1] em logits (SAM 2 usa threshold=0)
        logits = (mask_256 * 2.0 - 1.0) * LOGIT_SCALE   # [-10, +10]
        prompts["mask_input"] = logits[np.newaxis, :, :]  # (1, 256, 256)

    return prompts


# ---------------------------------------------------------------------------
# Inferência SAM 2 para uma imagem
# ---------------------------------------------------------------------------

def predict_one(
    predictor  : object,
    image      : np.ndarray,     # (H, W, 3) RGB uint8
    prompts    : dict,
    n_iter     : int = 1,        # iterações de refinamento (1 = sem refine)
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Roda SAM 2 para uma imagem com os prompts fornecidos.

    Parâmetros:
        n_iter : número de iterações. Se > 1, a máscara gerada é
                 re-alimentada como mask_input na iteração seguinte
                 (Cosegmentation Refinement do paper, via SAM 2).

    Retorna:
        best_mask  : (H, W) bool — máscara binária final
        best_iou   : float       — score de qualidade SAM
        low_res    : (256, 256)  — logits de baixa resolução (para refine externo)
    """
    predictor.set_image(image)

    box         = prompts.get("box",         None)
    pt_coords   = prompts.get("point_coords", None)
    pt_labels   = prompts.get("point_labels", None)
    mask_input  = prompts.get("mask_input",  None)

    # Usa multimask_output=True apenas quando o prompt é simples (box only)
    # Para prompts ricos (pontos + mask) uma única máscara é mais estável
    multi = (pt_coords is None and mask_input is None)

    for it in range(n_iter):
        masks, iou_scores, low_res_masks = predictor.predict(
            point_coords     = pt_coords,
            point_labels     = pt_labels,
            box              = box,
            mask_input       = mask_input,
            multimask_output = multi,
            return_logits    = False,
        )
        # Seleciona a máscara de maior IoU score
        best_idx  = int(np.argmax(iou_scores))
        best_mask = masks[best_idx].astype(bool)         # (H, W)
        best_iou  = float(iou_scores[best_idx])
        low_res   = low_res_masks[best_idx]              # (256, 256)

        # Refinamento iterativo: realimenta a low-res mask
        if it < n_iter - 1:
            mask_input = low_res[np.newaxis, :, :]       # (1, 256, 256)
            multi      = False                            # prompt rico agora

    return best_mask, best_iou, low_res


# ---------------------------------------------------------------------------
# Métricas internas
# ---------------------------------------------------------------------------

def compute_fg_coherence(
    patch_features  : np.ndarray,   # (N, P, D)
    masks           : np.ndarray,   # (N, H_max, W_max)
    patch_grid      : tuple[int, int],
    image_sizes     : np.ndarray,   # (N, 2), tamanhos originais
    resize_info     : "np.ndarray | None" = None,  # (N, 6), modo keep_ar
) -> float:
    """
    Mede coesão semântica cross-image por protótipos de foreground.

    A máscara é projetada para a mesma geometria usada na Etapa 1, incluindo
    resize e padding do modo keep_ar. Para cada imagem, calcula-se o embedding
    médio normalizado dos patches FG. O resultado é a similaridade coseno
    média apenas entre protótipos de imagens diferentes.
    """
    N, P, D = patch_features.shape
    H_p, W_p = patch_grid
    if H_p * W_p != P:
        raise ValueError(f"patch_grid={patch_grid} incompatível com P={P}.")
    if len(masks) != N or len(image_sizes) != N:
        raise ValueError("Número de máscaras, image_sizes e patch_features não coincide.")
    if resize_info is not None and len(resize_info) != N:
        raise ValueError("resize_info incompatível com o número de imagens.")

    prototypes = []
    for i in range(N):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        mask_real = np.asarray(masks[i, :H_i, :W_i], dtype=np.float32)
        mask_img = Image.fromarray((mask_real > 0.5).astype(np.uint8) * 255, mode="L")

        if resize_info is not None:
            resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = [
                int(x) for x in resize_info[i]
            ]
            resized = mask_img.resize((resized_w, resized_h), Image.NEAREST)
            canvas = Image.new("L", (canvas_w, canvas_h), 0)
            canvas.paste(resized, (pad_left, pad_top))
            mask_patch = np.asarray(
                canvas.resize((W_p, H_p), Image.NEAREST), dtype=np.uint8
            ) > 127
        else:
            mask_patch = np.asarray(
                mask_img.resize((W_p, H_p), Image.NEAREST), dtype=np.uint8
            ) > 127

        fg_idx = mask_patch.reshape(-1)
        if not fg_idx.any():
            continue
        feats = sk_normalize(np.asarray(patch_features[i][fg_idx], dtype=np.float32), axis=1)
        prototype = feats.mean(axis=0)
        norm = float(np.linalg.norm(prototype))
        if norm > 1e-8:
            prototypes.append(prototype / norm)

    if len(prototypes) < 2:
        return 0.0

    prototypes = np.vstack(prototypes)
    similarities = [
        float(prototypes[i] @ prototypes[j])
        for i in range(len(prototypes))
        for j in range(i + 1, len(prototypes))
    ]
    return float(np.mean(similarities)) if similarities else 0.0


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_sam_refinement(
    predictor      : object,
    image_paths    : list[str],
    image_sizes    : np.ndarray,
    fg_masks_e3    : np.ndarray,    # (N, H, W) — da Etapa 3
    seeds          : list[dict],
    patch_features : np.ndarray,    # (N, P, D) — da Etapa 1, para métricas
    patch_grid     : tuple[int, int],
    resize_info    : "np.ndarray | None" = None,
    prompt_mode    : str   = "box_points",
    n_iter         : int   = 1,
) -> dict:
    """
    Itera sobre as imagens da coleção, constrói os prompts e chama o SAM 2.
    """
    N       = len(image_paths)
    H_max   = int(image_sizes[:, 0].max())
    W_max   = int(image_sizes[:, 1].max())

    refined_masks  = np.zeros((N, H_max, W_max), dtype=np.float32)
    iou_scores_sam = np.zeros(N, dtype=np.float32)
    low_res_masks  = np.zeros((N, MASK_INPUT_SIZE, MASK_INPUT_SIZE), dtype=np.float32)

    print(f"\n[SAM 2] Processando {N} imagens  "
          f"(modo: {prompt_mode}, iter: {n_iter})...\n")

    for i, img_path in enumerate(image_paths):
        H_i = int(image_sizes[i, 0])
        W_i = int(image_sizes[i, 1])

        # Carrega imagem como RGB uint8
        image_rgb = np.array(Image.open(img_path).convert("RGB"))

        # Máscara coarse da Etapa 3 para esta imagem
        fg_mask_i = fg_masks_e3[i, :H_i, :W_i]

        # Monta prompts
        prompts = seeds_to_sam_prompts(seeds[i], fg_mask_i, prompt_mode)

        # Inferência
        try:
            best_mask, best_iou, low_res = predict_one(
                predictor, image_rgb, prompts, n_iter=n_iter
            )
        except Exception as e:
            print(f"  [AVISO] img[{i}] falhou: {e} — usando máscara da E3")
            best_mask = fg_mask_i > 0.5
            best_iou  = 0.0
            low_res   = np.zeros((MASK_INPUT_SIZE, MASK_INPUT_SIZE), dtype=np.float32)

        refined_masks[i, :H_i, :W_i] = best_mask.astype(np.float32)
        iou_scores_sam[i]              = best_iou
        low_res_masks[i]               = low_res

        fg_pct = best_mask.mean() * 100
        print(f"  [{i+1:>3}/{N}] {Path(img_path).name:38s}  "
              f"IoU={best_iou:.3f}  FG%={fg_pct:.0f}%")

    print(f"\n[Métricas] Calculando coesão FG cross-image por protótipos...")
    fg_coherence_e3 = compute_fg_coherence(
        patch_features, fg_masks_e3, patch_grid, image_sizes, resize_info
    )
    fg_coherence_e4 = compute_fg_coherence(
        patch_features, refined_masks, patch_grid, image_sizes, resize_info
    )
    fg_coherence_delta = fg_coherence_e4 - fg_coherence_e3
    print(f"  FG coherence E3 : {fg_coherence_e3:.4f}")
    print(f"  FG coherence E4 : {fg_coherence_e4:.4f}")
    print(f"  Delta E4-E3     : {fg_coherence_delta:+.4f}")
    print(f"  IoU SAM médio : {iou_scores_sam.mean():.3f}  "
          f"(> 0.8 = boa segmentação)")
    print(f"  FG% médio     : {refined_masks.mean()*100:.1f}%")

    return {
        "refined_masks" : refined_masks,              # (N, H_max, W_max)
        "iou_scores_sam": iou_scores_sam,             # (N,)
        "low_res_masks" : low_res_masks,              # (N, 256, 256)
        "fg_coherence"       : float(fg_coherence_e4),  # alias compatível
        "fg_coherence_e3"    : float(fg_coherence_e3),
        "fg_coherence_e4"    : float(fg_coherence_e4),
        "fg_coherence_delta" : float(fg_coherence_delta),
        "prompt_mode"   : prompt_mode,
    }


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def save_refined(result: dict, output_dir: Path,
                 image_paths: list[str], image_sizes: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "refined_masks.npz"
    np.savez_compressed(
        out,
        refined_masks  = result["refined_masks"],
        iou_scores_sam = result["iou_scores_sam"],
        low_res_masks  = result["low_res_masks"],
        fg_coherence       = np.array(result["fg_coherence_e4"]),
        fg_coherence_e3    = np.array(result["fg_coherence_e3"]),
        fg_coherence_e4    = np.array(result["fg_coherence_e4"]),
        fg_coherence_delta = np.array(result["fg_coherence_delta"]),
        prompt_mode    = np.array(result["prompt_mode"]),
        image_paths    = image_paths,
        image_sizes    = image_sizes,
    )
    print(f"\n[Salvo] {out}  ({out.stat().st_size/1024**2:.1f} MB)")
    return out


def load_refined(path: Path) -> dict:
    """Interface de carregamento para uso externo / avaliação."""
    data = np.load(path, allow_pickle=True)
    return {
        "refined_masks" : data["refined_masks"],
        "iou_scores_sam": data["iou_scores_sam"],
        "low_res_masks" : data["low_res_masks"],
        "fg_coherence"  : float(data["fg_coherence_e4"])
                          if "fg_coherence_e4" in data.files
                          else float(data["fg_coherence"]),
        "fg_coherence_e3": float(data["fg_coherence_e3"])
                           if "fg_coherence_e3" in data.files else None,
        "fg_coherence_e4": float(data["fg_coherence_e4"])
                           if "fg_coherence_e4" in data.files
                           else float(data["fg_coherence"]),
        "fg_coherence_delta": float(data["fg_coherence_delta"])
                              if "fg_coherence_delta" in data.files else None,
        "prompt_mode"   : str(data["prompt_mode"]),
        "image_paths"   : list(data["image_paths"]),
        "image_sizes"   : data["image_sizes"],
    }


# ---------------------------------------------------------------------------
# Sumário
# ---------------------------------------------------------------------------

def print_summary(result: dict, image_paths: list[str]):
    masks = result["refined_masks"]
    ious  = result["iou_scores_sam"]
    N     = len(image_paths)

    print("\n" + "=" * 65)
    print("  SUMÁRIO ETAPA 4")
    print("=" * 65)
    print(f"  Modo de prompt   : {result['prompt_mode']}")
    print(f"  FG coherence E3  : {result['fg_coherence_e3']:.4f}")
    print(f"  FG coherence E4  : {result['fg_coherence_e4']:.4f}")
    print(f"  Delta E4-E3      : {result['fg_coherence_delta']:+.4f}")
    print(f"  IoU SAM médio    : {ious.mean():.3f}  ±{ious.std():.3f}")
    print()
    print(f"  {'IMG':>4}  {'IoU':>6}  {'FG%':>6}  NOME")
    print(f"  {'─'*4}  {'─'*6}  {'─'*6}  {'─'*35}")
    for i in range(N):
        pct  = masks[i].mean() * 100
        bar  = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        name = Path(image_paths[i]).name[:35]
        print(f"  {i:>4}  {ious[i]:>6.3f}  {pct:>5.1f}%  {bar}  {name}")
    print("=" * 65)
    print()
    print("  O que refined_masks.npz guarda:")
    print("  • refined_masks   — máscaras finais em resolução de pixel")
    print("  • iou_scores_sam  — confiança SAM (usar como peso em avaliação)")
    print("  • fg_coherence_e3/e4 — coesão por protótipos antes/depois do SAM")
    print("  • fg_coherence_delta — ganho semântico da E4 sobre a E3")
    print()


# ---------------------------------------------------------------------------
# Exportação de imagens segmentadas individuais
# ---------------------------------------------------------------------------

def export_segmented_images(
    result      : dict,
    image_paths : list[str],
    image_sizes : np.ndarray,
    output_dir  : Path,
):
    """
    Salva cada imagem segmentada individualmente em output_dir/segmented/.

    Para cada imagem gera três arquivos:
      <nome>_overlay.jpg   — imagem original com máscara verde semitransparente
      <nome>_mask.png      — máscara binária pura (branco=FG, preto=BG)
      <nome>_cutout.png    — objeto recortado com fundo transparente (RGBA)

    Estrutura de saída:
      segmented/
        img001_overlay.jpg
        img001_mask.png
        img001_cutout.png
        img002_overlay.jpg
        ...
    """
    seg_dir = output_dir / "segmented_e4"
    seg_dir.mkdir(parents=True, exist_ok=True)

    refined = result["refined_masks"]   # (N, H_max, W_max)
    ious    = result["iou_scores_sam"]  # (N,)
    N       = len(image_paths)

    print(f"\n[Export] Salvando {N} imagens segmentadas em {seg_dir}/")

    for i, img_path in enumerate(image_paths):
        H_i = int(image_sizes[i, 0])
        W_i = int(image_sizes[i, 1])

        stem    = Path(img_path).stem
        img_arr = np.array(Image.open(img_path).convert("RGB"))
        mask    = refined[i, :H_i, :W_i]           # (H, W) float32 {0,1}
        mask_u8 = (mask * 255).astype(np.uint8)    # (H, W) uint8

        # 1. Overlay: imagem + máscara verde semitransparente
        overlay = img_arr.astype(np.float32)
        green   = np.zeros_like(img_arr, dtype=np.float32)
        green[:, :, 1] = 255.0
        alpha   = mask[:, :, None] * 0.5
        overlay = (overlay * (1 - alpha) + green * alpha).clip(0, 255).astype(np.uint8)
        Image.fromarray(overlay).save(seg_dir / f"{stem}_overlay.jpg", quality=92)

        # 2. Máscara binária pura
        Image.fromarray(mask_u8).save(seg_dir / f"{stem}_mask.png")

        # 3. Cutout: objeto com fundo transparente (RGBA)
        rgba         = np.zeros((H_i, W_i, 4), dtype=np.uint8)
        rgba[:, :, :3] = img_arr
        rgba[:, :,  3] = mask_u8
        Image.fromarray(rgba, mode="RGBA").save(seg_dir / f"{stem}_cutout.png")

        print(f"  [{i+1:>3}/{N}] {stem:35s}  IoU={ious[i]:.3f}  FG%={mask.mean()*100:.0f}%")

    print(f"[Export] Concluido -> {seg_dir}/  ({N*3} arquivos)")
    return seg_dir


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def visualize_refined(
    result         : dict,
    cosaliency     : dict,
    image_paths    : list[str],
    image_sizes    : np.ndarray,
    output_dir     : Path,
    max_images     : int = 8,
):
    """
    Quatro visualizações:

    1. Comparação E3 → E4 (coarse vs. refinada pelo SAM)
       A diferença entre as duas mostra o que o SAM corrigiu.

    2. Overlay final sobre a imagem original
       A saída da cosegmentação completa.

    3. Galeria de IoU scores
       Barras horizontais com os scores SAM por imagem.
       Permite identificar imagens onde o SAM teve baixa confiança.

    4. Comparação de máscara entre imagens (grid)
       Mostra as máscaras finais lado a lado para avaliar consistência.
       Uma boa cosegmentação deve ter FG% similar e visualmente coerente
       entre imagens.
    """
    refined  = result["refined_masks"]    # (N, H, W)
    ious     = result["iou_scores_sam"]   # (N,)
    fg_e3    = cosaliency["fg_mask_pixel"]  # (N, H, W) — E3 para comparação

    N_vis = min(len(image_paths), max_images)

    # ── Fig 1: E3 (coarse) → E4 (SAM) ──────────────────────────────────────
    fig, axes = plt.subplots(N_vis, 3, figsize=(13, 3.8 * N_vis))
    if N_vis == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "Etapa 4 — Refinamento SAM 2\n"
        "Esq: imagem original  |  Centro: E3 coarse  |  Dir: E4 refinada (SAM 2)",
        fontsize=11, y=1.01
    )
    for i in range(N_vis):
        H_i = int(image_sizes[i, 0])
        W_i = int(image_sizes[i, 1])
        img = Image.open(image_paths[i]).convert("RGB")

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(Path(image_paths[i]).name[:25], fontsize=7)
        axes[i, 0].axis("off")

        e3_i = fg_e3[i, :H_i, :W_i] if fg_e3.shape[1] >= H_i else fg_e3[i]
        axes[i, 1].imshow(e3_i, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, 1].set_title(f"E3 coarse  FG%={e3_i.mean()*100:.0f}%", fontsize=7)
        axes[i, 1].axis("off")

        r_i = refined[i, :H_i, :W_i]
        axes[i, 2].imshow(r_i, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, 2].set_title(f"SAM  IoU={ious[i]:.3f}  FG%={r_i.mean()*100:.0f}%", fontsize=7)
        axes[i, 2].axis("off")

    fig.tight_layout()
    out1 = output_dir / "viz_e3_vs_e4.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: Overlay final ─────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(N_vis, 2, figsize=(10, 3.8 * N_vis))
    if N_vis == 1:
        axes2 = axes2[np.newaxis, :]
    fig2.suptitle(
        "Cosegmentação Final (Etapa 4 — SAM 2)\n"
        "Verde = FG  |  IoU = confiança SAM",
        fontsize=11, y=1.01
    )
    for i in range(N_vis):
        H_i = int(image_sizes[i, 0])
        W_i = int(image_sizes[i, 1])
        img_arr = np.array(Image.open(image_paths[i]).convert("RGB"))
        r_i = refined[i, :H_i, :W_i]

        axes2[i, 0].imshow(img_arr)
        axes2[i, 0].set_title(Path(image_paths[i]).name[:28], fontsize=7)
        axes2[i, 0].axis("off")

        # Overlay verde
        overlay = img_arr.astype(np.float32)
        green   = np.zeros_like(img_arr, dtype=np.float32)
        green[:, :, 1] = 255
        alpha   = r_i[:, :, None] * 0.5
        overlay = (overlay * (1 - alpha) + green * alpha).clip(0, 255).astype(np.uint8)
        axes2[i, 1].imshow(overlay)
        axes2[i, 1].set_title(
            f"FG={r_i.mean()*100:.0f}%  IoU={ious[i]:.3f}",
            fontsize=7
        )
        axes2[i, 1].axis("off")

    fig2.tight_layout()
    out2 = output_dir / "viz_overlay_final.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # ── Fig 3: IoU bar chart ─────────────────────────────────────────────────
    N_all = len(image_paths)
    names = [Path(p).name[:20] for p in image_paths]
    colors = ["#2ecc71" if v > 0.8 else "#f39c12" if v > 0.6 else "#e74c3c"
              for v in ious]

    fig3, ax3 = plt.subplots(figsize=(9, max(3, N_all * 0.4 + 1)))
    bars = ax3.barh(range(N_all), ious, color=colors, edgecolor="white", height=0.7)
    ax3.set_yticks(range(N_all))
    ax3.set_yticklabels(names, fontsize=8)
    ax3.set_xlabel("IoU Score (SAM 2)", fontsize=9)
    ax3.set_xlim(0, 1.05)
    ax3.axvline(0.8, color="gray", ls="--", lw=1, alpha=0.6, label="0.8 (bom)")
    ax3.axvline(0.6, color="gray", ls=":",  lw=1, alpha=0.6, label="0.6 (razoável)")
    ax3.legend(fontsize=8)
    ax3.set_title(
        f"IoU Scores SAM 2  "
        f"(média={ious.mean():.3f}, verde≥0.8, laranja≥0.6, vermelho<0.6)\n"
        f"FG coherence={result['fg_coherence']:.4f}",
        fontsize=9
    )
    for bar, v in zip(bars, ious):
        ax3.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=7)
    fig3.tight_layout()
    out3 = output_dir / "viz_iou_scores.png"
    fig3.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig3)

    # ── Fig 4: Grade de máscaras finais ─────────────────────────────────────
    n_cols = 4
    n_rows = (N_vis + n_cols - 1) // n_cols
    fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    axes4_flat = np.array(axes4).flatten()
    fig4.suptitle(
        "Grade de máscaras finais (E4 — SAM 2)\n"
        "Consistência visual = boa cosegmentação",
        fontsize=10
    )
    for idx in range(len(axes4_flat)):
        ax = axes4_flat[idx]
        if idx >= N_vis:
            ax.axis("off")
            continue
        H_i = int(image_sizes[idx, 0])
        W_i = int(image_sizes[idx, 1])
        img_arr = np.array(Image.open(image_paths[idx]).convert("RGB"))
        r_i  = refined[idx, :H_i, :W_i]

        # Mini overlay
        overlay = img_arr.astype(np.float32)
        green   = np.zeros_like(img_arr, dtype=np.float32); green[:, :, 1] = 255
        alpha   = r_i[:, :, None] * 0.45
        overlay = (overlay * (1 - alpha) + green * alpha).clip(0, 255).astype(np.uint8)
        ax.imshow(overlay)
        ax.set_title(f"[{idx}] IoU={ious[idx]:.2f}  FG={r_i.mean()*100:.0f}%", fontsize=7)
        ax.axis("off")

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
        description="Coseg Etapa 4 — refinamento SAM 2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--features",     "-f", required=True,
                   help="features.npz da Etapa 1")
    p.add_argument("--cosaliency",   "-c", required=True,
                   help="cosaliency.npz da Etapa 3")
    p.add_argument("--output",       "-o", default=None,
                   help="Pasta de saída (default: mesma do features.npz)")
    p.add_argument("--model",        default=DEFAULT_MODEL,
                   choices=[
                       "facebook/sam2.1-hiera-tiny",
                       "facebook/sam2.1-hiera-small",
                       "facebook/sam2.1-hiera-base-plus",
                       "facebook/sam2.1-hiera-large",
                       "facebook/sam2-hiera-tiny",
                       "facebook/sam2-hiera-small",
                       "facebook/sam2-hiera-base-plus",
                       "facebook/sam2-hiera-large",
                   ],
                   help=f"Modelo SAM 2 (default: {DEFAULT_MODEL})")
    p.add_argument("--prompt_mode",  default="box_points",
                   choices=["box_only", "box_points", "box_points_mask"],
                   help="Estratégia de prompting (default: box_points)\n"
                        "  box_only        — apenas bounding box\n"
                        "  box_points      — box + pontos FG/BG\n"
                        "  box_points_mask — box + pontos + máscara coarse")
    p.add_argument("--n_iter",       type=int, default=1,
                   help="Iterações de refinamento SAM (default: 1)\n"
                        "  2 = realimenta a máscara na segunda rodada")
    p.add_argument("--visualize",    action="store_true")
    p.add_argument("--viz_max",      type=int, default=8)
    p.add_argument("--export",        action="store_true",
                   help="Salva cada imagem segmentada individualmente em segmented/")
    return p.parse_args()


def main():
    args = parse_args()

    feat_path  = Path(args.features)
    cos_path   = Path(args.cosaliency)
    output_dir = Path(args.output) if args.output else feat_path.parent

    # --- Carga ---------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Coseg — Etapa 4: Refinamento SAM 2")
    print(f"{'='*65}")

    step1 = load_step1(feat_path)
    step3 = load_step3(cos_path)

    # Valida consistência de imagens entre E1 e E3
    assert step1["image_paths"] == step3["image_paths"], (
        "[ERRO] image_paths da E1 e E3 não coincidem. "
        "Verifique se as etapas foram rodadas na mesma coleção."
    )

    image_paths    = step1["image_paths"]
    image_sizes    = step1["image_sizes"]
    patch_features = step1["patch_features"]
    patch_grid     = step1["patch_grid"]
    resize_info    = step1.get("resize_info")
    fg_masks_e3    = step3["fg_mask_pixel"]
    seeds          = step3["seeds"]

    N = len(image_paths)
    print(f"\n  Etapa 1   : {feat_path}")
    print(f"  Etapa 3   : {cos_path}")
    print(f"  Imagens   : {N}")
    print(f"  Modelo    : {args.model}")
    print(f"  Prompts   : {args.prompt_mode}")
    print(f"  Iterações : {args.n_iter}")
    print(f"  Saída     : {output_dir}")

    # --- SAM 2 ---------------------------------------------------------------
    predictor = load_sam2(args.model)

    # --- Pipeline ------------------------------------------------------------
    result = run_sam_refinement(
        predictor      = predictor,
        image_paths    = image_paths,
        image_sizes    = image_sizes,
        fg_masks_e3    = fg_masks_e3,
        seeds          = seeds,
        patch_features = patch_features,
        patch_grid     = patch_grid,
        resize_info    = resize_info,
        prompt_mode    = args.prompt_mode,
        n_iter         = args.n_iter,
    )

    save_refined(result, output_dir, image_paths, image_sizes)
    print_summary(result, image_paths)

    if args.export:
        export_segmented_images(result, image_paths, image_sizes, output_dir)

    if args.visualize:
        visualize_refined(result, step3, image_paths, image_sizes,
                          output_dir, max_images=args.viz_max)

    print(f"[Etapa 4 concluída]  →  {output_dir}/refined_masks.npz")
    print(f"\n  Pipeline completo:")
    print(f"    E1 features.npz  →  E2 correspondence.npz")
    print(f"    →  E3 cosaliency.npz  →  E4 refined_masks.npz")


if __name__ == "__main__":
    main()
