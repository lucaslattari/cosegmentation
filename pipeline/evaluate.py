"""
=============================================================================
Cosegmentation Pipeline — Avaliação vs. Ground Truth iCoseg
=============================================================================

COMO RODAR:

  # Passo 1: gera o índice de ground truth (uma vez)
  python evaluate.py --scan_gt icoseg/ground_truth --images_root icoseg/images

  # Passo 2: avalia as máscaras de qualquer etapa do pipeline
  python evaluate.py \
    --masks_dir  features/Pandas/segmented \
    --gt_index   gt_index.json \
    --output     results/Pandas

  # Com visualizações lado a lado:
  python evaluate.py \
    --masks_dir  features/Pandas/segmented \
    --gt_index   gt_index.json \
    --output     results/Pandas \
    --visualize

  # Comparar duas etapas (E4 vs E5):
  python evaluate.py --masks_dir features/Pandas/segmented_e4 --gt_index gt_index.json --output results/Pandas_E4
  python evaluate.py --masks_dir features/Pandas/segmented_e5 --gt_index gt_index.json --output results/Pandas_E5

  # Avaliar todas as classes de uma vez:
  python evaluate.py \
    --masks_root features \
    --gt_index   gt_index.json \
    --output     results/all

ESTRUTURA ESPERADA DO GROUND TRUTH (iCoseg):
  Opção A — pasta separada com mesma hierarquia das imagens (mais comum):
    icoseg/ground_truth/023 Pandas-.../img001.png
    icoseg/images/023 Pandas-.../img001.jpg

  Opção B — GT junto com as imagens (sufixo _gt ou _mask):
    icoseg/images/023 Pandas-.../img001.jpg
    icoseg/images/023 Pandas-.../img001_gt.png

  O GT deve ser imagem binária onde pixel > 128 = FG.

MÁSCARAS ESPERADAS (saída de --export nas etapas 4 ou 5):
  features/Pandas/segmented/
    img001_mask.png      ← máscara binária pura (branco=FG, preto=BG)
    img001_overlay.jpg   ← ignorado nesta avaliação
    img001_cutout.png    ← ignorado nesta avaliação

MÉTRICAS CALCULADAS:
  Accuracy   — % pixels corretamente classificados (FG + BG). Métrica
               principal do paper original (Lattari et al., ICIP 2015).
  Precision  — TP / (TP + FP): de tudo que predisse como FG, quanto era FG?
  Recall     — TP / (TP + FN): do FG real, quanto foi detectado?
  F1 / Dice  — média harmônica de P e R. Métrica balanceada mais comum.
  IoU        — TP / (TP + FP + FN): Jaccard index. Padrão em segmentação.

  São reportadas três agregações: média por imagem, macro por coleção e
  agregação global por pixels (soma de TP/TN/FP/FN).
  Nota sobre flip: as métricas principais NUNCA usam o GT para inverter FG/BG.
  O script reporta separadamente um oracle flip apenas como diagnóstico.

SAÍDA:
  results/<classe>/
    metrics.json        — métricas completas por imagem + médias
    metrics_summary.txt — tabela legível com sumário
    viz_comparison/     — (com --visualize) imagem + pred + GT + diff
    all_classes.json    — (com --masks_root) métricas agregadas por classe

=============================================================================
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
GT_EXTS  = {".png", ".bmp", ".tif", ".tiff"}
GT_SUFFIXES = ("_gt", "_mask", "_seg", "_ground_truth", "_label")


# ---------------------------------------------------------------------------
# Scan e índice de Ground Truth
# ---------------------------------------------------------------------------

def scan_gt(gt_root: Path, images_root: Path | None = None) -> dict:
    """
    Escaneia gt_root e constrói o índice de ground truth.

    Suporta as três variantes de organização do iCoseg:
      A) gt_root com mesma hierarquia de pastas que images_root
      B) GT flat (todos numa pasta só)
      C) GT com sufixo: img001_gt.png, img001_mask.png, etc.

    IMPORTANTE: diferente de versões anteriores, este scan NÃO resolve
    colisões de stem aqui (não faz "last write wins"). Guarda TODOS os
    candidatos por stem em `index_multi`; a resolução por classe acontece
    em build_gt_mapping(), que usa o nome da coleção/pasta como contexto
    de desambiguação. Isso evita associar silenciosamente uma imagem ao
    GT de outra classe quando nomes de arquivo se repetem entre coleções
    (ex.: "img001" existindo em Pandas/ e em Kite/).
    """
    index_multi: dict[str, list[str]] = {}

    for gt_file in sorted(gt_root.rglob("*")):
        if not gt_file.is_file():
            continue
        if gt_file.suffix.lower() not in GT_EXTS:
            continue

        stem = gt_file.stem
        # Remove sufixos conhecidos de GT
        for suf in GT_SUFFIXES:
            if stem.lower().endswith(suf):
                stem = stem[: -len(suf)]
                break

        index_multi.setdefault(stem, []).append(str(gt_file))

    n_ambiguous = sum(1 for v in index_multi.values() if len(v) > 1)
    if n_ambiguous:
        print(f"  [AVISO] {n_ambiguous} stem(s) aparecem em mais de um arquivo de GT "
              f"(provavelmente nomes repetidos entre classes).")
        print(f"          Serão resolvidos por classe em build_gt_mapping(); "
              f"casos não resolvíveis aparecerão em 'ambiguous', não serão adivinhados.")
        shown = 0
        for stem, paths in index_multi.items():
            if len(paths) > 1 and shown < 5:
                print(f"    {stem}: {len(paths)} ocorrências")
                shown += 1
        if n_ambiguous > 5:
            print(f"    ... e mais {n_ambiguous - 5}")

    return {"index_multi": index_multi}


def _norm_name(s: str) -> str:
    """Normaliza um nome de classe/pasta para comparação (lowercase, só alfanumérico)."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def build_gt_mapping(
    image_paths     : list[str],
    gt_index_multi  : dict[str, list[str]],
    collection_hint : str | None = None,
) -> tuple[dict[str, str], list[str], list[dict]]:
    """
    Para cada image_path encontra o GT correspondente pelo stem do nome.

    Quando um stem tem UM único candidato de GT, usa direto.
    Quando um stem tem MÚLTIPLOS candidatos (colisão de nome entre classes),
    desambigua comparando o nome da classe (collection_hint, ou a pasta-mãe
    da própria imagem se não for fornecido) com a pasta-mãe de cada candidato
    de GT. Só resolve se exatamente um candidato bater.

    Casos que continuam ambíguos após esse filtro NÃO são incluídos no
    mapping — são reportados separadamente em `ambiguous`, para que o
    chamador decida (e para nunca associar uma imagem ao GT errado
    silenciosamente).

    Aceita tanto o formato novo (gt_index_multi: stem -> [paths]) quanto,
    por compatibilidade, um dict legado stem -> path (será tratado como
    lista de 1 elemento).

    Retorna:
        mapping   : {img_path -> gt_path}
        not_found : lista de img_paths sem nenhum candidato de GT
        ambiguous : lista de {"img_path", "hint", "candidates"} não resolvidos
    """
    mapping: dict[str, str] = {}
    not_found: list[str] = []
    ambiguous: list[dict] = []

    for img_path in image_paths:
        p = Path(img_path)
        stem = p.stem
        # Remove sufixo _mask / _overlay / _cutout (saídas do pipeline)
        for suf in ("_mask", "_overlay", "_cutout"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break

        raw_candidates = gt_index_multi.get(stem)
        if raw_candidates is None:
            not_found.append(img_path)
            continue

        candidates = [raw_candidates] if isinstance(raw_candidates, str) else list(raw_candidates)

        if len(candidates) == 1:
            mapping[img_path] = candidates[0]
            continue

        hint = collection_hint or p.parent.name
        hint_norm = _norm_name(hint)
        matches = [
            c for c in candidates
            if hint_norm and (hint_norm in _norm_name(Path(c).parent.name)
                               or _norm_name(Path(c).parent.name) in hint_norm)
        ]

        if len(matches) == 1:
            mapping[img_path] = matches[0]
        else:
            ambiguous.append({
                "img_path": img_path,
                "hint": hint,
                "candidates": candidates,
            })

    return mapping, not_found, ambiguous


def save_gt_index(gt_scan: dict, image_paths: list[str],
                  gt_root: str, out_path: Path):
    """
    Salva gt_index.json com o índice multi-stem e, se images_root foi
    fornecido, um mapeamento de referência (já desambiguado por classe,
    usando a pasta-mãe de cada imagem como hint).
    """
    index_multi = gt_scan["index_multi"]

    if image_paths:
        mapping, not_found, ambiguous = build_gt_mapping(image_paths, index_multi)
    else:
        mapping, not_found, ambiguous = {}, [], []

    n_multi = sum(1 for v in index_multi.values() if len(v) > 1)
    data = {
        "version"              : "2.0",
        "gt_root"              : gt_root,
        "n_gt_files"           : sum(len(v) for v in index_multi.values()),
        "n_unique_stems"       : len(index_multi),
        "n_stems_with_collision": n_multi,
        "n_images_mapped"      : len(mapping),
        "n_images_missing"     : len(not_found),
        "n_images_ambiguous"   : len(ambiguous),
        "missing"              : [str(p) for p in not_found],
        "ambiguous"            : ambiguous,
        "mappings"             : mapping,        # img_path -> gt_path (referência)
        "gt_index_multi"       : index_multi,    # stem -> [gt_paths] (usado na avaliação)
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                        encoding="utf-8")


def load_gt_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_gt_index_multi(gt_index_json: dict) -> dict[str, list[str]]:
    """
    Extrai {stem: [gt_paths]} de um gt_index.json, aceitando tanto o
    formato novo (gt_index_multi) quanto o formato legado v1.0 (gt_index
    flat, stem -> path único).
    """
    if "gt_index_multi" in gt_index_json:
        return gt_index_json["gt_index_multi"]

    flat = gt_index_json.get("gt_index", gt_index_json)
    multi: dict[str, list[str]] = {}
    for k, v in flat.items():
        if isinstance(v, list):
            multi[k] = v
        else:
            multi[k] = [v]
    return multi


# ---------------------------------------------------------------------------
# Carregamento de máscaras
# ---------------------------------------------------------------------------

def load_mask(path: Path, target_hw: tuple | None = None) -> np.ndarray:
    """
    Carrega uma imagem como máscara binária (bool numpy array).
    pixel > 128 = FG para GT do iCoseg.
    pixel > 0.5 para máscaras do pipeline (já binárias).
    Se target_hw fornecido, faz resize com NEAREST (não interpola binário).
    """
    img = Image.open(path).convert("L")
    if target_hw:
        img = img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    return arr > 128


# Etapas do pipeline que a auto-detecção tem permissão de considerar.
# Etapa 6 foi uma tentativa de refinamento global que deu errado e nunca deve
# ser escolhida automaticamente, mesmo que a pasta segmented_e6 exista no disco.
MAX_AUTO_STAGE = 5


def find_class_masks_dir(cls_dir: Path, masks_subdir: str | None = None,
                          stage: str | None = None) -> Path | None:
    """
    Localiza a subpasta de máscaras dentro de uma pasta de classe.

    Prioridade:
      1. masks_subdir explícito (--masks_subdir) — nome literal da subpasta,
         usado como está, sem restrição (escape hatch manual).
      2. stage explícito (--stage e4 / --stage e5) — mapeia direto para
         "segmented_e4" / "segmented_e5". Se a pasta não existir para essa
         classe, retorna None (a classe é reportada como ausente, não cai
         silenciosamente para outro estágio).
      3. Auto-detecção: entre "segmented", "segmented_e4", ..., limitada a
         MAX_AUTO_STAGE. Escolhe o maior sufixo numérico disponível dentro
         desse limite. Pastas como "segmented_e6" (ou maiores) são
         ignoradas propositalmente, mesmo que existam — presume-se que
         qualquer etapa acima de MAX_AUTO_STAGE é experimental/quebrada
         até que alguém peça explicitamente por ela via masks_subdir.
      4. A própria cls_dir, se tiver arquivos *_mask.png direto (fallback
         para estruturas antigas sem subpasta).

    Retorna None se nada for encontrado.
    """
    if masks_subdir:
        cand = cls_dir / masks_subdir
        return cand if cand.exists() else None

    if stage:
        cand = cls_dir / f"segmented_{stage}"
        return cand if cand.exists() else None

    def _stage_num(name: str) -> int:
        if name == "segmented":
            return 0
        if "_e" in name:
            digits = "".join(ch for ch in name.split("_e")[-1] if ch.isdigit())
            return int(digits) if digits else -1
        return -1

    seg_dirs = [
        d for d in cls_dir.iterdir()
        if d.is_dir() and d.name.startswith("segmented")
        and 0 <= _stage_num(d.name) <= MAX_AUTO_STAGE
    ]
    if seg_dirs:
        return sorted(seg_dirs, key=lambda d: _stage_num(d.name))[-1]

    if any(cls_dir.glob("*_mask.png")):
        return cls_dir

    return None


def find_mask_files(masks_dir: Path) -> list[Path]:
    """
    Encontra todos os arquivos de máscara em masks_dir.
    Prioriza *_mask.png (saída padrão do pipeline) sobre outros.
    """
    mask_files = sorted(masks_dir.glob("*_mask.png"))
    if mask_files:
        return mask_files
    # Fallback: qualquer PNG/BMP na pasta que não seja overlay/cutout
    return sorted(
        f for f in masks_dir.iterdir()
        if f.suffix.lower() in GT_EXTS
        and "_overlay" not in f.name
        and "_cutout" not in f.name
    )


# ---------------------------------------------------------------------------
# Cálculo de métricas
# ---------------------------------------------------------------------------

def compute_metrics(
    pred_mask : np.ndarray,   # bool (H, W)
    gt_mask   : np.ndarray,   # bool (H, W)
) -> dict:
    """
    Calcula métricas pixel-wise entre pred e GT.

    IMPORTANTE:
      - As métricas principais são sempre calculadas sobre a predição original.
      - Também calcula, separadamente, um diagnóstico oracle em que FG/BG é
        invertido apenas se isso melhora a accuracy. Esse resultado NÃO deve ser
        usado como métrica principal de comparação com SOTA, pois consulta o GT.
    """
    def _metrics(p: np.ndarray, g: np.ndarray) -> dict:
        p, g = p.astype(bool), g.astype(bool)
        TP = int(( p &  g).sum())
        TN = int((~p & ~g).sum())
        FP = int(( p & ~g).sum())
        FN = int((~p &  g).sum())
        tot = p.size

        acc  = (TP + TN) / tot
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        iou  = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "iou": float(iou),
            "dice": float(f1),
            "tp": TP,
            "tn": TN,
            "fp": FP,
            "fn": FN,
            "n_pixels": int(tot),
        }

    raw = _metrics(pred_mask, gt_mask)
    flipped_metrics = _metrics(~pred_mask.astype(bool), gt_mask)
    oracle_applied = flipped_metrics["accuracy"] > raw["accuracy"]
    oracle = flipped_metrics if oracle_applied else raw.copy()

    raw["oracle_flip_applied"] = oracle_applied
    raw["oracle_accuracy"] = float(oracle["accuracy"])
    raw["oracle_precision"] = float(oracle["precision"])
    raw["oracle_recall"] = float(oracle["recall"])
    raw["oracle_f1"] = float(oracle["f1"])
    raw["oracle_iou"] = float(oracle["iou"])
    raw["oracle_tp"] = int(oracle["tp"])
    raw["oracle_tn"] = int(oracle["tn"])
    raw["oracle_fp"] = int(oracle["fp"])
    raw["oracle_fn"] = int(oracle["fn"])
    raw["fg_pred_ratio"] = float(pred_mask.mean())
    raw["fg_gt_ratio"] = float(gt_mask.mean())
    return raw


def aggregate_metrics(per_image: list[dict]) -> dict:
    """
    Agrega métricas em três protocolos complementares:

      image_mean: média simples das métricas por imagem.
      global_pixel: métricas obtidas somando TP/TN/FP/FN de todas as imagens.
      oracle_image_mean / oracle_global_pixel: versões diagnósticas com melhor
        permutação FG/BG escolhida usando o GT.

    O bloco legado de chaves por métrica (accuracy.mean etc.) é mantido apontando
    para image_mean, para compatibilidade com o restante do código.
    """
    if not per_image:
        return {}

    scalar_keys = ["accuracy", "precision", "recall", "f1", "iou",
                   "dice", "fg_pred_ratio", "fg_gt_ratio"]
    oracle_keys = ["oracle_accuracy", "oracle_precision", "oracle_recall",
                   "oracle_f1", "oracle_iou"]

    agg: dict = {}
    image_mean = {}
    for key in scalar_keys:
        vals = np.array([m[key] for m in per_image if key in m], dtype=float)
        stats = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
        agg[key] = stats  # compatibilidade
        image_mean[key] = stats

    oracle_image_mean = {}
    for key in oracle_keys:
        vals = np.array([m[key] for m in per_image], dtype=float)
        oracle_image_mean[key.replace("oracle_", "")] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

    def _from_counts(tp: int, tn: int, fp: int, fn: int) -> dict:
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        return {
            "accuracy": float(acc), "precision": float(prec),
            "recall": float(rec), "f1": float(f1), "iou": float(iou),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "n_pixels": int(total),
        }

    global_pixel = _from_counts(
        sum(m["tp"] for m in per_image),
        sum(m["tn"] for m in per_image),
        sum(m["fp"] for m in per_image),
        sum(m["fn"] for m in per_image),
    )
    oracle_global_pixel = _from_counts(
        sum(m["oracle_tp"] for m in per_image),
        sum(m["oracle_tn"] for m in per_image),
        sum(m["oracle_fp"] for m in per_image),
        sum(m["oracle_fn"] for m in per_image),
    )

    agg["image_mean"] = image_mean
    agg["global_pixel"] = global_pixel
    agg["oracle_image_mean"] = oracle_image_mean
    agg["oracle_global_pixel"] = oracle_global_pixel
    agg["n_images"] = len(per_image)
    agg["n_flipped"] = sum(1 for m in per_image if m.get("oracle_flip_applied", False))
    return agg


# ---------------------------------------------------------------------------
# Pipeline de avaliação para uma coleção
# ---------------------------------------------------------------------------

def evaluate_collection(
    masks_dir   : Path,
    gt_index    : dict,
    collection_name : str = "",
) -> dict:
    """
    Avalia todas as máscaras em masks_dir contra seus GT correspondentes.

    Retorna dict com:
      per_image   : lista de métricas por imagem
      aggregate   : médias e estatísticas da coleção
      collection  : nome da coleção
    """
    mask_files = find_mask_files(masks_dir)

    if not mask_files:
        print(f"  [AVISO] Nenhuma máscara encontrada em {masks_dir}")
        return {}

    gt_index_multi = get_gt_index_multi(gt_index)
    gt_mapping, not_found, ambiguous = build_gt_mapping(
        [str(f) for f in mask_files],
        gt_index_multi,
        collection_hint=collection_name,
    )

    if not_found:
        print(f"  [AVISO] {len(not_found)} máscara(s) sem GT:")
        for p in not_found[:5]:
            print(f"    {Path(p).name}")
        if len(not_found) > 5:
            print(f"    ... e mais {len(not_found)-5}")

    if ambiguous:
        print(f"  [ERRO-AMBIGUIDADE] {len(ambiguous)} máscara(s) têm mais de um GT candidato "
              f"e não puderam ser desambiguadas pelo nome da classe ('{collection_name}').")
        print(f"                      Excluídas da avaliação (não foram associadas ao GT errado).")
        for a in ambiguous[:5]:
            cand_names = [str(Path(c).parent.name) for c in a["candidates"]]
            print(f"    {Path(a['img_path']).name}  hint='{a['hint']}'  candidatos_em={cand_names}")
        if len(ambiguous) > 5:
            print(f"    ... e mais {len(ambiguous)-5}")

    per_image = []
    print(f"\n  Avaliando {len(gt_mapping)} imagens...")

    for mask_path_str, gt_path_str in sorted(gt_mapping.items()):
        mask_path = Path(mask_path_str)
        gt_path   = Path(gt_path_str)

        # Carrega GT e obtém dimensões
        gt_mask   = load_mask(gt_path)
        H_gt, W_gt = gt_mask.shape

        # Carrega pred no mesmo tamanho do GT
        pred_mask = load_mask(mask_path, target_hw=(H_gt, W_gt))

        metrics = compute_metrics(pred_mask, gt_mask)
        metrics["image_name"] = mask_path.stem
        metrics["mask_path"]  = mask_path_str
        metrics["gt_path"]    = gt_path_str
        per_image.append(metrics)

        flip_str = " [oracle flip]" if metrics["oracle_flip_applied"] else ""
        print(f"  {mask_path.name:40s}  "
              f"acc={metrics['accuracy']:.4f}  "
              f"iou={metrics['iou']:.4f}  "
              f"f1={metrics['f1']:.4f}{flip_str}")

    agg = aggregate_metrics(per_image)
    print(f"\n  Médias da coleção '{collection_name}':")
    print(f"    Accuracy  : {agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f}")
    print(f"    IoU       : {agg['iou']['mean']:.4f} ± {agg['iou']['std']:.4f}")
    print(f"    F1/Dice   : {agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}")
    print(f"    Precision : {agg['precision']['mean']:.4f}")
    print(f"    Recall    : {agg['recall']['mean']:.4f}")
    print(f"    Global pixel accuracy : {agg['global_pixel']['accuracy']:.4f}")
    print(f"    Oracle image accuracy : {agg['oracle_image_mean']['accuracy']['mean']:.4f}")
    if agg["n_flipped"] > 0:
        print(f"    [diag] {agg['n_flipped']}/{agg['n_images']} imagens melhorariam com oracle FG/BG flip")

    return {
        "collection"      : collection_name,
        "per_image"       : per_image,
        "aggregate"       : agg,
        "masks_dir"       : str(masks_dir),
        "n_not_found_gt"  : len(not_found),
        "n_ambiguous_gt"  : len(ambiguous),
        "ambiguous_gt"    : ambiguous,
    }


# ---------------------------------------------------------------------------
# Persistência dos resultados
# ---------------------------------------------------------------------------

def save_results(result: dict, output_dir: Path) -> tuple[Path, Path]:
    """Salva metrics.json e metrics_summary.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON completo
    json_path = output_dir / "metrics.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Sumário legível
    txt_path = output_dir / "metrics_summary.txt"
    agg = result.get("aggregate", {})
    per = result.get("per_image", [])
    coll = result.get("collection", "")

    lines = [
        "=" * 72,
        f"  AVALIAÇÃO — {coll}",
        "=" * 72,
        f"  Imagens avaliadas : {agg.get('n_images', 0)}",
        f"  Oracle flips      : {agg.get('n_flipped', 0)}",
        "",
        f"  {'MÉTRICA':<14}  {'MEAN':>8}  {'STD':>8}  {'MIN':>8}  {'MAX':>8}",
        f"  {'─'*14}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}",
    ]
    for key in ["accuracy", "precision", "recall", "f1", "iou"]:
        if key in agg:
            v = agg[key]
            lines.append(
                f"  {key.capitalize():<14}  "
                f"{v['mean']:>8.4f}  {v['std']:>8.4f}  "
                f"{v['min']:>8.4f}  {v['max']:>8.4f}"
            )
    lines += [
        "",
        f"  Global pixel accuracy : {agg.get('global_pixel', {}).get('accuracy', 0.0):.4f}",
        f"  Oracle image accuracy : {agg.get('oracle_image_mean', {}).get('accuracy', {}).get('mean', 0.0):.4f}",
        f"  Oracle pixel accuracy : {agg.get('oracle_global_pixel', {}).get('accuracy', 0.0):.4f}",
        "", f"  {'─'*72}", "",
        f"  {'IMG':>4}  {'ACC':>7}  {'IoU':>7}  {'F1':>7}  {'PREC':>7}  {'REC':>7}  NOME"
    ]
    lines.append(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*35}")
    for i, m in enumerate(per):
        flip = "~" if m.get("oracle_flip_applied") else " "
        lines.append(
            f"  {i:>4}  {m['accuracy']:>7.4f}  {m['iou']:>7.4f}  "
            f"{m['f1']:>7.4f}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  "
            f"{flip}{Path(m['image_name']).name[:35]}"
        )
    lines += ["", "  ~ = oracle flip melhoraria a accuracy (diagnóstico; GT não altera a métrica principal)", "=" * 72]

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[Salvo] {json_path}")
    print(f"[Salvo] {txt_path}")
    return json_path, txt_path


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def visualize_comparison(
    result     : dict,
    output_dir : Path,
    max_images : int = 12,
):
    """
    Para cada imagem gera uma figura com 4 colunas:
      1. Imagem original
      2. Pred (máscara do pipeline)
      3. GT (ground truth)
      4. Diff: verde=TP, vermelho=FP, azul=FN, cinza=TN

    Além disso gera:
      - viz_metrics_bar.png : barras comparativas de todas as métricas
      - viz_iou_sorted.png  : imagens ordenadas por IoU (fácil identificar problemas)
    """
    viz_dir = output_dir / "viz_comparison"
    viz_dir.mkdir(parents=True, exist_ok=True)

    per_image = result["per_image"]
    agg       = result["aggregate"]
    N_vis     = min(len(per_image), max_images)

    # ── Figuras por imagem ───────────────────────────────────────────────────
    print(f"\n[Viz] Gerando comparações para {N_vis} imagens...")

    for idx in range(N_vis):
        m          = per_image[idx]
        mask_path  = Path(m["mask_path"])
        gt_path    = Path(m["gt_path"])

        gt_mask    = load_mask(gt_path)
        H_gt, W_gt = gt_mask.shape
        pred_mask  = load_mask(mask_path, target_hw=(H_gt, W_gt))


        # Tenta encontrar imagem original (stem sem _mask)
        orig_stem = mask_path.stem.replace("_mask", "")
        orig_img  = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidates = [
                mask_path.parent.parent / f"{orig_stem}{ext}",
                mask_path.parent.parent.parent / f"{orig_stem}{ext}",
            ]
            for c in candidates:
                if c.exists():
                    orig_img = c; break
            if orig_img:
                break

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"{orig_stem}  |  acc={m['accuracy']:.4f}  "
            f"iou={m['iou']:.4f}  f1={m['f1']:.4f}"
            + ("  [oracle flip ajudaria]" if m.get("oracle_flip_applied") else ""),
            fontsize=9
        )

        # Col 0: imagem original (se encontrada) ou pred
        if orig_img:
            axes[0].imshow(Image.open(orig_img).convert("RGB").resize(
                (W_gt, H_gt), Image.BICUBIC
            ))
            axes[0].set_title("Original", fontsize=8)
        else:
            axes[0].imshow(pred_mask, cmap="gray")
            axes[0].set_title("Pred (sem original)", fontsize=8)
        axes[0].axis("off")

        # Col 1: pred
        axes[1].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title(f"Pred  FG%={pred_mask.mean()*100:.0f}%", fontsize=8)
        axes[1].axis("off")

        # Col 2: GT
        axes[2].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"GT  FG%={gt_mask.mean()*100:.0f}%", fontsize=8)
        axes[2].axis("off")

        # Col 3: diff colorido
        diff_rgb = np.zeros((H_gt, W_gt, 3), dtype=np.uint8)
        p = pred_mask.astype(bool)
        g = gt_mask.astype(bool)
        diff_rgb[ p &  g] = [0,   200,  0]    # TP  = verde
        diff_rgb[ p & ~g] = [200,  0,   0]    # FP  = vermelho
        diff_rgb[~p &  g] = [0,    0,  200]   # FN  = azul
        diff_rgb[~p & ~g] = [40,  40,  40]    # TN  = cinza escuro

        axes[3].imshow(diff_rgb)
        axes[3].set_title(
            f"Diff  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}",
            fontsize=7
        )
        legend_patches = [
            mpatches.Patch(color="#00C800", label="TP (correto FG)"),
            mpatches.Patch(color="#C80000", label="FP (falso FG)"),
            mpatches.Patch(color="#0000C8", label="FN (FG perdido)"),
            mpatches.Patch(color="#282828", label="TN (correto BG)"),
        ]
        axes[3].legend(handles=legend_patches, loc="lower right",
                       fontsize=6, framealpha=0.8)
        axes[3].axis("off")

        fig.tight_layout()
        out = viz_dir / f"{orig_stem}_eval.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ── Barras comparativas de métricas ──────────────────────────────────────
    metric_names  = ["accuracy", "precision", "recall", "f1", "iou"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1/Dice", "IoU"]
    means = [agg[k]["mean"] for k in metric_names]
    stds  = [agg[k]["std"]  for k in metric_names]
    colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#e74c3c"]

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    bars = ax2.bar(metric_labels, means, yerr=stds, color=colors,
                   capsize=5, width=0.6, edgecolor="white")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Score")
    ax2.set_title(
        f"Métricas — {result.get('collection', '')}  "
        f"(n={agg['n_images']} imagens)",
        fontsize=11
    )
    ax2.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig2.tight_layout()
    out2 = output_dir / "viz_metrics_bar.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # ── Imagens ordenadas por IoU ─────────────────────────────────────────────
    sorted_by_iou = sorted(per_image, key=lambda m: m["iou"])
    names_sorted  = [Path(m["image_name"]).name[:22] for m in sorted_by_iou]
    ious_sorted   = [m["iou"]      for m in sorted_by_iou]
    accs_sorted   = [m["accuracy"] for m in sorted_by_iou]

    fig3, ax3 = plt.subplots(figsize=(max(8, len(per_image)*0.45 + 1), 5))
    x = np.arange(len(sorted_by_iou))
    ax3.bar(x - 0.2, ious_sorted, 0.4, label="IoU",      color="#e74c3c", alpha=0.85)
    ax3.bar(x + 0.2, accs_sorted, 0.4, label="Accuracy", color="#2ecc71", alpha=0.85)
    ax3.axhline(agg["iou"]["mean"],      color="#e74c3c", ls="--", lw=1.2, alpha=0.6)
    ax3.axhline(agg["accuracy"]["mean"], color="#2ecc71", ls="--", lw=1.2, alpha=0.6)
    ax3.set_xticks(x); ax3.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Score"); ax3.set_ylim(0, 1.1)
    ax3.set_title(
        f"IoU e Accuracy por imagem (ordenado por IoU crescente)\n"
        f"Tracejado = média da coleção",
        fontsize=10
    )
    ax3.legend(fontsize=9); ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    out3 = output_dir / "viz_iou_sorted.png"
    fig3.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig3)

    outs = [out2, out3] + list(viz_dir.glob("*.png"))
    print(f"[Viz] {len(list(viz_dir.glob('*.png')))} figuras de comparação em {viz_dir}/")
    print(f"[Viz] viz_metrics_bar.png  |  viz_iou_sorted.png")
    return outs


# ---------------------------------------------------------------------------
# Avaliação de todas as classes (--masks_root)
# ---------------------------------------------------------------------------

def evaluate_all_classes(
    masks_root  : Path,
    gt_index    : dict,
    output_dir  : Path,
    visualize   : bool = False,
    max_images  : int  = 12,
    masks_subdir: str | None = None,
    stage       : str | None = None,
    classes     : list[str] | None = None,
) -> dict:
    """
    Itera sobre todas as subpastas de masks_root (uma por classe) e
    avalia cada uma. Gera all_classes.json com sumário comparativo.

    A subpasta de máscaras dentro de cada classe é localizada com
    find_class_masks_dir() — aceita "segmented", "segmented_e4",
    "segmented_e5" via auto-detecção (nunca além de MAX_AUTO_STAGE),
    um --stage explícito (e4/e5), ou um --masks_subdir literal (aceita
    caminhos com subpastas, ex: "border_topk/segmented_e5").

    Se `classes` for fornecido, avalia só essas classes (nomes exatos das
    pastas dentro de masks_root) — útil para testar uma variante nova só
    nas classes problemáticas antes de rodar nas 38 completas. Nomes que
    não existirem em masks_root são reportados como ausentes, não
    silenciosamente ignorados.
    """
    all_class_dirs = sorted(
        d for d in masks_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if classes:
        wanted = set(classes)
        found_names = {d.name for d in all_class_dirs}
        missing_classes = wanted - found_names
        if missing_classes:
            print(f"  [AVISO] {len(missing_classes)} classe(s) pedida(s) em --classes "
                  f"não existem em {masks_root}:")
            for m in sorted(missing_classes):
                print(f"    {m}")
        all_class_dirs = [d for d in all_class_dirs if d.name in wanted]

    class_dirs = sorted(
        d for d in all_class_dirs
        if find_class_masks_dir(d, masks_subdir, stage) is not None
    )

    if not class_dirs:
        print(f"[ERRO] Nenhuma classe com máscaras encontrada em {masks_root}"
              f"{f' para stage={stage}' if stage else ''}"
              f"{f' dentro do filtro --classes' if classes else ''}")
        print(f"       Esperado: {masks_root}/<Classe>/segmented*/*_mask.png "
              f"ou {masks_root}/<Classe>/*_mask.png")
        sys.exit(1)

    all_results = {}
    print(f"\n[Avaliação global] {len(class_dirs)} classes encontradas"
          f"{f' (filtradas de {len(classes)} pedidas)' if classes else ''}...")
    if stage:
        print(f"[Avaliação global] Usando explicitamente segmented_{stage} — outras etapas ignoradas.")
    elif not masks_subdir:
        print(f"[Avaliação global] Auto-detecção limitada a segmented_e{MAX_AUTO_STAGE} "
              f"ou anterior (etapas acima disso são ignoradas).")

    for cls_dir in class_dirs:
        cls_name  = cls_dir.name
        masks_dir = find_class_masks_dir(cls_dir, masks_subdir, stage)

        print(f"\n{'─'*60}")
        print(f"  Classe: {cls_name}")
        print(f"  Máscaras: {masks_dir}")

        cls_out = output_dir / cls_name
        result  = evaluate_collection(masks_dir, gt_index,
                                      collection_name=cls_name)
        if not result:
            continue

        save_results(result, cls_out)
        if visualize:
            visualize_comparison(result, cls_out, max_images=max_images)

        all_results[cls_name] = {
            "masks_dir"  : str(masks_dir),
            "n_images"   : result["aggregate"]["n_images"],
            "accuracy"   : result["aggregate"]["accuracy"]["mean"],
            "iou"        : result["aggregate"]["iou"]["mean"],
            "f1"         : result["aggregate"]["f1"]["mean"],
            "precision"  : result["aggregate"]["precision"]["mean"],
            "recall"     : result["aggregate"]["recall"]["mean"],
            "n_flipped"  : result["aggregate"]["n_flipped"],
            "global_pixel_accuracy": result["aggregate"]["global_pixel"]["accuracy"],
            "global_pixel_iou": result["aggregate"]["global_pixel"]["iou"],
            "oracle_accuracy": result["aggregate"]["oracle_image_mean"]["accuracy"]["mean"],
            "oracle_iou": result["aggregate"]["oracle_image_mean"]["iou"]["mean"],
            "oracle_f1": result["aggregate"]["oracle_image_mean"]["f1"]["mean"],
            "oracle_precision": result["aggregate"]["oracle_image_mean"]["precision"]["mean"],
            "oracle_recall": result["aggregate"]["oracle_image_mean"]["recall"]["mean"],
            "oracle_global_pixel_accuracy": result["aggregate"]["oracle_global_pixel"]["accuracy"],
            "oracle_global_pixel_iou": result["aggregate"]["oracle_global_pixel"]["iou"],
            "tp": sum(x["tp"] for x in result["per_image"]),
            "tn": sum(x["tn"] for x in result["per_image"]),
            "fp": sum(x["fp"] for x in result["per_image"]),
            "fn": sum(x["fn"] for x in result["per_image"]),
        }

    # Sumário global
    output_dir.mkdir(parents=True, exist_ok=True)
    all_json = output_dir / "all_classes.json"
    all_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # Tabela de sumário no terminal
    print(f"\n{'='*75}")
    print(f"  SUMÁRIO GLOBAL ({len(all_results)} classes)")
    print(f"{'='*75}")
    print(f"  {'CLASSE':<30}  {'N':>4}  {'ACC':>7}  {'IoU':>7}  {'F1':>7}  {'FLIP':>5}")
    print(f"  {'─'*30}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*5}")

    all_accs = []; all_ious = []; all_f1s = []
    all_oracle_accs = []; all_oracle_ious = []
    weighted_acc_num = weighted_iou_num = weighted_f1_num = 0.0
    weighted_oracle_acc_num = weighted_oracle_iou_num = 0.0
    total_images = 0
    total_tp = total_tn = total_fp = total_fn = 0
    for cls, m in sorted(all_results.items()):
        print(f"  {cls:<30}  {m['n_images']:>4}  "
              f"{m['accuracy']:>7.4f}  {m['iou']:>7.4f}  "
              f"{m['f1']:>7.4f}  {m['n_flipped']:>5}")
        all_accs.append(m["accuracy"]); all_ious.append(m["iou"]); all_f1s.append(m["f1"])
        all_oracle_accs.append(m["oracle_accuracy"]); all_oracle_ious.append(m["oracle_iou"])
        n = m["n_images"]; total_images += n
        weighted_acc_num += n * m["accuracy"]
        weighted_iou_num += n * m["iou"]
        weighted_f1_num += n * m["f1"]
        weighted_oracle_acc_num += n * m["oracle_accuracy"]
        weighted_oracle_iou_num += n * m["oracle_iou"]
        total_tp += m["tp"]; total_tn += m["tn"]; total_fp += m["fp"]; total_fn += m["fn"]

    image_mean_acc = weighted_acc_num / total_images if total_images else 0.0
    image_mean_iou = weighted_iou_num / total_images if total_images else 0.0
    image_mean_f1 = weighted_f1_num / total_images if total_images else 0.0
    image_mean_oracle_acc = weighted_oracle_acc_num / total_images if total_images else 0.0
    image_mean_oracle_iou = weighted_oracle_iou_num / total_images if total_images else 0.0
    pixel_total = total_tp + total_tn + total_fp + total_fn
    global_pixel_acc = (total_tp + total_tn) / pixel_total if pixel_total else 0.0

    print(f"  {'─'*30}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}")
    print(f"  {'MACRO POR COLEÇÃO':<30}  {'':>4}  {np.mean(all_accs):>7.4f}  {np.mean(all_ious):>7.4f}  {np.mean(all_f1s):>7.4f}")
    print(f"  {'MÉDIA POR IMAGEM':<30}  {total_images:>4}  {image_mean_acc:>7.4f}  {image_mean_iou:>7.4f}  {image_mean_f1:>7.4f}")
    print(f"  {'GLOBAL POR PIXEL':<30}  {'':>4}  {global_pixel_acc:>7.4f}")
    print(f"  {'─'*30}  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}")
    print(f"  {'[teto] ORACLE MACRO':<30}  {'':>4}  {np.mean(all_oracle_accs):>7.4f}  {np.mean(all_oracle_ious):>7.4f}  "
          f"{'':>7}  (diagnóstico — usa GT p/ resolver sign-flip; NÃO é métrica final)")
    print(f"  {'[teto] ORACLE MÉDIA/IMG':<30}  {total_images:>4}  {image_mean_oracle_acc:>7.4f}  {image_mean_oracle_iou:>7.4f}")
    print(f"{'='*75}")
    print(f"\n[Salvo] {all_json}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Avaliação de cosegmentação vs. ground truth iCoseg",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Geração do índice GT
    p.add_argument("--scan_gt",      metavar="GT_ROOT",
                   help="Escaneia GT_ROOT e gera gt_index.json")
    p.add_argument("--images_root",  metavar="IMG_ROOT", default=None,
                   help="Pasta raiz das imagens (usada com --scan_gt)")

    # Avaliação
    p.add_argument("--masks_dir",  "-m", default=None,
                   help="Pasta com máscaras de uma classe (ex: features/Pandas/segmented)")
    p.add_argument("--masks_root", default=None,
                   help="Pasta raiz com subpastas por classe (avalia todas)")
    p.add_argument("--masks_subdir", default=None,
                   help="Nome explícito da subpasta de máscaras dentro de cada classe "
                        "(ex: segmented_e5). Tem prioridade sobre --stage. "
                        "Único jeito de acessar etapas fora do intervalo automático (ex: e6).")
    p.add_argument("--stage", default=None, choices=["e4", "e5"],
                   help="Etapa explícita do pipeline a avaliar (segmented_e4 ou segmented_e5). "
                        "Se omitido, auto-detecta a mais avançada até e5 "
                        f"(nunca além de e{5} — etapas experimentais como e6 são sempre ignoradas "
                        "a menos que --masks_subdir seja usado).")
    p.add_argument("--classes", default=None,
                   help="Restringe --masks_root a um subconjunto de classes. Aceita: "
                        "(a) uma lista separada por vírgula com os nomes exatos das pastas "
                        "(ex: 'Kite,Pandas,Kendo-EKC_2008_Helsinki-SanSuiSo'), ou "
                        "(b) o caminho de um .txt com um nome de classe por linha. "
                        "Útil para testar uma variante nova só nas classes problemáticas "
                        "antes de rodar no dataset completo. Se omitido, avalia todas.")
    p.add_argument("--gt_index",   "-g", default="gt_index.json",
                   help="gt_index.json gerado por --scan_gt (default: gt_index.json)")
    p.add_argument("--output",     "-o", default="results",
                   help="Pasta de saída para métricas e visualizações")

    # Opções
    p.add_argument("--no_flip", action="store_true",
                   help="Compatibilidade legada. As métricas principais nunca usam flip; oracle flip é apenas diagnóstico.")
    p.add_argument("--visualize", action="store_true",
                   help="Gera visualizações comparativas")
    p.add_argument("--viz_max",   type=int, default=12,
                   help="Máx imagens nas visualizações (default: 12)")
    p.add_argument("--collection", default=None,
                   help="Nome da coleção (para labels nos gráficos)")
    return p.parse_args()


def parse_classes_arg(raw: str | None) -> list[str] | None:
    """
    Resolve --classes em uma lista de nomes de classe.
    Aceita um .txt (um nome por linha, ignora linhas vazias/'#' comentário)
    ou uma lista separada por vírgula direto na linha de comando.
    """
    if not raw:
        return None

    candidate_path = Path(raw)
    if candidate_path.suffix.lower() == ".txt":
        if not candidate_path.exists():
            print(f"[ERRO] Arquivo de classes não encontrado: {candidate_path}")
            sys.exit(1)
        names = [
            line.strip() for line in candidate_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        names = [n.strip() for n in raw.split(",") if n.strip()]

    if not names:
        print(f"[ERRO] --classes não resultou em nenhum nome de classe válido: {raw!r}")
        sys.exit(1)

    return names


def main():
    args = parse_args()

    # ── Modo 1: gera índice GT ────────────────────────────────────────────────
    if args.scan_gt:
        gt_root = Path(args.scan_gt)
        if not gt_root.exists():
            print(f"[ERRO] GT root não encontrado: {gt_root}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"  Scanning ground truth: {gt_root}")
        print(f"{'='*60}")

        gt_scan = scan_gt(gt_root)
        index_multi = gt_scan["index_multi"]
        n_gt_files = sum(len(v) for v in index_multi.values())
        print(f"  Encontrados {n_gt_files} arquivos de GT ({len(index_multi)} stems únicos)")

        # Se images_root fornecido, tenta fazer mapeamento direto (referência apenas;
        # a avaliação real por classe refaz isso com o nome da classe como hint)
        img_paths_all = []
        if args.images_root:
            img_root = Path(args.images_root)
            for f in sorted(img_root.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMG_EXTS:
                    img_paths_all.append(str(f))
            print(f"  Encontradas {len(img_paths_all)} imagens em {img_root}")

        out_path = Path("gt_index.json")
        save_gt_index(gt_scan, img_paths_all, str(gt_root), out_path)

        if img_paths_all:
            mapping, not_found, ambiguous = build_gt_mapping(img_paths_all, index_multi)
            print(f"\n  Mapeamentos: {len(mapping)}/{len(img_paths_all)}")
            if not_found:
                print(f"  Sem GT     : {len(not_found)}")
            if ambiguous:
                print(f"  Ambíguos   : {len(ambiguous)}  "
                      f"(serão resolvidos por classe na hora de avaliar cada coleção,"
                      f" com --masks_dir/--masks_root, onde o nome da pasta é o hint)")
        print(f"\n[Salvo] {out_path}")
        print(f"\n  Use:")
        print(f"    python evaluate.py --masks_dir <pasta_segmented> "
              f"--gt_index {out_path} --output results/<classe>")
        return

    # ── Validação: precisa de gt_index para avaliação ─────────────────────────
    gt_index_path = Path(args.gt_index)
    if not gt_index_path.exists():
        print(f"[ERRO] gt_index.json não encontrado: {gt_index_path}")
        print(f"       Gere primeiro com:")
        print(f"         python evaluate.py --scan_gt icoseg/ground_truth")
        sys.exit(1)

    gt_index = load_gt_index(gt_index_path)
    print(f"[GT] Índice carregado: {gt_index.get('n_found', len(gt_index.get('mappings', {})))} entradas")

    output_dir = Path(args.output)

    # ── Modo 2: avalia uma classe ─────────────────────────────────────────────
    if args.masks_dir:
        masks_dir = Path(args.masks_dir)
        if not masks_dir.exists():
            print(f"[ERRO] Pasta de máscaras não encontrada: {masks_dir}")
            sys.exit(1)

        collection = args.collection or masks_dir.parent.name
        print(f"\n{'='*60}")
        print(f"  Avaliando: {collection}")
        print(f"  Máscaras : {masks_dir}")
        print(f"  Saída    : {output_dir}")
        print(f"{'='*60}")

        result = evaluate_collection(
            masks_dir,
            gt_index,
            collection_name = collection,
        )

        if result:
            save_results(result, output_dir)
            if args.visualize:
                visualize_comparison(result, output_dir,
                                     max_images=args.viz_max)

    # ── Modo 3: avalia todas as classes ──────────────────────────────────────
    elif args.masks_root:
        masks_root = Path(args.masks_root)
        if not masks_root.exists():
            print(f"[ERRO] masks_root não encontrado: {masks_root}")
            sys.exit(1)

        evaluate_all_classes(
            masks_root,
            gt_index,
            output_dir,
            visualize    = args.visualize,
            max_images   = args.viz_max,
            masks_subdir = args.masks_subdir,
            stage        = args.stage,
            classes      = parse_classes_arg(args.classes),
        )

    else:
        print("[ERRO] Forneça --masks_dir, --masks_root, ou --scan_gt.")
        print("       Use -h para ajuda.")
        sys.exit(1)


if __name__ == "__main__":
    main()