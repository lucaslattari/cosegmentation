"""
=============================================================================
Cosegmentation Pipeline — Etapa 1: Extração de Features com DINOv2
=============================================================================

FLUXO RECOMENDADO:

  # 1A. Protocolo semântico legado (29 coleções; preserva experimentos anteriores)
  python step1_feature_extraction.py --scan datasets/icoseg/images --protocol semantic

  # 1B. Protocolo oficial iCoseg completo (38 grupos originais)
  python step1_feature_extraction.py --scan datasets/icoseg/images --protocol icoseg38

  # 2. Roda por nome de classe (match flexível, case-insensitive)
  python step1_feature_extraction.py --collection "Panda" --protocol semantic --output features/Panda
  python step1_feature_extraction.py --collection "Alaskan Brown Bear" --output features/Bear
  python step1_feature_extraction.py --collection "kite" --output features/Kite --img_size 518 --resize keep_ar --feature_mode l2 --visualize

  # Ainda funciona com path direto (sem JSON):
  python step1_feature_extraction.py --collection 'datasets/icoseg/images/023 Pandas-...'   --output features/Panda

  # Ver classes disponíveis no índice:
  python step1_feature_extraction.py --list --protocol semantic
  python step1_feature_extraction.py --list --protocol icoseg38

PROTOCOLOS:
  semantic  — comportamento legado: aplica CLASS_GROUP_OVERRIDES e agrupa
              álbuns semanticamente equivalentes (ex.: 4 pastas de Kite).
              Mantido para reprodutibilidade dos experimentos anteriores.
  icoseg38  — protocolo para comparação com a literatura: cada pasta original
              do iCoseg é tratada como um grupo independente. Gera exatamente
              uma entrada por pasta com imagens (38 no dataset completo).

SAÍDA (.npz):
  patch_features  : (N, P, D)   — N imagens, P patches, D dimensões
  cls_features    : (N, D)      — token CLS por imagem
  patch_grid      : (H_p, W_p)  — grade de patches
  image_paths     : lista de paths
  image_sizes     : (N, 2)      — tamanhos originais

=============================================================================
"""

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sk_normalize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DINOV2_MODEL   = "dinov2_vits14"
PATCH_SIZE     = 14
IMG_SIZE       = 518
EMBED_DIM      = 384
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)

# Índices separados por protocolo para preservar reprodutibilidade.
# O índice semântico mantém o nome legado; o protocolo iCoseg-38 usa arquivo
# próprio e nunca sobrescreve o índice dos experimentos anteriores.
INDEX_FILENAMES = {
    "semantic": "icoseg_index.json",
    "icoseg38": "icoseg_index_icoseg38.json",
}
DEFAULT_PROTOCOL = "semantic"


def index_filename_for_protocol(protocol: str) -> str:
    try:
        return INDEX_FILENAMES[protocol]
    except KeyError as exc:
        raise ValueError(
            f"Protocolo inválido: {protocol!r}. "
            f"Use um de: {', '.join(INDEX_FILENAMES)}"
        ) from exc


# ---------------------------------------------------------------------------
# Índice do dataset (gerado por --scan, consumido por --collection <nome>)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Agrupamento de classes — curadoria explícita
# ---------------------------------------------------------------------------
#
# _parse_folder_name() originalmente cortava o nome da pasta no primeiro "-",
# o que juntava incorretamente pastas com objetos DIFERENTES sob a mesma
# classe. Exemplos reais no iCoseg:
#   "025 Airshows-helicopter" e "025 Airshows-planes"            → helicóptero != avião
#   "039 Woman Soccer Players-Soccer-Dustin Hall-red" / "-white" → times diferentes
#
# Para evitar esse tipo de mistura silenciosa, o comportamento padrão agora é
# 1 pasta = 1 classe (nome completo da pasta, sem o número inicial). Isso bate
# com a estrutura do ground_truth/, que também mantém uma pasta por álbum.
#
# Agrupamentos multi-pasta só acontecem quando estão explicitamente listados
# abaixo — foram revisados manualmente e representam o mesmo objeto/espécie
# fotografado em locais diferentes. Para adicionar um novo agrupamento seguro,
# edite este dicionário:
#   "NomeDaClasse": ["prefixo do nome da pasta 1", "prefixo do nome da pasta 2", ...]
# O match é por prefixo (case-insensitive) do nome completo da pasta.
CLASS_GROUP_OVERRIDES: dict[str, list[str]] = {
    "Pandas":         ["023 Pandas", "029 Pandas"],
    "Kite":           ["032 Kite", "033 Kite", "034 Kite", "035 Kite"],
    "Stonehenge":     ["009 Stonehenge", "012 Stonehenge"],
    "Agra Taj Mahal": ["017 Agra Taj Mahal", "018 Agra Taj Mahal"],
    "Skating":        ["037 Skating", "038 Skating"],
    "Gymnastics": ["036 Gymnastics-1", "036 Gymnastics-2", "036 Gymnastics-3"],
}


def _default_class_name(name: str) -> str:
    """
    Nome de classe padrão para uma pasta sem agrupamento explícito:
    remove o número/prefixo inicial e usa o restante do nome como está,
    preservando qualquer sufixo (-helicopter, -red, -white, ...) que indique
    um objeto distinto, em vez de descartá-lo.

    '025 Airshows-helicopter'                         → 'Airshows-helicopter'
    '039 Woman Soccer Players-Soccer-Dustin Hall-red' → 'Woman Soccer Players-Soccer-Dustin Hall-red'
    'brown_bear'                                       → 'Brown Bear'
    """
    m = re.match(r"^\d+\s+(.+)$", name)
    if m:
        return m.group(1).strip()
    return name.replace("_", " ").strip().title()


def _match_override(folder_name: str) -> str | None:
    """Retorna o nome de classe agrupado se folder_name bater com algum
    prefixo curado em CLASS_GROUP_OVERRIDES, ou None caso contrário."""
    fl = folder_name.lower()
    for class_name, prefixes in CLASS_GROUP_OVERRIDES.items():
        for prefix in prefixes:
            if fl.startswith(prefix.lower()):
                return class_name
    return None


def build_index(root: Path, protocol: str = DEFAULT_PROTOCOL) -> dict:
    """
    Varre root/ e constrói o índice conforme o protocolo experimental.

    protocol="semantic":
      Mantém o comportamento legado. Pastas explicitamente listadas em
      CLASS_GROUP_OVERRIDES podem ser fundidas em uma mesma coleção semântica.

    protocol="icoseg38":
      Cada pasta original do dataset é uma coleção independente. Nenhum override
      é aplicado. Esse é o protocolo recomendado para comparação quantitativa
      com trabalhos que reportam resultados nas 38 classes/grupos do iCoseg.

    Retorno:
      { nome_da_colecao: [pasta1, ...] }

    No modo icoseg38, cada lista deve conter exatamente uma pasta.
    """
    if protocol not in INDEX_FILENAMES:
        raise ValueError(
            f"Protocolo inválido: {protocol!r}. "
            f"Use um de: {', '.join(INDEX_FILENAMES)}"
        )

    groups: dict[str, list[str]] = defaultdict(list)

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        has_images = any(
            f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
            for f in folder.iterdir()
        )
        if not has_images:
            continue

        if protocol == "icoseg38":
            # 1 pasta original = 1 grupo experimental.
            class_name = _default_class_name(folder.name)
        else:
            # Protocolo legado/semântico.
            class_name = _match_override(folder.name) or _default_class_name(folder.name)

        groups[class_name].append(str(folder))

    index = dict(sorted(groups.items()))

    if protocol == "icoseg38":
        merged = {k: v for k, v in index.items() if len(v) != 1}
        if merged:
            details = ", ".join(f"{k}={len(v)} pastas" for k, v in merged.items())
            raise RuntimeError(
                "O protocolo icoseg38 exige exatamente uma pasta por grupo, "
                f"mas foram detectadas colisões de nome: {details}. "
                "Ajuste _default_class_name() para manter nomes únicos."
            )

    return index


def save_index(index: dict, path: Path):
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def load_index(index_path: Path) -> dict:
    return json.loads(index_path.read_text(encoding="utf-8"))


def find_index(
    protocol: str = DEFAULT_PROTOCOL,
    start: Path = Path("."),
) -> Path | None:
    """Procura o arquivo de índice do protocolo atual e nos pais (até 3 níveis)."""
    filename = index_filename_for_protocol(protocol)
    for directory in [start] + list(start.parents)[:3]:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def resolve_collection(name_or_path: str, protocol: str = DEFAULT_PROTOCOL) -> list[Path]:
    """
    Recebe um nome de classe ('Panda', 'kite') OU um path direto.

    Retorna a lista de imagens encontradas.
    Ordem de resolução:
      1. Path existente no filesystem → usa diretamente
      2. Consulta o índice correspondente a `protocol` (match flexível)
      3. Erro com sugestões
    """
    # --- Caso 1: path direto ------------------------------------------------
    p = Path(name_or_path)
    if p.exists():
        if p.is_dir():
            imgs = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTS)
            if imgs:
                return imgs
            # Subpastas dentro do path (classe com múltiplos álbuns)
            imgs = []
            for sub in sorted(p.iterdir()):
                if sub.is_dir():
                    imgs.extend(sorted(f for f in sub.iterdir()
                                       if f.suffix.lower() in SUPPORTED_EXTS))
            if imgs:
                return imgs
        print(f"[ERRO] '{p}' existe mas não contém imagens.")
        sys.exit(1)

    # --- Caso 2: busca no índice -------------------------------------------
    index_path = find_index(protocol)
    if index_path is None:
        index_filename = index_filename_for_protocol(protocol)
        print(f"[ERRO] Pasta não encontrada: '{name_or_path}'")
        print(f"       E nenhum {index_filename} encontrado para protocol={protocol}.")
        print(f"       Gere o índice primeiro:")
        print(
            "         python step1_feature_extraction.py "
            f"--scan <pasta_icoseg> --protocol {protocol}"
        )
        sys.exit(1)

    index = load_index(index_path)
    match = _fuzzy_match(name_or_path, list(index.keys()))

    if match is None:
        print(f"[ERRO] Classe '{name_or_path}' não encontrada no índice.")
        print(f"       Classes disponíveis: {', '.join(index.keys())}")
        sys.exit(1)

    folders = [Path(f) for f in index[match]]
    imgs: list[Path] = []
    for folder in folders:
        imgs.extend(sorted(f for f in folder.iterdir()
                           if f.suffix.lower() in SUPPORTED_EXTS))

    if not imgs:
        print(f"[ERRO] Classe '{match}' encontrada no índice mas sem imagens.")
        sys.exit(1)

    if match.lower() != name_or_path.lower():
        print(f"[Info] '{name_or_path}' → classe '{match}'  "
              f"({len(folders)} álbum(s), {len(imgs)} imagens)")
    return imgs


def _fuzzy_match(query: str, keys: list[str]) -> str | None:
    """
    Match flexível entre query e lista de chaves:
      1. Exato (case-insensitive)
      2. Prefixo: 'Panda' casa com 'Pandas'
      3. Substring: 'bear' casa com 'Alaskan Brown Bear'

    A versão anterior retornava a PRIMEIRA chave que batesse com cada
    critério, o que causava dois problemas silenciosos:
      - nomes com erro de digitação (ex: 'Stonehenge 2', 'Kite Panda')
        casavam por prefixo com uma classe existente sem avisar;
      - uma query genérica (ex: 'Gymnastics') podia casar só com UM dos
        vários álbuns de uma classe multi-pasta (ex: 'Gymnastics-1'),
        descartando os demais sem qualquer aviso.

    Agora, quando mais de uma chave bate no mesmo critério, isso é tratado
    como ambíguo: ainda é feita uma escolha determinística (a chave mais
    curta), mas um aviso lista TODOS os candidatos, para que o problema
    apareça no log em vez de passar despercebido.
    """
    q = query.lower().strip()

    # 1) Exato — nunca é ambíguo
    for k in keys:
        if k.lower() == q:
            return k

    # 2) Prefixo — coleta todos os candidatos antes de decidir
    prefix_matches = [k for k in keys
                      if k.lower().startswith(q) or q.startswith(k.lower())]
    if prefix_matches:
        if len(prefix_matches) > 1:
            print(f"[Aviso] '{query}' casou por PREFIXO com {len(prefix_matches)} "
                  f"classes: {prefix_matches}")
            print(f"        Usando '{min(prefix_matches, key=len)}'. Se não era essa "
                  f"a intenção, use o nome completo da classe em --collection.")
        return min(prefix_matches, key=len)

    # 3) Substring — mesmo tratamento
    substr_matches = [k for k in keys if q in k.lower() or k.lower() in q]
    if substr_matches:
        if len(substr_matches) > 1:
            print(f"[Aviso] '{query}' casou por SUBSTRING com {len(substr_matches)} "
                  f"classes: {substr_matches}")
            print(f"        Usando '{min(substr_matches, key=len)}'. Se não era essa "
                  f"a intenção, use o nome completo da classe em --collection.")
        return min(substr_matches, key=len)

    return None


def cmd_scan(root_str: str, index_path: Path, protocol: str = DEFAULT_PROTOCOL):
    """Implementa --scan: descobre o dataset e salva o índice do protocolo."""
    root = Path(root_str)
    if not root.exists():
        print(f"[ERRO] Pasta não encontrada: {root}")
        sys.exit(1)

    index = build_index(root, protocol=protocol)
    if not index:
        print(f"[ERRO] Nenhuma subpasta com imagens encontrada em {root}")
        sys.exit(1)

    save_index(index, index_path)

    print(f"\n[Scan] Protocolo: {protocol}")
    print(f"[Scan] Índice salvo em: {index_path}")
    print(f"       {len(index)} classes/grupos encontrados:\n")
    print(f"  {'CLASSE':<30}  {'ÁLBUNS':>6}  {'IMAGENS':>7}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*7}")
    total = 0
    for cls, folders in index.items():
        n = sum(
            sum(1 for f in Path(fld).iterdir() if f.suffix.lower() in SUPPORTED_EXTS)
            for fld in folders
        )
        total += n
        print(f"  {cls:<30}  {len(folders):>6}  {n:>7}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*7}")
    n_folders = sum(len(v) for v in index.values())
    print(f"  {'TOTAL':<30}  {n_folders:>6}  {total:>7}")

    if protocol == "icoseg38":
        if n_folders == 38 and len(index) == 38:
            print("\n  [OK] Protocolo iCoseg-38 validado: 38 grupos independentes.")
        else:
            print(
                f"\n  [AVISO] O protocolo icoseg38 gerou {len(index)} grupos "
                f"a partir de {n_folders} pastas. O dataset iCoseg completo "
                "esperado para comparação SOTA possui 38 grupos originais."
            )

    print(
        "\n  Use: python step1_feature_extraction.py "
        f'--collection "<classe>" --protocol {protocol} ...'
    )


def cmd_list(index_path: Path | None, protocol: str = DEFAULT_PROTOCOL):
    """Implementa --list: mostra classes do índice do protocolo."""
    if index_path is None or not index_path.exists():
        filename = index_filename_for_protocol(protocol)
        print(
            f"[ERRO] {filename} não encontrado. "
            f"Rode --scan com --protocol {protocol} primeiro."
        )
        sys.exit(1)
    index = load_index(index_path)
    print(f"\nClasses em {index_path}:\n")
    for cls, folders in index.items():
        print(f"  {cls}  ({len(folders)} álbum(s))")
    print()


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def load_dinov2(model_name: str = DINOV2_MODEL) -> nn.Module:
    print(f"[DINOv2] Carregando {model_name}...")
    print("         (primeiro uso faz download ~85 MB — aguarde)")
    try:
        import torch.hub as hub_mod
        orig = getattr(hub_mod, "_validate_not_a_forked_repo", None)
        if orig:
            hub_mod._validate_not_a_forked_repo = lambda *a, **kw: None
        model = torch.hub.load("facebookresearch/dinov2", model_name,
                               pretrained=True, verbose=False)
        if orig:
            hub_mod._validate_not_a_forked_repo = orig
    except Exception as e:
        print(f"\n[ERRO] {e}")
        sys.exit(1)

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[DINOv2] Pronto ({device}, {sum(p.numel() for p in model.parameters())/1e6:.1f} M params)")
    return model


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------

def preprocess_image(image_path: Path, img_size: int = IMG_SIZE,
                     resize_mode: str = "square"):
    img = Image.open(image_path).convert("RGB")
    orig_hw = (img.height, img.width)

    # As etapas seguintes salvam patch_features como um array (N, P, D).
    # Portanto, todas as imagens precisam gerar a mesma quantidade P de patches.
    target_size = max(PATCH_SIZE, (img_size // PATCH_SIZE) * PATCH_SIZE)

    if resize_mode == "square":
        # Comportamento original: redimensiona diretamente para um quadrado.
        # Com IMG_SIZE=518 e PATCH_SIZE=14, isso gera uma grade 37×37.
        img = img.resize((target_size, target_size), Image.BICUBIC)
        resize_info = (target_size, target_size, 0, 0, target_size, target_size)

    elif resize_mode == "keep_ar":
        # Preserva a razão de aspecto e usa padding até target_size×target_size.
        # Isso evita distorcer a imagem e ainda mantém uma grade fixa de patches,
        # necessária para empilhar patch_features em um único array (N, P, D).
        scale = target_size / max(img.width, img.height)
        new_w = max(1, min(target_size, int(round(img.width * scale))))
        new_h = max(1, min(target_size, int(round(img.height * scale))))

        resized = img.resize((new_w, new_h), Image.BICUBIC)

        # Padding com a cor média da ImageNet. Depois da normalização,
        # essa região fica aproximadamente zerada, o que é menos intrusivo
        # do que padding preto ou branco para um backbone pré-treinado.
        pad_color = tuple(int(round(v * 255)) for v in IMAGENET_MEAN)
        canvas = Image.new("RGB", (target_size, target_size), pad_color)
        pad_left = (target_size - new_w) // 2
        pad_top = (target_size - new_h) // 2
        canvas.paste(resized, (pad_left, pad_top))
        img = canvas
        resize_info = (new_h, new_w, pad_top, pad_left, target_size, target_size)

    else:
        raise ValueError(
            f"resize_mode inválido: {resize_mode}. "
            "Use 'square' ou 'keep_ar'."
        )

    arr  = np.array(img, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std

    tensor = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, orig_hw, resize_info


# ---------------------------------------------------------------------------
# Extração
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model: nn.Module, image_paths: list[Path],
                     img_size: int = IMG_SIZE,
                     resize_mode: str = "square") -> dict:
    device = next(model.parameters()).device
    all_patch, all_cls, all_sizes, all_resize_info = [], [], [], []
    patch_grid = None

    print(f"\n[Extração] {len(image_paths)} imagens...")
    print(f"[Pré-processamento] img_size={img_size} | resize={resize_mode}")
    for i, path in enumerate(image_paths):
        tensor, orig_hw, resize_info = preprocess_image(
            path, img_size, resize_mode=resize_mode
        )
        tensor = tensor.to(device)

        out = model.get_intermediate_layers(tensor, n=1, return_class_token=True)
        patch_tokens, cls_token = out[0]
        patch_tokens = patch_tokens.squeeze(0).cpu().numpy()
        cls_token    = cls_token.squeeze(0).cpu().numpy()

        h_eff, w_eff = tensor.shape[2], tensor.shape[3]
        gh, gw = h_eff // PATCH_SIZE, w_eff // PATCH_SIZE
        current_grid = (gh, gw)

        if gh * gw != patch_tokens.shape[0]:
            raise RuntimeError(
                f"Grade inconsistente em {path.name}: "
                f"tensor={h_eff}x{w_eff}, grid={gh}x{gw}, "
                f"patch_tokens={patch_tokens.shape[0]}"
            )

        if patch_grid is None:
            patch_grid = current_grid
        elif current_grid != patch_grid:
            raise RuntimeError(
                f"Grade de patches diferente em {path.name}: "
                f"esperado {patch_grid}, obtido {current_grid}. "
                "Verifique o preprocessamento: todas as imagens devem ter o mesmo tamanho efetivo."
            )

        all_patch.append(patch_tokens)
        all_cls.append(cls_token)
        all_sizes.append(orig_hw)
        all_resize_info.append(resize_info)
        print(f"  [{i+1:>3}/{len(image_paths)}] {path.name:40s}  {orig_hw[0]}×{orig_hw[1]}")

    return {
        "patch_features": np.stack(all_patch),
        "cls_features"  : np.stack(all_cls),
        "patch_grid"    : np.array(patch_grid),
        "image_paths"   : [str(p) for p in image_paths],
        "image_sizes"   : np.array(all_sizes),
        "resize_mode"   : np.array(resize_mode),
        # Colunas: resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w
        "resize_info"   : np.array(all_resize_info, dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------
def postprocess_features(features: dict, feature_mode: str) -> dict:
    """
    Aplica pós-processamento nas features antes de salvar.

    Modos:
      raw        : não altera nada
      l2         : normaliza patch_features e cls_features
      pca64_l2   : PCA para 64 dimensões + L2
      pca128_l2  : PCA para 128 dimensões + L2
      pca256_l2  : PCA para 256 dimensões + L2
    """
    if feature_mode == "raw":
        return features

    pf = features["patch_features"].astype(np.float32)
    cf = features["cls_features"].astype(np.float32)

    N, P, D = pf.shape

    if feature_mode == "l2":
        features["patch_features"] = sk_normalize(
            pf.reshape(-1, D), axis=1
        ).reshape(N, P, D)

        features["cls_features"] = sk_normalize(cf, axis=1)
        features["feature_mode"] = np.array(feature_mode)
        return features

    if feature_mode.startswith("pca") and feature_mode.endswith("_l2"):
        n_components = int(feature_mode.replace("pca", "").replace("_l2", ""))

        max_components = min(n_components, N * P, D)
        if max_components < n_components:
            print(
                f"[Aviso] PCA solicitado com {n_components} componentes, "
                f"mas só é possível usar {max_components}."
            )
            n_components = max_components

        flat = pf.reshape(-1, D)

        print(f"[Feature mode] Aplicando PCA {D} → {n_components} + L2")

        pca = PCA(n_components=n_components, random_state=42)
        flat_pca = pca.fit_transform(flat)

        pf_pca = flat_pca.reshape(N, P, n_components)
        cf_pca = pca.transform(cf)

        pf_pca = sk_normalize(
            pf_pca.reshape(-1, n_components), axis=1
        ).reshape(N, P, n_components)

        cf_pca = sk_normalize(cf_pca, axis=1)

        features["patch_features"] = pf_pca.astype(np.float32)
        features["cls_features"] = cf_pca.astype(np.float32)
        features["pca_explained_variance_ratio"] = pca.explained_variance_ratio_
        features["feature_mode"] = np.array(feature_mode)

        return features

    raise ValueError(f"feature_mode desconhecido: {feature_mode}")

def save_features(features: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "features.npz"
    np.savez_compressed(out, **{k: v for k, v in features.items()
                                if k != "image_paths"},
                        image_paths=features["image_paths"])
    print(f"\n[Salvo] {out}  ({out.stat().st_size/1024**2:.1f} MB)")
    return out


def load_features(features_path: Path) -> dict:
    """
    Carrega o .npz da Etapa 1.

    IMPORTANTE: resize_info é obrigatório para o modo keep_ar. Sem ele não é
    possível derivar quais patches correspondem a padding, e patches sem
    conteúdo passam a ser tratados como conteúdo válido nas afinidades,
    na votação de correspondência e nas métricas.
    """
    data = np.load(features_path, allow_pickle=True)

    def _scalar(key: str, default):
        if key not in data.files:
            return default
        return str(data[key].item())

    features = {
        "patch_features": data["patch_features"],
        "cls_features"  : data["cls_features"],
        "patch_grid"    : tuple(data["patch_grid"]),
        "image_paths"   : list(data["image_paths"]),
        "image_sizes"   : data["image_sizes"],
        # Colunas: resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w
        "resize_info"   : data["resize_info"] if "resize_info" in data.files else None,
        "resize_mode"   : _scalar("resize_mode", "square"),
        # postprocess_features() retorna cedo no modo raw sem gravar a chave,
        # então a ausência de feature_mode significa raw.
        "feature_mode"  : _scalar("feature_mode", "raw"),
    }

    if features["resize_mode"] == "keep_ar" and features["resize_info"] is None:
        raise RuntimeError(
            f"{features_path} foi gerado com resize=keep_ar mas não contém "
            "resize_info. Sem essa informação os patches de padding não podem "
            "ser identificados. Regenere a Etapa 1 para esta coleção."
        )

    return features


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def visualize_features(features: dict, output_dir: Path, max_images: int = 6):
    pf   = features["patch_features"]
    clf  = features["cls_features"]
    grid = features["patch_grid"]
    paths = features["image_paths"]

    N    = min(len(paths), max_images)
    H_p, W_p = int(grid[0]), int(grid[1])

    # PCA global
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(pf[:N].reshape(-1, pf.shape[-1]))
    pca_min, pca_max = pca_coords.min(0), pca_coords.max(0)
    pca_rgb = ((pca_coords - pca_min) / (pca_max - pca_min + 1e-8)).reshape(N, H_p, W_p, 3)
    explained = pca.explained_variance_ratio_

    norms   = np.linalg.norm(pf[:N], axis=-1).reshape(N, H_p, W_p)

    cls_n   = sk_normalize(clf[:N], axis=1)
    pat_n   = sk_normalize(pf[:N].reshape(-1, pf.shape[-1]), axis=1).reshape(N, -1, pf.shape[-1])
    cos_map = np.einsum("nd,npd->np", cls_n, pat_n).reshape(N, H_p, W_p)

    fig = plt.figure(figsize=(18, 4.2 * N))
    fig.suptitle(
        f"Etapa 1 — Features DINOv2  |  grade {H_p}×{W_p}  |  "
        f"PCA: {sum(explained):.1%} variância",
        fontsize=12, y=1.01
    )
    col_titles = [
        "Imagem original",
        "PCA-RGB (mesma cor = feature similar)",
        "Norma L2 (ativação)",
        "Sim. coseno c/ CLS (saliência)",
    ]
    gs = gridspec.GridSpec(N, 4, figure=fig, hspace=0.35, wspace=0.08)

    for i in range(N):
        img_path = Path(paths[i])
        for col, (data, cmap, title) in enumerate([
            (None,        None,    col_titles[0]),
            (pca_rgb[i],  None,    col_titles[1]),
            (norms[i],    "magma", col_titles[2]),
            (cos_map[i],  "viridis", col_titles[3]),
        ]):
            ax = fig.add_subplot(gs[i, col])
            if col == 0:
                ax.imshow(Image.open(img_path).convert("RGB"))
                ax.set_title(img_path.name, fontsize=8)
            else:
                im = ax.imshow(data, cmap=cmap, interpolation="nearest")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                if i == 0:
                    ax.set_title(title, fontsize=9)
            ax.axis("off")

    out1 = output_dir / "viz_patch_features.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # CLS scatter
    pca2 = PCA(n_components=2, random_state=42)
    cls2d = pca2.fit_transform(clf)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    sc = ax2.scatter(cls2d[:, 0], cls2d[:, 1], c=range(len(cls2d)),
                     cmap="plasma", s=80, zorder=2)
    for j, p in enumerate(paths):
        ax2.annotate(Path(p).stem[:12], (cls2d[j, 0], cls2d[j, 1]),
                     fontsize=7, ha="center", va="bottom")
    plt.colorbar(sc, ax=ax2, label="índice da imagem")
    ax2.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%})")
    ax2.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%})")
    ax2.set_title("CLS tokens — cluster coeso = boa coleção para coseg", fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    out2 = output_dir / "viz_cls_scatter.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    print(f"[Viz] {out1}  |  {out2}")
    return out1, out2


# ---------------------------------------------------------------------------
# Sumário
# ---------------------------------------------------------------------------

def print_summary(features: dict):
    pf, clf, grid = (features["patch_features"],
                     features["cls_features"],
                     features["patch_grid"])
    N, P, D = pf.shape
    cls_n = sk_normalize(clf, axis=1)
    sim   = cls_n @ cls_n.T
    np.fill_diagonal(sim, 0)
    avg_sim = sim.sum() / (N * (N - 1)) if N > 1 else 0.0

    print("\n" + "=" * 55)
    print("  SUMÁRIO ETAPA 1")
    print("=" * 55)
    print(f"  Imagens          : {N}")
    print(f"  Grade de patches : {grid[0]}×{grid[1]} = {P} patches/img")
    print(f"  Dim embeddings   : {D}")
    print(f"  Norma L2 média   : {np.linalg.norm(pf, axis=-1).mean():.3f}")
    print(f"  Sim. CLS média   : {avg_sim:.3f}  "
          f"{'(coesa ✓)' if avg_sim > 0.5 else '(diversa)'}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Coseg Etapa 1 — extração de features DINOv2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--scan", metavar="ROOT",
                   help="Escaneia ROOT e gera o índice correspondente a --protocol")
    p.add_argument("--protocol",
                   default=DEFAULT_PROTOCOL,
                   choices=sorted(INDEX_FILENAMES.keys()),
                   help=(
                       "Protocolo de agrupamento do dataset: "
                       "semantic preserva os agrupamentos curados legados; "
                       "icoseg38 usa 1 pasta original = 1 grupo experimental "
                       "(recomendado para comparação com SOTA)."
                   ))
    p.add_argument("--list", action="store_true",
                   help="Lista classes disponíveis no icoseg_index.json")
    p.add_argument("--collection", "-c",
                   help='Nome da classe ("Panda") ou path direto para a pasta')
    p.add_argument("--output", "-o",
                   help="Pasta de saída (inferida do nome da classe se omitida)")
    p.add_argument("--output_base", default="features",
                   help="Base para output inferido (default: features/)")
    p.add_argument("--model", default=DINOV2_MODEL,
                   choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    p.add_argument("--feature_mode",
        default="raw",
        choices=[
            "raw",
            "l2",
            "pca64_l2",
            "pca128_l2",
            "pca256_l2",
        ],
        help="Pós-processamento das features: raw, l2, pca64_l2, pca128_l2 ou pca256_l2"
    )
    p.add_argument("--resize", default="square",
                   choices=["square", "keep_ar"],
                   help=(
                       "Modo de redimensionamento: "
                       "square distorce para img_size×img_size; "
                       "keep_ar preserva a razão de aspecto e completa com padding"
                   ))
    p.add_argument("--img_size", type=int, default=IMG_SIZE)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--viz_max", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()

    index_filename = index_filename_for_protocol(args.protocol)
    index_path = find_index(args.protocol) or Path(index_filename)

    # --- Modos especiais ---------------------------------------------------
    if args.scan:
        cmd_scan(args.scan, Path(index_filename), protocol=args.protocol)
        return

    if args.list:
        cmd_list(find_index(args.protocol), protocol=args.protocol)
        return

    if not args.collection:
        print("[ERRO] Informe --collection, --scan ou --list.")
        sys.exit(1)

    # --- Resolução da coleção ----------------------------------------------
    image_paths = resolve_collection(args.collection, protocol=args.protocol)

    if args.max_images:
        image_paths = image_paths[:args.max_images]

    # Infere output se não fornecido
    if args.output:
        output_dir = Path(args.output)
    else:
        # Usa o nome que o usuário digitou, sanitizado
        safe = re.sub(r"[^\w\s-]", "", args.collection).strip().replace(" ", "_")
        output_dir = Path(args.output_base) / safe

    # --- Execução ----------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Coseg — Etapa 1")
    print(f"{'='*55}")
    print(f"  Coleção  : {args.collection}  ({len(image_paths)} imagens)")
    print(f"  Protocolo: {args.protocol}")
    print(f"  Modelo   : {args.model}")
    print(f"  Img size : {args.img_size}")
    print(f"  Resize   : {args.resize}")
    print(f"  Features : {args.feature_mode}")
    print(f"  Saída    : {output_dir}")

    model    = load_dinov2(args.model)
    features = extract_features(
        model, image_paths, img_size=args.img_size, resize_mode=args.resize
    )
    features = postprocess_features(features, args.feature_mode)
    save_features(features, output_dir)
    print_summary(features)

    if args.visualize:
        visualize_features(features, output_dir, max_images=args.viz_max)

    print(f"\n[Etapa 1 concluída]  →  {output_dir}/features.npz")


if __name__ == "__main__":
    main()