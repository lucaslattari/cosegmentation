"""
=============================================================================
Cosegmentation Pipeline — Etapa 2: Cross-Image Correspondence
=============================================================================

PAPEL DESTA ETAPA NO PIPELINE:
  Entrada : features.npz da Etapa 1  (patch_features, cls_features, ...)
  Saída   : correspondence.npz       (fg_scores, pair_votes, top_pairs, ...)
              consumido pela Etapa 3 (Graph Cuts / SAM prompting)

COMO RODAR:
  python step2_correspondence.py --features features/Panda/features.npz
  python step2_correspondence.py --features features/Panda/features.npz --visualize
  python step2_correspondence.py --features features/Panda/features.npz --top_k 8 --visualize
  python step2_correspondence.py --features features/Panda/features.npz --orientation spatial

O QUE ESTA ETAPA FAZ:
  É o análogo moderno do "Global Clustering" do paper original (Lattari et al., ICIP 2015),
  mas operando em espaço semântico de alta dimensão (DINOv2 embeddings) em vez de
  descritores manuais de cor + textura Gabor.

  Algoritmo (TokenCut cross-image, inspirado em Wang et al., T-PAMI 2023):

  1. CLS SIMILARITY  — matrix N×N de similaridade coseno entre tokens globais.
                        Seleciona os top-K pares de imagens mais similares.
                        (Análogo: escolha de imagens candidatas no Global Clustering)

  2. CROSS-ATTENTION — para cada par (i, j): constrói um grafo combinado com
                        P_i + P_j nós (patches das duas imagens juntos).
                        Arestas = similaridade coseno entre todos os pares de patches.
                        (Análogo: Bhattacharyya distance entre Local Clusters)

  3. NORMALIZED CUT  — aplica NCut nesse grafo combinado.
                        O vetor de Fiedler (2º menor autovetor do Laplaciano normalizado)
                        particiona os patches em dois grupos coerentes.
                        (Análogo: Graph Cuts do paper, mas sem seeds manuais)

  4. ORIENTAÇÃO FG   — o corte produz dois lados sem rótulo. Por padrão, usamos
                        a saliência do CLS (comportamento histórico, --orientation cls).
                        Para ablação, --orientation spatial escolhe o lado mais
                        central/compacto nas duas imagens do par. O default permanece
                        'cls' para preservar integralmente o comportamento anterior.

  5. VOTAÇÃO         — cada par vota em quais patches de cada imagem são FG.
                        O score final fg_scores[i, p] é a fração de pares em que
                        o patch p da imagem i foi votado como FG.
                        (Análogo: "super clusters" com classificação por maioria)

  MASCARAMENTO DE PADDING (quando a Etapa 1 rodou com --resize keep_ar):
                        A entrada dos patches que caem inteiramente na faixa de
                        padding (cor média do ImageNet, ~zero após normalização)
                        é praticamente constante entre imagens, o que gera
                        afinidade cross-image artificialmente alta entre esses
                        patches — eles competem por "espaço" no NCut com os
                        patches de conteúdo real, especialmente danoso em
                        coleções com pouco sinal (objeto pequeno, muito clutter).
                        Por padrão, esses patches são excluídos do grafo de
                        afinidade e da orientação FG/BG de cada par, e seu voto
                        final é forçado a 0 (background) — nunca competem por FG.
                        Desative com --no_padding_mask se quiser o comportamento
                        antigo (ou para comparação A/B).

SAÍDA (correspondence.npz):
  fg_scores       : (N, P)       — score [0,1] de foreground por patch
  fg_scores_2d    : (N, H_p, W_p) — mesmo dado reshaped para visualização
  top_pairs       : (K, 3)       — [(i, j, cls_sim), ...] pares selecionados
  cls_sim_matrix  : (N, N)       — matriz de similaridade CLS completa
  pair_votes      : (K, N, P)    — votos individuais por par (debug/Etapa 3)
  image_paths     : lista de paths (espelhada da Etapa 1)
  patch_grid      : (H_p, W_p)

=============================================================================
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize as MplNorm
from PIL import Image
from sklearn.preprocessing import normalize as sk_normalize

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Interface com a Etapa 1
# ---------------------------------------------------------------------------

def load_step1(features_path: Path) -> dict:
    """Carrega o .npz produzido pela Etapa 1."""
    if not features_path.exists():
        print(f"[ERRO] features.npz não encontrado: {features_path}")
        print(f"       Rode a Etapa 1 primeiro:")
        print(f"         python step1_feature_extraction.py --collection <classe>")
        sys.exit(1)

    data = np.load(features_path, allow_pickle=True)
    feats = {
        "patch_features": data["patch_features"],        # (N, P, D)
        "cls_features"  : data["cls_features"],          # (N, D)
        "patch_grid"    : tuple(int(x) for x in data["patch_grid"]),  # (H_p, W_p)
        "image_paths"   : list(data["image_paths"]),
        "image_sizes"   : data["image_sizes"],           # (N, 2)
        # resize_info só existe se a Etapa 1 rodou com --resize keep_ar.
        # Colunas: resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w
        "resize_info"   : data["resize_info"] if "resize_info" in data.files else None,
    }

    N, P, D = feats["patch_features"].shape
    H_p, W_p = feats["patch_grid"]
    print(f"[Etapa 1 → 2] {N} imagens  |  grade {H_p}×{W_p} ({P} patches)  |  dim {D}")
    return feats


# ---------------------------------------------------------------------------
# Mascaramento de padding
# ---------------------------------------------------------------------------

def compute_content_mask(resize_info: "np.ndarray | None",
                         patch_grid: tuple[int, int]) -> "np.ndarray | None":
    """
    A partir de resize_info (só existe quando a Etapa 1 rodou com
    --resize keep_ar), calcula, por imagem, quais patches da grade caem
    dentro da imagem real e quais caem inteira ou parcialmente na faixa de
    padding.

    Um patch é considerado "conteúdo" se sua área tem QUALQUER interseção
    com a região da imagem real dentro do canvas (mesmo critério usado nos
    scripts de ablation para alinhar o GT). Patches de padding têm entrada
    constante (cor média do ImageNet, que vira ~zero após normalização) e
    tendem a ter afinidade cross-image artificialmente alta entre si,
    competindo por espaço no grafo NCut com os patches de conteúdo real —
    especialmente prejudicial em coleções com pouco sinal (objeto pequeno,
    cena com muito clutter).

    Retorna None se resize_info for None (Etapa 1 rodou em modo 'square',
    sem padding — não há nada para mascarar).
    """
    if resize_info is None:
        return None

    H_p, W_p = patch_grid
    N = resize_info.shape[0]
    canvas_h, canvas_w = int(resize_info[0, 4]), int(resize_info[0, 5])
    patch_h = canvas_h / H_p
    patch_w = canvas_w / W_p

    row_starts = np.arange(H_p) * patch_h
    col_starts = np.arange(W_p) * patch_w
    row_ends = row_starts + patch_h
    col_ends = col_starts + patch_w

    masks = np.zeros((N, H_p, W_p), dtype=bool)
    for i in range(N):
        resized_h, resized_w, pad_top, pad_left, _, _ = [int(x) for x in resize_info[i]]
        content_row = (row_ends > pad_top) & (row_starts < pad_top + resized_h)
        content_col = (col_ends > pad_left) & (col_starts < pad_left + resized_w)
        masks[i] = content_row[:, None] & content_col[None, :]

    return masks.reshape(N, -1)


# ---------------------------------------------------------------------------
# Passo 1 — CLS similarity matrix e seleção de pares
# ---------------------------------------------------------------------------

def compute_cls_similarity(cls_features: np.ndarray) -> np.ndarray:
    """
    Retorna a matriz N×N de similaridade coseno entre CLS tokens.
    Diagonal = 0 (não comparamos uma imagem com ela mesma).
    """
    cls_n = sk_normalize(cls_features, axis=1)   # (N, D)
    sim   = cls_n @ cls_n.T                       # (N, N)
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float32)


def select_top_pairs(cls_sim: np.ndarray, top_k: int) -> list[tuple[int, int, float]]:
    """
    Seleciona os top_k pares (i < j) com maior similaridade CLS.
    Retorna lista de (i, j, sim_score).
    """
    N = cls_sim.shape[0]
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j, float(cls_sim[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    selected = pairs[:top_k]

    print(f"\n[Pares] Top-{top_k} pares selecionados por similaridade CLS:")
    for rank, (i, j, s) in enumerate(selected):
        print(f"  {rank+1:>2}. img[{i:>2}] × img[{j:>2}]  sim={s:+.4f}")

    return selected


# ---------------------------------------------------------------------------
# Passo 2 — Cross-attention (grafo combinado por par)
# ---------------------------------------------------------------------------

def build_pair_affinity(feat_i: np.ndarray, feat_j: np.ndarray,
                        temperature: float = 0.05,
                        degree_alpha: float = 0.0) -> np.ndarray:
    """
    Constrói a matriz de afinidade (P_i + P_j) × (P_i + P_j) para um par.

    Blocos:
      W_ii : afinidade intra-imagem i  (coerência espacial local)
      W_jj : afinidade intra-imagem j
      W_ij : afinidade inter-imagem    (a correspondência que nos interessa)

    Nota: P_i e P_j aqui são o número de patches efetivamente passados
    (após o chamador excluir patches de padding, se aplicável) — não são
    necessariamente o total de patches da grade.

    A temperatura controla a nitidez das correspondências:
      - temperatura baixa  → correspondências mais esparsas/nítidas
      - temperatura alta   → correspondências mais suaves/difusas

    A matriz é simétrica e não-negativa (requisito do NCut).

    degree_alpha controla uma reponderação opcional por grau ANTES do NCut:

        W'_{ij} = W_{ij} / (d_i d_j)^degree_alpha

    onde d é o grau calculado na matriz W original.

      degree_alpha = 0.0  -> comportamento histórico, sem alteração
      degree_alpha = 0.5  -> normalização simétrica completa pré-NCut
      valores intermediários (ex. 0.10, 0.25) reduzem suavemente a
                             dominância de nós/regiões com grau muito alto

    Importante: o NCut continua aplicando sua normalização Laplaciana normal
    depois desta reponderação. Esta etapa é uma ablação estrutural do grafo,
    não uma substituição da normalização interna do NCut.
    """
    Pi, D = feat_i.shape
    Pj    = feat_j.shape[0]

    # Normaliza para similaridade coseno em [-1, 1]
    fi = sk_normalize(feat_i, axis=1)   # (Pi, D)
    fj = sk_normalize(feat_j, axis=1)   # (Pj, D)

    # Blocos de afinidade
    W_ii = fi @ fi.T              # (Pi, Pi)
    W_jj = fj @ fj.T              # (Pj, Pj)
    W_ij = fi @ fj.T              # (Pi, Pj) — correspondência cross-image

    # Aplica exp(sim / T) para tornar não-negativo e amplificar picos
    def _softpos(x):
        return np.exp(x / temperature)

    W_ii = _softpos(W_ii)
    W_jj = _softpos(W_jj)
    W_ij = _softpos(W_ij)

    # Remove auto-arestas (diagonal intra)
    np.fill_diagonal(W_ii, 0.0)
    np.fill_diagonal(W_jj, 0.0)

    # Monta matriz combinada (Pi+Pj) × (Pi+Pj)
    top    = np.concatenate([W_ii, W_ij], axis=1)       # (Pi, Pi+Pj)
    bottom = np.concatenate([W_ij.T, W_jj], axis=1)     # (Pj, Pi+Pj)
    W      = np.concatenate([top, bottom], axis=0)       # (Pi+Pj, Pi+Pj)

    # Reponderação opcional por grau antes do NCut.
    # alpha=0 preserva bit-a-bit a formulação anterior desta função.
    if degree_alpha > 0.0:
        if degree_alpha > 0.5:
            raise ValueError(
                f"degree_alpha={degree_alpha} fora da faixa recomendada [0, 0.5]."
            )
        d = np.maximum(W.sum(axis=1), 1e-12)
        scale = np.power(d, degree_alpha)
        W = W / (scale[:, None] * scale[None, :])

        # Garante simetria numérica após a reponderação.
        W = 0.5 * (W + W.T)

    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Passo 3 — Normalized Cut (Fiedler vector)
# ---------------------------------------------------------------------------

def normalized_cut(W: np.ndarray, n_cuts: int = 1) -> np.ndarray:
    """
    Computa os n_cuts+1 menores autovetores do Laplaciano normalizado simétrico.

    L_sym = I - D^{-1/2} W D^{-1/2}

    O 2º autovetor (Fiedler) dá a partição binária ótima do NCut.
    Retornar múltiplos autovetores permite partições hierárquicas (útil para debug).

    Retorna: (P, n_cuts+1) — autovetores, colunas 1..n_cuts são as partições
    """
    d          = W.sum(axis=1)                           # grau de cada nó
    d_safe     = np.maximum(d, 1e-8)
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)

    # L_sym via broadcasting (evita materializar D^{-1/2} como matriz densa)
    L_sym = np.eye(W.shape[0], dtype=np.float32) - \
            (d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :])

    # Eigendecomposição dos n_cuts+1 menores valores
    # np.linalg.eigh retorna em ordem crescente para matrizes simétricas
    # Para matrizes grandes, scipy.sparse.linalg.eigsh seria mais eficiente,
    # mas para (2P)×(2P) ≈ 2738×2738 o denso é suficiente (~0.5s por par na CPU)
    vals, vecs = np.linalg.eigh(L_sym)

    # Retorna os n_cuts+1 menores (índices 0..n_cuts)
    # Índice 0 = autovetor constante (todos iguais) = trivial
    # Índice 1 = Fiedler = partição binária
    return vecs[:, :n_cuts + 1].astype(np.float32)     # (P_total, n_cuts+1)


# ---------------------------------------------------------------------------
# Passo 4 — Orientação FG/BG
# ---------------------------------------------------------------------------

def _partition_from_fiedler(fiedler: np.ndarray) -> np.ndarray:
    """
    Converte o vetor de Fiedler em uma bipartição booleana sem orientação
    semântica. A mediana preserva o comportamento histórico da Etapa 2.
    """
    threshold = np.median(fiedler)
    return fiedler > threshold


def orient_partition_cls(
    fiedler: np.ndarray,
    saliency_i: np.ndarray,
    saliency_j: np.ndarray,
) -> np.ndarray:
    """
    Orientação histórica da Etapa 2.

    Escolhe como foreground o lado da bipartição com maior saliência CLS
    média, considerando conjuntamente as duas imagens do par.
    """
    saliency_all = np.concatenate([saliency_i, saliency_j])
    side_pos = _partition_from_fiedler(fiedler)

    sal_pos = saliency_all[side_pos].mean() if side_pos.any() else 0.0
    sal_neg = saliency_all[~side_pos].mean() if (~side_pos).any() else 0.0

    return side_pos if sal_pos >= sal_neg else ~side_pos


def _spatial_score_for_image_side(
    side_mask: np.ndarray,
    patch_indices: np.ndarray,
    patch_grid: tuple[int, int],
) -> float:
    """
    Score espacial de um lado da partição para UMA imagem.

    Replica a ideia usada na auditoria da Etapa 2:
      score = sqrt(centralidade * compacidade)

    - centralidade: favorece patches próximos ao centro da imagem;
    - compacidade: favorece regiões espacialmente concentradas.

    patch_indices contém os índices na grade COMPLETA, portanto o cálculo
    continua correto quando patches de padding foram removidos do grafo.
    """
    if not side_mask.any():
        return 0.0

    H_p, W_p = patch_grid
    rows = (patch_indices // W_p).astype(np.float32) / max(H_p - 1, 1)
    cols = (patch_indices % W_p).astype(np.float32) / max(W_p - 1, 1)

    rows_sel = rows[side_mask]
    cols_sel = cols[side_mask]

    max_dist = np.sqrt(0.5 ** 2 + 0.5 ** 2)
    dist_center = np.sqrt((rows_sel - 0.5) ** 2 + (cols_sel - 0.5) ** 2) / max_dist
    centrality = 1.0 - float(dist_center.mean())

    std_r = float(rows_sel.std())
    std_c = float(cols_sel.std())

    # Mesma normalização aproximada usada no analisador da Etapa 2.
    compactness = 1.0 - min(((std_r + std_c) / 2.0) / 0.289, 1.0)

    return float(np.sqrt(max(centrality, 0.0) * max(compactness, 0.0)))


def orient_partition_spatial(
    fiedler: np.ndarray,
    n_patches_i: int,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    patch_grid: tuple[int, int],
) -> np.ndarray:
    """
    Orientação espacial para ablação.

    O NCut é calculado sobre o par combinado, portanto a orientação precisa
    continuar sendo ÚNICA para as duas imagens. Para cada lado da bipartição,
    calculamos o score espacial separadamente em i e j e usamos a média dos
    scores das duas imagens. O lado mais central/compacto no par é definido
    como foreground.

    Isso preserva a arquitetura original: uma decisão de polaridade por par,
    sem usar GT e sem orientar cada imagem independentemente.
    """
    side_pos = _partition_from_fiedler(fiedler)

    pos_i = side_pos[:n_patches_i]
    pos_j = side_pos[n_patches_i:]

    score_pos_i = _spatial_score_for_image_side(pos_i, idx_i, patch_grid)
    score_pos_j = _spatial_score_for_image_side(pos_j, idx_j, patch_grid)
    score_neg_i = _spatial_score_for_image_side(~pos_i, idx_i, patch_grid)
    score_neg_j = _spatial_score_for_image_side(~pos_j, idx_j, patch_grid)

    score_pos = 0.5 * (score_pos_i + score_pos_j)
    score_neg = 0.5 * (score_neg_i + score_neg_j)

    return side_pos if score_pos >= score_neg else ~side_pos


def orient_partition(
    fiedler: np.ndarray,
    n_patches_i: int,
    saliency_i: np.ndarray,
    saliency_j: np.ndarray,
    *,
    method: str = "cls",
    idx_i: "np.ndarray | None" = None,
    idx_j: "np.ndarray | None" = None,
    patch_grid: "tuple[int, int] | None" = None,
) -> np.ndarray:
    """
    Seleciona a estratégia de orientação FG/BG.

    method='cls'
        Comportamento histórico e DEFAULT. Escolhe o lado com maior
        saliência CLS média.

    method='spatial'
        Estratégia experimental para ablação. Escolhe, de forma conjunta
        para o par, o lado mais central/compacto.

    O default 'cls' garante compatibilidade retroativa com chamadas antigas
    a orient_partition().
    """
    method = str(method).lower()

    if method == "cls":
        return orient_partition_cls(fiedler, saliency_i, saliency_j)

    if method == "spatial":
        if idx_i is None or idx_j is None or patch_grid is None:
            raise ValueError(
                "orientation='spatial' requer idx_i, idx_j e patch_grid."
            )
        return orient_partition_spatial(
            fiedler,
            n_patches_i,
            idx_i,
            idx_j,
            patch_grid,
        )

    raise ValueError(
        f"Método de orientação desconhecido: {method!r}. "
        "Use 'cls' ou 'spatial'."
    )


def compute_cls_saliency(patch_features: np.ndarray,
                         cls_features: np.ndarray) -> np.ndarray:
    """
    Similaridade coseno de cada patch com o CLS token da sua imagem.
    Retorna (N, P) — proxy de saliência sem supervisão.
    (Idêntico ao que a Etapa 1 usa nas visualizações.)
    """
    N, P, D = patch_features.shape
    cls_n   = sk_normalize(cls_features, axis=1)             # (N, D)
    pat_n   = sk_normalize(patch_features.reshape(-1, D), axis=1).reshape(N, P, D)
    cos_sim = np.einsum("nd,npd->np", cls_n, pat_n)          # (N, P)
    return cos_sim.astype(np.float32)


# ---------------------------------------------------------------------------
# Passo 5 — Votação e fg_scores finais
# ---------------------------------------------------------------------------

def aggregate_votes(pair_votes: np.ndarray,
                    top_pairs: list[tuple[int, int, float]],
                    N: int, P: int) -> np.ndarray:
    """
    Agrega os votos de todos os pares em fg_scores[N, P] ∈ [0, 1].

    Estratégia de ponderação: cada par vota com peso = cls_sim do par.
    Isso dá mais peso a pares de imagens genuinamente similares.

    Patches de padding, quando mascarados pelo chamador, chegam aqui com
    voto sempre 0 (nunca participaram do NCut) — a média ponderada resulta
    em fg_scores=0.0 (background confiante) para eles, e não no fallback
    "incerto" de 0.5 aplicado a patches nunca votados de fato.

    pair_votes: (K, N, P) — voto binário por par
    Retorna: fg_scores (N, P)
    """
    weights = np.array([s for (_, _, s) in top_pairs], dtype=np.float32)
    # Normaliza pesos para soma = 1 por imagem
    # (uma imagem pode aparecer em poucos pares — normalizamos pelo total de peso recebido)
    weight_sum = np.zeros((N, P), dtype=np.float32)
    fg_weighted = np.zeros((N, P), dtype=np.float32)

    for k, (i, j, sim) in enumerate(top_pairs):
        w = max(sim, 0.0)   # ignora pares negativamente similares
        fg_weighted[i] += pair_votes[k, i] * w
        fg_weighted[j] += pair_votes[k, j] * w
        weight_sum[i]  += w
        weight_sum[j]  += w

    # Evita divisão por zero (patches nunca votados ficam com score 0.5 = incerto).
    # np.divide com where= computa a divisão só onde weight_sum>0, evitando o
    # 0/0 que np.where(cond, a/b, fallback) calcularia primeiro (e descartaria
    # depois) — mesmo resultado, sem o RuntimeWarning "invalid value encountered
    # in divide".
    fg_scores = np.divide(
        fg_weighted, weight_sum,
        out=np.full_like(fg_weighted, 0.5),
        where=weight_sum > 0,
    )
    return fg_scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_correspondence(
    patch_features : np.ndarray,          # (N, P, D)
    cls_features   : np.ndarray,          # (N, D)
    patch_grid     : tuple[int, int],      # (H_p, W_p)
    top_k          : int   = 10,
    temperature    : float = 0.05,
    content_mask   : "np.ndarray | None" = None,   # (N, P) bool — True = patch real
    orientation    : str = "cls",                  # "cls" (default histórico) | "spatial"
    degree_alpha   : float = 0.0,                    # reponderação por grau pré-NCut
) -> dict:
    """
    Executa o pipeline completo de correspondência cross-image.
    Retorna um dicionário com todos os artefatos intermediários e finais.

    Se content_mask for fornecida, patches de padding (fora dela) são
    excluídos do grafo de afinidade do NCut e da orientação FG/BG de cada
    par — participam apenas patches de conteúdo real. O voto final desses
    patches é forçado a 0 (background), consistente com o GT (que também
    trata a área de padding como fundo).

    orientation='cls' preserva o comportamento histórico. orientation='spatial'
    ativa apenas a nova heurística de orientação.

    degree_alpha controla a reponderação pré-NCut do grafo de afinidade:
      0.0  = baseline histórico
      0.10/0.25 = normalização parcial, alvo principal desta ablação
      0.50 = normalização simétrica forte
    """
    N, P, D = patch_features.shape
    H_p, W_p = patch_grid

    if not (0.0 <= degree_alpha <= 0.5):
        raise ValueError(
            f"degree_alpha deve estar em [0, 0.5], recebido: {degree_alpha}"
        )

    if content_mask is not None:
        frac_padding = 1.0 - float(content_mask.mean())
        if frac_padding > 1e-6:
            print(f"[Padding] {frac_padding:.1%} dos patches identificados como padding "
                  f"(resize_info do modo keep_ar) — excluídos do grafo NCut e da orientação.")

    # Pré-computa saliência CLS (usada na orientação FG/BG)
    cls_saliency = compute_cls_saliency(patch_features, cls_features)  # (N, P)

    print(f"\n[Passo 1] Similaridade CLS ({N}×{N})...")
    cls_sim   = compute_cls_similarity(cls_features)
    top_pairs = select_top_pairs(cls_sim, top_k)
    K         = len(top_pairs)

    # Armazena votos individuais de cada par: pair_votes[k, i/j, :] = voto binário
    pair_votes = np.zeros((K, N, P), dtype=np.float32)

    print(f"\n[Passos 2-4] Cross-attention + NCut + orientação FG ({K} pares)...")
    print(f"[Orientação] método = {orientation}")
    for rank, (i, j, sim_score) in enumerate(top_pairs):
        feat_i = patch_features[i]   # (P, D)
        feat_j = patch_features[j]   # (P, D)
        sal_i  = cls_saliency[i]     # (P,)
        sal_j  = cls_saliency[j]     # (P,)

        if content_mask is not None:
            idx_i = np.flatnonzero(content_mask[i])
            idx_j = np.flatnonzero(content_mask[j])
            if idx_i.size == 0 or idx_j.size == 0:
                # Salvaguarda: imagem sem nenhum patch de conteúdo detectado
                # (não deveria acontecer na prática) — cai de volta para
                # todos os patches, para não travar o pipeline.
                idx_i = np.arange(P)
                idx_j = np.arange(P)
        else:
            idx_i = np.arange(P)
            idx_j = np.arange(P)

        feat_i_c = feat_i[idx_i]
        feat_j_c = feat_j[idx_j]
        sal_i_c  = sal_i[idx_i]
        sal_j_c  = sal_j[idx_j]
        Pi_c     = idx_i.size

        # Grafo combinado, só com patches de conteúdo (2*Pi_c/Pj_c em vez de 2P)
        W = build_pair_affinity(
            feat_i_c,
            feat_j_c,
            temperature=temperature,
            degree_alpha=degree_alpha,
        )

        # NCut → Fiedler vector
        eigvecs = normalized_cut(W, n_cuts=1)
        fiedler = eigvecs[:, 1]      # (Pi_c+Pj_c,)  — a partição principal

        # Orientação FG via saliência (só entre patches de conteúdo)
        fg_mask = orient_partition(
            fiedler,
            Pi_c,
            sal_i_c,
            sal_j_c,
            method=orientation,
            idx_i=idx_i,
            idx_j=idx_j,
            patch_grid=patch_grid,
        )                             # (Pi_c+Pj_c,) booleano

        # Separa votos das duas imagens (ainda em espaço de conteúdo)
        fg_i_c = fg_mask[:Pi_c].astype(np.float32)
        fg_j_c = fg_mask[Pi_c:].astype(np.float32)

        # Espalha de volta para o espaço completo (P,). Patches de padding
        # ficam em 0 (background) por construção — nunca participaram da
        # decisão do NCut.
        fg_i = np.zeros(P, dtype=np.float32)
        fg_j = np.zeros(P, dtype=np.float32)
        fg_i[idx_i] = fg_i_c
        fg_j[idx_j] = fg_j_c

        pair_votes[rank, i] = fg_i
        pair_votes[rank, j] = fg_j

        # Log de progresso
        pct_fg_i = fg_i.mean() * 100
        pct_fg_j = fg_j.mean() * 100
        print(f"  [{rank+1:>2}/{K}] par ({i},{j})  sim={sim_score:+.4f}  "
              f"FG%: img[{i}]={pct_fg_i:.0f}%  img[{j}]={pct_fg_j:.0f}%")

    print(f"\n[Passo 5] Agregando votos ponderados por similaridade CLS...")
    fg_scores    = aggregate_votes(pair_votes, top_pairs, N, P)   # (N, P)
    fg_scores_2d = fg_scores.reshape(N, H_p, W_p)                 # (N, H_p, W_p)

    # Estatísticas rápidas
    mean_fg = fg_scores.mean(axis=1)   # fração média de FG por imagem
    print(f"  FG scores médios por imagem: "
          f"min={mean_fg.min():.2f}  max={mean_fg.max():.2f}  "
          f"média={mean_fg.mean():.2f}")

    return {
        "fg_scores"      : fg_scores,           # (N, P)
        "fg_scores_2d"   : fg_scores_2d,        # (N, H_p, W_p)
        "cls_sim_matrix" : cls_sim,             # (N, N)
        "top_pairs"      : np.array([(i, j, s) for i, j, s in top_pairs]),  # (K, 3)
        "pair_votes"     : pair_votes,          # (K, N, P)
        "cls_saliency"   : cls_saliency,        # (N, P) — para referência na Etapa 3
        "patch_grid"     : np.array(patch_grid),
        "orientation"    : str(orientation),
        "degree_alpha"   : float(degree_alpha),
    }


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def save_correspondence(result: dict, output_dir: Path,
                        image_paths: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "correspondence.npz"
    np.savez_compressed(
        out,
        fg_scores      = result["fg_scores"],
        fg_scores_2d   = result["fg_scores_2d"],
        cls_sim_matrix = result["cls_sim_matrix"],
        top_pairs      = result["top_pairs"],
        pair_votes     = result["pair_votes"],
        cls_saliency   = result["cls_saliency"],
        patch_grid     = result["patch_grid"],
        image_paths    = image_paths,
        orientation    = np.array(result.get("orientation", "cls")),
        degree_alpha   = np.array(result.get("degree_alpha", 0.0), dtype=np.float32),
    )
    size_mb = out.stat().st_size / 1024 ** 2
    print(f"\n[Salvo] {out}  ({size_mb:.1f} MB)")
    return out


def load_correspondence(path: Path) -> dict:
    """Interface de carregamento para a Etapa 3."""
    data = np.load(path, allow_pickle=True)
    return {
        "fg_scores"      : data["fg_scores"],
        "fg_scores_2d"   : data["fg_scores_2d"],
        "cls_sim_matrix" : data["cls_sim_matrix"],
        "top_pairs"      : data["top_pairs"],
        "pair_votes"     : data["pair_votes"],
        "cls_saliency"   : data["cls_saliency"],
        "patch_grid"     : tuple(int(x) for x in data["patch_grid"]),
        "image_paths"    : list(data["image_paths"]),
        "orientation"    : str(data["orientation"].item()) if "orientation" in data.files else "cls",
        "degree_alpha"   : float(data["degree_alpha"].item()) if "degree_alpha" in data.files else 0.0,
    }


# ---------------------------------------------------------------------------
# Sumário
# ---------------------------------------------------------------------------

def print_summary(result: dict, image_paths: list[str]):
    fg    = result["fg_scores"]      # (N, P)
    pairs = result["top_pairs"]      # (K, 3)
    N, P  = fg.shape

    print("\n" + "=" * 58)
    print("  SUMÁRIO ETAPA 2")
    print("=" * 58)
    print(f"  Imagens                : {N}")
    print(f"  Patches por imagem     : {P}")
    print(f"  Pares processados      : {len(pairs)}")
    print(f"  Sim. CLS média (pares) : {pairs[:, 2].mean():.4f}")
    print(f"  Orientação FG/BG       : {result.get('orientation', 'cls')}")
    print(f"  Degree alpha pré-NCut  : {result.get('degree_alpha', 0.0):.3f}")
    print()
    print(f"  {'IMG':>4}  {'FG%':>6}  {'NOME'}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*35}")
    for i in range(N):
        pct = fg[i].mean() * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        name = Path(image_paths[i]).name[:35]
        print(f"  {i:>4}  {pct:>5.1f}%  {bar}  {name}")
    print("=" * 58)
    print()
    print("  O que fg_scores guardam para a Etapa 3:")
    print("  • fg_scores[i, p] ∈ [0,1] — probabilidade de FG do patch p")
    print("    0.5 = incerto (patch não participou de nenhum par)")
    print("    0.0 = background confiante (inclui patches de padding, se mascarados)")
    print("    > 0.5 = provavelmente FG  |  < 0.5 = provavelmente BG")
    print("  • pair_votes[k, i, :] — votos individuais por par (debug)")
    print("  • cls_saliency  — saliência DINOv2 bruta (referência)")
    print()


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def visualize_correspondence(result: dict, image_paths: list[str],
                              output_dir: Path, max_images: int = 8):
    """
    Gera quatro visualizações:

    1. FG scores por imagem (mapa de calor em cima da imagem original)
       → O que a Etapa 2 "acha" que é objeto de interesse em cada imagem.
         Compara diretamente com a saliência CLS da Etapa 1.

    2. Matriz de similaridade CLS (N×N heatmap)
       → Mostra quão similares as imagens são globalmente.
         Pares escuros = muito diferentes; pares claros = candidatos ao co-objeto.

    3. Detalhe de um par: grafo de correspondência (top-matches por patch)
       → Para o par mais similar, mostra os N patches de img_i com sua
         melhor correspondência em img_j. Patches da mesma cor = correspondentes.

    4. FG score vs. saliência CLS (scatter plot por imagem)
       → Relação entre a saliência do modelo individual (Etapa 1) e o score
         cross-image (Etapa 2). Correlação alta = as duas concordam = bom sinal.
    """
    fg_2d     = result["fg_scores_2d"]    # (N, H_p, W_p)
    cls_sal   = result["cls_saliency"].reshape(fg_2d.shape)  # (N, H_p, W_p)
    cls_sim   = result["cls_sim_matrix"]  # (N, N)
    top_pairs = result["top_pairs"]       # (K, 3)
    H_p, W_p  = result["patch_grid"]

    N_vis = min(len(image_paths), max_images)

    # ── Fig 1: FG scores por imagem ─────────────────────────────────────────
    fig, axes = plt.subplots(N_vis, 3, figsize=(13, 3.5 * N_vis))
    if N_vis == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "Etapa 2 — FG Scores (cross-image correspondence)\n"
        "Esq: imagem original  |  Centro: saliência CLS (Etapa 1)  |"
        "  Dir: FG score Etapa 2",
        fontsize=11, y=1.01
    )
    for i in range(N_vis):
        img = Image.open(image_paths[i]).convert("RGB")

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(Path(image_paths[i]).name[:28], fontsize=8)
        axes[i, 0].axis("off")

        im1 = axes[i, 1].imshow(cls_sal[i], cmap="viridis",
                                 interpolation="nearest", vmin=0, vmax=1)
        axes[i, 1].set_title(f"CLS saliency (Etapa 1)", fontsize=8)
        axes[i, 1].axis("off")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(fg_2d[i], cmap="RdYlGn",
                                 interpolation="nearest", vmin=0, vmax=1)
        axes[i, 2].set_title(f"FG score Etapa 2  (mean={fg_2d[i].mean():.2f})", fontsize=8)
        axes[i, 2].axis("off")
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    out1 = output_dir / "viz_fg_scores.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Fig 2: Matriz de similaridade CLS ───────────────────────────────────
    N = len(image_paths)
    fig2, ax = plt.subplots(figsize=(min(N * 0.6 + 1, 14), min(N * 0.6 + 1, 14)))
    im = ax.imshow(cls_sim, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [Path(p).stem[:12] for p in image_paths]
    ax.set_xticks(range(N)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels(labels, fontsize=7)
    # Marca os pares selecionados
    K_show = min(len(top_pairs), 10)
    for k in range(K_show):
        i, j = int(top_pairs[k, 0]), int(top_pairs[k, 1])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor="lime", lw=2))
        ax.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1,
                                   fill=False, edgecolor="lime", lw=2))
    ax.set_title(
        f"Similaridade CLS entre imagens  (verde = pares selecionados top-{K_show})",
        fontsize=10
    )
    fig2.tight_layout()
    out2 = output_dir / "viz_cls_similarity.png"
    fig2.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # ── Fig 3: Detalhe do melhor par (correspondências de patches) ───────────
    if len(top_pairs) > 0:
        best_i = int(top_pairs[0, 0])
        best_j = int(top_pairs[0, 1])
        best_s = float(top_pairs[0, 2])

        # Votes desse par
        fg_i_2d = result["pair_votes"][0, best_i].reshape(H_p, W_p)
        fg_j_2d = result["pair_votes"][0, best_j].reshape(H_p, W_p)

        fig3, axes3 = plt.subplots(2, 3, figsize=(13, 8))
        fig3.suptitle(
            f"Etapa 2 — Detalhe do par mais similar\n"
            f"img[{best_i}] × img[{best_j}]  (CLS sim = {best_s:.4f})",
            fontsize=11
        )
        for col, (idx, fg_map) in enumerate([(best_i, fg_i_2d), (best_j, fg_j_2d)]):
            img = Image.open(image_paths[idx]).convert("RGB")
            axes3[0, col].imshow(img)
            axes3[0, col].set_title(f"img[{idx}]: {Path(image_paths[idx]).name[:25]}", fontsize=8)
            axes3[0, col].axis("off")

            im_ = axes3[1, col].imshow(fg_map, cmap="RdYlGn",
                                        interpolation="nearest", vmin=0, vmax=1)
            axes3[1, col].set_title(f"NCut FG vote  (FG%={fg_map.mean()*100:.0f}%)", fontsize=8)
            axes3[1, col].axis("off")
            plt.colorbar(im_, ax=axes3[1, col], fraction=0.046, pad=0.04)

        # Terceira coluna: diferença de scores FG entre os dois
        axes3[0, 2].imshow(np.abs(fg_i_2d - fg_j_2d), cmap="Reds",
                           interpolation="nearest", vmin=0, vmax=1)
        axes3[0, 2].set_title("|FG_i − FG_j|  (discordância entre o par)", fontsize=8)
        axes3[0, 2].axis("off")

        # FG score agregado (ambas as imagens sobrepostas)
        fg_agg = (fg_i_2d + fg_j_2d) / 2
        im_ = axes3[1, 2].imshow(fg_agg, cmap="RdYlGn",
                                  interpolation="nearest", vmin=0, vmax=1)
        axes3[1, 2].set_title("FG score médio do par", fontsize=8)
        axes3[1, 2].axis("off")
        plt.colorbar(im_, ax=axes3[1, 2], fraction=0.046, pad=0.04)

        fig3.tight_layout()
        out3 = output_dir / "viz_best_pair.png"
        fig3.savefig(out3, dpi=120, bbox_inches="tight")
        plt.close(fig3)
    else:
        out3 = None

    # ── Fig 4: FG score vs. saliência CLS (scatter por imagem) ─────────────
    fig4, axes4 = plt.subplots(
        1, min(N_vis, 4),
        figsize=(4 * min(N_vis, 4), 4),
        squeeze=False
    )
    fig4.suptitle(
        "FG score (Etapa 2) vs. Saliência CLS (Etapa 1)\n"
        "Correlação alta = as duas perspectivas concordam",
        fontsize=10
    )
    for idx in range(min(N_vis, 4)):
        ax = axes4[0, idx]
        x = cls_sal[idx].flatten()
        y = fg_2d[idx].flatten()
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
        # Density scatter (hex)
        ax.hexbin(x, y, gridsize=25, cmap="YlOrRd", mincnt=1)
        ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.axvline(float(np.median(x)), color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.set_xlabel("CLS saliency (E1)", fontsize=8)
        ax.set_ylabel("FG score (E2)", fontsize=8)
        ax.set_title(f"img[{idx}]  r={corr:+.2f}", fontsize=9)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    fig4.tight_layout()
    out4 = output_dir / "viz_score_correlation.png"
    fig4.savefig(out4, dpi=120, bbox_inches="tight")
    plt.close(fig4)

    # Relatório
    outs = [out1, out2, out3, out4]
    print("\n[Viz]")
    for o in outs:
        if o and o.exists():
            print(f"  {o}  ({o.stat().st_size // 1024} KB)")
    return outs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Coseg Etapa 2 — cross-image correspondence (NCut + votação)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--features", "-f", required=True,
        help="Path para features.npz da Etapa 1\n"
             "Ex: features/Panda/features.npz"
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Pasta de saída (default: mesma pasta do features.npz)"
    )
    p.add_argument(
        "--top_k", type=int, default=10,
        help="Número de pares a processar (default: 10)\n"
             "Mais pares = mais robusto, mas mais lento"
    )
    p.add_argument(
        "--temperature", type=float, default=0.05,
        help="Temperatura do softmax de afinidade (default: 0.05)\n"
             "Menor = correspondências mais nítidas"
    )
    p.add_argument(
        "--orientation",
        choices=["cls", "spatial"],
        default="cls",
        help="Heurística para orientar os dois lados do NCut como FG/BG.\n"
             "  cls     = comportamento histórico/default: maior saliência CLS média\n"
             "  spatial = ablação: lado mais central e compacto no par\n"
             "O default 'cls' preserva integralmente o comportamento anterior."
    )
    p.add_argument(
        "--degree_alpha",
        type=float,
        default=0.0,
        help="Reponderação por grau aplicada ao grafo antes do NCut.\n"
             "  0.0  = baseline histórico (default)\n"
             "  0.10 = normalização parcial suave\n"
             "  0.25 = normalização parcial moderada\n"
             "  0.50 = normalização simétrica forte\n"
             "Faixa permitida: [0, 0.5]."
    )
    p.add_argument(
        "--no_padding_mask", action="store_true",
        help="Desativa o mascaramento de patches de padding (modo --resize keep_ar\n"
             "da Etapa 1) na construção do grafo NCut. Por padrão, patches de\n"
             "padding são excluídos do grafo e forçados a background."
    )
    p.add_argument(
        "--visualize", action="store_true",
        help="Gera visualizações interpretáveis"
    )
    p.add_argument(
        "--viz_max", type=int, default=8,
        help="Máx imagens nas visualizações (default: 8)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    features_path = Path(args.features)
    output_dir    = Path(args.output) if args.output else features_path.parent

    # --- Carrega Etapa 1 ---------------------------------------------------
    feats = load_step1(features_path)

    patch_features = feats["patch_features"]   # (N, P, D)
    cls_features   = feats["cls_features"]     # (N, D)
    patch_grid     = feats["patch_grid"]       # (H_p, W_p)
    image_paths    = feats["image_paths"]
    resize_info    = feats.get("resize_info")
    N              = len(image_paths)

    top_k = min(args.top_k, N * (N - 1) // 2)  # nunca mais pares do que existem

    content_mask = None
    if not args.no_padding_mask:
        content_mask = compute_content_mask(resize_info, patch_grid)

    print(f"\n{'='*58}")
    print(f"  Coseg — Etapa 2: Cross-Image Correspondence")
    print(f"{'='*58}")
    print(f"  Features  : {features_path}")
    print(f"  Imagens   : {N}  |  Patches: {patch_features.shape[1]}")
    print(f"  Top-K     : {top_k} pares")
    print(f"  Temperatura: {args.temperature}")
    print(f"  Orientação : {args.orientation}")
    print(f"  Degree α   : {args.degree_alpha}")
    if resize_info is None:
        print(f"  Padding   : sem resize_info (modo 'square' na Etapa 1) — nada a mascarar")
    elif content_mask is None:
        print(f"  Padding   : mascaramento desativado (--no_padding_mask)")
    else:
        print(f"  Padding   : mascarando {1.0 - float(content_mask.mean()):.1%} dos patches (padding)")
    print(f"  Saída     : {output_dir}")

    # --- Pipeline ----------------------------------------------------------
    result = run_correspondence(
        patch_features = patch_features,
        cls_features   = cls_features,
        patch_grid     = patch_grid,
        top_k          = top_k,
        temperature    = args.temperature,
        content_mask   = content_mask,
        orientation    = args.orientation,
        degree_alpha   = args.degree_alpha,
    )

    save_correspondence(result, output_dir, image_paths)
    print_summary(result, image_paths)

    if args.visualize:
        visualize_correspondence(result, image_paths, output_dir,
                                 max_images=args.viz_max)

    print(f"[Etapa 2 concluída]  →  {output_dir}/correspondence.npz")
    print(f"\n  Próximo passo (Etapa 3):")
    print(f"    from step2_correspondence import load_correspondence")
    print(f"    corr = load_correspondence('{output_dir}/correspondence.npz')")
    print(f"    # corr['fg_scores'] (N, P) — seeds para Graph Cuts / SAM")


if __name__ == "__main__":
    main()