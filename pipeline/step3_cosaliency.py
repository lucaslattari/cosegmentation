"""
=============================================================================
Cosegmentation Pipeline — Etapa 3: Co-Saliency via Spectral Clustering
=============================================================================

PAPEL DESTA ETAPA NO PIPELINE:
  Entrada : features.npz       (Etapa 1)
            correspondence.npz (Etapa 2)
  Saída   : cosaliency.npz     — máscaras binárias por imagem + seeds para
              a Etapa 4 (SAM prompting / Graph Cuts de alta resolução)

COMO RODAR:
  python step3_cosaliency.py --features features/Panda/features.npz \
                             --correspondence features/Panda/correspondence.npz

  python step3_cosaliency.py --features features/Panda/features.npz \
                             --correspondence features/Panda/correspondence.npz \
                             --visualize

  # Ajuste de parâmetros:
  python step3_cosaliency.py -f features/Kite/features.npz \
                             -c features/Kite/correspondence.npz \
                             --n_clusters 6 --k_neighbors 12 --visualize

  # Ajuste do prior de saliência espacial (novo):
  #   --saliency_alpha 0.0  → só score da E2 (comportamento antigo)
  #   --saliency_alpha 1.0  → só saliência espacial
  #   --saliency_alpha 0.5  → média balanceada (default)
  python step3_cosaliency.py -f ... -c ... --saliency_alpha 0.6 --visualize

O QUE ESTA ETAPA FAZ E POR QUE É DIFERENTE DA ETAPA 2:

  A Etapa 2 operou em PARES de imagens (grafo 2P × 2P) e acumulou votos.
  O resultado — fg_scores (N, P) — é um score contínuo e probabilístico.

  A Etapa 3 opera na COLEÇÃO INTEIRA simultaneamente (grafo global N×P nós),
  o que captura estruturas que só emergem quando todas as imagens são
  consideradas ao mesmo tempo. É o análogo moderno do "super cluster" do
  paper original (Lattari et al., ICIP 2015), mas usando spectral embedding
  em vez de GMM com distância de Bhattacharyya.

  Algoritmo:

  A. GRAFO GLOBAL kNN
     Constrói um grafo esparso onde cada patch é um nó e as arestas são as
     k similaridades coseno mais altas com outros patches (de qualquer imagem).
     Custo: O(N·P·k) arestas — viável até coleções de ~50 imagens.

  B. SPECTRAL EMBEDDING
     Computa os k menores autovetores do Laplaciano normalizado simétrico
     usando eigsh esparso (scipy). Cada patch vira um vetor k-dimensional
     onde a distância euclidiana = distância geodésica no grafo.

  C. K-MEANS ESPECTRAL
     Agrupa os patches no espaço espectral em n_clusters clusters globais.

  D. SALIÊNCIA ESPACIAL POR CLUSTER (novo)
     Para cada cluster, calcula dois critérios complementares ao score da E2:

     1. Centralidade: distância média dos patches do cluster ao centro da
        imagem. Clusters de objeto tendem a estar mais centralizados que
        clusters de céu/chão/borda.

     2. Compacidade: inverso da dispersão espacial normalizada dos patches.
        Objetos formam regiões compactas; fundos estruturados (bandas
        horizontais de céu/chão) têm patches espalhados por toda a imagem.

     O score final de FG é uma combinação convexa:
       score_final = (1 - α) * score_e2 + α * score_spatial
     onde α = saliency_alpha (default 0.5).

     Isso resolve o problema de clusters de fundo estruturado (bandas de
     céu, chão, vegetação periférica) que têm score_e2 alto porque aparecem
     consistentemente nas duas imagens, mas não são o objeto de interesse.

  E. MAPEAMENTO FG/BG
     Clusters com score_final > threshold = FG, demais = BG.

  F. REFINAMENTO INTRA-IMAGEM — EXPERIMENTAL E DESATIVADO POR PADRÃO

    Opcionalmente executa um novo corte NCut por imagem. Essa operação não é
    um refinamento estritamente condicionado pela máscara coarse e pode substituir
    uma máscara global de boa qualidade por uma partição local inferior.

    Ative explicitamente com --refine.

  G. UPSAMPLE + SUAVIZAÇÃO
     A máscara binária por patch é reamostrada para o tamanho original.

  H. EMPTY-FOREGROUND ADAPTIVE BYPASS (validado em PASCAL + controles iCoseg)
     Após a decisão normal da E3, cada imagem é inspecionada individualmente.
     Se a máscara E3 não contiver NENHUM patch FG válido, considera-se que a
     etapa entrou em um estado degenerado de foreground vazio. Nesse caso,
     somente para essa imagem, a máscara é substituída pelos patches com os
     maiores scores da E2B (top-20% por padrão).

     Regra padrão validada:
       - se E3 tem >= 1 patch FG válido: mantém E3 integralmente;
       - se E3 tem 0 patches FG válidos: usa top-20% dos scores E2B.

     O gatilho é propositalmente exato (máscara vazia), e não FG <= 1%.
     Na ablação, FG <= 1% degradou Kite-Brighton; o gatilho de máscara vazia
     preservou os controles iCoseg e manteve os ganhos no PASCAL.

     Desative com:
       --empty_fg_bypass none

     Ajuste a fração com:
       --empty_fg_top_fraction 0.20

  I. SEEDS PARA ETAPA 4
     Bounding box, pontos FG, pontos BG prontos para o SAM.

  MASCARAMENTO DE PADDING (quando a Etapa 1 rodou com --resize keep_ar):
     A entrada dos patches que caem inteiramente na faixa de padding (cor
     média do ImageNet, ~zero após normalização) é praticamente constante
     entre imagens, gerando afinidade cross-image artificialmente alta.
     Sem mascaramento, esses patches podem consumir clusters inteiros do
     orçamento de n_clusters (confirmado empiricamente: coleções com muito
     padding e objeto pequeno chegaram a gastar 4 de 5 clusters só com
     padding, deixando zero clusters de conteúdo real disponíveis para FG).
     Por padrão, esses patches são excluídos tanto do grafo global (Passo A)
     quanto do NCut de refino intra-imagem (Passo F), e são sempre forçados
     a background na máscara final. Desative com --no_padding_mask.

SAÍDA (cosaliency.npz):
  fg_mask_patch      : (N, H_p, W_p)  — máscara binária em resolução de patch
  fg_mask_pixel      : (N, H, W)      — máscara upsampled (tamanho original)
  fg_score_global    : (N, P)         — score espectral contínuo (pré-threshold)
  cluster_labels     : (N, P)         — índice de cluster por patch (patches de
                        padding, quando mascarados, recebem o rótulo arbitrário
                        0 apenas para manter o array indexável — a máscara final
                        os força para background independentemente disso)
  cluster_scores     : (n_clusters,)  — score FG final por cluster
  cluster_scores_e2  : (n_clusters,)  — score bruto da E2 (novo)
  cluster_scores_sp  : (n_clusters,)  — score espacial (novo)
  cluster_is_fg      : (n_clusters,)  — bool FG/BG
  n_clusters_used    : int
  empty_fg_bypass_mode        : str — "e2_topk" ou "none"
  empty_fg_top_fraction       : float — fração E2B usada no fallback
  empty_fg_bypass_triggered   : (N,) bool — imagens em que o fallback disparou
  empty_fg_bypass_trigger_count: int — total de imagens corrigidas
  empty_fg_before_patch_fraction: (N,) — fração FG válida antes do bypass
  empty_fg_after_patch_fraction : (N,) — fração FG válida depois do bypass
  seeds              : lista de dicts — {bbox, fg_points, bg_points} por imagem
  image_paths        : lista de paths
  image_sizes        : (N, 2)

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
from scipy.sparse import csr_matrix, diags as sp_diags
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as sk_normalize

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Interfaces de carga (Etapas 1 e 2)
# ---------------------------------------------------------------------------

def load_step1(path: Path) -> dict:
    if not path.exists():
        print(f"[ERRO] features.npz não encontrado: {path}")
        print(f"       Rode: python step1_feature_extraction.py --collection <classe>")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    return {
        "patch_features": data["patch_features"],                   # (N, P, D)
        "cls_features"  : data["cls_features"],                     # (N, D)
        "patch_grid"    : tuple(int(x) for x in data["patch_grid"]),
        "image_paths"   : list(data["image_paths"]),
        "image_sizes"   : data["image_sizes"],                      # (N, 2)
        # resize_info só existe se a Etapa 1 rodou com --resize keep_ar.
        # Colunas: resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w
        "resize_info"   : data["resize_info"] if "resize_info" in data.files else None,
    }


def load_step2(path: Path) -> dict:
    if not path.exists():
        print(f"[ERRO] correspondence.npz não encontrado: {path}")
        print(f"       Rode: python step2_correspondence.py --features <path>")
        sys.exit(1)
    data = np.load(path, allow_pickle=True)
    return {
        "fg_scores"    : data["fg_scores"],      # (N, P)
        "cls_saliency" : data["cls_saliency"],   # (N, P)
        "patch_grid"   : tuple(int(x) for x in data["patch_grid"]),
        "image_paths"  : list(data["image_paths"]),
    }


# ---------------------------------------------------------------------------
# Mascaramento de padding
# ---------------------------------------------------------------------------

def compute_content_mask(resize_info: "np.ndarray | None",
                         patch_grid: tuple) -> "np.ndarray | None":
    """
    A partir de resize_info (só existe quando a Etapa 1 rodou com
    --resize keep_ar), calcula, por imagem, quais patches da grade caem
    dentro da imagem real e quais caem inteira ou parcialmente na faixa de
    padding. Mesma lógica usada em step2_correspondence.py — duplicada
    aqui de propósito, seguindo o padrão de scripts autocontidos usado em
    todo o pipeline (step1/step2/step3/step4/step5 não compartilham um
    módulo utilitário comum).

    Retorna None se resize_info for None (Etapa 1 rodou em modo 'square',
    sem padding).
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


def compute_bypass_content_mask(
    resize_info: "np.ndarray | None",
    patch_grid: tuple,
) -> "np.ndarray | None":
    """
    Máscara de conteúdo usada EXCLUSIVAMENTE pelo empty-FG bypass.

    Esta lógica replica exatamente a ablação que validou o mecanismo:
    um patch é considerado válido se o CENTRO do patch cai dentro da região
    real da imagem no canvas keep_ar.

    Ela é deliberadamente separada de compute_content_mask(), que considera
    sobreposição parcial e continua sendo usada normalmente pelo grafo
    espectral da E3. Assim, incorporar o bypass não altera a E3 já calibrada.

    Retorna:
        (N, P) bool, ou None quando não existe resize_info.
    """
    if resize_info is None:
        return None

    H_p, W_p = patch_grid
    N = resize_info.shape[0]
    masks = np.zeros((N, H_p, W_p), dtype=bool)

    for i in range(N):
        resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = [
            float(x) for x in resize_info[i]
        ]
        ys = (np.arange(H_p) + 0.5) * canvas_h / H_p
        xs = (np.arange(W_p) + 0.5) * canvas_w / W_p

        valid_y = (ys >= pad_top) & (ys < pad_top + resized_h)
        valid_x = (xs >= pad_left) & (xs < pad_left + resized_w)
        masks[i] = valid_y[:, None] & valid_x[None, :]

    return masks.reshape(N, -1)


def apply_empty_fg_bypass(
    fg_masks_patch_flat: np.ndarray,
    fg_scores_e2: np.ndarray,
    bypass_content_mask: "np.ndarray | None" = None,
    mode: str = "e2_topk",
    top_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Corrige SOMENTE o estado degenerado em que uma imagem sai da E3 sem
    nenhum patch foreground válido.

    mode="e2_topk":
      - se houver >= 1 patch FG válido, preserva a máscara E3 sem alteração;
      - se houver 0 patches FG válidos, seleciona diretamente os top-K scores
        da E2/E2B daquela imagem, com K = ceil(top_fraction * patches_válidos).

    mode="none":
      - não modifica nenhuma imagem.

    O gatilho é exato: zero patches FG. Não existe limiar de 1% aqui.

    Retorna:
      masks_out       : (N, P) float32
      triggered       : (N,) bool
      fg_frac_before  : (N,) float32, calculada apenas em patches válidos
      fg_frac_after   : (N,) float32, calculada apenas em patches válidos
    """
    if mode not in {"none", "e2_topk"}:
        raise ValueError(f"empty_fg_bypass mode desconhecido: {mode!r}")
    if not (0.0 < float(top_fraction) <= 1.0):
        raise ValueError(
            "empty_fg_top_fraction deve estar no intervalo (0, 1]. "
            f"Recebido: {top_fraction}"
        )

    masks = np.asarray(fg_masks_patch_flat, dtype=np.float32).copy()
    scores = np.asarray(fg_scores_e2, dtype=np.float32)

    if masks.shape != scores.shape:
        raise ValueError(
            "Máscara E3 e scores E2/E2B incompatíveis no bypass: "
            f"{masks.shape} vs {scores.shape}"
        )

    N, P = masks.shape
    if bypass_content_mask is None:
        valid_all = np.ones((N, P), dtype=bool)
    else:
        valid_all = np.asarray(bypass_content_mask, dtype=bool)
        if valid_all.shape != (N, P):
            raise ValueError(
                "bypass_content_mask incompatível: "
                f"esperado {(N, P)}, recebido {valid_all.shape}"
            )

    triggered = np.zeros(N, dtype=bool)
    fg_frac_before = np.zeros(N, dtype=np.float32)
    fg_frac_after = np.zeros(N, dtype=np.float32)

    for i in range(N):
        valid = valid_all[i]
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            # Salvaguarda: se a máscara de conteúdo estiver vazia por alguma
            # inconsistência de resize_info, usa todos os patches.
            valid = np.ones(P, dtype=bool)
            valid_idx = np.arange(P)

        fg_before = (masks[i] > 0.5) & valid
        fg_frac_before[i] = float(fg_before.sum()) / float(valid_idx.size)

        if mode == "e2_topk" and fg_before.sum() == 0:
            triggered[i] = True

            n_take = max(1, int(np.ceil(float(top_fraction) * valid_idx.size)))
            score_i = np.nan_to_num(
                scores[i],
                nan=-np.inf,
                posinf=np.finfo(np.float32).max,
                neginf=-np.inf,
            )
            local_order = np.argpartition(score_i[valid_idx], -n_take)[-n_take:]
            chosen = valid_idx[local_order]

            masks[i] = 0.0
            masks[i, chosen] = 1.0

        # Patches fora da área válida do bypass nunca são introduzidos pelo
        # fallback. Se a E3 não disparou, a máscara original é preservada.
        fg_after = (masks[i] > 0.5) & valid
        fg_frac_after[i] = float(fg_after.sum()) / float(valid_idx.size)

    return masks, triggered, fg_frac_before, fg_frac_after


def compute_adaptive_n_clusters(
    base_n_clusters: int,
    content_mask: "np.ndarray | None",
    max_n_clusters: "int | None" = 12,
) -> int:
    """
    Quando o mascaramento de padding remove uma fração grande de patches do
    grafo, o orçamento de clusters calibrado para o grafo COM padding
    (base_n_clusters) fica insuficiente para representar o conteúdo real
    sem fragmentá-lo em pedaços que cruzam o threshold FG individualmente
    (confirmado empiricamente: Ferrari foi de 1 para 3 clusters FG após o
    mascaramento com n_clusters fixo, prejudicando patch_iou; escalar
    n_clusters recuperou ~75% da perda em Ferrari e Track and Field).

    Escala pela fração de conteúdo restante:
        n_clusters_efetivo = round(base_n_clusters / fração_de_conteúdo)

    Ex: 33% de padding (fração de conteúdo = 0.67) → ~1.5x mais clusters.

    Não recompõe perfeitamente o resultado sem máscara nas classes
    testadas (recupera ~75%, não 100%) — é uma mitigação, não uma
    correção completa do efeito de fragmentação.
    """
    if content_mask is None:
        return base_n_clusters

    content_fraction = float(content_mask.mean())
    if content_fraction >= 0.999:
        return base_n_clusters  # sem padding relevante, nada a escalar

    n = int(round(base_n_clusters / max(content_fraction, 0.1)))
    n = max(n, base_n_clusters)  # nunca reduz abaixo do valor pedido
    if max_n_clusters is not None:
        n = min(n, max_n_clusters)
    return n

def load_gt_index(gt_index_path: Path):  # pragma: no cover — não usado em produção
    pass


# ---------------------------------------------------------------------------
# Passo A — Grafo global kNN esparso
# ---------------------------------------------------------------------------

def build_global_graph(
    patch_features: np.ndarray,
    k_neighbors: int = 10,
    content_idx: "np.ndarray | None" = None,
) -> csr_matrix:
    """
    Constrói o grafo esparso kNN sobre os patches da coleção.

    Se content_idx for fornecido (índices flat, no espaço N*P, dos patches
    que NÃO são padding), o grafo é construído apenas sobre esses patches —
    patches de padding (entrada quase constante, cor média do ImageNet)
    nunca entram no grafo e não competem por clusters com o conteúdo real.
    Removê-los inteiramente (em vez de só zerar sua afinidade) evita criar
    dezenas de componentes desconectados no grafo, o que corromperia a
    seleção do vetor de Fiedler na etapa de embedding espectral.

    Se content_idx for None, comportamento idêntico ao original (todos os
    N*P patches participam).

    Retorna a matriz de afinidade esparsa M × M, onde M = N*P (sem máscara)
    ou M = len(content_idx) (com máscara).
    """
    N, P, D = patch_features.shape
    total   = N * P

    all_feats = sk_normalize(patch_features.reshape(total, D), axis=1)  # (total, D)

    if content_idx is not None:
        feats = all_feats[content_idx]
    else:
        feats = all_feats
    M = feats.shape[0]

    rows, cols, data_v = [], [], []
    for i in range(M):
        sims      = feats[i] @ feats.T   # (M,)
        sims[i]   = -1.0                           # exclui auto-aresta
        top_k_idx = np.argpartition(sims, -k_neighbors)[-k_neighbors:]

        for j in top_k_idx:
            w = float(sims[j])
            if w > 0:
                rows.extend([i,   int(j)])
                cols.extend([int(j), i  ])
                data_v.extend([w, w])

    W = csr_matrix((data_v, (rows, cols)), shape=(M, M))

    n_edges = W.nnz // 2
    if content_idx is not None:
        print(f"  Grafo: {M:,} nós de conteúdo (de {total:,} patches totais, "
              f"{total - M:,} de padding excluídos)  |  {n_edges:,} arestas  "
              f"(k={k_neighbors} vizinhos por patch)")
    else:
        print(f"  Grafo: {M:,} nós  |  {n_edges:,} arestas  "
              f"(k={k_neighbors} vizinhos por patch)")
    return W


# ---------------------------------------------------------------------------
# Passo B — Spectral embedding (eigsh esparso)
# ---------------------------------------------------------------------------

def spectral_embedding(W: csr_matrix, n_components: int = 8) -> np.ndarray:
    """
    Computa os n_components menores autovetores do Laplaciano normalizado.

    L_sym = I - D^{-1/2} W D^{-1/2}

    Usa scipy.sparse.linalg.eigsh que é O(nnz · n_components).

    O autovetor 0 é trivial (constante). Os autovetores 1..n_components formam
    o embedding espectral onde a geometria do grafo é preservada.

    Retorna: (M, n_components) — embedding normalizado por linha, onde M é
    o número de nós do grafo (patches de conteúdo, se mascarado).
    """
    total = W.shape[0]
    d      = np.array(W.sum(axis=1)).flatten()
    D_inv  = sp_diags(1.0 / np.sqrt(np.maximum(d, 1e-8)))
    L_sym  = sp_diags(np.ones(total)) - D_inv @ W @ D_inv

    eigenvalues, eigenvectors = eigsh(
        L_sym,
        k       = n_components + 1,
        which   = "SM",
        tol     = 1e-4,
        maxiter = 1000,
    )

    order = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, order]

    embed = sk_normalize(eigenvectors[:, 1: n_components + 1], axis=1)

    print(f"  Autovalores (1..{n_components}): "
          f"{eigenvalues[order][1:n_components+1].round(4)}")
    return embed.astype(np.float32)    # (M, n_components)


# ---------------------------------------------------------------------------
# Passo C — K-means espectral
# ---------------------------------------------------------------------------

def spectral_kmeans(embed: np.ndarray, n_clusters: int,
                    random_state: int = 42) -> np.ndarray:
    """
    Agrupa os patches no espaço espectral em n_clusters grupos.

    Retorna: labels (M,) — índice de cluster por patch (M = patches de
    conteúdo, se mascarado).
    """
    km = KMeans(
        n_clusters  = n_clusters,
        n_init      = 10,
        random_state = random_state,
    )
    labels = km.fit_predict(embed)

    counts = np.bincount(labels, minlength=n_clusters)
    print(f"  Clusters (tamanho): "
          + "  ".join(f"C{i}={c}" for i, c in enumerate(counts)))
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Passo D — Saliência espacial por cluster
# ---------------------------------------------------------------------------

def compute_spatial_saliency(
    labels_flat : np.ndarray,   # (M,) — M = N·P ou len(content_idx)
    patch_grid  : tuple,        # (H_p, W_p)
    n_images    : int,
    n_clusters  : int,
    content_idx : "np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula dois scores espaciais por cluster:

    1. Centralidade (score_centrality):
       Para cada cluster, calcula a distância média de seus patches ao
       centro da imagem, normalizada pelo raio máximo. Inverte para que
       clusters centrais tenham score alto.

       Intuição: objetos de interesse tendem a estar no centro da imagem
       (viés do fotógrafo). Fundos estruturados como céu e chão ocupam
       as bordas/topo/base.

    2. Compacidade (score_compactness):
       Mede a dispersão espacial dos patches de um cluster. Calcula o
       desvio padrão da posição normalizada dos patches.
       Inverte para que clusters compactos tenham score alto.

       Intuição: objetos formam blobs compactos. Bandas horizontais de
       fundo (céu, chão, grama) têm alta dispersão horizontal.

    O score espacial final é a média geométrica dos dois:
       score_spatial = sqrt(score_centrality * score_compactness)

    Se content_idx for fornecido, as coordenadas espaciais (linha/coluna
    normalizadas por patch) são calculadas sobre a grade completa e depois
    filtradas para os mesmos índices, garantindo alinhamento com
    labels_flat (que também está no espaço de conteúdo).

    Retorna:
        score_spatial     : (n_clusters,) — score combinado [0, 1]
        score_centrality  : (n_clusters,) — centralidade [0, 1]
        score_compactness : (n_clusters,) — compacidade [0, 1]
    """
    H_p, W_p = patch_grid
    N = n_images

    # Grade de coordenadas normalizadas por patch [0, 1]
    row_coords = np.repeat(np.arange(H_p), W_p) / (H_p - 1 + 1e-8)   # (P,)
    col_coords = np.tile(np.arange(W_p), H_p)   / (W_p - 1 + 1e-8)   # (P,)

    # Replica para todas as imagens (N·P,)
    rows_all = np.tile(row_coords, N)   # (N·P,)
    cols_all = np.tile(col_coords, N)   # (N·P,)

    if content_idx is not None:
        rows_all = rows_all[content_idx]
        cols_all = cols_all[content_idx]

    # Distância ao centro (0.5, 0.5), normalizada para [0, 1]
    max_dist = np.sqrt(0.5**2 + 0.5**2)   # ≈ 0.707
    dist_center = np.sqrt((rows_all - 0.5)**2 + (cols_all - 0.5)**2) / max_dist

    score_centrality  = np.zeros(n_clusters, dtype=np.float32)
    score_compactness = np.zeros(n_clusters, dtype=np.float32)

    for c in range(n_clusters):
        mask = labels_flat == c
        if not mask.any():
            # Cluster vazio: penaliza ambos
            score_centrality[c]  = 0.0
            score_compactness[c] = 0.0
            continue

        # Centralidade: 1 - distância média ao centro
        mean_dist = float(dist_center[mask].mean())
        score_centrality[c] = 1.0 - mean_dist

        # Compacidade: 1 - desvio padrão da posição
        # std máximo possível para distribuição uniforme ≈ 0.289 (std de U[0,1])
        std_rows = float(rows_all[mask].std())
        std_cols = float(cols_all[mask].std())
        mean_std = (std_rows + std_cols) / 2.0
        max_std  = 0.289
        score_compactness[c] = 1.0 - min(mean_std / max_std, 1.0)

    # Score combinado: média geométrica (penaliza quem falha em qualquer um)
    score_spatial = np.sqrt(score_centrality * score_compactness)

    print(f"\n  Scores espaciais por cluster:")
    print(f"  {'Cluster':>8}  {'Centralidade':>13}  {'Compacidade':>12}  {'Espacial':>10}")
    print(f"  {'─'*8}  {'─'*13}  {'─'*12}  {'─'*10}")
    for c in range(n_clusters):
        print(f"  C{c:>6}    {score_centrality[c]:>12.3f}   {score_compactness[c]:>11.3f}   "
              f"{score_spatial[c]:>9.3f}")

    return score_spatial, score_centrality, score_compactness


def compute_border_prior(
    labels_flat : np.ndarray,   # (M,)
    patch_grid  : tuple,        # (H_p, W_p)
    n_images    : int,
    n_clusters  : int,
    content_idx : "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Calcula, por cluster, um prior geométrico baseado em toque de borda —
    alternativa/complemento ao prior espacial de compute_spatial_saliency(),
    desenhada especificamente para o caso em que a saliência CLS (score_e2)
    falha por o fundo ser mais consistente entre as imagens do que o
    próprio objeto (ex.: fotos de time de futebol — o gramado nunca muda,
    os jogadores sim).

    Fração de patches do cluster que tocam a borda da grade (linha/coluna 0
    ou máxima), invertida para que clusters LONGE da borda tenham score
    alto (mais parecido com objeto). Diferente do prior espacial, este não
    depende de nenhuma comparação estatística entre imagens — é geometria
    pura por cluster — por isso continua confiável mesmo quando a
    consistência cross-image (que tanto score_e2 quanto score_spatial
    acabam dependendo indiretamente) está enganada pelo fundo.

    Mesma convenção de coordenadas de compute_spatial_saliency (grade do
    canvas, não da foto real dentro do padding) — mesma limitação
    conhecida em coleções com muito padding, mantida por consistência.

    Validado via ablação (ablation/analyze_geometric_priors.py) nas 14
    classes do subset icoseg16: o ranking deste score sozinho empata com
    score_e2 em 9/14 classes e é a correção mais forte encontrada nas duas
    classes com esse tipo de falha (Liverpool, Red Sox Players). Ver
    --fg_decision_mode.

    Retorna:
        score_border : (n_clusters,) — score [0, 1], 1 = longe da borda
    """
    H_p, W_p = patch_grid
    N = n_images

    row_idx = np.repeat(np.arange(H_p), W_p)   # (P,)
    col_idx = np.tile(np.arange(W_p), H_p)     # (P,)
    rows_all = np.tile(row_idx, N)             # (N·P,)
    cols_all = np.tile(col_idx, N)             # (N·P,)

    if content_idx is not None:
        rows_all = rows_all[content_idx]
        cols_all = cols_all[content_idx]

    on_border = (
        (rows_all == 0) | (rows_all == H_p - 1) |
        (cols_all == 0) | (cols_all == W_p - 1)
    )

    score_border = np.zeros(n_clusters, dtype=np.float32)
    for c in range(n_clusters):
        mask = labels_flat == c
        if not mask.any():
            score_border[c] = 0.0
            continue
        border_frac = float(on_border[mask].mean())
        score_border[c] = 1.0 - border_frac

    print(f"\n  Scores de borda por cluster:")
    print(f"  {'Cluster':>8}  {'Borda':>8}")
    print(f"  {'─'*8}  {'─'*8}")
    for c in range(n_clusters):
        print(f"  C{c:>6}    {score_border[c]:>7.3f}")

    return score_border


# ---------------------------------------------------------------------------
# Passo E — Mapeamento cluster → FG/BG
# ---------------------------------------------------------------------------

def map_clusters_to_fg(
    labels_flat    : np.ndarray,   # (M,)
    fg_scores_e2   : np.ndarray,   # (M,) já no mesmo espaço de labels_flat
                                    # (patches de conteúdo apenas, se mascarado)
    score_spatial  : np.ndarray,   # (n_clusters,) — prior espacial
    n_clusters     : int,
    fg_threshold   : float = 0.5,
    saliency_alpha : float = 0.5,  # peso do prior espacial
    fg_decision_mode : str = "saliency",   # "saliency" (default) | "e2_border_topk"
    score_border   : "np.ndarray | None" = None,   # (n_clusters,), obrigatório se
                                                     # fg_decision_mode="e2_border_topk"
    e2_border_weight : float = 0.3,
    topk_fraction    : float = 0.30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cada cluster recebe um score FG, com dois mecanismos de decisão
    possíveis:

    fg_decision_mode="saliency" (DEFAULT — comportamento original, inalterado):
        score_e2[c]    = média dos fg_scores da E2 para os patches do cluster c
        score_final[c] = (1 - α) * score_e2[c] + α * score_spatial[c]
        cluster_is_fg[c] = score_final[c] > fg_threshold

    fg_decision_mode="e2_border_topk" (experimental — ver compute_border_prior):
        score_final[c] = (1 - e2_border_weight) * score_e2[c]
                        + e2_border_weight * score_border[c]
        cluster_is_fg  = top-K clusters por score_final, onde
                        K = ceil(topk_fraction * n_clusters)

        Diferente do modo "saliency", a decisão aqui é por RANKING (top-K),
        não por limiar sobre o valor do score — validado via ablação que um
        limiar fixo não é robusto entre classes quando score_border entra
        na combinação (a escala do score muda). topk_fraction=0.30 e
        e2_border_weight=0.3 foram os valores validados em
        ablation/analyze_geometric_priors.py sobre as 14 classes do subset
        icoseg16: melhoram a média geral e o pior caso (classes com fundo
        mais consistente que o objeto, ex. multidões/gramado) às custas de
        uma queda real em 2 das 14 classes testadas (Kite, Statue) — não é
        um substituto estrito do modo "saliency", é uma troca. NÃO é o
        default; precisa ser habilitado explicitamente.

    O prior espacial (modo "saliency") penaliza clusters periféricos/não-
    compactos que tinham score_e2 alto mas não correspondem ao objeto de
    interesse. O prior de borda (modo "e2_border_topk") ataca um caso
    diferente: quando o PRÓPRIO score_e2 já está errado (aponta o fundo
    como FG), não só o "peso" dele precisa mudar.

    Retorna:
        score_e2      : (n_clusters,) — score bruto da E2
        score_final   : (n_clusters,) — score combinado
        cluster_is_fg : (n_clusters,) — bool FG/BG
        fg_score_map  : (M,)          — score contínuo mapeado por patch
    """
    fg_flat   = fg_scores_e2.flatten()    # (M,)
    score_e2  = np.array([
        fg_flat[labels_flat == c].mean() if (labels_flat == c).any() else 0.5
        for c in range(n_clusters)
    ], dtype=np.float32)

    if fg_decision_mode == "e2_border_topk":
        if score_border is None:
            raise ValueError(
                "fg_decision_mode='e2_border_topk' requer score_border "
                "(ver compute_border_prior())."
            )
        score_final = (1.0 - e2_border_weight) * score_e2 + e2_border_weight * score_border
        k = max(1, int(np.ceil(topk_fraction * n_clusters)))
        order = np.argsort(-score_final)
        top_idx = order[:k]
        cluster_is_fg = np.zeros(n_clusters, dtype=bool)
        cluster_is_fg[top_idx] = True
        decision_label = f"top-{k}/{n_clusters} por score_final (fração={topk_fraction:.2f})"
    elif fg_decision_mode == "saliency":
        # Combinação convexa (comportamento original, inalterado)
        score_final = (1.0 - saliency_alpha) * score_e2 + saliency_alpha * score_spatial
        cluster_is_fg = score_final > fg_threshold
        decision_label = f"limiar={fg_threshold:.2f}"
    else:
        raise ValueError(f"fg_decision_mode desconhecido: {fg_decision_mode!r}")

    fg_score_map  = score_final[labels_flat]   # (M,)

    print(f"\n  Scores por cluster  (modo={fg_decision_mode}, {decision_label}):")
    print(f"  {'Cluster':>8}  {'Score E2':>10}  {'Espacial/Borda':>15}  {'Final':>8}  {'Decisão':>8}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*15}  {'─'*8}  {'─'*8}")
    prior_col = score_spatial if fg_decision_mode == "saliency" else score_border
    for c in range(n_clusters):
        tag = "→ FG" if cluster_is_fg[c] else "→ BG"
        prior_val = prior_col[c] if prior_col is not None else float("nan")
        print(f"  C{c:>6}    {score_e2[c]:>9.3f}   {prior_val:>14.3f}   "
              f"{score_final[c]:>7.3f}   {tag}")

    n_fg = cluster_is_fg.sum()
    print(f"\n  {n_fg}/{n_clusters} clusters classificados como FG")
    return score_e2, score_final, cluster_is_fg, fg_score_map


# ---------------------------------------------------------------------------
# Passo F — Refinamento intra-imagem via NCut local
# ---------------------------------------------------------------------------

def ncut_intra_refine(
    feat_i         : np.ndarray,   # (P, D)
    fg_prior       : np.ndarray,   # (P,) — fg_score da E2 para esta imagem
    temperature    : float = 0.07,
    content_mask_i : "np.ndarray | None" = None,   # (P,) bool — True = patch real
) -> np.ndarray:
    """
    Aplica um NCut local na imagem i para refinar a borda da máscara coarse.

    O grafo intra-imagem tem apenas P nós (não N·P), então a eigendecomp
    densa é rápida (<0.1s para P=1369).

    Inovação: as arestas são pesadas pelo produto da similaridade coseno
    com a 'concordância de prior':
        w(p,q) = cos_sim(p,q) · (1 - |fg_prior[p] - fg_prior[q]|)

    Orientação: o lado do Fiedler com maior fg_prior médio = FG.

    Se content_mask_i for fornecida, patches de padding são excluídos do
    grafo local (mesmo raciocínio do grafo global da Etapa 3 e do NCut
    par-a-par da Etapa 2) e ficam forçados a 0 (background) na máscara
    final, sem participar do corte.

    Retorna: fg_mask (P,) float32 ∈ {0, 1}
    """
    P = feat_i.shape[0]

    if content_mask_i is not None and content_mask_i.sum() < P:
        idx = np.flatnonzero(content_mask_i)
        if idx.size == 0:
            idx = np.arange(P)  # salvaguarda: nenhum patch de conteúdo detectado
    else:
        idx = np.arange(P)

    feat_c  = feat_i[idx]
    prior_c = fg_prior[idx]

    fi  = sk_normalize(feat_c, axis=1)
    W   = np.maximum(fi @ fi.T, 0.0)
    np.fill_diagonal(W, 0.0)

    prior_agree = 1.0 - np.abs(prior_c[:, None] - prior_c[None, :])
    W = W * prior_agree

    d     = W.sum(axis=1)
    d_inv = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    L_sym = np.eye(len(feat_c), dtype=np.float32) - \
            (d_inv[:, None] * W * d_inv[None, :])

    _, vecs = np.linalg.eigh(L_sym)
    fiedler = vecs[:, 1]

    side = fiedler > np.median(fiedler)
    if prior_c[side].mean() < prior_c[~side].mean():
        side = ~side

    fg_content = side.astype(np.float32)

    fg_full = np.zeros(P, dtype=np.float32)
    fg_full[idx] = fg_content
    return fg_full


# ---------------------------------------------------------------------------
# Passo G — Upsample para tamanho original
# ---------------------------------------------------------------------------

def upsample_masks(
    fg_masks_patch : np.ndarray,          # (N, H_p, W_p)
    image_sizes    : np.ndarray,          # (N, 2) — (H_orig, W_orig)
    resize_info    : "np.ndarray | None" = None,  # (N, 6), só existe no modo keep_ar
    threshold      : float = 0.5,
) -> np.ndarray:
    """
    Reamostra a máscara de resolução de patch para o tamanho original.

    Se resize_info for fornecido (modo keep_ar da Etapa 1), primeiro recorta
    fora a região de padding antes de redimensionar — do contrário a máscara
    fica esticada/deslocada, pois a grade de patches cobre o canvas quadrado
    inteiro (imagem + padding), não só a imagem real.
    """
    N       = fg_masks_patch.shape[0]
    H_max   = int(image_sizes[:, 0].max())
    W_max   = int(image_sizes[:, 1].max())
    output  = np.zeros((N, H_max, W_max), dtype=np.float32)

    for i in range(N):
        H_orig, W_orig = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        mask_img = Image.fromarray(
            (fg_masks_patch[i] * 255).astype(np.uint8)
        )

        if resize_info is not None:
            resized_h, resized_w, pad_top, pad_left, canvas_h, canvas_w = \
                [int(x) for x in resize_info[i]]

            # 1) upsample da grade de patches para o canvas completo (com padding)
            canvas_mask = mask_img.resize((canvas_w, canvas_h), Image.BILINEAR)
            canvas_arr  = np.array(canvas_mask)

            # 2) recorta só a região onde a imagem real foi colada,
            #    descartando as faixas de padding do modo keep_ar
            crop = canvas_arr[pad_top:pad_top + resized_h,
                              pad_left:pad_left + resized_w]

            # 3) resize do recorte (já na proporção certa) para o tamanho original
            up = np.array(
                Image.fromarray(crop).resize((W_orig, H_orig), Image.BILINEAR)
            ) / 255.0
        else:
            # Modo "square": sem padding, resize direto (comportamento antigo)
            up = np.array(
                mask_img.resize((W_orig, H_orig), Image.BILINEAR)
            ) / 255.0

        hard = (up > threshold).astype(np.float32)
        output[i, :H_orig, :W_orig] = hard

    return output


# ---------------------------------------------------------------------------
# Passo H — Seeds para Etapa 4 (SAM prompting)
# ---------------------------------------------------------------------------

def extract_seeds(
    fg_masks_pixel : np.ndarray,   # (N, H, W)
    image_sizes    : np.ndarray,   # (N, 2)
    n_fg_points    : int = 3,
    n_bg_points    : int = 3,
) -> list[dict]:
    N = fg_masks_pixel.shape[0]
    seeds = []

    for i in range(N):
        H_i, W_i = int(image_sizes[i, 0]), int(image_sizes[i, 1])
        mask = fg_masks_pixel[i, :H_i, :W_i]

        fg_pts = np.argwhere(mask > 0.5)
        bg_pts = np.argwhere(mask < 0.5)

        if len(fg_pts) > 0:
            r0, c0 = fg_pts.min(axis=0)
            r1, c1 = fg_pts.max(axis=0)
            bbox = [int(c0), int(r0), int(c1), int(r1)]
        else:
            bbox = [0, 0, W_i, H_i]

        if len(fg_pts) >= n_fg_points:
            idx   = np.linspace(0, len(fg_pts) - 1, n_fg_points, dtype=int)
            fg_sampled = fg_pts[idx].tolist()
        else:
            fg_sampled = fg_pts.tolist()

        corners = np.array([
            [0, 0], [0, W_i - 1], [H_i - 1, 0], [H_i - 1, W_i - 1],
            [H_i // 2, 0], [H_i // 2, W_i - 1],
            [0, W_i // 2], [H_i - 1, W_i // 2],
        ])
        corner_bg = [pt for pt in corners
                     if 0 <= pt[0] < H_i and 0 <= pt[1] < W_i
                     and mask[pt[0], pt[1]] < 0.5]
        if len(corner_bg) >= n_bg_points:
            bg_sampled = corner_bg[:n_bg_points]
        elif len(bg_pts) >= n_bg_points:
            idx = np.linspace(0, len(bg_pts) - 1, n_bg_points, dtype=int)
            bg_sampled = bg_pts[idx].tolist()
        else:
            bg_sampled = bg_pts.tolist()

        seeds.append({
            "bbox"      : bbox,
            "fg_points" : [pt if isinstance(pt, list) else pt.tolist()
                           for pt in fg_sampled],
            "bg_points" : [pt if isinstance(pt, list) else pt.tolist()
                           for pt in bg_sampled],
            "fg_ratio"  : float(mask.mean()),
        })

    return seeds


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_cosaliency(
    patch_features : np.ndarray,          # (N, P, D)
    fg_scores_e2   : np.ndarray,          # (N, P) — da Etapa 2
    cls_saliency   : np.ndarray,          # (N, P) — da Etapa 2
    patch_grid     : tuple[int, int],
    image_sizes    : np.ndarray,          # (N, 2)
    resize_info    : "np.ndarray | None" = None,  # (N, 6), vem da Etapa 1
    content_mask   : "np.ndarray | None" = None,  # (N, P) bool — True = patch real
    k_neighbors    : int   = 10,
    n_spectral     : int   = 8,
    n_clusters     : int   = 5,
    fg_threshold   : float = 0.5,
    saliency_alpha : float = 0.5,
    refine         : bool  = False,
    fg_decision_mode : str = "saliency",
    e2_border_weight : float = 0.3,
    topk_fraction    : float = 0.30,
    empty_fg_bypass_mode : str = "e2_topk",
    empty_fg_top_fraction : float = 0.20,
    bypass_content_mask   : "np.ndarray | None" = None,
) -> dict:
    N, P, D = patch_features.shape
    H_p, W_p = patch_grid
    total = N * P

    content_idx = None
    if content_mask is not None:
        content_idx = np.flatnonzero(content_mask.reshape(-1))
        frac_padding = 1.0 - (content_idx.size / total)
        if frac_padding > 1e-6:
            print(f"[Padding] {frac_padding:.1%} dos patches identificados como padding "
                  f"— excluídos do grafo global e do refino intra-imagem.")

    print(f"\n[Passo A] Construindo grafo global ({N}×{P}={total:,} nós)...")
    W_global = build_global_graph(patch_features, k_neighbors=k_neighbors,
                                  content_idx=content_idx)

    print(f"\n[Passo B] Spectral embedding ({n_spectral} componentes)...")
    embed = spectral_embedding(W_global, n_components=n_spectral)   # (M, n_spectral)

    print(f"\n[Passo C] K-means espectral ({n_clusters} clusters)...")
    labels_content = spectral_kmeans(embed, n_clusters=n_clusters)      # (M,)

    print(f"\n[Passo D] Calculando prior de saliência espacial...")
    score_spatial, score_centrality, score_compactness = compute_spatial_saliency(
        labels_content, patch_grid, N, n_clusters, content_idx=content_idx
    )

    score_border = None
    if fg_decision_mode == "e2_border_topk":
        print(f"\n[Passo D2] Calculando prior de borda (fg_decision_mode=e2_border_topk)...")
        score_border = compute_border_prior(
            labels_content, patch_grid, N, n_clusters, content_idx=content_idx
        )

    print(f"\n[Passo E] Mapeamento cluster → FG/BG (α={saliency_alpha:.2f})...")
    if content_idx is not None:
        fg_scores_e2_for_map = fg_scores_e2.reshape(-1)[content_idx]
    else:
        fg_scores_e2_for_map = fg_scores_e2

    score_e2, score_final, cluster_is_fg, fg_score_map_content = map_clusters_to_fg(
        labels_content, fg_scores_e2_for_map, score_spatial, n_clusters,
        fg_threshold, saliency_alpha,
        fg_decision_mode=fg_decision_mode, score_border=score_border,
        e2_border_weight=e2_border_weight, topk_fraction=topk_fraction,
    )

    # Espalha de volta para o espaço completo (N·P). Patches de padding
    # (fora de content_idx) recebem rótulo de cluster arbitrário (0) apenas
    # para manter os arrays em formato válido para indexação/visualização —
    # a máscara final os força explicitamente para background logo abaixo,
    # e fg_score_global também fica em 0.0 (confiante em BG) para eles,
    # independentemente do que "cluster 0" representa.
    if content_idx is not None:
        labels_full = np.zeros(total, dtype=np.int64)
        labels_full[content_idx] = labels_content
        fg_score_global_flat = np.zeros(total, dtype=np.float32)
        fg_score_global_flat[content_idx] = fg_score_map_content
        is_padding_flat = np.ones(total, dtype=bool)
        is_padding_flat[content_idx] = False
    else:
        labels_full = labels_content
        fg_score_global_flat = fg_score_map_content
        is_padding_flat = np.zeros(total, dtype=bool)

    labels_img = labels_full.reshape(N, P)
    fg_score_global_img = fg_score_global_flat.reshape(N, P)

    fg_mask_coarse_flat = cluster_is_fg[labels_full].astype(np.float32)
    fg_mask_coarse_flat[is_padding_flat] = 0.0
    fg_mask_coarse = fg_mask_coarse_flat.reshape(N, P)

    if refine:
        print(f"\n[Passo F] Refinamento NCut intra-imagem ({N} imagens)...")
        fg_mask_refined = np.zeros((N, P), dtype=np.float32)
        for i in range(N):
            content_mask_i = content_mask[i] if content_mask is not None else None
            fg_mask_refined[i] = ncut_intra_refine(
                patch_features[i], fg_scores_e2[i], content_mask_i=content_mask_i
            )
            pct = fg_mask_refined[i].mean() * 100
            print(f"  img[{i:>2}]  FG%={pct:.0f}%")
    else:
        fg_mask_refined = fg_mask_coarse
        print("\n[Passo F] Refinamento desativado (--no_refine)")

    # ------------------------------------------------------------------
    # Passo G — Empty-Foreground Adaptive Bypass
    # ------------------------------------------------------------------
    print(
        f"\n[Passo G] Empty-Foreground Adaptive Bypass "
        f"(mode={empty_fg_bypass_mode}, top_fraction={empty_fg_top_fraction:.2f})..."
    )
    (
        fg_mask_refined,
        empty_fg_bypass_triggered,
        empty_fg_before_patch_fraction,
        empty_fg_after_patch_fraction,
    ) = apply_empty_fg_bypass(
        fg_mask_refined,
        fg_scores_e2,
        bypass_content_mask=bypass_content_mask,
        mode=empty_fg_bypass_mode,
        top_fraction=empty_fg_top_fraction,
    )

    n_triggered = int(empty_fg_bypass_triggered.sum())
    if empty_fg_bypass_mode == "none":
        print("  Bypass desativado.")
    elif n_triggered == 0:
        print("  Nenhuma imagem com foreground E3 vazio — nenhuma alteração.")
    else:
        print(
            f"  Bypass ativado em {n_triggered}/{N} imagem(ns): "
            + ", ".join(str(i) for i in np.flatnonzero(empty_fg_bypass_triggered))
        )
        for i in np.flatnonzero(empty_fg_bypass_triggered):
            print(
                f"    img[{i:>2}] FG válido: "
                f"{empty_fg_before_patch_fraction[i]*100:.2f}% -> "
                f"{empty_fg_after_patch_fraction[i]*100:.2f}%"
            )

    fg_mask_patch = fg_mask_refined.reshape(N, H_p, W_p)            # (N, H_p, W_p)

    print(f"\n[Passo H] Upsample para tamanho original...")
    fg_mask_pixel = upsample_masks(fg_mask_patch, image_sizes, resize_info=resize_info)  # (N, H, W)
    print(f"  Máscaras: {fg_mask_pixel.shape}  "
          f"FG% médio = {fg_mask_pixel.mean()*100:.1f}%")

    print(f"\n[Passo I] Extraindo seeds para Etapa 4 (SAM)...")
    seeds = extract_seeds(fg_mask_pixel, image_sizes)
    for i, s in enumerate(seeds):
        print(f"  img[{i:>2}]  FG%={s['fg_ratio']*100:.0f}%  "
              f"bbox={s['bbox']}  "
              f"fg_pts={len(s['fg_points'])}  bg_pts={len(s['bg_points'])}")

    return {
        "fg_mask_patch"      : fg_mask_patch,
        "fg_mask_pixel"      : fg_mask_pixel,
        "fg_score_global"    : fg_score_global_img,
        "fg_mask_coarse"     : fg_mask_coarse.reshape(N, H_p, W_p),
        "cluster_labels"     : labels_img,
        "cluster_scores"     : score_final,        # score combinado (FG decision)
        "cluster_scores_e2"  : score_e2,           # score bruto E2
        "cluster_scores_sp"  : score_spatial,      # score espacial
        "cluster_scores_centrality"  : score_centrality,
        "cluster_scores_compactness" : score_compactness,
        "cluster_scores_border"      : score_border,   # None se fg_decision_mode="saliency"
        "cluster_is_fg"      : cluster_is_fg,
        "n_clusters_used"    : int(n_clusters),
        "saliency_alpha"     : float(saliency_alpha),
        "fg_decision_mode"   : fg_decision_mode,
        "e2_border_weight"   : float(e2_border_weight),
        "topk_fraction"      : float(topk_fraction),
        "empty_fg_bypass_mode"         : empty_fg_bypass_mode,
        "empty_fg_top_fraction"        : float(empty_fg_top_fraction),
        "empty_fg_bypass_triggered"    : empty_fg_bypass_triggered,
        "empty_fg_bypass_trigger_count": int(empty_fg_bypass_triggered.sum()),
        "empty_fg_before_patch_fraction": empty_fg_before_patch_fraction,
        "empty_fg_after_patch_fraction" : empty_fg_after_patch_fraction,
        "spectral_embed"     : embed,
        "seeds"              : seeds,
    }


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def save_cosaliency(result: dict, output_dir: Path,
                    image_paths: list[str], image_sizes: np.ndarray) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "cosaliency.npz"

    seeds_arr = np.empty(len(result["seeds"]), dtype=object)
    for i, s in enumerate(result["seeds"]):
        seeds_arr[i] = s

    np.savez_compressed(
        out,
        fg_mask_patch               = result["fg_mask_patch"],
        fg_mask_pixel               = result["fg_mask_pixel"],
        fg_score_global             = result["fg_score_global"],
        fg_mask_coarse              = result["fg_mask_coarse"],
        cluster_labels              = result["cluster_labels"],
        cluster_scores              = result["cluster_scores"],
        cluster_scores_e2           = result["cluster_scores_e2"],
        cluster_scores_sp           = result["cluster_scores_sp"],
        cluster_scores_centrality   = result["cluster_scores_centrality"],
        cluster_scores_compactness  = result["cluster_scores_compactness"],
        cluster_scores_border       = (result["cluster_scores_border"]
                                       if result["cluster_scores_border"] is not None
                                       else np.array([])),
        cluster_is_fg               = result["cluster_is_fg"],
        n_clusters_used             = np.array(result["n_clusters_used"]),
        saliency_alpha              = np.array(result["saliency_alpha"]),
        fg_decision_mode            = result["fg_decision_mode"],
        e2_border_weight            = np.array(result["e2_border_weight"]),
        topk_fraction                = np.array(result["topk_fraction"]),
        empty_fg_bypass_mode         = result["empty_fg_bypass_mode"],
        empty_fg_top_fraction        = np.array(result["empty_fg_top_fraction"]),
        empty_fg_bypass_triggered    = result["empty_fg_bypass_triggered"],
        empty_fg_bypass_trigger_count= np.array(result["empty_fg_bypass_trigger_count"]),
        empty_fg_before_patch_fraction = result["empty_fg_before_patch_fraction"],
        empty_fg_after_patch_fraction  = result["empty_fg_after_patch_fraction"],
        seeds                       = seeds_arr,
        image_paths                 = image_paths,
        image_sizes                 = image_sizes,
    )
    print(f"\n[Salvo] {out}  ({out.stat().st_size/1024**2:.1f} MB)")
    return out


def load_cosaliency(path: Path) -> dict:
    """Interface de carregamento para a Etapa 4."""
    data = np.load(path, allow_pickle=True)

    def _opt(key, default=None):
        return data[key] if key in data.files else default

    return {
        "fg_mask_patch"              : data["fg_mask_patch"],
        "fg_mask_pixel"              : data["fg_mask_pixel"],
        "fg_score_global"            : data["fg_score_global"],
        "fg_mask_coarse"             : data["fg_mask_coarse"],
        "cluster_labels"             : data["cluster_labels"],
        "cluster_scores"             : data["cluster_scores"],
        "cluster_scores_e2"          : _opt("cluster_scores_e2"),
        "cluster_scores_sp"          : _opt("cluster_scores_sp"),
        "cluster_scores_centrality"  : _opt("cluster_scores_centrality"),
        "cluster_scores_compactness" : _opt("cluster_scores_compactness"),
        "cluster_scores_border"      : _opt("cluster_scores_border"),
        "cluster_is_fg"              : data["cluster_is_fg"],
        "n_clusters_used"            : int(data["n_clusters_used"]),
        "saliency_alpha"             : float(_opt("saliency_alpha", np.array(0.5))),
        "fg_decision_mode"           : str(_opt("fg_decision_mode", "saliency")),
        "e2_border_weight"           : float(_opt("e2_border_weight", np.array(0.3))),
        "topk_fraction"               : float(_opt("topk_fraction", np.array(0.30))),
        "empty_fg_bypass_mode"        : str(_opt("empty_fg_bypass_mode", "none")),
        "empty_fg_top_fraction"       : float(_opt("empty_fg_top_fraction", np.array(0.20))),
        "empty_fg_bypass_triggered"   : _opt(
            "empty_fg_bypass_triggered",
            np.zeros(len(data["image_paths"]), dtype=bool),
        ),
        "empty_fg_bypass_trigger_count": int(
            _opt("empty_fg_bypass_trigger_count", np.array(0))
        ),
        "empty_fg_before_patch_fraction": _opt(
            "empty_fg_before_patch_fraction",
            np.array([], dtype=np.float32),
        ),
        "empty_fg_after_patch_fraction": _opt(
            "empty_fg_after_patch_fraction",
            np.array([], dtype=np.float32),
        ),
        "seeds"                      : list(data["seeds"]),
        "image_paths"                : list(data["image_paths"]),
        "image_sizes"                : data["image_sizes"],
    }


# ---------------------------------------------------------------------------
# Sumário
# ---------------------------------------------------------------------------

def print_summary(result: dict, image_paths: list[str]):
    mask  = result["fg_mask_patch"]
    N     = len(image_paths)
    nc    = result["n_clusters_used"]
    n_fg  = result["cluster_is_fg"].sum()

    print("\n" + "=" * 62)
    print("  SUMÁRIO ETAPA 3")
    print("=" * 62)
    print(f"  Imagens           : {N}")
    print(f"  Clusters globais  : {nc}  ({n_fg} FG, {nc-n_fg} BG)")
    print(f"  saliency_alpha    : {result['saliency_alpha']:.2f}")
    print(
        f"  empty_fg_bypass  : {result.get('empty_fg_bypass_mode', 'none')} "
        f"(top_fraction={result.get('empty_fg_top_fraction', 0.20):.2f}, "
        f"triggered={result.get('empty_fg_bypass_trigger_count', 0)}/{N})"
    )
    print()
    print(f"  {'IMG':>4}  {'FG%':>6}  {'BBOX':>22}  NOME")
    print(f"  {'─'*4}  {'─'*6}  {'─'*22}  {'─'*28}")
    for i in range(N):
        pct  = mask[i].mean() * 100
        bbox = result["seeds"][i]["bbox"]
        name = Path(image_paths[i]).name[:28]
        print(f"  {i:>4}  {pct:>5.1f}%  {str(bbox):>22}  {name}")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Visualizações
# ---------------------------------------------------------------------------

def plot_mask_evolution(result: dict, image_paths: list[str],
                        image_sizes: np.ndarray, output_dir: Path, max_images: int = 8):
    if plt is None:
        return
    fg_patch  = result["fg_mask_patch"]
    fg_coarse = result["fg_mask_coarse"]
    fg_pixel  = result["fg_mask_pixel"]
    N_vis = min(len(image_paths), max_images)

    fig, axes = plt.subplots(N_vis, 4, figsize=(15, 3.5 * N_vis))
    if N_vis == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(
        "Etapa 3 — Evolução da máscara: Coarse → Refined → Pixel",
        fontsize=11, y=1.01
    )
    for i in range(N_vis):
        H_i = int(image_sizes[i, 0]); W_i = int(image_sizes[i, 1])
        img = Image.open(image_paths[i]).convert("RGB")
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(Path(image_paths[i]).name[:22], fontsize=7)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(fg_coarse[i], cmap="RdYlGn", vmin=0, vmax=1,
                          interpolation="nearest")
        axes[i, 1].set_title(f"Coarse  FG%={fg_coarse[i].mean()*100:.0f}%", fontsize=7)
        axes[i, 1].axis("off")
        axes[i, 2].imshow(fg_patch[i], cmap="RdYlGn", vmin=0, vmax=1,
                          interpolation="nearest")
        axes[i, 2].set_title(f"Refined  FG%={fg_patch[i].mean()*100:.0f}%", fontsize=7)
        axes[i, 2].axis("off")
        axes[i, 3].imshow(fg_pixel[i, :H_i, :W_i], cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, 3].set_title(f"Pixel  FG%={fg_pixel[i,:H_i,:W_i].mean()*100:.0f}%", fontsize=7)
        axes[i, 3].axis("off")
    fig.tight_layout()
    out1 = output_dir / "viz_mask_evolution.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out1


def plot_clusters(result: dict, image_paths: list[str], output_dir: Path, max_images: int = 8):
    """
    Mostra os clusters globais por imagem. Patches de padding (quando
    mascarados) recebem rótulo arbitrário 0 apenas para fins de indexação
    — este gráfico os desenha sobrepostos em PRETO, independentemente da
    cor do cluster 0, para que não sejam confundidos com conteúdo real.
    """
    if plt is None:
        return
    fg_patch = result["fg_mask_patch"]
    labels   = result["cluster_labels"]
    cluster_is_fg = result["cluster_is_fg"]
    nc       = result["n_clusters_used"]
    H_p, W_p = fg_patch.shape[1], fg_patch.shape[2]
    N_vis    = min(len(image_paths), max_images)

    # patches de padding são identificados aqui via fg_score_global == 0
    # combinado com fg_mask_coarse == 0 não é confiável isoladamente (conteúdo
    # real também pode ser BG legitimamente); usamos diretamente a máscara de
    # conteúdo, se disponível, calculada pelo chamador — como este helper não
    # a recebe, aproximamos via ausência de participação: não há como
    # distinguir aqui sem info extra, então este gráfico deliberadamente NÃO
    # tenta destacar padding (ver plot_clusters_with_padding, chamado quando
    # content_mask está disponível no momento da visualização).
    cmap_clusters = plt.get_cmap("tab10", nc)
    fig, axes = plt.subplots(1, N_vis, figsize=(3 * N_vis, 4))
    if N_vis == 1:
        axes = [axes]
    fig.suptitle(
        f"Clusters globais ({nc} clusters)  —  "
        + ", ".join(f"C{i}={'FG' if cluster_is_fg[i] else 'BG'}" for i in range(nc)),
        fontsize=10
    )
    legend_patches = [
        mpatches.Patch(color=cmap_clusters(i),
                       label=f"C{i} {'FG' if cluster_is_fg[i] else 'BG'}")
        for i in range(nc)
    ]
    for i in range(N_vis):
        lmap = labels[i].reshape(H_p, W_p).astype(float) / max(nc - 1, 1)
        axes[i].imshow(lmap, cmap=cmap_clusters, vmin=0, vmax=1,
                       interpolation="nearest")
        axes[i].set_title(f"img[{i}]", fontsize=8)
        axes[i].axis("off")
    axes[-1].legend(handles=legend_patches, loc="lower right",
                     fontsize=7, framealpha=0.8)
    fig.tight_layout()
    out2 = output_dir / "viz_clusters.png"
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out2


def plot_overlay(result: dict, image_paths: list[str], image_sizes: np.ndarray,
                 output_dir: Path, max_images: int = 8):
    if plt is None:
        return
    fg_pixel = result["fg_mask_pixel"]
    N_vis = min(len(image_paths), max_images)

    fig, axes = plt.subplots(N_vis, 2, figsize=(9, 3.8 * N_vis))
    if N_vis == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Etapa 3 — Máscara final sobreposta à imagem original", fontsize=11)
    for i in range(N_vis):
        H_i = int(image_sizes[i, 0]); W_i = int(image_sizes[i, 1])
        img_arr = np.array(Image.open(image_paths[i]).convert("RGB").resize(
            (W_i, H_i), Image.BICUBIC
        ))
        mask_i = fg_pixel[i, :H_i, :W_i]
        axes[i, 0].imshow(img_arr)
        axes[i, 0].set_title(Path(image_paths[i]).name[:28], fontsize=7)
        axes[i, 0].axis("off")
        overlay = img_arr.copy().astype(np.float32)
        green_channel = np.zeros_like(img_arr, dtype=np.float32)
        green_channel[:, :, 1] = 255
        alpha_ch = mask_i[:, :, None] * 0.5
        overlay = (overlay * (1 - alpha_ch) + green_channel * alpha_ch).astype(np.uint8)
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"Overlay  FG={mask_i.mean()*100:.0f}%", fontsize=7)
        axes[i, 1].axis("off")
    fig.tight_layout()
    out3 = output_dir / "viz_overlay.png"
    fig.savefig(out3, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out3


def plot_seeds(result: dict, image_paths: list[str], image_sizes: np.ndarray,
              output_dir: Path, max_images: int = 6):
    if plt is None:
        return
    seeds = result["seeds"]
    n_show = min(len(image_paths), max_images)
    n_cols = 3
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes_flat = np.array(axes).flatten()
    fig.suptitle("Seeds para Etapa 4 (SAM)\n"
                 "Verde=FG  Vermelho=BG  Amarelo=bbox", fontsize=11)
    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        if idx >= n_show:
            ax.axis("off"); continue
        i = idx
        H_i = int(image_sizes[i, 0]); W_i = int(image_sizes[i, 1])
        img_arr = np.array(Image.open(image_paths[i]).convert("RGB").resize(
            (W_i, H_i), Image.BICUBIC
        ))
        ax.imshow(img_arr)
        s = seeds[i]
        if s["bbox"]:
            x0, y0, x1, y1 = s["bbox"]
            rect = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor="yellow", facecolor="none"
            )
            ax.add_patch(rect)
        for pt in s["fg_points"]:
            ax.plot(pt[1], pt[0], "g+", markersize=12, markeredgewidth=2)
        for pt in s["bg_points"]:
            ax.plot(pt[1], pt[0], "rx", markersize=10, markeredgewidth=2)
        ax.set_title(f"img[{i}]  FG={s['fg_ratio']*100:.0f}%", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out4 = output_dir / "viz_seeds.png"
    fig.savefig(out4, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out4


def plot_saliency_prior(result: dict, output_dir: Path):
    if plt is None:
        return
    score_e2   = result.get("cluster_scores_e2")
    score_sp   = result.get("cluster_scores_sp")
    score_cent = result.get("cluster_scores_centrality")
    score_comp = result.get("cluster_scores_compactness")
    score_fin  = result["cluster_scores"]
    cluster_is_fg = result["cluster_is_fg"]
    alpha = result.get("saliency_alpha", 0.5)
    nc = result["n_clusters_used"]

    if score_e2 is None or score_sp is None:
        return

    cluster_ids = np.arange(nc)
    bar_w = 0.22
    fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5))
    fig5.suptitle(
        f"Prior de Saliência Espacial por Cluster  (α={alpha:.2f})\n"
        f"score_final = (1-α)·score_E2 + α·score_espacial",
        fontsize=11
    )

    ax = axes5[0]
    ax.bar(cluster_ids - bar_w,     score_e2,   width=bar_w, label="Score E2",        color="#4472C4", alpha=0.85)
    ax.bar(cluster_ids,              score_sp,   width=bar_w, label="Espacial (√C·K)", color="#ED7D31", alpha=0.85)
    ax.bar(cluster_ids + bar_w,      score_fin,  width=bar_w, label="Final (combinado)", color="#70AD47", alpha=0.85)
    ax.axhline(0.5, color="black", ls="--", lw=1.2, alpha=0.6, label="threshold=0.5")
    ax.set_xticks(cluster_ids)
    ax.set_xticklabels(
        [f"C{c}\n{'FG' if cluster_is_fg[c] else 'BG'}" for c in cluster_ids], fontsize=9
    )
    ax.set_ylabel("Score [0, 1]"); ax.set_ylim(0, 1.05)
    ax.set_title("Score E2 vs Espacial vs Final")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    ax2 = axes5[1]
    if score_cent is not None and score_comp is not None:
        ax2.bar(cluster_ids - bar_w/2, score_cent, width=bar_w,
                label="Centralidade", color="#9B59B6", alpha=0.85)
        ax2.bar(cluster_ids + bar_w/2, score_comp, width=bar_w,
                label="Compacidade",  color="#E74C3C", alpha=0.85)
        ax2.set_xticks(cluster_ids)
        ax2.set_xticklabels(
            [f"C{c}\n{'FG' if cluster_is_fg[c] else 'BG'}" for c in cluster_ids], fontsize=9
        )
        ax2.set_ylabel("Score [0, 1]"); ax2.set_ylim(0, 1.05)
        ax2.set_title("Decomposição do Score Espacial")
        ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Centralidade/Compacidade\nnão disponíveis",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=11)
        ax2.axis("off")

    fig5.tight_layout()
    out5 = output_dir / "viz_saliency_prior.png"
    fig5.savefig(out5, dpi=120, bbox_inches="tight")
    plt.close(fig5)
    return out5


def visualize_cosaliency(result: dict, image_paths: list[str],
                         image_sizes: np.ndarray,
                         output_dir: Path, max_images: int = 8):
    outs = [
        plot_mask_evolution(result, image_paths, image_sizes, output_dir, max_images),
        plot_clusters(result, image_paths, output_dir, max_images),
        plot_overlay(result, image_paths, image_sizes, output_dir, max_images),
        plot_seeds(result, image_paths, image_sizes, output_dir, min(max_images, 6)),
        plot_saliency_prior(result, output_dir),
    ]
    outs = [o for o in outs if o is not None]
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
        description="Coseg Etapa 3 — co-saliency via spectral clustering global",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--features",        "-f", required=True,
                   help="features.npz da Etapa 1")
    p.add_argument("--correspondence",  "-c", required=True,
                   help="correspondence.npz da Etapa 2")
    p.add_argument("--output",          "-o", default=None,
                   help="Pasta de saída (default: mesma do features.npz)")
    p.add_argument("--k_neighbors",     type=int,   default=10,
                   help="Vizinhos no grafo kNN (default: 10)")
    p.add_argument("--n_spectral",      type=int,   default=8,
                   help="Nº de componentes espectrais (default: 8)")
    p.add_argument("--n_clusters",      type=int,   default=5,
                   help="Nº de clusters K-means (default: 5)")
    p.add_argument("--fg_threshold",    type=float, default=0.5,
                   help="Threshold FG/BG no score de cluster (default: 0.5)")
    p.add_argument("--saliency_alpha",  type=float, default=0.5,
                   help="Peso do prior espacial no score final [0=só E2, 1=só espacial] (default: 0.5)")
    p.add_argument("--fg_decision_mode", type=str, default="saliency",
                   choices=["saliency", "e2_border_topk"],
                   help="Mecanismo de decisão FG/BG por cluster. 'saliency' (default) é o "
                        "comportamento original (prior espacial + limiar). 'e2_border_topk' "
                        "é experimental (ver compute_border_prior/map_clusters_to_fg) — "
                        "combina score da E2 com um prior de toque de borda e decide por "
                        "top-K no ranking, não por limiar. Validado como melhoria de média/"
                        "pior-caso em ablação nas 14 classes do icoseg16, com custo real em "
                        "2 delas (Kite, Statue) — não é o default.")
    p.add_argument("--e2_border_weight", type=float, default=0.3,
                   help="Só usado com --fg_decision_mode e2_border_topk. Peso do prior de "
                        "borda no score final [0=só E2, 1=só borda] (default: 0.3, valor "
                        "validado em ablação).")
    p.add_argument("--topk_fraction", type=float, default=0.30,
                   help="Só usado com --fg_decision_mode e2_border_topk. Fração dos clusters "
                        "(por ranking do score final) marcados como FG (default: 0.30, valor "
                        "validado em ablação).")
    p.add_argument(
        "--empty_fg_bypass",
        type=str,
        default="e2_topk",
        choices=["none", "e2_topk"],
        help="Fallback para o estado degenerado em que uma imagem sai da E3 com zero "
             "patches FG válidos. 'e2_topk' (default validado) usa diretamente os "
             "top-K scores da E2/E2B somente nessas imagens. 'none' desativa."
    )
    p.add_argument(
        "--empty_fg_top_fraction",
        type=float,
        default=0.20,
        help="Fração dos patches válidos selecionada por score E2/E2B quando "
             "--empty_fg_bypass=e2_topk (default: 0.20 = top-20%%)."
    )
    refine_group = p.add_mutually_exclusive_group()

    refine_group.add_argument(
        "--refine",
        dest="refine",
        action="store_true",
        help="Ativa experimentalmente o refinamento NCut intra-imagem."
    )

    refine_group.add_argument(
        "--no_refine",
        dest="refine",
        action="store_false",
        help="Desativa o refinamento NCut intra-imagem."
    )

    p.set_defaults(refine=False)
    p.add_argument("--no_padding_mask", action="store_true",
                   help="Desativa o mascaramento de patches de padding (modo --resize\n"
                        "keep_ar da Etapa 1) no grafo global e no refino intra-imagem.\n"
                        "Por padrão, o mascaramento é ativado sempre que resize_info\n"
                        "estiver disponível.")
    p.add_argument("--no_adaptive_clusters", action="store_true",
                   help="Desativa o escalonamento automático de n_clusters quando o\n"
                        "mascaramento de padding remove parte relevante dos patches.\n"
                        "Por padrão, n_clusters é escalado por 1/fração_de_conteúdo,\n"
                        "com teto em --max_n_clusters_adaptive.")
    p.add_argument("--max_n_clusters_adaptive", type=int, default=12,
                   help="Teto para o escalonamento adaptativo de n_clusters (default: 12).")
    p.add_argument("--visualize",       action="store_true")
    p.add_argument("--viz_max",         type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    feat_path  = Path(args.features)
    corr_path  = Path(args.correspondence)
    output_dir = Path(args.output) if args.output else feat_path.parent

    print(f"\n{'='*62}")
    print(f"  Coseg — Etapa 3: Co-Saliency via Spectral Clustering")
    print(f"{'='*62}")

    step1 = load_step1(feat_path)
    step2 = load_step2(corr_path)

    if step1["patch_grid"] != step2["patch_grid"]:
        raise ValueError(
            "A grade de patches da Etapa 1 não coincide com a da Etapa 2: "
            f"{step1['patch_grid']} vs {step2['patch_grid']}."
        )

    paths_e1 = [str(path) for path in step1["image_paths"]]
    paths_e2 = [str(path) for path in step2["image_paths"]]

    if paths_e1 != paths_e2:
        raise ValueError(
            "features.npz e correspondence.npz possuem imagens diferentes "
            "ou estão em ordens diferentes."
        )

    N = len(paths_e1)
    P = step1["patch_grid"][0] * step1["patch_grid"][1]

    if step2["fg_scores"].shape != (N, P):
        raise ValueError(
            "fg_scores da Etapa 2 incompatível com as features: "
            f"esperado {(N, P)}, recebido {step2['fg_scores'].shape}."
        )

    patch_features = step1["patch_features"]
    image_sizes    = step1["image_sizes"]
    image_paths    = step1["image_paths"]
    patch_grid     = step1["patch_grid"]
    resize_info    = step1.get("resize_info")
    fg_scores_e2   = step2["fg_scores"]
    cls_saliency   = step2["cls_saliency"]

    N = len(image_paths)

    content_mask = None
    bypass_content_mask = None
    if not args.no_padding_mask:
        content_mask = compute_content_mask(resize_info, patch_grid)
        # Regra de conteúdo específica do bypass, idêntica à ablação validada.
        bypass_content_mask = compute_bypass_content_mask(resize_info, patch_grid)

    if not (0.0 < args.empty_fg_top_fraction <= 1.0):
        raise ValueError(
            "--empty_fg_top_fraction deve estar no intervalo (0, 1]. "
            f"Recebido: {args.empty_fg_top_fraction}"
        )

    n_clusters_effective = args.n_clusters
    if content_mask is not None and not args.no_adaptive_clusters:
        n_clusters_effective = compute_adaptive_n_clusters(
            args.n_clusters, content_mask,
            max_n_clusters=args.max_n_clusters_adaptive,
        )

    print(f"\n  Etapa 1: {feat_path}")
    print(f"  Etapa 2: {corr_path}")
    print(f"  Imagens : {N}  |  Grade: {patch_grid[0]}×{patch_grid[1]}")
    print(f"  Saída   : {output_dir}")
    if resize_info is None:
        print(f"  Padding        : sem resize_info (modo 'square' na Etapa 1) — nada a mascarar")
    elif content_mask is None:
        print(f"  Padding        : mascaramento desativado (--no_padding_mask)")
    else:
        print(f"  Padding        : mascarando {1.0 - float(content_mask.mean()):.1%} dos patches (padding)")
    print(f"\n  Parâmetros:")
    print(f"    k_neighbors    = {args.k_neighbors}")
    print(f"    n_spectral     = {args.n_spectral}")
    print(f"    n_clusters     = {args.n_clusters}"
          + (f"  (efetivo após escalonamento adaptativo: {n_clusters_effective})"
             if n_clusters_effective != args.n_clusters else ""))
    print(f"    fg_threshold   = {args.fg_threshold}")
    print(f"    saliency_alpha = {args.saliency_alpha}")
    print(f"    fg_decision_mode = {args.fg_decision_mode}")
    if args.fg_decision_mode == "e2_border_topk":
        print(f"    e2_border_weight = {args.e2_border_weight}")
        print(f"    topk_fraction    = {args.topk_fraction}")
    print(f"    empty_fg_bypass = {args.empty_fg_bypass}")
    print(f"    empty_fg_top_fraction = {args.empty_fg_top_fraction}")
    print(f"    refine         = {args.refine}")

    result = run_cosaliency(
        patch_features = patch_features,
        fg_scores_e2   = fg_scores_e2,
        cls_saliency   = cls_saliency,
        patch_grid     = patch_grid,
        image_sizes    = image_sizes,
        resize_info    = resize_info,
        content_mask   = content_mask,
        k_neighbors    = args.k_neighbors,
        n_spectral     = args.n_spectral,
        n_clusters     = n_clusters_effective,
        fg_threshold   = args.fg_threshold,
        saliency_alpha = args.saliency_alpha,
        refine         = args.refine,
        fg_decision_mode = args.fg_decision_mode,
        e2_border_weight = args.e2_border_weight,
        topk_fraction    = args.topk_fraction,
        empty_fg_bypass_mode = args.empty_fg_bypass,
        empty_fg_top_fraction = args.empty_fg_top_fraction,
        bypass_content_mask   = bypass_content_mask,
    )

    save_cosaliency(result, output_dir, image_paths, image_sizes)
    print_summary(result, image_paths)

    if args.visualize:
        visualize_cosaliency(result, image_paths, image_sizes,
                             output_dir, max_images=args.viz_max)

    print(f"[Etapa 3 concluída]  →  {output_dir}/cosaliency.npz")
    print(f"\n  Próximo passo (Etapa 4 — SAM prompting):")
    print(f"    from step3_cosaliency import load_cosaliency")
    print(f"    cos = load_cosaliency('{output_dir}/cosaliency.npz')")
    
if __name__ == "__main__":
    main()