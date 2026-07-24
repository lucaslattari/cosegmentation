#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Pipeline completo com E2 degree + E2B background suppression
# =============================================================================
#
# CONFIGURAÇÃO PADRÃO VALIDADA:
#   E2: orientation=spatial, degree_alpha=0.10
#   E2B: recurring-background suppression, lambda=0.50, gamma=0.00
#
#   Essa configuração obteve os melhores resultados globais entre as variantes
#   testadas no iCoseg completo. As saídas continuam isoladas por experimento.
#
# USO:
#   bash run_pipeline.sh [OPÇÕES]
#   bash run_pipeline.sh [OPÇÕES] --set icoseg16
#   bash run_pipeline.sh [OPÇÕES] --set full
#   bash run_pipeline.sh [OPÇÕES] --set icoseg38
#   bash run_pipeline.sh [OPÇÕES] --set pascal
#   bash run_pipeline.sh [OPÇÕES] --set msrc
#   bash run_pipeline.sh [OPÇÕES] Pandas Kite Ferrari
#
# EXEMPLOS:
#   bash run_pipeline.sh
#   bash run_pipeline.sh --set icoseg16
#   bash run_pipeline.sh --set full
#   bash run_pipeline.sh Pandas Kite
#   bash run_pipeline.sh --from-step 3 --set icoseg16
#   bash run_pipeline.sh --steps 1,2 --set full
#   bash run_pipeline.sh --dry-run --set full
#   bash run_pipeline.sh --force --visualize Pandas
#   bash run_pipeline.sh --no-sam --set full
#   bash run_pipeline.sh --export --set icoseg16
#   bash run_pipeline.sh --set full --export
#   bash run_pipeline.sh --set icoseg38 --export
#   bash run_pipeline.sh --set icoseg38 --affected-only --export
#
# OPÇÕES:
#   --set NOME          Set: icoseg16, full, icoseg38, pascal ou msrc (default: icoseg16)
#                       full = protocolo semântico legado; icoseg38 = 38 grupos originais
#   --affected-only     Com --set icoseg38, executa somente os 15 grupos que eram
#                       fundidos no protocolo semântico (164 imagens)
#   --degree-alpha A    Expoente da reponderação por grau da E2 (default: 0.10)\n#   --bg-suppression M  E2B: none ou recurrence (default: recurrence)\n#   --bg-lambda X       E2B: força base da supressão (default: 0.50)\n#   --bg-gamma X        E2B: proteção FG-vs-BG, 0..1 (default: 0.00)\n#                       Valores recomendados para ablação: 0, 0.10, 0.25, 0.50\n#   --e3-empty-fg-bypass M   E3: none ou e2_topk (default: e2_topk; só máscara vazia)
#   --e3-empty-fg-top-fraction X  Fração E2B no fallback (default: 0.20)
#   --exp-suffix NOME   Sufixo pro nome do experimento (mesma lista de
#                       classes do --set, saída em pasta separada). Ex.:
#                       --set icoseg16 --exp-suffix geoprior grava em
#                       features/icoseg16_geoprior/, sem tocar em
#                       features/icoseg16/. Use junto com
#                       E3_FG_DECISION_MODE=e2_border_topk para testar o
#                       modo experimental sem sobrescrever o resultado
#                       já validado.
#   --steps N,M,...     Etapas a executar (default: 1,2,3,4,5,eval)
#   --from-step N       Executa etapas N até o final
#   --force             Reprocessa mesmo se saída já existe
#   --dry-run           Imprime os comandos sem executar
#   --visualize         Ativa --visualize em todas as etapas
#   --export            Ativa --export em E4 e E5
#   --no-sam            Pula etapas 4 e 5 (não requer SAM 2)
#   --no-eval           Pula avaliação contra ground truth
#   --no-lock           Não usa lock automático contra execuções simultâneas
#   --max-images N      Limita imagens por coleção
#   selected/all        Atalhos compatíveis para --set icoseg16 / --set full
#   --help              Mostra esta ajuda
#
# CHECKPOINTS / RETOMADA:
#   - Cada etapa só é pulada se a saída existir E passar em validação básica.
#   - Estados por classe ficam em features/<set>/.state/<classe>/{E1..E5,EVAL}.done|running|failed
#   - Para retomar após interrupção, rode o mesmo comando sem --force.
#   - Use --force apenas para reprocessar saídas válidas.
#
# NOTA SOBRE CAMINHOS:
#   Este script pode ser chamado de qualquer diretório (bash pipeline/run_pipeline.sh,
#   cd pipeline && bash run_pipeline.sh, ./run_pipeline.sh, etc). Os scripts Python
#   irmãos (step1_feature_extraction.py, evaluate.py, ...) são sempre resolvidos
#   relativos à pasta onde este script está ($SCRIPT_DIR), e os dados/saídas
#   (icoseg/, gt_index.json, features/, results/, logs/) são sempre resolvidos
#   relativos à raiz do projeto ($PROJECT_ROOT, um nível acima de pipeline/).
#   O iCoseg é esperado em datasets/icoseg/{images,ground_truth}.
#
# =============================================================================

# Não usar set -e aqui pois booleanos false causam exit imediato.
# Cada comando crítico é verificado explicitamente com if/then.
set -uo pipefail

# =============================================================================
# CONFIGURAÇÕES — edite aqui antes de rodar
# =============================================================================

# Resolve a raiz do projeto e a pasta deste script de forma robusta,
# independente de onde/como o script for chamado.
# pipeline/run_pipeline.sh está um nível abaixo da raiz (datasets/, gt_index.json,
# icoseg_index.json), então subimos um diretório a partir do script para achar
# a raiz, e usamos SCRIPT_DIR para localizar os demais scripts .py irmãos.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Dataset iCoseg agora fica sob datasets/icoseg/.
# A raiz pode ser sobrescrita via ICOSEG_DATASET_ROOT, se necessário.
ICOSEG_DATASET_ROOT="${ICOSEG_DATASET_ROOT:-$PROJECT_ROOT/datasets/icoseg}"
ICOSEG_ROOT="$ICOSEG_DATASET_ROOT/images"
GT_ROOT="$ICOSEG_DATASET_ROOT/ground_truth"

# Dataset ativo. Para iCoseg, aponta para ICOSEG_ROOT. Para PASCAL/MSRC,
# aponta para datasets/<nome>/images e usa 1 subpasta = 1 grupo.
DATASET_ROOT="$ICOSEG_ROOT"
COLLECTION_MODE="indexed"
PASCAL_COSEG_ROOT="${PASCAL_COSEG_ROOT:-$PROJECT_ROOT/datasets/PASCAL}"
MSRC_COSEG_ROOT="${MSRC_COSEG_ROOT:-$PROJECT_ROOT/datasets/msrc_coseg}"
FEATURES_ROOT="${FEATURES_ROOT:-$PROJECT_ROOT/features}"
RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs}"
FEATURES_BASE=""
RESULTS_BASE=""
STATE_BASE=""
GT_INDEX="$PROJECT_ROOT/gt_index.json"
ICOSEG_INDEX="$PROJECT_ROOT/icoseg_index.json"
STEP1_PROTOCOL="semantic"
AFFECTED_ONLY="no"

PYTHON="${PYTHON:-python3}"
LOG_DIR=""

# Set experimental padrão. Pode ser sobrescrito por:
#   ICOSEG_SET=full bash run_pipeline.sh
#   bash run_pipeline.sh --set full
ICOSEG_SET="${ICOSEG_SET:-icoseg16}"
EXP_NAME=""
# Sufixo opcional pro nome do experimento — permite rodar o MESMO --set
# (mesma lista de classes) numa pasta de saída separada, sem tocar no
# experimento original. Ex.: --set icoseg16 --exp-suffix geoprior grava em
# features/icoseg16_geoprior/ em vez de features/icoseg16/.
EXP_SUFFIX="${EXP_SUFFIX:-}"
EXP_SUFFIX_WAS_EXPLICIT="no"
[[ -n "$EXP_SUFFIX" ]] && EXP_SUFFIX_WAS_EXPLICIT="yes"

DINOV2_MODEL="dinov2_vits14"
DINO_IMG_SIZE=518
DINO_RESIZE="keep_ar"
# O preprocessamento atual já redimensiona para quadrado fixo derivado de --img_size.
# A camada usada é a última camada intermediária porque extract_features chama
# get_intermediate_layers(..., n=1, return_class_token=True).
DINO_FEATURE_MODE="l2"

# SAM_MODEL calibrado via sweep de ablação (ablation/run_step4_sam_sweep.sh)
# sobre as mesmas 6 classes de validação, com prompt_mode/n_iter já fixados
# em box_points_mask/2 (vencedores do sweep anterior). Critério: maior
# gt_iou_mean MÉDIO entre as 6 classes. sam2.1-hiera-large venceu (0.647 vs
# 0.635 do small), mas o ganho NÃO é uniforme: é puxado majoritariamente por
# Gymnastics (+0.093), com Taj Mahal levemente pior (-0.030) e Kite
# praticamente inalterado (-0.002) — Kite não melhora com modelo SAM maior
# porque o gargalo dele está no prompt recebido (Etapas 2/3), não na
# capacidade do SAM. sam2.1-hiera-large é sensivelmente mais lento por
# imagem que o small; se o volume de imagens em produção for grande, vale
# medir o tempo de inferência antes de manter este default.
SAM_MODEL="facebook/sam2.1-hiera-large"

# E2_TOP_K / E2_TEMPERATURE calibrados via sweep de ablação
# (ablation/run_step2_correspondence_sweep.sh) sobre as mesmas 6 classes do
# iCoseg usadas para calibrar a Etapa 1 e a Etapa 3, alimentado pela config
# única de Etapa 1 já validada (vits14/518/keep_ar/l2). Critério: maior
# gt_best_iou MÉDIO entre as 6 classes (mesmo critério de generalização
# usado nas outras etapas, não o melhor por classe individual).
# top_k=60 foi onde a taxa de ganho por passo caiu de forma clara pela
# primeira vez (delta/step: 0.0034 -> 0.0026 -> 0.0026 -> 0.0019) E onde as
# duas classes mais fracas da validação (Brown Bear N=5, Taj Mahal N=10)
# atingiram o teto físico de pares possíveis (C(N,2) pares total) — não dá
# pra melhorá-las mais só aumentando top_k, então parar aqui não deixa
# ganho na mesa nas classes que mais precisavam. temperature=0.05 dominou
# 0.02 e 0.10 em todo valor de top_k testado (já era o default).
E2_TOP_K=60
E2_TEMPERATURE=0.05

# Reponderação parcial por grau antes do NCut.
# W'_{ij} = W_{ij} / (d_i d_j)^alpha
#
# Configuração padrão validada no iCoseg completo:
#   degree_alpha = 0.10
#
# Comparação global com orientação spatial:
#   0.00 -> IoU média 0.6827
#   0.10 -> IoU média 0.7080  (melhor entre os valores testados)
#   0.25 -> IoU média 0.6949
#   0.50 -> IoU média 0.6557
#
# Pode ser sobrescrito por:
#   E2_DEGREE_ALPHA=0.25 bash pipeline/run_pipeline.sh --set full --export
# ou pela CLI:
#   bash pipeline/run_pipeline.sh --degree-alpha 0.25 --set full --export
E2_DEGREE_ALPHA="${E2_DEGREE_ALPHA:-0.10}"

# Heurística de orientação FG/BG da Etapa 2.
# Configuração padrão validada atualmente:
#   spatial -> lado mais central/compacto da bipartição NCut
#
# O modo cls continua disponível para ablações via:
#   E2_ORIENTATION=cls bash pipeline/run_pipeline.sh ...
#
# Valores aceitos pelo step2_correspondence.py:
#   cls     -> maior saliência CLS média
#   spatial -> lado mais central/compacto da bipartição NCut
E2_ORIENTATION="${E2_ORIENTATION:-spatial}"

# E2B — supressão adaptativa de background recorrente.
#
# Configuração padrão validada no iCoseg completo:
#   mode   = recurrence
#   lambda = 0.50
#   gamma  = 0.00
#
# Resultado global observado:
#   accuracy ≈ 0.9262
#   IoU      ≈ 0.7562
#   F1       ≈ 0.8090
#
# A variante gamma=0.75 ficou abaixo globalmente, apesar de proteger melhor
# algumas classes específicas. Por isso gamma=0.00 passa a ser o default.
#
# Para desativar a etapa:
#   --bg-suppression none
#
# A etapa gera correspondence_bg.npz; E3 usa esse arquivo quando E2B está ativa.
E2_BG_SUPPRESSION="${E2_BG_SUPPRESSION:-recurrence}"
E2_BG_LAMBDA="${E2_BG_LAMBDA:-0.50}"
E2_BG_GAMMA="${E2_BG_GAMMA:-0.00}"
E2_BG_BORDER_WIDTH="${E2_BG_BORDER_WIDTH:-2}"
E2_BG_TOP_FG_FRACTION="${E2_BG_TOP_FG_FRACTION:-0.10}"
# E3_N_CLUSTERS / E3_K_NEIGHBORS / E3_SALIENCY_ALPHA calibrados via sweep de
# ablação (ablation/run_step3_cosaliency_sweep.sh) sobre as mesmas 6 classes
# do iCoseg usadas em toda a calibração (3 "fáceis": Taj Mahal, Balloon,
# Panda; 3 "difíceis": Brown Bear, Kite, Gymnastics), com GT. Critério:
# maior patch_iou_refined MÉDIO entre as 6 classes (não o melhor por classe
# individualmente — isso seria overfitting às classes testadas e não
# generalizaria para outras coleções do iCoseg ou datasets futuros).
#
# IMPORTANTE: esta calibração final foi feita alimentando o sweep de E3 com
# a config ÚNICA e já validada de E1 (vits14/518/keep_ar/l2) e E2
# (top_k=60/temperature=0.05) — não mais os .npz "melhores por classe" de
# uma rodada anterior. Essa rodada anterior (E1/E2 escolhidos por classe)
# tinha apontado nc=8/kn=10/alpha=0.75 como vencedor (patch_iou médio
# 0.552), mas era um resultado contaminado: a Etapa 3 tinha se ajustado às
# idiossincrasias daquela mistura específica de inputs, não ao que o
# pipeline de fato produz com uma config única e consistente. Refeito com a
# cadeia E1->E2->E3 inteira consistente, o vencedor real é outro:
#   nc=8, kn=15, alpha=0.5   (patch_iou médio 0.563, pior caso 0.394)
# contra 0.462 médio / 0.000 pior-caso (Kite) do default original
# nc=5/kn=10/alpha=0.5, e 0.494 médio / 0.161 pior-caso do resultado
# intermediário contaminado acima. Kite, que antes falhava por completo
# (IoU=0.000), passa a IoU=0.394; Panda, que tinha regredido na versão
# contaminada (0.698->0.419), se recupera e melhora (0.698->0.796) na
# versão consistente — confirma que aquela regressão era artefato de
# inputs inconsistentes, não um trade-off real do método.
E3_N_CLUSTERS=8
E3_K_NEIGHBORS=15
E3_N_SPECTRAL=8
E3_SALIENCY_ALPHA=0.5
# fg_decision_mode="e2_border_topk" é EXPERIMENTAL — testado em ablação
# (ablation/analyze_geometric_priors.py) nas 14 classes do icoseg16.
# Melhora média e pior-caso em relação ao modo "saliency" (produção),
# resolvendo bem as classes com fundo mais consistente que o objeto
# (multidões/gramado), mas com custo real em 2 das 14 classes testadas
# (Kite, Statue) — não é o default, precisa ser habilitado explicitamente
# via E3_FG_DECISION_MODE=e2_border_topk. Ver step3_cosaliency.py
# --fg_decision_mode para detalhes. e2_border_weight/topk_fraction só têm
# efeito quando fg_decision_mode=e2_border_topk (valores validados em
# ablação, não mudar sem re-validar).
E3_FG_DECISION_MODE="${E3_FG_DECISION_MODE:-saliency}"
E3_E2_BORDER_WEIGHT="${E3_E2_BORDER_WEIGHT:-0.3}"
E3_TOPK_FRACTION="${E3_TOPK_FRACTION:-0.30}"

# E3 — Empty-Foreground Adaptive Bypass.
#
# Configuração validada:
#   mode         = e2_topk
#   top_fraction = 0.20
#   trigger      = EXATAMENTE zero patches FG válidos na saída normal da E3
#
# Resultado PASCAL completo observado:
#   IoU macro: 0.4186 -> 0.5836
#   16/20 classes melhoraram, 4/20 ficaram iguais, 0/20 pioraram.
#
# Controles iCoseg com trigger exato (máscara vazia):
#   82/82 imagens preservadas nos 6 controles testados.
#
# Importante: o trigger antigo FG <= 1% NÃO é usado. Ele causou regressão em
# Kite-Brighton. O mecanismo atual só intervém em estado degenerado inequívoco:
# nenhuma região foreground selecionada pela E3.
#
# Para desativar:
#   E3_EMPTY_FG_BYPASS=none bash pipeline/run_pipeline.sh ...
# ou:
#   --e3-empty-fg-bypass none
E3_EMPTY_FG_BYPASS="${E3_EMPTY_FG_BYPASS:-e2_topk}"
E3_EMPTY_FG_TOP_FRACTION="${E3_EMPTY_FG_TOP_FRACTION:-0.20}"

E4_PROMPT_MODE="box_points_mask"
E4_N_ITER=2
# E5_MAX_ITER / E5_CONVERGE_THRESH / E5_N_FG_PTS / E5_N_BG_PTS calibrados via
# sweep de ablação (ablation/run_step5_iterative_sweep.sh) sobre as mesmas 6
# classes de validação, alimentado pela config já fixada de E1/E4. Critério:
# maior best_gt_iou MÉDIO entre as 6 classes.
#
# E5_MAX_ITER precisou subir bem mais que o esperado: Balloon e Taj Mahal só
# convergem de verdade em 15 e 17 iterações respectivamente (as outras 4
# classes convergem sozinhas bem antes disso e não sofrem o custo extra,
# graças ao converge_thresh). Um teto baixo (o antigo default=4) cortava
# essas duas classes bem antes do pico — Balloon perdia ~0.25 de IoU,
# Taj Mahal ~0.20, só por parar cedo demais. 20 dá margem de sobra acima do
# maior ponto de convergência observado (17).
#
# n_fg_pts/n_bg_pts=8 (o teto testado no sweep) venceu por média sem piorar
# nenhuma classe de forma real — ganhos grandes em Taj Mahal (+0.13) e
# Balloon (+0.08) sobre o valor testado mais baixo (3), neutro (ruído) em
# Brown Bear/Panda.
#
# Ao contrário do refine da Etapa 3 (que ficou desligado por piorar
# resultados), a Etapa 5 — devidamente calibrada — ajudou as 6 classes sem
# exceção, sem sinal de convergência-para-o-erro (checado via mask_history:
# a curva de IoU-vs-GT nunca caiu, só desacelerou até estabilizar).
E5_MAX_ITER=20
E5_CONVERGE_THRESH=0.99
E5_N_FG_PTS=8
E5_N_BG_PTS=8

# =============================================================================
# ESTADO INTERNO (todas globais, sem set -e para booleanos)
# =============================================================================

STEPS="1,2,2b,3,4,5,eval"
FORCE="no"
DRY_RUN="no"
VISUALIZE="no"
EXPORT="no"
NO_SAM="no"
NO_EVAL="no"
LOCK="yes"
MAX_IMAGES=""
VIZ_FLAG=""
EXPORT_FLAG=""
CLASSES=()
MANUAL_CLASSES="no"

# Subconjunto escolhido para este experimento.
# Observação: os nomes precisam bater com as chaves/pastas do iCoseg.
ICOSEG_SELECTED_CLASSES=(
    "Alaskan Brown Bear"
    "Red Sox Players"
    "Stonehenge"
    "Liverpool"
    "Ferrari"
    "Taj Mahal"
    "Elephant"
    "Panda"
    "Kite"
    "Gymnastics"
    "Skating"
    "Balloon"
    "Statue"
    "Brown Bear"
)


# Grupos originais que eram fundidos por CLASS_GROUP_OVERRIDES no protocolo
# semântico. No protocolo icoseg38, cada entrada abaixo deve ser processada
# independentemente. Total esperado: 15 grupos / 164 imagens.
ICOSEG38_AFFECTED_CLASSES=(
    "Stonehenge-Stongehenge-rjt208"
    "Stonehenge-Salisbury Cathedral and Stonehenge-Sou'wester"
    "Agra Taj Mahal-Taj Mahal-LeszekZadlo"
    "Agra Taj Mahal-Inde du Nord 2004-Mhln"
    "Pandas-Tai-Land 6708-TAIwiffic"
    "Pandas-National Zoo, Washington DC-smata2"
    "Kite-Brighton kite Festival-jadepike4"
    "Kite-kitekid"
    "Kite-Margate Kite Festival 08-only lines"
    "Kite-Colt Park 6-1-08-rreed3"
    "Gymnastics-1"
    "Gymnastics-2"
    "Gymnastics-3"
    "Skating-Skating-Rich Moffitt"
    "Skating-ISU Grand Prix 2007-albertosepe"
)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

N_OK=0; N_SKIP=0; N_FAIL=0
FAILED_CLASSES=()
LOCK_FILE=""

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${RESET} $*"; }
ok()   { echo -e "${GREEN}  ✓ $*${RESET}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${RESET}"; }
err()  { echo -e "${RED}  ✗ $*${RESET}"; }
bold() { echo -e "${BOLD}$*${RESET}"; }

step_enabled() {
    [[ ",$STEPS," == *",$1,"* ]]
}

run_cmd() {
    # Executa o comando como array de argumentos, SEM eval.
    #
    # Versões anteriores juntavam tudo em uma string e usavam eval, o que
    # quebrava nomes contendo apóstrofo, por exemplo:
    #   Stonehenge-Salisbury Cathedral and Stonehenge-Sou'wester
    #
    # Os call sites legados envolvem alguns argumentos em aspas simples
    # literais ("'$VAR'"). Removemos apenas esse par externo e preservamos
    # integralmente o conteúdo interno do argumento.
    local log_file="$1"
    shift

    local -a cmd=()
    local arg

    for arg in "$@"; do
        if [[ ${#arg} -ge 2 && "${arg:0:1}" == "'" && "${arg: -1}" == "'" ]]; then
            arg="${arg:1:${#arg}-2}"
        fi
        cmd+=("$arg")
    done

    local printable=""
    printf -v printable '%q ' "${cmd[@]}"
    printable="${printable% }"

    if [[ "$DRY_RUN" == "yes" ]]; then
        echo -e "${YELLOW}  [dry-run]${RESET} $printable"
        return 0
    fi

    echo -e "  ${CYAN}▶${RESET} $printable"

    if "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
        return 0
    else
        return 1
    fi
}

print_help() {
    sed -n '/^# USO:/,/^# =============================================================================/p' "$0" | sed 's/^# \?//'
    exit 0
}

# =============================================================================
# VALIDAÇÃO DE SAÍDAS / CHECKPOINTS
# =============================================================================

valid_npz() {
    local file="$1"
    [[ -s "$file" ]] || return 1

    "$PYTHON" - "$file" >/dev/null 2>&1 <<'PYEOF'
import sys
import numpy as np

path = sys.argv[1]
try:
    data = np.load(path, allow_pickle=True)
    if len(data.files) == 0:
        raise RuntimeError("NPZ vazio")
except Exception:
    raise SystemExit(1)
PYEOF
}

valid_json() {
    local file="$1"
    [[ -s "$file" ]] || return 1

    "$PYTHON" - "$file" >/dev/null 2>&1 <<'PYEOF'
import sys
import json

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        json.load(f)
except Exception:
    raise SystemExit(1)
PYEOF
}

valid_step3_output() {
    # Além de validar o NPZ, garante que um checkpoint antigo sem o novo
    # empty-FG bypass não seja silenciosamente reutilizado quando o mecanismo
    # está habilitado. Isso é importante mesmo quando o usuário fornece um
    # --exp-suffix já existente.
    local file="$1"
    valid_npz "$file" || return 1

    "$PYTHON" - "$file" "$E3_EMPTY_FG_BYPASS" "$E3_EMPTY_FG_TOP_FRACTION" >/dev/null 2>&1 <<'PYEOF'
import sys
import numpy as np

path, desired_mode, desired_top = sys.argv[1], sys.argv[2], float(sys.argv[3])

try:
    d = np.load(path, allow_pickle=True)
    files = set(d.files)

    if "empty_fg_bypass_mode" not in files:
        # Saídas legadas são aceitáveis apenas se o bypass atual está desligado.
        raise SystemExit(0 if desired_mode == "none" else 1)

    mode = str(d["empty_fg_bypass_mode"].item())
    top = float(d["empty_fg_top_fraction"].item()) if "empty_fg_top_fraction" in files else 0.20

    if mode != desired_mode:
        raise SystemExit(1)

    if desired_mode == "e2_topk" and abs(top - desired_top) > 1e-8:
        raise SystemExit(1)

except Exception:
    raise SystemExit(1)
PYEOF
}

mask_dir_has_masks() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    find "$dir" -name "*_mask.png" -print -quit 2>/dev/null | grep -q .
}

valid_step4_output() {
    local feat_dir="$1"
    valid_npz "$feat_dir/refined_masks.npz" || return 1

    if [[ "$EXPORT" == "yes" ]]; then
        mask_dir_has_masks "$feat_dir/segmented_e4" || return 1
    fi

    return 0
}

valid_step5_output() {
    local feat_dir="$1"
    valid_npz "$feat_dir/iterative_masks.npz" || return 1

    if [[ "$EXPORT" == "yes" ]]; then
        mask_dir_has_masks "$feat_dir/segmented_e5" || return 1
    fi

    return 0
}

should_run_output() {
    # Uso:
    #   should_run_output E1 features.npz valid_npz "$FEAT_DIR/features.npz"
    #   should_run_output E4 refined_masks.npz valid_step4_output "$FEAT_DIR"
    local step="$1"
    local label="$2"
    local validator="$3"
    shift 3

    if [[ "$FORCE" == "yes" ]]; then
        return 0
    fi

    if "$validator" "$@"; then
        warn "$step: saída válida já existe ($label) — pulando (use --force para reprocessar)"
        return 1
    fi

    warn "$step: saída ausente, inválida ou incompleta ($label) — executando"
    return 0
}

should_run_output_or_dirty() {
    # Se uma etapa upstream foi realmente recalculada nesta classe, não reutiliza
    # checkpoints downstream potencialmente obsoletos.
    #
    # Uso:
    #   should_run_output_or_dirty "$UPSTREAM_DIRTY" E4 ... validator args...
    local upstream_dirty="$1"
    shift

    if [[ "$upstream_dirty" == "yes" && "$FORCE" != "yes" ]]; then
        local step="$1"
        local label="$2"
        warn "$step: etapa upstream foi recalculada — reprocessando ($label)"
        return 0
    fi

    should_run_output "$@"
}

# =============================================================================
# ESTADO .running / .done / .failed
# =============================================================================

state_dir_for_class() {
    local safe_name="$1"
    echo "$STATE_BASE/$safe_name"
}

write_state_file() {
    local safe_name="$1"
    local step="$2"
    local status="$3"
    local dir
    dir=$(state_dir_for_class "$safe_name")
    mkdir -p "$dir"

    {
        echo "status=$status"
        echo "set=$ICOSEG_SET"
        echo "experiment=$EXP_NAME"
        echo "class=$safe_name"
        echo "step=$step"
        echo "timestamp=$(date -Is)"
        echo "host=$(hostname 2>/dev/null || echo unknown)"
        echo "pid=$$"
    } > "$dir/$step.$status"
}

mark_running() {
    [[ "$DRY_RUN" == "yes" ]] && return 0
    local safe_name="$1"
    local step="$2"
    local dir
    dir=$(state_dir_for_class "$safe_name")
    mkdir -p "$dir"
    rm -f "$dir/$step.done" "$dir/$step.failed"
    write_state_file "$safe_name" "$step" "running"
}

mark_done() {
    [[ "$DRY_RUN" == "yes" ]] && return 0
    local safe_name="$1"
    local step="$2"
    local dir
    dir=$(state_dir_for_class "$safe_name")
    mkdir -p "$dir"
    rm -f "$dir/$step.running" "$dir/$step.failed"
    write_state_file "$safe_name" "$step" "done"
}

mark_failed() {
    [[ "$DRY_RUN" == "yes" ]] && return 0
    local safe_name="$1"
    local step="$2"
    local dir
    dir=$(state_dir_for_class "$safe_name")
    mkdir -p "$dir"
    rm -f "$dir/$step.running" "$dir/$step.done"
    write_state_file "$safe_name" "$step" "failed"
}

finish_validated_step() {
    local safe_name="$1"
    local step="$2"
    local label="$3"
    local validator="$4"
    shift 4

    if [[ "$DRY_RUN" == "yes" ]]; then
        ok "[dry-run] $label"
        return 0
    fi

    if "$validator" "$@"; then
        mark_done "$safe_name" "$step"
        ok "$label"
        return 0
    fi

    mark_failed "$safe_name" "$step"
    err "$step gerou saída inválida ou incompleta: $label"
    return 1
}

skip_valid_step() {
    local safe_name="$1"
    local step="$2"
    mark_done "$safe_name" "$step"
    N_SKIP=$((N_SKIP + 1))
}

select_segmented_dir() {
    # Define SEG_DIR e SEG_LABEL como globais da função chamadora.
    local feat_dir="$1"

    SEG_DIR=""
    SEG_LABEL=""

    if mask_dir_has_masks "$feat_dir/segmented_e5"; then
        SEG_DIR="$feat_dir/segmented_e5"
        SEG_LABEL="E5"
    elif mask_dir_has_masks "$feat_dir/segmented_e4"; then
        SEG_DIR="$feat_dir/segmented_e4"
        SEG_LABEL="E4"
    elif mask_dir_has_masks "$feat_dir/segmented"; then
        SEG_DIR="$feat_dir/segmented"
        SEG_LABEL="legacy"
    fi
}

acquire_lock() {
    if [[ "$LOCK" != "yes" ]]; then
        return 0
    fi
    if [[ "$DRY_RUN" == "yes" ]]; then
        return 0
    fi

    if ! command -v flock >/dev/null 2>&1; then
        warn "flock não encontrado; seguindo sem lock automático"
        return 0
    fi

    mkdir -p "$FEATURES_BASE"
    LOCK_FILE="$FEATURES_BASE/.run_pipeline.lock"
    exec 9>"$LOCK_FILE"

    if ! flock -n 9; then
        err "Já existe uma execução ativa para este set: $ICOSEG_SET"
        echo "Lock: $LOCK_FILE"
        echo "Use --no-lock somente se tiver certeza de que não há outro pipeline rodando."
        exit 1
    fi
}

# =============================================================================
# PARSE DE ARGUMENTOS
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                print_help
                ;;
            --steps)
                STEPS="$2"; shift 2
                ;;
            --from-step)
                case "$2" in
                    1) STEPS="1,2,2b,3,4,5,eval" ;;
                    2) STEPS="2,2b,3,4,5,eval"   ;;
                    3) STEPS="3,4,5,eval"      ;;
                    4) STEPS="4,5,eval"        ;;
                    5) STEPS="5,eval"          ;;
                    *) echo "from-step inválido: $2"; exit 1 ;;
                esac
                shift 2
                ;;
            --force)
                FORCE="yes"; shift
                ;;
            --dry-run)
                DRY_RUN="yes"; shift
                ;;
            --visualize)
                VISUALIZE="yes"; VIZ_FLAG="--visualize"; shift
                ;;
            --export)
                EXPORT="yes"; EXPORT_FLAG="--export"; shift
                ;;
            --no-sam)
                NO_SAM="yes"
                STEPS="1,2,2b,3,eval"
                shift
                ;;
            --no-eval)
                NO_EVAL="yes"
                STEPS="${STEPS/,eval/}"
                shift
                ;;
            --no-lock)
                LOCK="no"; shift
                ;;
            --affected-only)
                AFFECTED_ONLY="yes"; shift
                ;;
            --max-images)
                MAX_IMAGES="--max_images $2"; shift 2
                ;;
            --set|--dataset-set)
                ICOSEG_SET="$2"; shift 2
                ;;
            --set=*|--dataset-set=*)
                ICOSEG_SET="${1#*=}"; shift
                ;;
            --degree-alpha)
                E2_DEGREE_ALPHA="$2"; shift 2
                ;;
            --degree-alpha=*)
                E2_DEGREE_ALPHA="${1#*=}"; shift
                ;;
            --bg-suppression)
                E2_BG_SUPPRESSION="$2"; shift 2
                ;;
            --bg-suppression=*)
                E2_BG_SUPPRESSION="${1#*=}"; shift
                ;;
            --bg-lambda)
                E2_BG_LAMBDA="$2"; shift 2
                ;;
            --bg-lambda=*)
                E2_BG_LAMBDA="${1#*=}"; shift
                ;;
            --bg-gamma)
                E2_BG_GAMMA="$2"; shift 2
                ;;
            --bg-gamma=*)
                E2_BG_GAMMA="${1#*=}"; shift
                ;;
            --e3-empty-fg-bypass)
                E3_EMPTY_FG_BYPASS="$2"; shift 2
                ;;
            --e3-empty-fg-bypass=*)
                E3_EMPTY_FG_BYPASS="${1#*=}"; shift
                ;;
            --e3-empty-fg-top-fraction)
                E3_EMPTY_FG_TOP_FRACTION="$2"; shift 2
                ;;
            --e3-empty-fg-top-fraction=*)
                E3_EMPTY_FG_TOP_FRACTION="${1#*=}"; shift
                ;;
            --exp-suffix)
                EXP_SUFFIX="$2"; EXP_SUFFIX_WAS_EXPLICIT="yes"; shift 2
                ;;
            --exp-suffix=*)
                EXP_SUFFIX="${1#*=}"; EXP_SUFFIX_WAS_EXPLICIT="yes"; shift
                ;;
            all|full|icoseg_full)
                ICOSEG_SET="full"; shift
                ;;
            icoseg38|full38|official38)
                ICOSEG_SET="icoseg38"; shift
                ;;
            pascal|pascal_voc|pascal-voc)
                ICOSEG_SET="pascal"; shift
                ;;
            msrc|msrc7)
                ICOSEG_SET="msrc"; shift
                ;;
            selected|icoseg16|subset)
                ICOSEG_SET="icoseg16"; shift
                ;;
            -*)
                echo "Opção desconhecida: $1  (use --help)"; exit 1
                ;;
            *)
                CLASSES+=("$1")
                MANUAL_CLASSES="yes"
                shift
                ;;
        esac
    done
}

# =============================================================================
# CONFIGURAÇÃO DO SET EXPERIMENTAL
# =============================================================================

configure_experiment_dirs() {
    case "$ICOSEG_SET" in
        icoseg16|selected|subset)
            ICOSEG_SET="icoseg16"
            EXP_NAME="icoseg16"
            STEP1_PROTOCOL="semantic"
            ICOSEG_INDEX="$PROJECT_ROOT/icoseg_index.json"
            DATASET_ROOT="$ICOSEG_ROOT"
            GT_ROOT="$ICOSEG_DATASET_ROOT/ground_truth"
            GT_INDEX="$PROJECT_ROOT/gt_index.json"
            COLLECTION_MODE="indexed"
            ;;
        full|all|icoseg_full)
            ICOSEG_SET="full"
            EXP_NAME="icoseg_full"
            STEP1_PROTOCOL="semantic"
            ICOSEG_INDEX="$PROJECT_ROOT/icoseg_index.json"
            DATASET_ROOT="$ICOSEG_ROOT"
            GT_ROOT="$ICOSEG_DATASET_ROOT/ground_truth"
            GT_INDEX="$PROJECT_ROOT/gt_index.json"
            COLLECTION_MODE="indexed"
            ;;
        icoseg38|full38|official38)
            ICOSEG_SET="icoseg38"
            EXP_NAME="icoseg38"
            STEP1_PROTOCOL="icoseg38"
            ICOSEG_INDEX="$PROJECT_ROOT/icoseg_index_icoseg38.json"
            DATASET_ROOT="$ICOSEG_ROOT"
            GT_ROOT="$ICOSEG_DATASET_ROOT/ground_truth"
            GT_INDEX="$PROJECT_ROOT/gt_index.json"
            COLLECTION_MODE="indexed"
            ;;
        pascal|pascal_voc|pascal-voc)
            ICOSEG_SET="pascal"
            EXP_NAME="pascal_coseg"
            STEP1_PROTOCOL="semantic"
            DATASET_ROOT="$PASCAL_COSEG_ROOT/images"
            GT_ROOT="$PASCAL_COSEG_ROOT/GT"
            GT_INDEX="$PASCAL_COSEG_ROOT/gt_index.json"
            COLLECTION_MODE="folder"
            ;;
        msrc|msrc7)
            ICOSEG_SET="msrc"
            EXP_NAME="msrc_coseg"
            STEP1_PROTOCOL="semantic"
            DATASET_ROOT="$MSRC_COSEG_ROOT/images"
            GT_ROOT="$MSRC_COSEG_ROOT/ground_truth"
            GT_INDEX="$MSRC_COSEG_ROOT/gt_index.json"
            COLLECTION_MODE="folder"
            ;;
        *)
            err "Set inválido: $ICOSEG_SET"
            echo "Use: --set icoseg16, --set full, --set icoseg38, --set pascal ou --set msrc"
            exit 1
            ;;
    esac

    if [[ "$AFFECTED_ONLY" == "yes" && "$ICOSEG_SET" != "icoseg38" ]]; then
        err "--affected-only só pode ser usado com --set icoseg38"
        exit 1
    fi

    case "$E2_BG_SUPPRESSION" in
        none|recurrence) ;;
        *)
            err "Modo --bg-suppression inválido: $E2_BG_SUPPRESSION"
            echo "Use: none ou recurrence"
            exit 1
            ;;
    esac

    if ! "$PYTHON" - "$E2_BG_LAMBDA" "$E2_BG_GAMMA" >/dev/null 2>&1 <<'PYEOF'
import sys
lam = float(sys.argv[1])
gamma = float(sys.argv[2])
assert lam >= 0.0
assert 0.0 <= gamma <= 1.0
PYEOF
    then
        err "Parâmetros E2B inválidos: lambda=$E2_BG_LAMBDA gamma=$E2_BG_GAMMA"
        exit 1
    fi

    case "$E3_EMPTY_FG_BYPASS" in
        none|e2_topk) ;;
        *)
            err "Modo --e3-empty-fg-bypass inválido: $E3_EMPTY_FG_BYPASS"
            echo "Use: none ou e2_topk"
            exit 1
            ;;
    esac

    if ! "$PYTHON" - "$E3_EMPTY_FG_TOP_FRACTION" >/dev/null 2>&1 <<'PYEOF'
import sys
x = float(sys.argv[1])
assert 0.0 < x <= 1.0
PYEOF
    then
        err "Parâmetro E3 empty-FG inválido: top_fraction=$E3_EMPTY_FG_TOP_FRACTION"
        echo "Use um valor no intervalo (0, 1]. Ex.: 0.20"
        exit 1
    fi

    # Se o usuário não fornecer --exp-suffix, cria um sufixo que registra
    # a configuração estrutural do experimento e evita colisões entre runs.
    if [[ -z "$EXP_SUFFIX" ]]; then
        local alpha_tag lambda_tag gamma_tag
        alpha_tag=$(echo "$E2_DEGREE_ALPHA" | tr '.' 'p')
        lambda_tag=$(echo "$E2_BG_LAMBDA" | tr '.' 'p')
        gamma_tag=$(echo "$E2_BG_GAMMA" | tr '.' 'p')

        if [[ "$E2_BG_SUPPRESSION" == "recurrence" ]]; then
            EXP_SUFFIX="e2_spatial_degree_a${alpha_tag}_bg_recurrence_l${lambda_tag}_g${gamma_tag}"
        else
            EXP_SUFFIX="e2_spatial_degree_a${alpha_tag}"
        fi

        if [[ "$E3_EMPTY_FG_BYPASS" == "e2_topk" ]]; then
            local bypass_pct
            bypass_pct=$("$PYTHON" - "$E3_EMPTY_FG_TOP_FRACTION" <<'PYEOF'
import sys
print(int(round(float(sys.argv[1]) * 100)))
PYEOF
)
            EXP_SUFFIX="${EXP_SUFFIX}_e3bypass_top${bypass_pct}_t0"
        fi
    fi

    if [[ -n "$EXP_SUFFIX" ]]; then
        EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
    fi

    FEATURES_BASE="$FEATURES_ROOT/$EXP_NAME"
    RESULTS_BASE="$RESULTS_ROOT/$EXP_NAME"
    LOG_DIR="$LOG_ROOT/$EXP_NAME"
    STATE_BASE="$FEATURES_BASE/.state"
}

# =============================================================================
# DESCOBERTA DE CLASSES
# =============================================================================

get_all_classes() {
    if [[ "$COLLECTION_MODE" == "folder" ]]; then
        if [[ ! -d "$DATASET_ROOT" ]]; then
            err "Dataset não encontrado: $DATASET_ROOT"
            exit 1
        fi
        find "$DATASET_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
        return
    fi

    if [[ ! -f "$ICOSEG_INDEX" ]]; then
        log "Gerando $ICOSEG_INDEX via --scan (protocol=$STEP1_PROTOCOL)..."
        # O E1 salva o índice no diretório atual; executamos a partir da raiz do
        # projeto para garantir que ele seja criado em $PROJECT_ROOT.
        if ! (cd "$PROJECT_ROOT" &&               $PYTHON "$SCRIPT_DIR/step1_feature_extraction.py"                   --scan "$ICOSEG_ROOT"                   --protocol "$STEP1_PROTOCOL"); then
            err "Não foi possível gerar $ICOSEG_INDEX"
            exit 1
        fi
    fi

    $PYTHON - "$ICOSEG_INDEX" <<'PYEOF'
import json, sys
idx = json.load(open(sys.argv[1], encoding="utf-8"))
for cls in sorted(idx.keys()):
    print(cls)
PYEOF
}

resolve_classes() {
    # Classes passadas manualmente na CLI têm prioridade sobre --set.
    if [[ ${#CLASSES[@]} -gt 0 ]]; then
        log "Classes definidas manualmente: ${#CLASSES[@]}"
        for cls in "${CLASSES[@]}"; do
            echo "    • $cls"
        done
        return
    fi

    case "$ICOSEG_SET" in
        icoseg16)
            CLASSES=("${ICOSEG_SELECTED_CLASSES[@]}")
            log "Usando subconjunto iCoseg atual: ${#CLASSES[@]} classes"
            ;;
        full)
            mapfile -t CLASSES < <(get_all_classes)
            log "Usando iCoseg completo no protocolo semântico: ${#CLASSES[@]} classes"
            ;;
        icoseg38)
            # Garante que o índice oficial exista antes de selecionar classes.
            if [[ "$AFFECTED_ONLY" == "yes" ]]; then
                get_all_classes >/dev/null
                CLASSES=("${ICOSEG38_AFFECTED_CLASSES[@]}")
                log "Usando somente grupos afetados do iCoseg-38: ${#CLASSES[@]} grupos"
            else
                mapfile -t CLASSES < <(get_all_classes)
                log "Usando protocolo iCoseg-38 completo: ${#CLASSES[@]} grupos"
            fi
            ;;
        pascal)
            mapfile -t CLASSES < <(get_all_classes)
            log "Usando PASCAL-VOC CoSeg: ${#CLASSES[@]} grupos"
            if [[ ${#CLASSES[@]} -ne 20 ]]; then
                err "PASCAL CoSeg deveria conter 20 grupos; encontrados ${#CLASSES[@]}."
                exit 1
            fi

            # Valida o pacote preparado antes de iniciar DINO/SAM.
            local n_pascal_images n_pascal_gt
            n_pascal_images=$(find "$DATASET_ROOT" -type f \( \
                -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
                -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \
            \) | wc -l)
            n_pascal_gt=$(find "$GT_ROOT" -type f \( \
                -iname '*.png' -o -iname '*.bmp' \
                -o -iname '*.tif' -o -iname '*.tiff' \
            \) | wc -l)

            if [[ "$n_pascal_images" -ne 1037 ]]; then
                err "PASCAL CoSeg deveria conter 1037 imagens; encontradas $n_pascal_images."
                exit 1
            fi
            if [[ "$n_pascal_gt" -ne 1037 ]]; then
                err "PASCAL CoSeg deveria conter 1037 GTs; encontrados $n_pascal_gt."
                exit 1
            fi

            ok "PASCAL validado: 20 classes, 1037 imagens e 1037 GTs"
            ;;
        msrc)
            mapfile -t CLASSES < <(get_all_classes)
            log "Usando MSRC CoSeg: ${#CLASSES[@]} grupos"
            if [[ ${#CLASSES[@]} -ne 7 ]]; then
                err "MSRC-70 deveria conter 7 grupos; encontrados ${#CLASSES[@]}."
                exit 1
            fi
            ;;
        *)
            err "Set inválido: $ICOSEG_SET"
            echo "Use: --set icoseg16, --set full, --set icoseg38, --set pascal ou --set msrc"
            exit 1
            ;;
    esac

    if [[ "$ICOSEG_SET" == "icoseg38" ]]; then
        # Valida que todas as classes selecionadas existem no índice oficial.
        if ! "$PYTHON" - "$ICOSEG_INDEX" "${CLASSES[@]}" <<'PYEOF'
import json, sys
index_path, *classes = sys.argv[1:]
idx = json.load(open(index_path, encoding="utf-8"))
missing = [c for c in classes if c not in idx]
if missing:
    print("[ERRO] Classes ausentes no índice iCoseg-38:", file=sys.stderr)
    for c in missing:
        print(f"  - {c}", file=sys.stderr)
    raise SystemExit(1)
PYEOF
        then
            exit 1
        fi
    fi

    for cls in "${CLASSES[@]}"; do
        echo "    • $cls"
    done
}

# =============================================================================
# PIPELINE POR CLASSE
# =============================================================================

run_class() {
    local CLASS="$1"
    local E1_COLLECTION="$CLASS"
    if [[ "$COLLECTION_MODE" == "folder" ]]; then
        E1_COLLECTION="$DATASET_ROOT/$CLASS"
    fi

    local SAFE_NAME
    SAFE_NAME=$(echo "$CLASS" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    local FEAT_DIR="$FEATURES_BASE/$SAFE_NAME"
    local RES_DIR="$RESULTS_BASE/$SAFE_NAME"
    local CORR_FOR_E3="$FEAT_DIR/correspondence.npz"
    if [[ "$E2_BG_SUPPRESSION" != "none" ]]; then
        CORR_FOR_E3="$FEAT_DIR/correspondence_bg.npz"
    fi

    # Quando uma etapa é realmente recalculada, todas as etapas downstream
    # habilitadas nesta execução devem ser refeitas para evitar checkpoints
    # semanticamente obsoletos.
    local UPSTREAM_DIRTY="no"

    mkdir -p "$LOG_DIR" "$FEAT_DIR" "$RES_DIR" "$STATE_BASE/$SAFE_NAME"
    local LOG_FILE="$LOG_DIR/${SAFE_NAME}_$(date '+%Y%m%d_%H%M%S').log"
    # Cria o log e registra início
    echo "=== Pipeline: $CLASS ===" > "$LOG_FILE"
    echo "Início: $(date)" >> "$LOG_FILE"
    echo "Set: $ICOSEG_SET" >> "$LOG_FILE"
    echo "E1 protocol: $STEP1_PROTOCOL" >> "$LOG_FILE"
    echo "Features: $FEAT_DIR" >> "$LOG_FILE"
    echo "Results: $RES_DIR" >> "$LOG_FILE"
    echo "E2 orientation: $E2_ORIENTATION" >> "$LOG_FILE"
    echo "E2 degree alpha: $E2_DEGREE_ALPHA" >> "$LOG_FILE"
    echo "E2B suppression: $E2_BG_SUPPRESSION" >> "$LOG_FILE"
    echo "E2B lambda: $E2_BG_LAMBDA" >> "$LOG_FILE"
    echo "E2B gamma: $E2_BG_GAMMA" >> "$LOG_FILE"
    echo "E3 empty-FG bypass: $E3_EMPTY_FG_BYPASS" >> "$LOG_FILE"
    echo "E3 empty-FG top fraction: $E3_EMPTY_FG_TOP_FRACTION" >> "$LOG_FILE"

    bold "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    bold "  Classe: $CLASS"
    bold "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Etapa 1 ───────────────────────────────────────────────────────────────
    if step_enabled 1; then
        log "Etapa 1 — Feature Extraction (DINOv2)"
        if should_run_output_or_dirty "$UPSTREAM_DIRTY" "E1" "features.npz" valid_npz "$FEAT_DIR/features.npz"; then
            mark_running "$SAFE_NAME" "E1"
            if run_cmd "$LOG_FILE" $PYTHON "'$SCRIPT_DIR/step1_feature_extraction.py'" \
                    --collection "'$E1_COLLECTION'" \
                    --protocol "$STEP1_PROTOCOL" \
                    --output "'$FEAT_DIR'" \
                    --model "$DINOV2_MODEL" \
                    --img_size "$DINO_IMG_SIZE" \
                    --resize "$DINO_RESIZE" \
                    --feature_mode "$DINO_FEATURE_MODE" \
                    $MAX_IMAGES $VIZ_FLAG; then
                if ! finish_validated_step "$SAFE_NAME" "E1" "features.npz" valid_npz "$FEAT_DIR/features.npz"; then
                    FAILED_CLASSES+=("$CLASS:E1")
                    return 1
                fi
                UPSTREAM_DIRTY="yes"
            else
                mark_failed "$SAFE_NAME" "E1"
                err "Etapa 1 falhou — veja: $LOG_FILE"
                FAILED_CLASSES+=("$CLASS:E1")
                return 1
            fi
        else
            skip_valid_step "$SAFE_NAME" "E1"
        fi
    fi

    # ── Etapa 2 ───────────────────────────────────────────────────────────────
    if step_enabled 2; then
        log "Etapa 2 — Cross-Image Correspondence"
        if should_run_output_or_dirty "$UPSTREAM_DIRTY" "E2" "correspondence.npz" valid_npz "$FEAT_DIR/correspondence.npz"; then
            mark_running "$SAFE_NAME" "E2"
            if run_cmd "$LOG_FILE" $PYTHON "'$SCRIPT_DIR/step2_correspondence.py'" \
                    --features "'$FEAT_DIR/features.npz'" \
                    --output "'$FEAT_DIR'" \
                    --top_k "$E2_TOP_K" \
                    --temperature "$E2_TEMPERATURE" \
                    --orientation "$E2_ORIENTATION" \
                    --degree_alpha "$E2_DEGREE_ALPHA" \
                    $VIZ_FLAG; then
                if ! finish_validated_step "$SAFE_NAME" "E2" "correspondence.npz" valid_npz "$FEAT_DIR/correspondence.npz"; then
                    FAILED_CLASSES+=("$CLASS:E2")
                    return 1
                fi
                UPSTREAM_DIRTY="yes"
            else
                mark_failed "$SAFE_NAME" "E2"
                err "Etapa 2 falhou — veja: $LOG_FILE"
                FAILED_CLASSES+=("$CLASS:E2")
                return 1
            fi
        else
            skip_valid_step "$SAFE_NAME" "E2"
        fi
    fi

    # ── Etapa 2B — Background Suppression ────────────────────────────────────
    # Executada somente quando --bg-suppression != none.
    if [[ "$E2_BG_SUPPRESSION" != "none" ]] && step_enabled 2b; then
        log "Etapa 2B — Adaptive Recurring-Background Suppression"
        if should_run_output_or_dirty "$UPSTREAM_DIRTY" "E2B" "correspondence_bg.npz" valid_npz "$FEAT_DIR/correspondence_bg.npz"; then
            mark_running "$SAFE_NAME" "E2B"
            if run_cmd "$LOG_FILE" $PYTHON "'$SCRIPT_DIR/step2b_background_suppression.py'" \
                    --features "'$FEAT_DIR/features.npz'" \
                    --correspondence "'$FEAT_DIR/correspondence.npz'" \
                    --output "'$FEAT_DIR'" \
                    --mode "$E2_BG_SUPPRESSION" \
                    --lambda-bg "$E2_BG_LAMBDA" \
                    --gamma "$E2_BG_GAMMA" \
                    --border-width "$E2_BG_BORDER_WIDTH" \
                    --top-fg-fraction "$E2_BG_TOP_FG_FRACTION"; then
                if ! finish_validated_step "$SAFE_NAME" "E2B" "correspondence_bg.npz" valid_npz "$FEAT_DIR/correspondence_bg.npz"; then
                    FAILED_CLASSES+=("$CLASS:E2B")
                    return 1
                fi
                UPSTREAM_DIRTY="yes"
            else
                mark_failed "$SAFE_NAME" "E2B"
                err "Etapa 2B falhou — veja: $LOG_FILE"
                FAILED_CLASSES+=("$CLASS:E2B")
                return 1
            fi
        else
            skip_valid_step "$SAFE_NAME" "E2B"
        fi
    fi

    # ── Etapa 3 ───────────────────────────────────────────────────────────────
    if step_enabled 3; then
        log "Etapa 3 — Co-Saliency via Spectral Clustering"
        if should_run_output_or_dirty "$UPSTREAM_DIRTY" "E3" "cosaliency.npz" valid_step3_output "$FEAT_DIR/cosaliency.npz"; then
            mark_running "$SAFE_NAME" "E3"

            local -a E3_EXTRA_FLAGS=()
            if [[ "$E3_FG_DECISION_MODE" != "saliency" ]]; then
                E3_EXTRA_FLAGS+=(
                    --e2_border_weight "$E3_E2_BORDER_WEIGHT"
                    --topk_fraction "$E3_TOPK_FRACTION"
                )
            fi
            E3_EXTRA_FLAGS+=(
                --empty_fg_bypass "$E3_EMPTY_FG_BYPASS"
                --empty_fg_top_fraction "$E3_EMPTY_FG_TOP_FRACTION"
            )
            if [[ -n "$VIZ_FLAG" ]]; then
                E3_EXTRA_FLAGS+=("$VIZ_FLAG")
            fi

            if run_cmd "$LOG_FILE" "$PYTHON" "$SCRIPT_DIR/step3_cosaliency.py" \
                    --features "$FEAT_DIR/features.npz" \
                    --correspondence "$CORR_FOR_E3" \
                    --output "$FEAT_DIR" \
                    --n_clusters "$E3_N_CLUSTERS" \
                    --k_neighbors "$E3_K_NEIGHBORS" \
                    --n_spectral "$E3_N_SPECTRAL" \
                    --saliency_alpha "$E3_SALIENCY_ALPHA" \
                    --fg_decision_mode "$E3_FG_DECISION_MODE" \
                    "${E3_EXTRA_FLAGS[@]}"; then
                if ! finish_validated_step "$SAFE_NAME" "E3" "cosaliency.npz" valid_step3_output "$FEAT_DIR/cosaliency.npz"; then
                    FAILED_CLASSES+=("$CLASS:E3")
                    return 1
                fi
                UPSTREAM_DIRTY="yes"
            else
                mark_failed "$SAFE_NAME" "E3"
                err "Etapa 3 falhou — veja: $LOG_FILE"
                FAILED_CLASSES+=("$CLASS:E3")
                return 1
            fi
        else
            skip_valid_step "$SAFE_NAME" "E3"
        fi
    fi

    # ── Etapa 4 ───────────────────────────────────────────────────────────────
    if step_enabled 4; then
        log "Etapa 4 — Refinamento SAM 2"
        if should_run_output_or_dirty "$UPSTREAM_DIRTY" "E4" "refined_masks.npz / segmented_e4" valid_step4_output "$FEAT_DIR"; then
            mark_running "$SAFE_NAME" "E4"
            if run_cmd "$LOG_FILE" $PYTHON "'$SCRIPT_DIR/step4_sam_refinement.py'" \
                    --features "'$FEAT_DIR/features.npz'" \
                    --cosaliency "'$FEAT_DIR/cosaliency.npz'" \
                    --output "'$FEAT_DIR'" \
                    --model "$SAM_MODEL" \
                    --prompt_mode "$E4_PROMPT_MODE" \
                    --n_iter "$E4_N_ITER" \
                    $EXPORT_FLAG $VIZ_FLAG; then
                if ! finish_validated_step "$SAFE_NAME" "E4" "refined_masks.npz / segmented_e4" valid_step4_output "$FEAT_DIR"; then
                    FAILED_CLASSES+=("$CLASS:E4")
                    return 1
                fi
                UPSTREAM_DIRTY="yes"
            else
                mark_failed "$SAFE_NAME" "E4"
                err "Etapa 4 falhou — veja: $LOG_FILE"
                FAILED_CLASSES+=("$CLASS:E4")
                return 1
            fi
        else
            skip_valid_step "$SAFE_NAME" "E4"
        fi
    fi

    # ── Etapa 5 ───────────────────────────────────────────────────────────────
    if step_enabled 5; then
        log "Etapa 5 — Refinamento Iterativo"
        if should_run_output_or_dirty "$UPSTREAM_DIRTY" "E5" "iterative_masks.npz / segmented_e5" valid_step5_output "$FEAT_DIR"; then
            mark_running "$SAFE_NAME" "E5"
            if run_cmd "$LOG_FILE" $PYTHON "'$SCRIPT_DIR/step5_iterative.py'" \
                    --features "'$FEAT_DIR/features.npz'" \
                    --refined "'$FEAT_DIR/refined_masks.npz'" \
                    --output "'$FEAT_DIR'" \
                    --model "$SAM_MODEL" \
                    --prompt_mode "$E4_PROMPT_MODE" \
                    --max_iter "$E5_MAX_ITER" \
                    --converge_thresh "$E5_CONVERGE_THRESH" \
                    --n_fg_pts "$E5_N_FG_PTS" \
                    --n_bg_pts "$E5_N_BG_PTS" \
                    --no_mask_history \
                    $EXPORT_FLAG $VIZ_FLAG; then
                if ! finish_validated_step "$SAFE_NAME" "E5" "iterative_masks.npz / segmented_e5" valid_step5_output "$FEAT_DIR"; then
                    FAILED_CLASSES+=("$CLASS:E5")
                    return 1
                fi
                UPSTREAM_DIRTY="yes"
            else
                mark_failed "$SAFE_NAME" "E5"
                err "Etapa 5 falhou — veja: $LOG_FILE"
                FAILED_CLASSES+=("$CLASS:E5")
                return 1
            fi
        else
            skip_valid_step "$SAFE_NAME" "E5"
        fi
    fi

    # ── Avaliação ─────────────────────────────────────────────────────────────
    if step_enabled eval && [[ "$NO_EVAL" == "no" ]]; then
        log "Avaliação vs. Ground Truth"

        local SEG_DIR=""
        local SEG_LABEL=""
        select_segmented_dir "$FEAT_DIR"

        if [[ -z "$SEG_DIR" ]]; then
            mark_failed "$SAFE_NAME" "EVAL"
            warn "Nenhuma pasta de máscaras exportadas encontrada em $FEAT_DIR"
            warn "Esperado: segmented_e5/, segmented_e4/ ou segmented/ com arquivos *_mask.png"
            warn "Use --export nas etapas 4/5 para gerar máscaras avaliáveis"
        elif [[ ! -f "$GT_INDEX" ]]; then
            mark_failed "$SAFE_NAME" "EVAL"
            warn "GT index ausente: $GT_INDEX"
            warn "Gere com: $PYTHON '$SCRIPT_DIR/evaluate.py' --scan_gt '$GT_ROOT' --images_root '$DATASET_ROOT'"
        else
            log "Avaliando máscaras de $SEG_LABEL: $SEG_DIR"
            mark_running "$SAFE_NAME" "EVAL"

            if run_cmd "$LOG_FILE" $PYTHON "'$SCRIPT_DIR/evaluate.py'" \
                    --masks_dir "'$SEG_DIR'" \
                    --gt_index "'$GT_INDEX'" \
                    --output "'$RES_DIR'" \
                    --collection "'$CLASS'" \
                    $VIZ_FLAG; then
                if [[ "$DRY_RUN" == "yes" ]]; then
                    ok "[dry-run] metrics.json [$SEG_LABEL]"
                elif valid_json "$RES_DIR/metrics.json"; then
                    mark_done "$SAFE_NAME" "EVAL"
                    ok "metrics.json [$SEG_LABEL]"
                else
                    mark_failed "$SAFE_NAME" "EVAL"
                    warn "Avaliação executou, mas metrics.json não foi gerado ou está inválido — veja: $LOG_FILE"
                    FAILED_CLASSES+=("$CLASS:EVAL")
                fi
            else
                mark_failed "$SAFE_NAME" "EVAL"
                warn "Avaliação falhou (não fatal) — veja: $LOG_FILE"
                FAILED_CLASSES+=("$CLASS:EVAL")
            fi
        fi

        # ── Avaliação comparativa E4 vs E5 (suplementar, não afeta o checkpoint
        # principal) — permite quantificar quanto a Etapa 5 (refinamento
        # iterativo) ganha sobre a Etapa 4 (SAM de uma passada só), avaliando
        # as duas separadamente em vez de só a "melhor disponível".
        if [[ -f "$GT_INDEX" ]]; then
            for stage_pair in "e4:segmented_e4" "e5:segmented_e5"; do
                stage_tag="${stage_pair%%:*}"
                stage_dir_name="${stage_pair##*:}"
                stage_seg_dir="$FEAT_DIR/$stage_dir_name"
                stage_res_dir="$RES_DIR/$stage_tag"

                if ! mask_dir_has_masks "$stage_seg_dir"; then
                    continue
                fi
                if [[ "$FORCE" != "yes" && -f "$stage_res_dir/metrics.json" ]] && valid_json "$stage_res_dir/metrics.json"; then
                    continue
                fi

                mkdir -p "$stage_res_dir"
                log "Avaliação comparativa [$stage_tag]: $stage_seg_dir"
                if run_cmd "$LOG_FILE" $PYTHON "'$SCRIPT_DIR/evaluate.py'" \
                        --masks_dir "'$stage_seg_dir'" \
                        --gt_index "'$GT_INDEX'" \
                        --output "'$stage_res_dir'" \
                        --collection "'$CLASS ($stage_tag)'"; then
                    if [[ "$DRY_RUN" != "yes" ]] && valid_json "$stage_res_dir/metrics.json"; then
                        ok "metrics.json [$stage_tag]"
                    else
                        warn "Avaliação comparativa [$stage_tag] não gerou metrics.json válido — veja: $LOG_FILE"
                    fi
                else
                    warn "Avaliação comparativa [$stage_tag] falhou (não fatal) — veja: $LOG_FILE"
                fi
            done
        fi
    fi

    log "Log completo: $LOG_FILE"
    return 0
}

# =============================================================================
# SETUP GT INDEX
# =============================================================================

setup_gt_index() {
    if [[ "$NO_EVAL" == "yes" ]]; then return; fi
    if ! step_enabled eval; then return; fi

    if [[ ! -f "$GT_INDEX" ]]; then
        if [[ -d "$GT_ROOT" ]]; then
            log "Gerando $GT_INDEX..."
            mkdir -p "$(dirname "$GT_INDEX")"
            if ! (
                cd "$(dirname "$GT_INDEX")" &&                 $PYTHON "$SCRIPT_DIR/evaluate.py"                     --scan_gt "$GT_ROOT"                     --images_root "$DATASET_ROOT"
            ); then
                warn "Falha ao gerar $GT_INDEX; avaliação poderá ser pulada"
            fi
        else
            warn "Ground truth não encontrado: $GT_ROOT"
            warn "Avaliação será pulada automaticamente"
        fi
    else
        ok "$GT_INDEX já existe"
    fi
}

# =============================================================================
# SUMÁRIO FINAL
# =============================================================================

print_summary() {
    bold "\n╔══════════════════════════════════════════════════════════════╗"
    bold "  SUMÁRIO DO PIPELINE"
    bold "╚══════════════════════════════════════════════════════════════╝"
    echo
    echo "  Set:          $ICOSEG_SET"
    echo "  E1 protocol:  $STEP1_PROTOCOL"
    echo "  Índice:       $ICOSEG_INDEX"
    echo "  Affected only:$AFFECTED_ONLY"
    echo "  Manual classes:$MANUAL_CLASSES"
    echo "  Classes:      ${#CLASSES[@]}"
    echo "  Etapas:       $STEPS"
    echo "  E3 empty-FG:  $E3_EMPTY_FG_BYPASS | top_fraction=$E3_EMPTY_FG_TOP_FRACTION | trigger=exact-zero"
    echo "  OK:           $N_OK"
    echo "  Puladas:      $N_SKIP (etapas com saída válida)"
    echo "  Falhas:       $N_FAIL"
    echo "  Checkpoints:  $STATE_BASE"
    [[ -n "$LOCK_FILE" ]] && echo "  Lock:         $LOCK_FILE"

    if [[ ${#FAILED_CLASSES[@]} -gt 0 ]]; then
        echo
        err "Classes/etapas com falha ou avaliação incompleta:"
        for f in "${FAILED_CLASSES[@]}"; do
            echo "    ✗ $f"
        done
    fi

    if [[ -d "$STATE_BASE" ]]; then
        local running failed
        running=$(find "$STATE_BASE" -name "*.running" 2>/dev/null | wc -l | tr -d ' ')
        failed=$(find "$STATE_BASE" -name "*.failed" 2>/dev/null | wc -l | tr -d ' ')
        echo
        echo "  Estado atual: $running running | $failed failed"
        if [[ "$failed" != "0" ]]; then
            echo "  Para listar falhas: find '$STATE_BASE' -name '*.failed'"
        fi
    fi

    # Tabela de métricas se houver resultados
    local found_metrics="no"
    for json_file in "$RESULTS_BASE"/*/metrics.json; do
        [[ -f "$json_file" ]] || continue
        if [[ "$found_metrics" == "no" ]]; then
            echo
            bold "  Métricas (accuracy / IoU / F1):"
            echo "  ─────────────────────────────────────────────────────────"
            printf "  %-30s  %8s  %8s  %8s\n" "CLASSE" "ACCURACY" "IoU" "F1"
            echo "  ─────────────────────────────────────────────────────────"
            found_metrics="yes"
        fi
        local cls_name
        cls_name=$(basename "$(dirname "$json_file")")
        $PYTHON - "$json_file" "$cls_name" << 'PYEOF' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
agg = d.get('aggregate', {})
acc = agg.get('accuracy', {}).get('mean', 0)
iou = agg.get('iou',      {}).get('mean', 0)
f1  = agg.get('f1',       {}).get('mean', 0)
cls = d.get('collection', sys.argv[2])
print(f'  {cls:<30}  {acc:>8.4f}  {iou:>8.4f}  {f1:>8.4f}')
PYEOF
    done

    if [[ "$found_metrics" == "yes" ]]; then
        echo "  ─────────────────────────────────────────────────────────"
    fi

    # Agrega métricas de todas as classes num resumo consolidado
    if [[ -f "$SCRIPT_DIR/aggregate_results.py" ]]; then
        echo
        $PYTHON "$SCRIPT_DIR/aggregate_results.py" \
            --results "$RESULTS_BASE" \
            --sort iou \
            --csv "$RESULTS_BASE/resumo_agregado.csv" \
            --json "$RESULTS_BASE/resumo_agregado.json"
    fi

    echo
    log "Concluído às $(date '+%H:%M:%S')"
}

# =============================================================================
# MAIN
# =============================================================================

main() {
    parse_args "$@"
    configure_experiment_dirs

    mkdir -p "$FEATURES_BASE" "$RESULTS_BASE" "$LOG_DIR" "$STATE_BASE"
    acquire_lock

    bold "\n╔══════════════════════════════════════════════════════════════╗"
    bold "  COSEGMENTATION PIPELINE"
    bold "╚══════════════════════════════════════════════════════════════╝"
    echo
    echo "  Set:          $ICOSEG_SET"
    echo "  E1 protocol:  $STEP1_PROTOCOL"
    echo "  Índice:       $ICOSEG_INDEX"
    echo "  Affected only:$AFFECTED_ONLY"
    echo "  Etapas:       $STEPS"
    echo "  Raiz projeto: $PROJECT_ROOT"
    echo "  Dataset root: $DATASET_ROOT"
    echo "  GT root:      $GT_ROOT"
    echo "  Features:     $FEATURES_BASE"
    echo "  Results:      $RESULTS_BASE"
    echo "  Logs:         $LOG_DIR"
    echo "  Checkpoints:  $STATE_BASE"
    echo "  Visualizar:   $VISUALIZE"
    echo "  Export:       $EXPORT"
    echo "  Force:        $FORCE"
    echo "  Dry-run:      $DRY_RUN"
    echo "  Lock:         $LOCK"
    echo "  DINOv2:       $DINOV2_MODEL | img_size=$DINO_IMG_SIZE | resize=$DINO_RESIZE | feature_mode=$DINO_FEATURE_MODE"
    echo "  E2:           top_k=$E2_TOP_K | temperature=$E2_TEMPERATURE | orientation=$E2_ORIENTATION | degree_alpha=$E2_DEGREE_ALPHA"
    echo "  E2B:          mode=$E2_BG_SUPPRESSION | lambda=$E2_BG_LAMBDA | gamma=$E2_BG_GAMMA | border_width=$E2_BG_BORDER_WIDTH | top_fg=$E2_BG_TOP_FG_FRACTION"
    echo "  E3:           clusters=$E3_N_CLUSTERS | k_neighbors=$E3_K_NEIGHBORS | spectral=$E3_N_SPECTRAL | alpha=$E3_SALIENCY_ALPHA"
    echo "  E3 empty-FG:  mode=$E3_EMPTY_FG_BYPASS | top_fraction=$E3_EMPTY_FG_TOP_FRACTION | trigger=exact-zero"

    if [[ "$DRY_RUN" == "yes" ]]; then
        warn "MODO DRY-RUN: nenhum comando será executado"
    fi

    # Avisa sobre scripts ausentes mas não aborta (checa relativo a SCRIPT_DIR,
    # não ao cwd, já que os scripts são invocados via caminho absoluto).
    for script in step1_feature_extraction.py step2_correspondence.py \
                  step2b_background_suppression.py step3_cosaliency.py step4_sam_refinement.py \
                  step5_iterative.py evaluate.py; do
        if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
            warn "Script não encontrado: $SCRIPT_DIR/$script"
        fi
    done

    resolve_classes

    # As validações de cardinalidade 38/15 só se aplicam quando o próprio
    # pipeline seleciona automaticamente o conjunto. Quando o usuário passa
    # uma ou mais classes manualmente na CLI, qualquer subconjunto válido do
    # índice icoseg38 deve poder ser executado.
    if [[ "$MANUAL_CLASSES" != "yes" && "$ICOSEG_SET" == "icoseg38" && "$AFFECTED_ONLY" != "yes" && ${#CLASSES[@]} -ne 38 ]]; then
        err "O protocolo icoseg38 deveria conter 38 grupos, mas foram encontrados ${#CLASSES[@]}."
        err "Revise $ICOSEG_INDEX antes de executar o experimento."
        exit 1
    fi

    if [[ "$MANUAL_CLASSES" != "yes" && "$ICOSEG_SET" == "icoseg38" && "$AFFECTED_ONLY" == "yes" && ${#CLASSES[@]} -ne 15 ]]; then
        err "--affected-only deveria selecionar 15 grupos, mas selecionou ${#CLASSES[@]}."
        exit 1
    fi

    if [[ ${#CLASSES[@]} -eq 0 ]]; then
        err "Nenhuma classe encontrada"
        exit 1
    fi

    log "Classes a processar (${#CLASSES[@]}): ${CLASSES[*]}"

    setup_gt_index

    for CLASS in "${CLASSES[@]}"; do
        if run_class "$CLASS"; then
            N_OK=$((N_OK + 1))
        else
            N_FAIL=$((N_FAIL + 1))
        fi
    done

    print_summary
}

main "$@"