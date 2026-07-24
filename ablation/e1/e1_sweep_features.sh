#!/usr/bin/env bash
set -Eeuo pipefail
# Força separador decimal como ponto em todas as chamadas awk/printf,
# independente do locale do sistema (necessário em pt_BR).
export LC_NUMERIC=C

# =============================================================================
# Sweep da Etapa 1 — 14 classes selecionadas OU todas as classes do iCoseg
#
# Diferenças em relação ao script original (all_classes):
#   - A lista de classes é hardcoded em SELECTED_CLASSES[] abaixo por padrão.
#     O JSON de índice ainda é necessário para o extrator resolver os paths,
#     mas NÃO é lido para enumerar classes A MENOS QUE -A/--all-classes seja
#     usado (ver abaixo).
#   - A flag -A (ou --all-classes) lê TODAS as classes de icoseg_index.json
#     dinamicamente, ignorando a lista hardcoded — equivalente ao --set full
#     do run_pipeline.sh, mas para o sweep de ablação da Etapa 1.
#   - A flag -k (regex) ainda pode ser usada para filtrar dentro da lista
#     (hardcoded ou completa, dependendo de -A).
#   - A flag -n (max classes) ainda pode ser usada para testes rápidos.
#   - CORRIGIDO: argumento de resize passado ao extrator como --resize
#     (não --resize_mode, que não existe no step1_feature_extraction.py).
#
# Uso básico (14 classes curadas):
#   chmod +x run_step1_feature_sweep.sh
#   ./run_step1_feature_sweep.sh \
#     -j ./icoseg_index.json \
#     -s ./step1_feature_extraction.py \
#     -a ./analyze_step1_features.py \
#     -o ./features_sweep_selected \
#     -m 10
#
# Uso para TODAS as classes do iCoseg:
#   ./run_step1_feature_sweep.sh \
#     -j ./icoseg_index.json \
#     -s ./step1_feature_extraction.py \
#     -a ./analyze_step1_features.py \
#     -o ./features_sweep_all \
#     -g ./gt_index.json \
#     -A
#
# Variáveis de ambiente úteis:
#   PYTHON_BIN=python3
#   MODELS_STR="dinov2_vits14 dinov2_vitb14 dinov2_vitl14"
#   IMG_SIZES_STR="448 518 672"
#   FEATURE_MODES_STR="raw l2 pca128_l2 pca256_l2"
#   RESIZE_MODES_STR="keep_ar square"   (passado ao extrator como --resize)
#   LAYER_MODES_STR="last"              (só se o extrator suportar --layer_mode)
#   RUN_ANALYZER=0                      # pula analyze_step1_features
#   GT_INDEX=/caminho/gt_index.json     # passado ao analisador como --gt_index
#   NO_PADDING_MASK=0                   # 1 = passa --no_padding_mask ao analisador
#                                        # (desativa exclusão de patches de padding
#                                        #  nas métricas; default é mascarar)
#   MIN_CLASSES_FOR_AGGREGATE=all | número
#   RESUME=1                            # reaproveita .npz válidos já concluídos
#   VALIDATE_EXISTING=1                 # valida .npz antes de pular extração
#   REUSE_ANALYZER_REPORTS=1            # reaproveita relatórios do analisador já existentes
# =============================================================================

# ----------------------------------------------------------------------------
# 14 classes selecionadas — edite aqui se quiser ajustar a lista.
# Ignorada inteiramente quando -A/--all-classes é usado.
# ----------------------------------------------------------------------------
# 14 classes curadas do iCoseg (1 pasta = 1 objeto/álbum, exceto os grupos
# explicitamente unificados em CLASS_GROUP_OVERRIDES no step1_feature_extraction.py).
# Removidas "Stonehenge 2" e "Kite Panda": não existem no índice e casavam
# por acidente via fuzzy match com "Stonehenge" e "Kite", respectivamente.
SELECTED_CLASSES=(
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

# ----------------------------------------------------------------------------
# Defaults dos parâmetros CLI
# ----------------------------------------------------------------------------
INDEX_JSON="./icoseg_index.json"
STEP1_SCRIPT="./step1_feature_extraction.py"
ANALYZER_SCRIPT="./analyze_step1_features.py"
OUT_ROOT="./features_sweep_selected"
MAX_IMAGES=""
MAX_CLASSES=""
CLASS_REGEX=""
FORCE=0
VIZ=0
VIZ_MAX=6
ALL_CLASSES=0
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ANALYZER="${RUN_ANALYZER:-1}"
GT_INDEX="${GT_INDEX:-}"
NO_PADDING_MASK="${NO_PADDING_MASK:-0}"
RESUME="${RESUME:-1}"
VALIDATE_EXISTING="${VALIDATE_EXISTING:-1}"
REUSE_ANALYZER_REPORTS="${REUSE_ANALYZER_REPORTS:-1}"

# ----------------------------------------------------------------------------
# Ajuda
# ----------------------------------------------------------------------------
usage() {
  cat <<USAGE
Uso:
  $0 -j ./icoseg_index.json [opções]

Opções:
  -j  JSON de índice. Necessário para o extrator resolver os paths.
      Default: ./icoseg_index.json
  -s  Script da etapa 1. Default: ./step1_feature_extraction.py
  -a  Script analisador. Default: ./analyze_step1_features.py
  -o  Pasta raiz de saída. Default: ./features_sweep_selected
  -g  Caminho para gt_index.json. Passado ao analisador como --gt_index.
      Também pode ser definido via variável de ambiente GT_INDEX.
  -m  Máximo de imagens por classe/execução. Ex.: -m 10
  -n  Máximo de classes da lista (útil para teste rápido). Ex.: -n 3
  -k  Regex para filtrar dentro da lista (hardcoded ou completa, ver -A).
      Ex.: -k 'Panda|Bear'
  -A  Todas as classes do iCoseg: lê dinamicamente as chaves de
      icoseg_index.json (via --scan, se o índice ainda não existir) em vez
      de usar a lista hardcoded SELECTED_CLASSES[]. Equivalente ao
      --set full do run_pipeline.sh, mas para este sweep.
  -f  Força reprocessamento mesmo se o .npz já existir
  -v  Gera visualizações do extrator (se suportado)
  -h  Ajuda

Variáveis de ambiente úteis:
  PYTHON_BIN=python3
  MODELS_STR="dinov2_vits14 dinov2_vitb14 dinov2_vitl14"
  IMG_SIZES_STR="448 518 672"
  FEATURE_MODES_STR="raw l2 pca128_l2 pca256_l2"
  RESIZE_MODES_STR="keep_ar square"
  LAYER_MODES_STR="last"
  RUN_ANALYZER=0
  GT_INDEX=/caminho/gt_index.json
  NO_PADDING_MASK=0   # 1 desativa a exclusão de padding no analisador
  MIN_CLASSES_FOR_AGGREGATE=all | número
  RESUME=1
  VALIDATE_EXISTING=1
  REUSE_ANALYZER_REPORTS=1

Exemplos:
  # Teste rápido: só Panda e Bear, 10 imagens, ViT-S/14, L2, com GT
  MODELS_STR="dinov2_vits14" FEATURE_MODES_STR="l2" IMG_SIZES_STR="518" \\
  $0 -j ./icoseg_index.json -g ./gt_index.json -m 10 -k 'Panda|Bear'

  # As 14 classes curadas, pulando o analisador
  RUN_ANALYZER=0 $0 -j ./icoseg_index.json -m 10

  # TODAS as classes do iCoseg, com GT
  $0 -j ./icoseg_index.json -g ./gt_index.json -o ./features_sweep_all -A

  # Grid completo com GT, todas as classes
  MODELS_STR="dinov2_vits14 dinov2_vitb14" IMG_SIZES_STR="448 518 672" \\
  FEATURE_MODES_STR="l2 pca128_l2" RESIZE_MODES_STR="keep_ar square" \\
  $0 -j ./icoseg_index.json -g ./gt_index.json -o ./features_sweep_all -A
USAGE
}

while getopts ":j:s:a:o:g:m:n:k:fvAh" opt; do
  case "$opt" in
    j) INDEX_JSON="$OPTARG" ;;
    s) STEP1_SCRIPT="$OPTARG" ;;
    a) ANALYZER_SCRIPT="$OPTARG" ;;
    o) OUT_ROOT="$OPTARG" ;;
    g) GT_INDEX="$OPTARG" ;;
    m) MAX_IMAGES="$OPTARG" ;;
    n) MAX_CLASSES="$OPTARG" ;;
    k) CLASS_REGEX="$OPTARG" ;;
    f) FORCE=1 ;;
    v) VIZ=1 ;;
    A) ALL_CLASSES=1 ;;
    h) usage; exit 0 ;;
    :) echo "[ERRO] Opção -$OPTARG requer argumento." >&2; usage; exit 2 ;;
    \?) echo "[ERRO] Opção inválida: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

# Suporte à variante longa --all-classes (getopts só trata opções curtas).
for arg in "$@"; do
  if [[ "$arg" == "--all-classes" ]]; then
    ALL_CLASSES=1
  fi
done

if [[ ! -f "$STEP1_SCRIPT" ]]; then
  echo "[ERRO] Script da etapa 1 não encontrado: $STEP1_SCRIPT" >&2
  exit 1
fi

# Se -A foi pedido e o índice ainda não existe, gera via --scan usando a
# pasta apontada por INDEX_JSON como raiz de imagens não é possível (o
# índice não existe ainda), então avisamos e paramos — o usuário precisa
# ter rodado "step1_feature_extraction.py --scan <pasta_icoseg>" ao menos
# uma vez, ou apontar -j para um icoseg_index.json já existente.
if [[ "$ALL_CLASSES" -eq 1 && ! -f "$INDEX_JSON" ]]; then
  echo "[ERRO] -A/--all-classes requer um icoseg_index.json já existente." >&2
  echo "       Gere-o primeiro com:" >&2
  echo "         $PYTHON_BIN '$STEP1_SCRIPT' --scan <pasta_icoseg/images>" >&2
  echo "       e então aponte -j para o icoseg_index.json gerado (raiz do projeto)." >&2
  exit 1
fi

if [[ ! -f "$INDEX_JSON" ]]; then
  echo "[ERRO] JSON de índice não encontrado: $INDEX_JSON" >&2
  exit 1
fi

# Caminhos absolutos.
INDEX_JSON="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$INDEX_JSON")"
STEP1_SCRIPT="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$STEP1_SCRIPT")"
ANALYZER_SCRIPT="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$ANALYZER_SCRIPT")"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUT_ROOT")"

# Resolve GT_INDEX para path absoluto, se fornecido.
if [[ -n "$GT_INDEX" ]]; then
  if [[ ! -f "$GT_INDEX" ]]; then
    echo "[ERRO] gt_index.json não encontrado: $GT_INDEX" >&2
    exit 1
  fi
  GT_INDEX="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$GT_INDEX")"
  echo "[GT] Usando ground truth: $GT_INDEX"
fi

INDEX_DIR="$(dirname "$INDEX_JSON")"
INDEX_BASE="$(basename "$INDEX_JSON")"
TEMP_INDEX_LINK=""

cleanup() {
  if [[ -n "$TEMP_INDEX_LINK" && -L "$TEMP_INDEX_LINK" ]]; then
    rm -f "$TEMP_INDEX_LINK"
  fi
}
trap cleanup EXIT

# O step1_feature_extraction.py procura especificamente icoseg_index.json.
if [[ "$INDEX_BASE" != "icoseg_index.json" ]]; then
  if [[ ! -e "$INDEX_DIR/icoseg_index.json" ]]; then
    ln -s "$INDEX_JSON" "$INDEX_DIR/icoseg_index.json"
    TEMP_INDEX_LINK="$INDEX_DIR/icoseg_index.json"
    echo "[Info] Criei link temporário: $TEMP_INDEX_LINK -> $INDEX_JSON"
  else
    echo "[Aviso] Já existe $INDEX_DIR/icoseg_index.json; o extrator usará esse arquivo." >&2
  fi
fi

# ----------------------------------------------------------------------------
# Define a lista base de classes: hardcoded (default) ou todas do índice (-A).
# ----------------------------------------------------------------------------
if [[ "$ALL_CLASSES" -eq 1 ]]; then
  mapfile -t BASE_CLASSES < <("$PYTHON_BIN" -c "
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    idx = json.load(f)
for k in sorted(idx.keys()):
    print(k)
" "$INDEX_JSON")

  if [[ "${#BASE_CLASSES[@]}" -eq 0 ]]; then
    echo "[ERRO] -A/--all-classes ativo, mas $INDEX_JSON não contém nenhuma classe." >&2
    exit 1
  fi
  echo "[Info] -A/--all-classes ativo: ${#BASE_CLASSES[@]} classes lidas de $INDEX_JSON" \
       "(lista hardcoded SELECTED_CLASSES[] ignorada)."
else
  BASE_CLASSES=("${SELECTED_CLASSES[@]}")
fi

# ----------------------------------------------------------------------------
# Aplica filtro regex e limite -n sobre a lista base (hardcoded ou completa).
# ----------------------------------------------------------------------------
CLASSES=()
for cls in "${BASE_CLASSES[@]}"; do
  if [[ -n "$CLASS_REGEX" ]]; then
    if ! echo "$cls" | grep -qiE "$CLASS_REGEX"; then
      continue
    fi
  fi
  CLASSES+=("$cls")
done

if [[ -n "$MAX_CLASSES" ]]; then
  CLASSES=("${CLASSES[@]:0:$MAX_CLASSES}")
fi

if [[ "${#CLASSES[@]}" -eq 0 ]]; then
  echo "[ERRO] Nenhuma classe restou após aplicar os filtros." >&2
  if [[ "$ALL_CLASSES" -eq 1 ]]; then
    echo "       Classes lidas do índice: ${BASE_CLASSES[*]}" >&2
  else
    echo "       Classes na lista: ${SELECTED_CLASSES[*]}" >&2
  fi
  exit 1
fi

# ----------------------------------------------------------------------------
# Detecta argumentos suportados pelo extrator.
#
# ATENÇÃO: o step1_feature_extraction.py usa --resize (não --resize_mode).
# O script original tinha um bug aqui: detectava --resize_mode e também
# passava --resize_mode ao extrator, que nunca reconhecia o argumento.
# Corrigido: detectamos --resize e passamos --resize ao extrator.
# ----------------------------------------------------------------------------
STEP1_HELP="$($PYTHON_BIN "$STEP1_SCRIPT" --help 2>&1 || true)"

supports_arg() {
  local arg="$1"
  grep -q -- "$arg" <<< "$STEP1_HELP"
}

SUPPORTS_FEATURE_MODE=0
SUPPORTS_RESIZE=0       # argumento --resize (não --resize_mode)
SUPPORTS_LAYER_MODE=0
SUPPORTS_VISUALIZE=0

supports_arg "--feature_mode" && SUPPORTS_FEATURE_MODE=1
supports_arg "--resize"        && SUPPORTS_RESIZE=1
supports_arg "--layer_mode"    && SUPPORTS_LAYER_MODE=1
supports_arg "--visualize"     && SUPPORTS_VISUALIZE=1

# ----------------------------------------------------------------------------
# Grade experimental.
# ----------------------------------------------------------------------------
read -r -a MODELS        <<< "${MODELS_STR:-dinov2_vits14 dinov2_vitb14 dinov2_vitl14}"
read -r -a IMG_SIZES     <<< "${IMG_SIZES_STR:-448 518 672}"
read -r -a FEATURE_MODES <<< "${FEATURE_MODES_STR:-raw l2 pca128_l2 pca256_l2}"
read -r -a RESIZE_MODES  <<< "${RESIZE_MODES_STR:-keep_ar}"
read -r -a LAYER_MODES   <<< "${LAYER_MODES_STR:-last}"

if [[ "$SUPPORTS_FEATURE_MODE" -eq 0 ]]; then
  echo "[Info] Extrator não suporta --feature_mode. Usando feature_mode=raw no nome dos arquivos."
  FEATURE_MODES=("raw")
fi

if [[ "$SUPPORTS_RESIZE" -eq 0 ]]; then
  if [[ "${#RESIZE_MODES[@]}" -gt 1 || "${RESIZE_MODES[0]}" != "keep_ar" ]]; then
    echo "[Info] Extrator não suporta --resize. Ignorando RESIZE_MODES_STR." >&2
  fi
  RESIZE_MODES=("keep_ar")
fi

if [[ "$SUPPORTS_LAYER_MODE" -eq 0 ]]; then
  if [[ "${#LAYER_MODES[@]}" -gt 1 || "${LAYER_MODES[0]}" != "last" ]]; then
    echo "[Info] Extrator não suporta --layer_mode. Ignorando LAYER_MODES_STR." >&2
  fi
  LAYER_MODES=("last")
fi

if [[ "$VIZ" -eq 1 && "$SUPPORTS_VISUALIZE" -eq 0 ]]; then
  echo "[Info] Extrator não suporta --visualize. Ignorando -v." >&2
  VIZ=0
fi

ANALYZER_TARGET_MODE="${ANALYZER_TARGET_MODE:-features}"
ANALYZER_ARGS_TEMPLATE="${ANALYZER_ARGS_TEMPLATE:---features {target}}"

# Injeta --gt_index no template se GT_INDEX estiver definido e o template ainda não contiver.
if [[ -n "$GT_INDEX" ]] && ! grep -q "gt_index" <<< "$ANALYZER_ARGS_TEMPLATE"; then
  ANALYZER_ARGS_TEMPLATE="$ANALYZER_ARGS_TEMPLATE --gt_index $GT_INDEX"
fi

# Injeta --no_padding_mask no template se NO_PADDING_MASK=1 e o template
# ainda não contiver (default é deixar o analisador mascarar o padding).
if [[ "$NO_PADDING_MASK" == "1" ]] && ! grep -q "no_padding_mask" <<< "$ANALYZER_ARGS_TEMPLATE"; then
  ANALYZER_ARGS_TEMPLATE="$ANALYZER_ARGS_TEMPLATE --no_padding_mask"
  echo "[Info] NO_PADDING_MASK=1: analisador rodará com --no_padding_mask (padding NÃO será excluído)."
fi

# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
sanitize() {
  local s="$1"
  s="${s//\\//_}"
  s="${s//\//_}"
  s="${s// /_}"
  s="$(printf '%s' "$s" | tr -cd '[:alnum:]_.-')"
  printf '%s' "$s"
}

format_duration() {
  local total="${1:-0}"
  if [[ "$total" -lt 0 ]]; then total=0; fi
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

now_human() {
  date "+%H:%M:%S do dia %d/%m/%Y"
}

eta_human() {
  local remaining_sec="${1:-0}"
  if [[ "$remaining_sec" -lt 0 ]]; then remaining_sec=0; fi
  local eta_epoch=$(( $(date +%s) + remaining_sec ))
  date -d "@$eta_epoch" "+%H:%M do dia %d/%m/%Y"
}

append_timing_row() {
  local run_index="$1"
  local total_runs="$2"
  local class_name="$3"
  local model="$4"
  local img_size="$5"
  local resize_mode="$6"
  local layer_mode="$7"
  local feature_mode="$8"
  local status="$9"
  local started_at="${10}"
  local finished_at="${11}"
  local elapsed_sec="${12}"
  local avg_sec_per_run="${13}"
  local remaining_runs="${14}"
  local estimated_remaining_sec="${15}"
  local estimated_finish_at="${16}"
  local feature_file="${17}"
  local extract_log="${18}"
  local analyzer_report="${19}"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$run_index" "$total_runs" "$class_name" "$model" "$img_size" "$resize_mode" "$layer_mode" "$feature_mode" \
    "$status" "$started_at" "$finished_at" "$elapsed_sec" "$(format_duration "$elapsed_sec")" "$avg_sec_per_run" \
    "$remaining_runs" "$estimated_remaining_sec" "$(format_duration "$estimated_remaining_sec")" "$estimated_finish_at" \
    "$feature_file" "$extract_log" "$analyzer_report" >> "$TIMING_TSV"
}

# ----------------------------------------------------------------------------
# Estrutura de diretórios de saída
# ----------------------------------------------------------------------------
RUNS_ROOT="$OUT_ROOT/runs"
NPZ_ROOT="$OUT_ROOT/npz"
REPORT_ROOT="$OUT_ROOT/reports"
LOG_ROOT="$OUT_ROOT/logs"
STATUS_ROOT="$OUT_ROOT/status"
mkdir -p "$RUNS_ROOT" "$NPZ_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$STATUS_ROOT"

METRICS_TSV="$OUT_ROOT/metrics.tsv"
AGG_TSV="$OUT_ROOT/aggregate_by_config.tsv"
TOP10_AGG_TSV="$OUT_ROOT/top10_aggregate.tsv"
TOP10_AGG_MD="$OUT_ROOT/top10_aggregate.md"
BEST_BY_CLASS_TSV="$OUT_ROOT/best_by_class.tsv"
TIMING_TSV="$OUT_ROOT/timing.tsv"

printf "class_name\tfeature_file\tmodel\timg_size\tresize_mode\tlayer_mode\tfeature_mode\tN\tP\tD\tgrid_h\tgrid_w\tcls_mean\tcls_std\tcls_min\tcls_max\tcross_nn_mean\tcross_nn_std\tpatch_norm_mean\tpatch_norm_std\tpatch_norm_cv\tpca3_var\tproxy_score\tanalyzer_report\n" > "$METRICS_TSV"
printf "run_index\ttotal_runs\tclass_name\tmodel\timg_size\tresize_mode\tlayer_mode\tfeature_mode\tstatus\tstarted_at\tfinished_at\telapsed_sec\telapsed_hms\tavg_sec_per_run\tremaining_runs\testimated_remaining_sec\testimated_remaining_hms\testimated_finish_at\tfeature_file\textract_log\tanalyzer_report\n" > "$TIMING_TSV"

# ----------------------------------------------------------------------------
# Validação de .npz
# ----------------------------------------------------------------------------
npz_is_valid() {
  local npz="$1"
  [[ -f "$npz" ]] || return 1

  if [[ "$VALIDATE_EXISTING" -eq 0 ]]; then
    return 0
  fi

  "$PYTHON_BIN" - "$npz" <<'PY_NPZ_VALIDATE' >/dev/null 2>&1
import sys
import numpy as np
from pathlib import Path
p = Path(sys.argv[1])
try:
    data = np.load(p, allow_pickle=True)
    required = ["patch_features", "cls_features", "patch_grid", "image_paths", "image_sizes"]
    for key in required:
        if key not in data:
            raise ValueError(f"missing {key}")
    pf = data["patch_features"]
    cf = data["cls_features"]
    if pf.ndim != 3:
        raise ValueError("patch_features must have shape (N,P,D)")
    if cf.ndim != 2:
        raise ValueError("cls_features must have shape (N,D)")
    if pf.shape[0] != cf.shape[0] or pf.shape[2] != cf.shape[1]:
        raise ValueError("inconsistent patch/cls shapes")
except Exception as e:
    print(e, file=sys.stderr)
    sys.exit(1)
PY_NPZ_VALIDATE
}

atomic_stage_npz() {
  local src="$1"
  local dst="$2"
  local tmp="${dst}.tmp.$$"

  rm -f "$tmp"

  if ! ln "$src" "$tmp" 2>/dev/null; then
    cp "$src" "$tmp"
  fi

  if ! npz_is_valid "$tmp"; then
    rm -f "$tmp"
    echo "[ERRO] .npz temporário inválido: $tmp" >&2
    return 1
  fi

  mv -f "$tmp" "$dst"
}

write_done_marker() {
  local marker="$1"
  local feature_file="$2"
  local status="$3"
  mkdir -p "$(dirname "$marker")"
  {
    echo "status=$status"
    echo "finished_at=$(now_human)"
    echo "feature_file=$feature_file"
  } > "${marker}.tmp.$$"
  mv -f "${marker}.tmp.$$" "$marker"
}

# ----------------------------------------------------------------------------
# Cálculo de métricas proxy
# ----------------------------------------------------------------------------
compute_metrics() {
  local class_name="$1"
  local npz_file="$2"
  local model="$3"
  local img_size="$4"
  local resize_mode="$5"
  local layer_mode="$6"
  local feature_mode="$7"
  local analyzer_report="$8"

  "$PYTHON_BIN" - "$class_name" "$npz_file" "$model" "$img_size" "$resize_mode" "$layer_mode" "$feature_mode" "$analyzer_report" <<'PY'
import sys
from pathlib import Path
import numpy as np

class_name, npz_file, model, img_size, resize_mode, layer_mode, feature_mode, analyzer_report = sys.argv[1:]

def l2_normalize(x, eps=1e-12):
    x = x.astype(np.float32, copy=False)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

def sample_rows(x, max_rows=5000, seed=123):
    n = x.shape[0]
    if n <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_rows, replace=False)
    return x[idx]

try:
    data = np.load(npz_file, allow_pickle=True)
    pf = data["patch_features"].astype(np.float32, copy=False)
    cf = data["cls_features"].astype(np.float32, copy=False)
    grid = data["patch_grid"]

    N, P, D = pf.shape
    grid_h, grid_w = int(grid[0]), int(grid[1])

    if N > 1:
        cn = l2_normalize(cf)
        sim = cn @ cn.T
        mask = ~np.eye(N, dtype=bool)
        off = sim[mask]
        cls_mean = float(np.mean(off))
        cls_std  = float(np.std(off))
        cls_min  = float(np.min(off))
        cls_max  = float(np.max(off))
    else:
        cls_mean = cls_std = cls_min = cls_max = 0.0

    norms = np.linalg.norm(pf, axis=-1)
    patch_norm_mean = float(np.mean(norms))
    patch_norm_std  = float(np.std(norms))
    patch_norm_cv   = float(patch_norm_std / (patch_norm_mean + 1e-12))

    rng = np.random.default_rng(123)
    per_img = min(P, 128)
    sampled = []
    for i in range(N):
        if P <= per_img:
            idx = np.arange(P)
        else:
            idx = rng.choice(P, size=per_img, replace=False)
        sampled.append(l2_normalize(pf[i, idx, :]))

    pair_scores = []
    if N > 1:
        for i in range(N):
            ai = sampled[i]
            for j in range(i + 1, N):
                bj = sampled[j]
                s = ai @ bj.T
                score = 0.5 * (float(np.mean(np.max(s, axis=1))) + float(np.mean(np.max(s, axis=0))))
                pair_scores.append(score)

    if pair_scores:
        cross_nn_mean = float(np.mean(pair_scores))
        cross_nn_std  = float(np.std(pair_scores))
    else:
        cross_nn_mean = cross_nn_std = 0.0

    X = sample_rows(pf.reshape(-1, D), max_rows=5000, seed=456)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        _, S, _ = np.linalg.svd(X, full_matrices=False)
        var = S**2
        pca3_var = float(var[:3].sum() / (var.sum() + 1e-12))
    except Exception:
        pca3_var = 0.0

    proxy_score = (
        0.55 * cross_nn_mean
        + 0.35 * cls_mean
        - 0.10 * cross_nn_std
        - 0.05 * patch_norm_cv
    )

    values = [
        class_name, str(Path(npz_file)), model, str(img_size), resize_mode, layer_mode, feature_mode,
        str(N), str(P), str(D), str(grid_h), str(grid_w),
        f"{cls_mean:.6f}", f"{cls_std:.6f}", f"{cls_min:.6f}", f"{cls_max:.6f}",
        f"{cross_nn_mean:.6f}", f"{cross_nn_std:.6f}",
        f"{patch_norm_mean:.6f}", f"{patch_norm_std:.6f}", f"{patch_norm_cv:.6f}",
        f"{pca3_var:.6f}", f"{proxy_score:.6f}", str(Path(analyzer_report))
    ]
    print("\t".join(values))
except Exception as e:
    print(f"[ERRO_METRICS] {npz_file}: {e}", file=sys.stderr)
    raise
PY
}

# ----------------------------------------------------------------------------
# Execução do analisador externo
# ----------------------------------------------------------------------------
run_analyzer() {
  local target="$1"
  local report_file="$2"

  if [[ "$REUSE_ANALYZER_REPORTS" -eq 1 && "$FORCE" -eq 0 && -s "$report_file" ]]; then
    echo "[Analyzer] Reaproveitando relatório existente: $report_file"
    return 0
  fi

  if [[ "$RUN_ANALYZER" -eq 0 ]]; then
    echo "[Analyzer] Desativado por RUN_ANALYZER=0." > "$report_file"
    return 0
  fi

  if [[ ! -f "$ANALYZER_SCRIPT" ]]; then
    echo "[Analyzer] Script não encontrado: $ANALYZER_SCRIPT" > "$report_file"
    echo "[Analyzer] Pulei analyze_step1_features para este caso." >> "$report_file"
    return 0
  fi

  local args_string="${ANALYZER_ARGS_TEMPLATE//\{target\}/$target}"

  # Por padrão, analyze_step1_features.py grava feature_quality.json/.md em
  # <pasta_do_.npz>/feature_quality — a MESMA pasta para toda config da
  # mesma classe (o nome do .npz muda por STEM, mas seu diretório pai não).
  # Sem um --output explícito por config, rodar o grid completo (múltiplos
  # model/img_size/feature_mode) faz cada execução sobrescrever o relatório
  # estruturado (JSON/MD/plots) da execução anterior da mesma classe — só o
  # último config testado sobrevive em disco, mesmo que best_by_class.tsv
  # aponte outro como vencedor. Corrigido: se o template não especificar
  # --output, deriva um diretório própria por STEM a partir do próprio .npz
  # (ex.: .../NPZ_ROOT/<classe>/<STEM>_quality/).
  if ! grep -q -- "--output" <<< "$args_string"; then
    local per_run_output="${target%.npz}_quality"
    args_string="$args_string --output $per_run_output"
  fi

  # IMPORTANTE (resiliência a queda de energia / interrupção):
  # report_file é o que REUSE_ANALYZER_REPORTS usa para decidir se esta
  # config já foi analisada. Se escrevêssemos diretamente nele durante a
  # execução do Python, uma interrupção no meio da análise deixaria um
  # arquivo não-vazio em disco — no próximo resume, o REUSE_ANALYZER_REPORTS
  # interpretaria isso como "já analisado" e pularia a config
  # silenciosamente, mesmo com o relatório de qualidade incompleto ou
  # ausente. Por isso todo o conteúdo é escrito num arquivo temporário e só
  # promovido (mv atômico) para report_file depois que o processo Python
  # termina — com sucesso ou com falha registrada, nunca no meio do caminho.
  local tmp_report="${report_file}.tmp.$$"
  echo "[Analyzer] $PYTHON_BIN $ANALYZER_SCRIPT $args_string" > "$tmp_report"

  set +e
  # shellcheck disable=SC2086
  "$PYTHON_BIN" "$ANALYZER_SCRIPT" $args_string >> "$tmp_report" 2>&1
  local status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    echo "" >> "$tmp_report"
    echo "[AVISO] O analisador falhou com ANALYZER_ARGS_TEMPLATE='$ANALYZER_ARGS_TEMPLATE'." >> "$tmp_report"
    echo "[AVISO] Tentando fallback posicional: $PYTHON_BIN $ANALYZER_SCRIPT $target" >> "$tmp_report"
    set +e
    "$PYTHON_BIN" "$ANALYZER_SCRIPT" "$target" >> "$tmp_report" 2>&1
    status=$?
    set -e
  fi

  mv -f "$tmp_report" "$report_file"

  if [[ "$status" -ne 0 ]]; then
    echo "[AVISO] analyze_step1_features falhou para $target. Veja: $report_file" >&2
  fi

  return 0
}

# ----------------------------------------------------------------------------
# Agregação final
# ----------------------------------------------------------------------------
make_aggregate() {
  local total_classes="$1"
  local min_classes="$2"

  "$PYTHON_BIN" - "$METRICS_TSV" "$AGG_TSV" "$TOP10_AGG_TSV" "$TOP10_AGG_MD" "$BEST_BY_CLASS_TSV" "$total_classes" "$min_classes" <<'PY'
import csv
import math
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

metrics_tsv, agg_tsv, top10_tsv, top10_md, best_by_class_tsv = map(Path, sys.argv[1:6])
total_classes = int(sys.argv[6])
min_classes_arg = sys.argv[7]
min_classes = total_classes if min_classes_arg == "all" else int(min_classes_arg)

with metrics_tsv.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

rows = [r for r in rows if r.get("proxy_score") not in (None, "")]

# Melhor configuração por classe.
best_by_class = {}
for r in rows:
    c = r["class_name"]
    if c not in best_by_class or float(r["proxy_score"]) > float(best_by_class[c]["proxy_score"]):
        best_by_class[c] = r

best_fields = [
    "class_name", "model", "img_size", "resize_mode", "layer_mode", "feature_mode",
    "N", "P", "D", "cls_mean", "cross_nn_mean", "patch_norm_cv", "pca3_var", "proxy_score", "feature_file"
]
with best_by_class_tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, delimiter="\t", fieldnames=best_fields)
    w.writeheader()
    for c in sorted(best_by_class):
        r = best_by_class[c]
        w.writerow({k: r.get(k, "") for k in best_fields})

# Agrega por configuração.
groups = defaultdict(list)
for r in rows:
    key = (r["model"], r["img_size"], r["resize_mode"], r["layer_mode"], r["feature_mode"])
    groups[key].append(r)

agg_rows = []
for key, gr in groups.items():
    classes = sorted({r["class_name"] for r in gr})
    proxies    = [float(r["proxy_score"]) for r in gr]
    cls_means  = [float(r["cls_mean"]) for r in gr]
    cross_means= [float(r["cross_nn_mean"]) for r in gr]
    norm_cvs   = [float(r["patch_norm_cv"]) for r in gr]
    pca3       = [float(r["pca3_var"]) for r in gr]
    Ns = [float(r["N"]) for r in gr]
    Ps = [float(r["P"]) for r in gr]
    Ds = [float(r["D"]) for r in gr]

    def mean(xs):
        return sum(xs) / len(xs) if xs else math.nan
    def stdev(xs):
        return stats.pstdev(xs) if len(xs) > 1 else 0.0

    agg_rows.append({
        "model": key[0],
        "img_size": key[1],
        "resize_mode": key[2],
        "layer_mode": key[3],
        "feature_mode": key[4],
        "classes_done": len(classes),
        "classes_total": total_classes,
        "coverage": len(classes) / total_classes if total_classes else 0.0,
        "mean_proxy_score": mean(proxies),
        "std_proxy_score": stdev(proxies),
        "min_proxy_score": min(proxies),
        "max_proxy_score": max(proxies),
        "mean_cls_mean": mean(cls_means),
        "mean_cross_nn_mean": mean(cross_means),
        "mean_patch_norm_cv": mean(norm_cvs),
        "mean_pca3_var": mean(pca3),
        "mean_N": mean(Ns),
        "mean_P": mean(Ps),
        "mean_D": mean(Ds),
    })

agg_rows_all = sorted(agg_rows, key=lambda r: (r["classes_done"], r["mean_proxy_score"]), reverse=True)
qualified = [r for r in agg_rows if int(r["classes_done"]) >= min_classes]
if not qualified:
    qualified = agg_rows
    effective_min = 1
else:
    effective_min = min_classes
qualified.sort(key=lambda r: float(r["mean_proxy_score"]), reverse=True)
top10 = qualified[:10]

agg_fields = [
    "model", "img_size", "resize_mode", "layer_mode", "feature_mode",
    "classes_done", "classes_total", "coverage",
    "mean_proxy_score", "std_proxy_score", "min_proxy_score", "max_proxy_score",
    "mean_cls_mean", "mean_cross_nn_mean", "mean_patch_norm_cv", "mean_pca3_var",
    "mean_N", "mean_P", "mean_D",
]

with agg_tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, delimiter="\t", fieldnames=agg_fields)
    w.writeheader()
    for r in agg_rows_all:
        out = dict(r)
        for k, v in list(out.items()):
            if isinstance(v, float):
                out[k] = f"{v:.6f}"
        w.writerow(out)

with top10_tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, delimiter="\t", fieldnames=agg_fields)
    w.writeheader()
    for r in top10:
        out = dict(r)
        for k, v in list(out.items()):
            if isinstance(v, float):
                out[k] = f"{v:.6f}"
        w.writerow(out)

with top10_md.open("w", encoding="utf-8") as f:
    f.write("# Top 10 agregado — configurações da Etapa 1\n\n")
    f.write("> Ranking pela média de `proxy_score` entre classes. Use como triagem, não como métrica final.\n\n")
    f.write(f"Classes avaliadas: **{total_classes}**. Mínimo para ranking: **{effective_min}**.\n\n")
    f.write("| # | model | img_size | resize | layer | feature | classes | coverage | mean_proxy | std_proxy | mean_cls | mean_cross_nn | mean_norm_cv | mean_pca3_var | mean_N | mean_P | mean_D |\n")
    f.write("|---:|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(top10, 1):
        f.write(
            f"| {i} | {r['model']} | {r['img_size']} | {r['resize_mode']} | {r['layer_mode']} | {r['feature_mode']} "
            f"| {r['classes_done']}/{r['classes_total']} | {r['coverage']:.2%} "
            f"| {r['mean_proxy_score']:.4f} | {r['std_proxy_score']:.4f} "
            f"| {r['mean_cls_mean']:.4f} | {r['mean_cross_nn_mean']:.4f} "
            f"| {r['mean_patch_norm_cv']:.4f} | {r['mean_pca3_var']:.4f} "
            f"| {r['mean_N']:.1f} | {r['mean_P']:.1f} | {r['mean_D']:.1f} |\n"
        )

print(f"[Aggregate] {agg_tsv}")
print(f"[Top10 agregado] {top10_tsv}")
print(f"[Top10 agregado] {top10_md}")
print(f"[Melhor por classe] {best_by_class_tsv}")
PY
}

# ----------------------------------------------------------------------------
# Loop principal
# ----------------------------------------------------------------------------
TOTAL_CLASSES="${#CLASSES[@]}"
MIN_CLASSES_FOR_AGGREGATE="${MIN_CLASSES_FOR_AGGREGATE:-all}"
TOTAL_CONFIGS=$(( ${#MODELS[@]} * ${#IMG_SIZES[@]} * ${#RESIZE_MODES[@]} * ${#LAYER_MODES[@]} * ${#FEATURE_MODES[@]} ))
TOTAL_RUNS=$(( TOTAL_CLASSES * TOTAL_CONFIGS ))
COUNT=0
SWEEP_START_TS=$(date +%s)
SWEEP_START_HUMAN=$(now_human)

cat <<INFO
============================================================
Sweep Etapa 1 — $([ "$ALL_CLASSES" -eq 1 ] && echo "TODAS as classes do iCoseg" || echo "classes selecionadas")
JSON de índice         : $INDEX_JSON
Pasta do índice        : $INDEX_DIR
Ground truth (GT)      : ${GT_INDEX:-não fornecido}
Modo de classes        : $([ "$ALL_CLASSES" -eq 1 ] && echo "TODAS (-A)" || echo "lista hardcoded")
Classes a processar    : $TOTAL_CLASSES
  $(printf '  %s\n' "${CLASSES[@]}")
Script etapa 1         : $STEP1_SCRIPT
Analyzer               : $ANALYZER_SCRIPT
Padding mask analyzer  : $([ "$NO_PADDING_MASK" == "1" ] && echo "DESATIVADO (--no_padding_mask)" || echo "ativo (default)")
Saída                  : $OUT_ROOT
Total configs/classe   : $TOTAL_CONFIGS
Total execuções        : $TOTAL_RUNS
Início                 : $SWEEP_START_HUMAN
Resume habilitado      : $RESUME
Valida .npz existentes : $VALIDATE_EXISTING
Reusa relatórios análise: $REUSE_ANALYZER_REPORTS
Max imagens/classe     : ${MAX_IMAGES:-todas}
Filtro regex           : ${CLASS_REGEX:-nenhum}
Suporta --feature_mode : $SUPPORTS_FEATURE_MODE
Suporta --resize       : $SUPPORTS_RESIZE
Suporta --layer_mode   : $SUPPORTS_LAYER_MODE
Models                 : ${MODELS[*]}
Img sizes              : ${IMG_SIZES[*]}
Feature modes          : ${FEATURE_MODES[*]}
Resize modes            : ${RESIZE_MODES[*]}
Layer modes             : ${LAYER_MODES[*]}
============================================================
INFO

for class_name in "${CLASSES[@]}"; do
  safe_class="$(sanitize "$class_name")"
  class_runs_dir="$RUNS_ROOT/$safe_class"
  class_npz_dir="$NPZ_ROOT/$safe_class"
  class_report_dir="$REPORT_ROOT/$safe_class"
  class_log_dir="$LOG_ROOT/$safe_class"
  class_status_dir="$STATUS_ROOT/$safe_class"
  mkdir -p "$class_runs_dir" "$class_npz_dir" "$class_report_dir" "$class_log_dir" "$class_status_dir"

  echo ""
  echo "############################################################"
  echo "Classe: $class_name"
  echo "############################################################"

  for model in "${MODELS[@]}"; do
    for img_size in "${IMG_SIZES[@]}"; do
      for resize_mode in "${RESIZE_MODES[@]}"; do
        for layer_mode in "${LAYER_MODES[@]}"; do
          for feature_mode in "${FEATURE_MODES[@]}"; do
            COUNT=$((COUNT + 1))

            STEM="${safe_class}__model-${model}__img-${img_size}__resize-${resize_mode}__layer-${layer_mode}__feat-${feature_mode}"
            RUN_DIR="$class_runs_dir/$STEM"
            FLAT_NPZ="$class_npz_dir/${STEM}.npz"
            EXTRACT_LOG="$class_log_dir/${STEM}.extract.log"
            ANALYZER_REPORT="$class_report_dir/${STEM}.analysis.txt"
            DONE_MARKER="$class_status_dir/${STEM}.done"

            echo ""
            echo "[$COUNT/$TOTAL_RUNS] $class_name :: model=$model img=$img_size resize=$resize_mode feature=$feature_mode"

            RUN_START_TS=$(date +%s)
            RUN_START_HUMAN=$(now_human)
            RUN_STATUS="ok"

            EXISTING_VALID=0
            if [[ "$RESUME" -eq 1 && "$FORCE" -eq 0 && -f "$FLAT_NPZ" ]]; then
              if npz_is_valid "$FLAT_NPZ"; then
                EXISTING_VALID=1
              else
                echo "[Aviso] .npz existente inválido/corrompido. Removendo e reprocessando: $FLAT_NPZ" >&2
                rm -f "$FLAT_NPZ" "$DONE_MARKER"
              fi
            fi

            if [[ "$EXISTING_VALID" -eq 1 ]]; then
              if [[ -f "$DONE_MARKER" ]]; then
                echo "[Resume] Concluído anteriormente. Pulando extração: $FLAT_NPZ"
                RUN_STATUS="resumed_done"
              else
                echo "[Resume] .npz válido encontrado, sem marcador .done. Reaproveitando: $FLAT_NPZ"
                RUN_STATUS="resumed_npz"
              fi
            else
              rm -rf "$RUN_DIR" "${FLAT_NPZ}.tmp."*
              mkdir -p "$RUN_DIR"

              cmd=("$PYTHON_BIN" "$STEP1_SCRIPT"
                --collection "$class_name"
                --output "$RUN_DIR"
                --model "$model"
                --img_size "$img_size")

              if [[ -n "$MAX_IMAGES" ]]; then
                cmd+=(--max_images "$MAX_IMAGES")
              fi

              if [[ "$VIZ" -eq 1 ]]; then
                cmd+=(--visualize --viz_max "$VIZ_MAX")
              fi

              if [[ "$SUPPORTS_FEATURE_MODE" -eq 1 ]]; then
                cmd+=(--feature_mode "$feature_mode")
              fi

              # CORRIGIDO: --resize (não --resize_mode) conforme step1_feature_extraction.py
              if [[ "$SUPPORTS_RESIZE" -eq 1 ]]; then
                cmd+=(--resize "$resize_mode")
              fi

              if [[ "$SUPPORTS_LAYER_MODE" -eq 1 ]]; then
                cmd+=(--layer_mode "$layer_mode")
              fi

              echo "[Run] ${cmd[*]}"
              set +e
              ( cd "$INDEX_DIR" && "${cmd[@]}" ) > "$EXTRACT_LOG" 2>&1
              status=$?
              set -e

              if [[ "$status" -ne 0 ]]; then
                echo "[ERRO] Extração falhou. Veja: $EXTRACT_LOG" >&2
                RUN_STATUS="failed_extract"
                RUN_END_TS=$(date +%s)
                RUN_END_HUMAN=$(now_human)
                RUN_ELAPSED=$((RUN_END_TS - RUN_START_TS))
                SWEEP_ELAPSED=$((RUN_END_TS - SWEEP_START_TS))
                AVG_SEC=$(awk -v elapsed="$SWEEP_ELAPSED" -v count="$COUNT" 'BEGIN { if (count > 0) printf "%.2f", elapsed / count; else printf "0.00" }')
                REMAINING_RUNS=$((TOTAL_RUNS - COUNT))
                EST_REMAINING=$(awk -v avg="$AVG_SEC" -v rem="$REMAINING_RUNS" 'BEGIN { printf "%d", avg * rem }')
                ETA_HUMAN=$(eta_human "$EST_REMAINING")
                append_timing_row "$COUNT" "$TOTAL_RUNS" "$class_name" "$model" "$img_size" "$resize_mode" "$layer_mode" "$feature_mode" "$RUN_STATUS" "$RUN_START_HUMAN" "$RUN_END_HUMAN" "$RUN_ELAPSED" "$AVG_SEC" "$REMAINING_RUNS" "$EST_REMAINING" "$ETA_HUMAN" "$FLAT_NPZ" "$EXTRACT_LOG" "$ANALYZER_REPORT"
                echo "[Tempo] execução: $(format_duration "$RUN_ELAPSED") | média: ${AVG_SEC}s/run | restante: $(format_duration "$EST_REMAINING") | término previsto: $ETA_HUMAN"
                continue
              fi

              if [[ ! -f "$RUN_DIR/features.npz" ]]; then
                echo "[ERRO] Extração terminou mas features.npz não encontrado em $RUN_DIR" >&2
                RUN_STATUS="failed_missing_npz"
                RUN_END_TS=$(date +%s)
                RUN_END_HUMAN=$(now_human)
                RUN_ELAPSED=$((RUN_END_TS - RUN_START_TS))
                SWEEP_ELAPSED=$((RUN_END_TS - SWEEP_START_TS))
                AVG_SEC=$(awk -v elapsed="$SWEEP_ELAPSED" -v count="$COUNT" 'BEGIN { if (count > 0) printf "%.2f", elapsed / count; else printf "0.00" }')
                REMAINING_RUNS=$((TOTAL_RUNS - COUNT))
                EST_REMAINING=$(awk -v avg="$AVG_SEC" -v rem="$REMAINING_RUNS" 'BEGIN { printf "%d", avg * rem }')
                ETA_HUMAN=$(eta_human "$EST_REMAINING")
                append_timing_row "$COUNT" "$TOTAL_RUNS" "$class_name" "$model" "$img_size" "$resize_mode" "$layer_mode" "$feature_mode" "$RUN_STATUS" "$RUN_START_HUMAN" "$RUN_END_HUMAN" "$RUN_ELAPSED" "$AVG_SEC" "$REMAINING_RUNS" "$EST_REMAINING" "$ETA_HUMAN" "$FLAT_NPZ" "$EXTRACT_LOG" "$ANALYZER_REPORT"
                echo "[Tempo] execução: $(format_duration "$RUN_ELAPSED") | média: ${AVG_SEC}s/run | restante: $(format_duration "$EST_REMAINING") | término previsto: $ETA_HUMAN"
                continue
              fi

              atomic_stage_npz "$RUN_DIR/features.npz" "$FLAT_NPZ"
              echo "[OK] $FLAT_NPZ"
            fi

            if [[ "$ANALYZER_TARGET_MODE" == "dir" ]]; then
              ANALYZER_TARGET="$RUN_DIR"
              if [[ ! -f "$RUN_DIR/features.npz" ]]; then
                ANALYZER_TARGET="$FLAT_NPZ"
              fi
            else
              ANALYZER_TARGET="$FLAT_NPZ"
            fi

            run_analyzer "$ANALYZER_TARGET" "$ANALYZER_REPORT"
            compute_metrics "$class_name" "$FLAT_NPZ" "$model" "$img_size" "$resize_mode" "$layer_mode" "$feature_mode" "$ANALYZER_REPORT" >> "$METRICS_TSV"
            write_done_marker "$DONE_MARKER" "$FLAT_NPZ" "$RUN_STATUS"

            RUN_END_TS=$(date +%s)
            RUN_END_HUMAN=$(now_human)
            RUN_ELAPSED=$((RUN_END_TS - RUN_START_TS))
            SWEEP_ELAPSED=$((RUN_END_TS - SWEEP_START_TS))
            AVG_SEC=$(awk -v elapsed="$SWEEP_ELAPSED" -v count="$COUNT" 'BEGIN { if (count > 0) printf "%.2f", elapsed / count; else printf "0.00" }')
            REMAINING_RUNS=$((TOTAL_RUNS - COUNT))
            EST_REMAINING=$(awk -v avg="$AVG_SEC" -v rem="$REMAINING_RUNS" 'BEGIN { printf "%d", avg * rem }')
            ETA_HUMAN=$(eta_human "$EST_REMAINING")

            append_timing_row "$COUNT" "$TOTAL_RUNS" "$class_name" "$model" "$img_size" "$resize_mode" "$layer_mode" "$feature_mode" "$RUN_STATUS" "$RUN_START_HUMAN" "$RUN_END_HUMAN" "$RUN_ELAPSED" "$AVG_SEC" "$REMAINING_RUNS" "$EST_REMAINING" "$ETA_HUMAN" "$FLAT_NPZ" "$EXTRACT_LOG" "$ANALYZER_REPORT"

            PCT=$(awk -v count="$COUNT" -v total="$TOTAL_RUNS" 'BEGIN { if (total > 0) printf "%.2f", 100 * count / total; else printf "0.00" }')
            echo "[Tempo] execução: $(format_duration "$RUN_ELAPSED") | progresso: $COUNT/$TOTAL_RUNS (${PCT}%)"
            echo "[Estimativa] média: ${AVG_SEC}s/run | restante: $(format_duration "$EST_REMAINING") | término previsto: $ETA_HUMAN"
          done
        done
      done
    done
  done

  # Atualiza agregado parcial ao fim de cada classe.
  make_aggregate "$TOTAL_CLASSES" "$MIN_CLASSES_FOR_AGGREGATE"
done

make_aggregate "$TOTAL_CLASSES" "$MIN_CLASSES_FOR_AGGREGATE"

cat <<DONE

============================================================
Concluído.
Features nomeadas        : $NPZ_ROOT
Relatórios por execução  : $REPORT_ROOT
Logs de extração         : $LOG_ROOT
Marcadores de conclusão  : $STATUS_ROOT
Métricas completas       : $METRICS_TSV
Agregado por config      : $AGG_TSV
Top 10 agregado TSV      : $TOP10_AGG_TSV
Top 10 agregado Markdown : $TOP10_AGG_MD
Melhor config por classe : $BEST_BY_CLASS_TSV
Tempos por execução      : $TIMING_TSV
============================================================
DONE
