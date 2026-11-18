#!/usr/bin/env bash
set -Eeuo pipefail
export LC_NUMERIC=C

# =============================================================================
# Sweep da Etapa 4 — Refinamento SAM 2
#
# Itera sobre pares (features.npz E1, cosaliency.npz E3) já existentes e
# roda step4_sam_refinement.py + analyze_step4_sam_refinement.py para cada
# combinação de (par_e1_e3, prompt_mode, n_iter, sam_model).
#
# Estrutura esperada:
#   <step1_npz_root>/<classe>/<stem_e1>.npz
#   <step3_npz_root>/<classe>/<stem_e3>.npz  (stem_e3 contém stem_e1 como prefixo)
#
# Uso básico:
#   chmod +x run_step4_sam_sweep.sh
#   dos2unix run_step4_sam_sweep.sh
#   ./run_step4_sam_sweep.sh \
#     -1 ./features_sweep_selected/npz \
#     -3 ./cosaliency_sweep/npz \
#     -s ./step4_sam_refinement.py \
#     -a ./analyze_step4_sam_refinement.py \
#     -o ./sam_sweep \
#     -g ./gt_index.json
#
# Variáveis de ambiente úteis:
#   PYTHON_BIN=python3
#   PROMPT_MODES_STR="box_only box_points box_points_mask"
#   N_ITER_STR="1 2"
#   SAM_MODELS_STR="facebook/sam2.1-hiera-small"
#   STEP1_CONFIG_FILTER=""   # substring para filtrar .npz da E1
#   STEP3_CONFIG_FILTER=""   # substring para filtrar .npz da E3
#   RUN_ANALYZER=1
#   RESUME=1
#   VALIDATE_EXISTING=1
#   REUSE_ANALYZER_REPORTS=1
#   MIN_CLASSES_FOR_AGGREGATE=all
# =============================================================================

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

# Sanidade: este sweep sempre rodou sobre este conjunto fixo de 14 classes.
# Se a lista for editada, o aviso evita rodar silenciosamente sobre um
# subconjunto diferente do que foi usado nos sweeps anteriores (E1/E3).
EXPECTED_N_SELECTED_CLASSES=14
if [[ "${#SELECTED_CLASSES[@]}" -ne "$EXPECTED_N_SELECTED_CLASSES" ]]; then
  echo "[AVISO] SELECTED_CLASSES tem ${#SELECTED_CLASSES[@]} classes (esperado: $EXPECTED_N_SELECTED_CLASSES). Confirme se isso é intencional." >&2
fi

# Defaults
STEP1_NPZ_ROOT="./features_sweep_selected/npz"
STEP3_NPZ_ROOT="./cosaliency_sweep/npz"
STEP4_SCRIPT="./step4_sam_refinement.py"
ANALYZER_SCRIPT="./analyze_step4_sam_refinement.py"
OUT_ROOT="./sam_sweep"
GT_INDEX="${GT_INDEX:-}"
MAX_CLASSES=""
CLASS_REGEX=""
STEP1_CONFIG_FILTER="${STEP1_CONFIG_FILTER:-}"
STEP3_CONFIG_FILTER="${STEP3_CONFIG_FILTER:-}"
FORCE=0
VIZ=0
VIZ_MAX=8
EXPORT=0
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ANALYZER="${RUN_ANALYZER:-1}"
RESUME="${RESUME:-1}"
VALIDATE_EXISTING="${VALIDATE_EXISTING:-1}"
REUSE_ANALYZER_REPORTS="${REUSE_ANALYZER_REPORTS:-1}"

usage() {
  cat <<USAGE
Uso:
  $0 -1 <npz_e1> -3 <npz_e3> [opções]

Opções:
  -1  Pasta raiz dos .npz da Etapa 1. Default: ./features_sweep_selected/npz
  -3  Pasta raiz dos .npz da Etapa 3. Default: ./cosaliency_sweep/npz
  -s  Script da etapa 4. Default: ./step4_sam_refinement.py
  -a  Script analisador. Default: ./analyze_step4_sam_refinement.py
  -o  Pasta raiz de saída. Default: ./sam_sweep
  -g  Caminho para gt_index.json.
  -n  Máximo de classes (teste rápido). Ex.: -n 2
  -k  Regex para filtrar classes. Ex.: -k 'Panda|Bear'
  -f  Força reprocessamento
  -v  Gera visualizações (--visualize)
  -e  Exporta imagens segmentadas individuais (--export)
  -h  Ajuda

Variáveis de ambiente úteis:
  PYTHON_BIN=python3
  PROMPT_MODES_STR="box_only box_points box_points_mask"
  N_ITER_STR="1 2"
  SAM_MODELS_STR="facebook/sam2.1-hiera-small"
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2"
  STEP3_CONFIG_FILTER="nc-5__kn-10__fgt-0.5"
  GT_INDEX=/caminho/gt_index.json
  RUN_ANALYZER=1
  RESUME=1

Exemplos:
  # Teste rápido: 1 classe, 1 par, só box_points, 1 iteração
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2" \\
  STEP3_CONFIG_FILTER="nc-5__kn-10__fgt-0.5" \\
  PROMPT_MODES_STR="box_points" N_ITER_STR="1" \\
  $0 -1 ./features_sweep_selected/npz -3 ./cosaliency_sweep/npz \\
     -g ./gt_index.json -n 1

  # Sweep completo de modos de prompt
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2" \\
  STEP3_CONFIG_FILTER="nc-6__kn-10__fgt-0.5" \\
  PROMPT_MODES_STR="box_only box_points box_points_mask" N_ITER_STR="1 2" \\
  $0 -1 ./features_sweep_selected/npz -3 ./cosaliency_sweep/npz -g ./gt_index.json
USAGE
}

while getopts ":1:3:s:a:o:g:n:k:fveh" opt; do
  case "$opt" in
    1) STEP1_NPZ_ROOT="$OPTARG" ;;
    3) STEP3_NPZ_ROOT="$OPTARG" ;;
    s) STEP4_SCRIPT="$OPTARG" ;;
    a) ANALYZER_SCRIPT="$OPTARG" ;;
    o) OUT_ROOT="$OPTARG" ;;
    g) GT_INDEX="$OPTARG" ;;
    n) MAX_CLASSES="$OPTARG" ;;
    k) CLASS_REGEX="$OPTARG" ;;
    f) FORCE=1 ;;
    v) VIZ=1 ;;
    e) EXPORT=1 ;;
    h) usage; exit 0 ;;
    :) echo "[ERRO] Opção -$OPTARG requer argumento." >&2; usage; exit 2 ;;
    \?) echo "[ERRO] Opção inválida: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

for d in "$STEP1_NPZ_ROOT" "$STEP3_NPZ_ROOT"; do
  [[ -d "$d" ]] || { echo "[ERRO] Pasta não encontrada: $d" >&2; exit 1; }
done
[[ -f "$STEP4_SCRIPT" ]] || { echo "[ERRO] Script E4 não encontrado: $STEP4_SCRIPT" >&2; exit 1; }

abs() { "$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }
STEP1_NPZ_ROOT="$(abs "$STEP1_NPZ_ROOT")"
STEP3_NPZ_ROOT="$(abs "$STEP3_NPZ_ROOT")"
STEP4_SCRIPT="$(abs "$STEP4_SCRIPT")"
ANALYZER_SCRIPT="$(abs "$ANALYZER_SCRIPT")"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(abs "$OUT_ROOT")"

if [[ -n "$GT_INDEX" ]]; then
  [[ -f "$GT_INDEX" ]] || { echo "[ERRO] gt_index.json não encontrado: $GT_INDEX" >&2; exit 1; }
  GT_INDEX="$(abs "$GT_INDEX")"
  echo "[GT] Usando ground truth: $GT_INDEX"
fi

# Filtra classes
CLASSES=()
for cls in "${SELECTED_CLASSES[@]}"; do
  [[ -n "$CLASS_REGEX" ]] && ! echo "$cls" | grep -qiE "$CLASS_REGEX" && continue
  CLASSES+=("$cls")
done
[[ -n "$MAX_CLASSES" ]] && CLASSES=("${CLASSES[@]:0:$MAX_CLASSES}")
[[ "${#CLASSES[@]}" -eq 0 ]] && { echo "[ERRO] Nenhuma classe após filtros." >&2; exit 1; }

# Grade experimental E4
read -r -a PROMPT_MODES <<< "${PROMPT_MODES_STR:-box_only box_points box_points_mask}"
read -r -a N_ITERS      <<< "${N_ITER_STR:-1 2}"
read -r -a SAM_MODELS   <<< "${SAM_MODELS_STR:-facebook/sam2.1-hiera-small}"

# Utilitários
sanitize() {
  local s="$1"
  s="${s//\\//_}"; s="${s//\//_}"; s="${s// /_}"
  printf '%s' "$s" | tr -cd '[:alnum:]_.-'
}
sanitize_model() {
  # facebook/sam2.1-hiera-small → sam2.1-hiera-small
  printf '%s' "${1##*/}" | tr -cd '[:alnum:]._-'
}
format_duration() {
  local t="${1:-0}"; [[ "$t" -lt 0 ]] && t=0
  printf "%02d:%02d:%02d" $((t/3600)) $(((t%3600)/60)) $((t%60))
}
now_human()  { date "+%H:%M:%S do dia %d/%m/%Y"; }
eta_human()  {
  local rem="${1:-0}"; [[ "$rem" -lt 0 ]] && rem=0
  date -d "@$(( $(date +%s) + rem ))" "+%H:%M do dia %d/%m/%Y"
}

npz_is_valid_e4() {
  local npz="$1"
  [[ -f "$npz" ]] || return 1
  [[ "$VALIDATE_EXISTING" -eq 0 ]] && return 0
  "$PYTHON_BIN" - "$npz" <<'PY' >/dev/null 2>&1
import sys, numpy as np
data = np.load(sys.argv[1], allow_pickle=True)
for k in ["refined_masks","iou_scores_sam","fg_coherence","prompt_mode","image_paths","image_sizes"]:
    if k not in data: raise ValueError(f"missing {k}")
PY
}

write_done_marker() {
  local marker="$1" npz="$2" status="$3"
  mkdir -p "$(dirname "$marker")"
  { echo "status=$status"; echo "finished_at=$(now_human)"; echo "npz=$npz"; } \
    > "${marker}.tmp.$$"
  mv -f "${marker}.tmp.$$" "$marker"
}

# Estrutura de saída
NPZ_ROOT="$OUT_ROOT/npz"
REPORT_ROOT="$OUT_ROOT/reports"
LOG_ROOT="$OUT_ROOT/logs"
STATUS_ROOT="$OUT_ROOT/status"
mkdir -p "$NPZ_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$STATUS_ROOT"

METRICS_TSV="$OUT_ROOT/metrics.tsv"
AGG_TSV="$OUT_ROOT/aggregate_by_config.tsv"
TOP10_TSV="$OUT_ROOT/top10_aggregate.tsv"
TOP10_MD="$OUT_ROOT/top10_aggregate.md"
BEST_BY_CLASS_TSV="$OUT_ROOT/best_by_class.tsv"
TIMING_TSV="$OUT_ROOT/timing.tsv"

printf "class_name\tstep1_stem\tstep3_stem\tprompt_mode\tn_iter\tsam_model\tN\tsam_iou_mean\tsam_iou_std\tsam_iou_n_low\tfg_coherence\tfg_frac_mean\tspatial_coh\te3_fg_frac\te4_fg_frac\tgt_iou_mean\tiou_gain_mean\tn_improved\tn_degraded\te3_iou_mean_gt\te4_iou_mean_gt\tproxy_score\tstep4_npz\tanalyzer_report\n" > "$METRICS_TSV"
printf "run_index\ttotal_runs\tclass_name\tstep1_stem\tstep3_stem\tprompt_mode\tn_iter\tsam_model\tstatus\tstarted_at\tfinished_at\telapsed_sec\telapsed_hms\tavg_sec\tremaining_runs\test_sec\test_hms\test_finish\tstep4_npz\tlog\tanalyzer_report\n" > "$TIMING_TSV"

# Descobre pares (e1_npz, e3_npz) por classe
declare -A CLASS_PAIRS_MAP

for cls in "${CLASSES[@]}"; do
  safe_cls="$(sanitize "$cls")"
  e1_dir="$STEP1_NPZ_ROOT/$safe_cls"
  e3_dir="$STEP3_NPZ_ROOT/$safe_cls"
  [[ -d "$e1_dir" && -d "$e3_dir" ]] || continue

  pairs=()
  while IFS= read -r -d '' e1_npz; do
    e1_base="$(basename "$e1_npz" .npz)"
    [[ -n "$STEP1_CONFIG_FILTER" ]] && ! echo "$e1_base" | grep -q "$STEP1_CONFIG_FILTER" && continue
    # Alguns sweeps de E1 prefixam o stem com "<Classe>__" (redundante, já que
    # o arquivo já está na pasta da classe); os de E3 nem sempre repetem esse
    # prefixo. Tira o prefixo "<safe_cls>__" de ambos antes de comparar, para
    # não depender de qual dos dois geradores decidiu incluí-lo.
    e1_match="$e1_base"
    [[ "$e1_match" == "${safe_cls}__"* ]] && e1_match="${e1_match#${safe_cls}__}"
    while IFS= read -r -d '' e3_npz; do
      e3_base="$(basename "$e3_npz" .npz)"
      [[ -n "$STEP3_CONFIG_FILTER" ]] && ! echo "$e3_base" | grep -q "$STEP3_CONFIG_FILTER" && continue
      e3_match="$e3_base"
      [[ "$e3_match" == "${safe_cls}__"* ]] && e3_match="${e3_match#${safe_cls}__}"
      # E3 stem (sem prefixo de classe) contém E1 stem (sem prefixo de classe)
      # como prefixo -- mesma convenção das etapas anteriores.
      [[ "$e3_match" == "${e1_match}"* ]] || continue
      pairs+=("${e1_npz}|${e3_npz}")
    done < <(find "$e3_dir" -maxdepth 1 -name "*.npz" -print0 | sort -z)
  done < <(find "$e1_dir" -maxdepth 1 -name "*.npz" -print0 | sort -z)

  if [[ "${#pairs[@]}" -eq 0 ]]; then
    echo "[Aviso] Nenhum par E1/E3 para '$cls'" >&2; continue
  fi
  CLASS_PAIRS_MAP["$cls"]="$(printf '%s\n' "${pairs[@]}")"
done

# Contagem total
TOTAL_RUNS=0
for cls in "${CLASSES[@]}"; do
  [[ -z "${CLASS_PAIRS_MAP[$cls]+x}" ]] && continue
  n_pairs=0
  while IFS= read -r line; do [[ -n "$line" ]] && n_pairs=$((n_pairs+1)); done \
    <<< "${CLASS_PAIRS_MAP[$cls]}"
  TOTAL_RUNS=$(( TOTAL_RUNS + n_pairs * ${#PROMPT_MODES[@]} * ${#N_ITERS[@]} * ${#SAM_MODELS[@]} ))
done

TOTAL_CLASSES="${#CLASSES[@]}"
MIN_CLASSES_FOR_AGGREGATE="${MIN_CLASSES_FOR_AGGREGATE:-all}"
COUNT=0
SWEEP_START_TS=$(date +%s)

cat <<INFO
============================================================
Sweep Etapa 4 — SAM 2 Refinement
Pasta .npz Etapa 1     : $STEP1_NPZ_ROOT
Pasta .npz Etapa 3     : $STEP3_NPZ_ROOT
Ground truth (GT)      : ${GT_INDEX:-não fornecido}
Filtro E1              : ${STEP1_CONFIG_FILTER:-nenhum}
Filtro E3              : ${STEP3_CONFIG_FILTER:-nenhum}
Script etapa 4         : $STEP4_SCRIPT
Analyzer               : $ANALYZER_SCRIPT
Saída                  : $OUT_ROOT
Classes                : $TOTAL_CLASSES
Total execuções        : $TOTAL_RUNS
Prompt modes           : ${PROMPT_MODES[*]}
N iterations           : ${N_ITERS[*]}
SAM models             : ${SAM_MODELS[*]}
Início                 : $(now_human)
Resume                 : $RESUME
============================================================
INFO

# Cálculo de métricas proxy
compute_step4_metrics() {
  local class_name="$1" step1_npz="$2" step3_npz="$3"
  local prompt_mode="$4" n_iter="$5" sam_model="$6"
  local step4_npz="$7" analyzer_report="$8"

  "$PYTHON_BIN" - \
    "$class_name" "$step1_npz" "$step3_npz" \
    "$prompt_mode" "$n_iter" "$sam_model" \
    "$step4_npz" "$analyzer_report" <<'PY'
import sys, json, math
from pathlib import Path
import numpy as np

(class_name, step1_npz, step3_npz, prompt_mode, n_iter,
 sam_model, step4_npz, analyzer_report) = sys.argv[1:]

def safe(v):
    try:
        v = float(v)
        return f"{v:.6f}" if not (math.isnan(v) or math.isinf(v)) else "nan"
    except Exception:
        return "nan"

try:
    data = np.load(step4_npz, allow_pickle=True)
    refined   = data["refined_masks"].astype(np.float32)
    ious_sam  = data["iou_scores_sam"].astype(np.float32)
    fg_coh    = float(data["fg_coherence"])
    image_sizes = data["image_sizes"]

    N = len(refined)

    # Fração FG
    fracs = [float(refined[i, :int(image_sizes[i,0]), :int(image_sizes[i,1])].mean())
             for i in range(N)]
    fg_frac_mean = float(np.mean(fracs))

    # Coesão espacial
    spatial_coh = "nan"
    try:
        from scipy.ndimage import label as ndimage_label
        lc_fracs = []
        for i in range(N):
            H_i, W_i = int(image_sizes[i,0]), int(image_sizes[i,1])
            mask = (refined[i, :H_i, :W_i] > 0.5).astype(np.uint8)
            if mask.sum() == 0: lc_fracs.append(0.0); continue
            labeled, n = ndimage_label(mask)
            if n > 0:
                sizes = [(labeled == k).sum() for k in range(1, n+1)]
                lc_fracs.append(float(max(sizes) / mask.sum()))
            else:
                lc_fracs.append(0.0)
        spatial_coh = safe(float(np.mean(lc_fracs)))
    except Exception:
        pass

    sam_iou_mean = safe(float(np.mean(ious_sam)))
    sam_iou_std  = safe(float(np.std(ious_sam)))
    n_low = int(np.sum(ious_sam < 0.60))

    # E3 fração para comparação
    e3_frac = "nan"
    try:
        e3_data = np.load(step3_npz, allow_pickle=True)
        if "fg_mask_pixel" in e3_data.files:
            fg_e3 = e3_data["fg_mask_pixel"].astype(np.float32)
            e3_fracs = [float(fg_e3[i, :int(image_sizes[i,0]),
                                    :int(image_sizes[i,1])].mean())
                        for i in range(min(N, len(fg_e3)))]
            e3_frac = safe(float(np.mean(e3_fracs)))
    except Exception:
        pass

    # Lê métricas GT do json do analisador.
    # O diretório é único por configuração: <stem>__quality/step4_quality.json
    gt_iou = gain = n_improved = n_degraded = "nan"
    e3_iou_gt = e4_iou_gt = "nan"
    stem = Path(analyzer_report).stem.replace(".analysis", "")
    json_path = Path(analyzer_report).parent / f"{stem}__quality" / "step4_quality.json"
    if json_path.exists():
        try:
            d = json.loads(json_path.read_text(encoding="utf-8"))
            m = d.get("metrics", {})
            gt_iou      = safe(m.get("gt_iou_mean"))
            gain        = safe(m.get("iou_gain_mean"))
            n_improved  = str(m.get("n_images_improved", "nan"))
            n_degraded  = str(m.get("n_images_degraded", "nan"))
            # Baseline pareado: IoU real da E3 (pré-SAM) e da E4, calculado
            # sobre exatamente o mesmo subconjunto de imagens com GT válido
            # usado para o gain — é o número certo para a linha de baseline
            # da tabela de ablação (gt_iou_mean acima não é necessariamente
            # pareado com E3 se cosaliency.npz tiver imagens a mais/a menos).
            e3_iou_gt   = safe(m.get("e3_iou_mean"))
            e4_iou_gt   = safe(m.get("e4_iou_mean"))
        except Exception:
            pass

    # Proxy score: premia iou_sam alto, coesão semântica, coesão espacial
    s_iou  = max(0.0, min(1.0, float(sam_iou_mean) if sam_iou_mean != "nan" else 0.0))
    s_coh  = max(0.0, min(1.0, float(spatial_coh) if spatial_coh != "nan" else 0.0))
    s_fcoh = max(0.0, min(1.0, (fg_coh + 0.3) / 0.8))  # normaliza aprox. para [0,1]
    s_low  = max(0.0, 1.0 - n_low / max(N, 1))
    proxy  = 0.40 * s_iou + 0.25 * s_coh + 0.20 * s_fcoh + 0.15 * s_low

    step1_stem = Path(step1_npz).stem
    step3_stem = Path(step3_npz).stem
    model_short = sam_model.split("/")[-1]

    values = [
        class_name, step1_stem, step3_stem,
        prompt_mode, str(n_iter), model_short,
        str(N), sam_iou_mean, sam_iou_std, str(n_low),
        safe(fg_coh), safe(fg_frac_mean), spatial_coh,
        e3_frac, safe(fg_frac_mean),
        gt_iou, gain, str(n_improved), str(n_degraded),
        e3_iou_gt, e4_iou_gt,
        safe(proxy), str(Path(step4_npz)), str(Path(analyzer_report)),
    ]
    print("\t".join(values))
except Exception as e:
    print(f"[ERRO_METRICS] {step4_npz}: {e}", file=sys.stderr)
    raise
PY
}

# Analisador
run_analyzer() {
  local step4_npz="$1" step3_npz="$2" report_file="$3"

  if [[ "$REUSE_ANALYZER_REPORTS" -eq 1 && "$FORCE" -eq 0 && -s "$report_file" ]]; then
    echo "[Analyzer] Reaproveitando: $report_file"; return 0
  fi
  if [[ "$RUN_ANALYZER" -eq 0 ]]; then
    echo "[Analyzer] Desativado." > "$report_file"; return 0
  fi
  if [[ ! -f "$ANALYZER_SCRIPT" ]]; then
    echo "[Analyzer] Script não encontrado." > "$report_file"; return 0
  fi

  # Diretório único por configuração: deriva do nome do report_file (que já contém o STEM).
  # Isso evita que execuções diferentes da mesma classe sobrescrevam o step4_quality.json.
  local stem
  stem="$(basename "$report_file" .analysis.txt)"
  local out_dir
  out_dir="$(dirname "$report_file")/${stem}__quality"

  local analyzer_args=(
    --refined    "$step4_npz"
    --output     "$out_dir"
  )
  [[ -n "$step3_npz" && -f "$step3_npz" ]] && analyzer_args+=(--cosaliency "$step3_npz")
  [[ -n "$GT_INDEX" ]]                      && analyzer_args+=(--gt_index   "$GT_INDEX")

  # IMPORTANTE (resiliência a queda de energia / interrupção): report_file é
  # o que REUSE_ANALYZER_REPORTS usa para decidir se esta config já foi
  # analisada. Se escrevêssemos diretamente nele durante a execução do
  # Python, uma interrupção no meio da análise deixaria um arquivo
  # não-vazio em disco — no próximo resume, isso seria interpretado como
  # "já analisado" e a config seria pulada silenciosamente, mesmo com o
  # relatório de qualidade incompleto ou ausente. Por isso todo o conteúdo
  # é escrito num arquivo temporário e só promovido (mv atômico) para
  # report_file depois que o processo Python termina.
  local tmp_report="${report_file}.tmp.$$"
  echo "[Analyzer] $PYTHON_BIN $ANALYZER_SCRIPT ${analyzer_args[*]}" > "$tmp_report"
  set +e
  "$PYTHON_BIN" "$ANALYZER_SCRIPT" "${analyzer_args[@]}" >> "$tmp_report" 2>&1
  local status=$?
  set -e

  mv -f "$tmp_report" "$report_file"

  [[ "$status" -ne 0 ]] && echo "[AVISO] Analisador falhou. Veja: $report_file" >&2
  return 0
}

# Agregação
make_aggregate() {
  local total_classes="$1" min_classes="$2"
  "$PYTHON_BIN" - \
    "$METRICS_TSV" "$AGG_TSV" "$TOP10_TSV" "$TOP10_MD" \
    "$BEST_BY_CLASS_TSV" "$total_classes" "$min_classes" <<'PY'
import csv, math, statistics as stats, sys
from collections import defaultdict
from pathlib import Path

metrics_tsv, agg_tsv, top10_tsv, top10_md, best_tsv = map(Path, sys.argv[1:6])
total_cls   = int(sys.argv[6])
min_cls_arg = sys.argv[7]

def safe(s):
    try:
        v = float(s)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    except Exception:
        return None

with metrics_tsv.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
rows = [r for r in rows if safe(r.get("proxy_score")) is not None]

# Critério de ranking: quando há GT (iou_gain_mean real da E3->E4), ele SEMPRE
# prevalece sobre o proxy_score (que só mede confiança do SAM + heurísticas de
# coesão, sem contato com ground truth). proxy_score vira fallback/desempate
# para linhas sem GT, ou desempate entre linhas que empatam em iou_gain_mean.
def row_rank_value(r):
    gain = safe(r.get("iou_gain_mean"))
    proxy = safe(r.get("proxy_score"))
    proxy = proxy if proxy is not None else float("-inf")
    if gain is not None:
        return (1, gain, proxy)
    return (0, proxy, proxy)

def row_rank_basis(r):
    return "gt_iou_gain" if safe(r.get("iou_gain_mean")) is not None else "proxy_score"

# Melhor por classe
best_by_cls = {}
for r in rows:
    c = r["class_name"]
    if c not in best_by_cls or row_rank_value(r) > row_rank_value(best_by_cls[c]):
        best_by_cls[c] = r

best_fields = [
    "class_name","step1_stem","step3_stem","prompt_mode","n_iter","sam_model",
    "N","sam_iou_mean","fg_coherence","fg_frac_mean","spatial_coh",
    "gt_iou_mean","iou_gain_mean","n_improved","n_degraded","proxy_score",
    "rank_basis","step4_npz",
]
with best_tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, delimiter="\t", fieldnames=best_fields)
    w.writeheader()
    for c in sorted(best_by_cls):
        r = dict(best_by_cls[c])
        r["rank_basis"] = row_rank_basis(r)
        w.writerow({k: r.get(k,"") for k in best_fields})

# Agrega por (prompt_mode, n_iter, sam_model)
groups = defaultdict(list)
for r in rows:
    key = (r["prompt_mode"], r["n_iter"], r["sam_model"])
    groups[key].append(r)

def mean(xs): return sum(xs)/len(xs) if xs else math.nan
def stdev(xs): return stats.pstdev(xs) if len(xs)>1 else 0.0

agg_rows = []
for (pm, ni, sm), gr in groups.items():
    classes = sorted({r["class_name"] for r in gr})
    def col(k): return [float(r[k]) for r in gr if safe(r.get(k)) is not None]

    mean_gain = mean(col("iou_gain_mean"))
    agg_rows.append({
        "prompt_mode": pm, "n_iter": ni, "sam_model": sm,
        "classes_done": len(classes), "classes_total": total_cls,
        "coverage": len(classes)/total_cls if total_cls else 0.0,
        "mean_proxy":       mean(col("proxy_score")),
        "std_proxy":        stdev(col("proxy_score")),
        "mean_sam_iou":     mean(col("sam_iou_mean")),
        "mean_fg_coherence":mean(col("fg_coherence")),
        "mean_spatial_coh": mean(col("spatial_coh")),
        "mean_gt_iou":      mean(col("gt_iou_mean")),
        "std_gt_iou":       stdev(col("gt_iou_mean")),
        "mean_iou_gain":    mean_gain,
        # Desvio-padrão do ganho ENTRE CLASSES (não entre imagens) — mede se
        # o ganho é consistente pelo dataset ou puxado por poucas classes.
        "std_iou_gain":     stdev(col("iou_gain_mean")),
        "mean_e4_iou_gt":   mean(col("e4_iou_mean_gt")),
        "n_gt_classes":     len(col("iou_gain_mean")),
        "rank_basis":       "gt_iou_gain" if not math.isnan(mean_gain) else "proxy_score",
    })

# Baseline E3 (sem refinamento SAM): não depende de prompt_mode/n_iter/model,
# então tira-se a média de e3_iou_mean_gt uma vez por classe (deduplicado),
# não por linha (que repetiria o mesmo valor de E3 em cada config testada).
baseline_e3_by_cls = defaultdict(list)
for r in rows:
    v = safe(r.get("e3_iou_mean_gt"))
    if v is not None:
        baseline_e3_by_cls[r["class_name"]].append(v)
baseline_e3_values = [mean(vs) for vs in baseline_e3_by_cls.values()]
baseline_e3_mean = mean(baseline_e3_values)
baseline_e3_std  = stdev(baseline_e3_values)
baseline_n_cls   = len(baseline_e3_values)

min_cls = total_cls if min_cls_arg == "all" else int(min_cls_arg)

# Mesmo critério do ranking por linha: ganho real de IoU (GT) tem prioridade
# sobre proxy_score sempre que a config tiver cobertura de GT; proxy_score só
# decide quando não há GT disponível para nenhuma classe daquela config.
def group_rank_value(r):
    gain = r["mean_iou_gain"]
    proxy = r["mean_proxy"]
    proxy = proxy if not math.isnan(proxy) else float("-inf")
    if not math.isnan(gain):
        return (1, gain, proxy)
    return (0, proxy, proxy)

agg_all = sorted(agg_rows, key=lambda r: (r["classes_done"], group_rank_value(r)), reverse=True)
qualified = [r for r in agg_rows if int(r["classes_done"]) >= min_cls] or agg_rows
qualified.sort(key=group_rank_value, reverse=True)
top10 = qualified[:10]

agg_fields = [
    "prompt_mode","n_iter","sam_model",
    "classes_done","classes_total","coverage","n_gt_classes",
    "mean_proxy","std_proxy",
    "mean_sam_iou","mean_fg_coherence","mean_spatial_coh",
    "mean_gt_iou","std_gt_iou","mean_iou_gain","std_iou_gain",
    "mean_e4_iou_gt","rank_basis",
]

def fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}" if not (math.isnan(v) or math.isinf(v)) else "nan"
    return str(v)

for path, data in [(agg_tsv, agg_all), (top10_tsv, top10)]:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=agg_fields)
        w.writeheader()
        for r in data:
            w.writerow({k: fmt(r.get(k,"")) for k in agg_fields})

def s(r, k):
    v = r.get(k, math.nan)
    return f"{v:.4f}" if isinstance(v,float) and not (math.isnan(v) or math.isinf(v)) else "n/a"

with top10_md.open("w", encoding="utf-8") as f:
    f.write("# Top 10 agregado — configurações da Etapa 4 (SAM 2)\n\n")
    f.write("> Ranking por `mean_iou_gain` (ganho real de IoU E3->E4 contra GT) quando há "
            "ground truth disponível; `mean_proxy` (confiança SAM + coesão) é usado apenas "
            "como fallback para configs sem GT. Ver coluna `rank_basis`.\n\n")
    f.write(f"Classes avaliadas: **{total_cls}**. Mínimo para ranking: **{min_cls}**.\n\n")
    if baseline_n_cls > 0:
        f.write(
            f"**Baseline (E3, sem refinamento SAM2)**: IoU médio contra GT = "
            f"**{baseline_e3_mean:.4f}** (±{baseline_e3_std:.4f}, n={baseline_n_cls} classes "
            f"com GT). Use esta linha como referência para o ganho absoluto (`mean_gain` "
            f"abaixo é o delta E4−E3; IoU absoluto da E4 é `mean_e4_iou_gt`).\n\n"
        )
    else:
        f.write(
            "**Baseline (E3) indisponível** — nenhuma classe com GT pareado encontrada. "
            "`mean_gain`/`mean_e4_iou_gt` abaixo não têm base de comparação; confira "
            "`gt_index.json`.\n\n"
        )
    f.write("| # | prompt_mode | n_iter | model | classes (GT) | cov | rank_basis | "
            "mean_proxy | mean_sam_iou | mean_fg_coh | mean_spatial | "
            "mean_gt_iou±std | mean_e4_iou | mean_gain±std |\n")
    f.write("|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(top10, 1):
        gt_iou_cell = f"{s(r,'mean_gt_iou')}±{s(r,'std_gt_iou')}" if not math.isnan(r.get("mean_gt_iou", math.nan)) else "n/a"
        gain_cell   = f"{s(r,'mean_iou_gain')}±{s(r,'std_iou_gain')}" if not math.isnan(r.get("mean_iou_gain", math.nan)) else "n/a"
        f.write(f"| {i} | {r['prompt_mode']} | {r['n_iter']} | {r['sam_model']} "
                f"| {r['n_gt_classes']}/{r['classes_done']} | {r['coverage']:.1%} "
                f"| {r['rank_basis']} "
                f"| {s(r,'mean_proxy')} | {s(r,'mean_sam_iou')} | {s(r,'mean_fg_coherence')} "
                f"| {s(r,'mean_spatial_coh')} | {gt_iou_cell} | {s(r,'mean_e4_iou_gt')} | {gain_cell} |\n")

print(f"[Aggregate] {agg_tsv}")
print(f"[Top10] {top10_tsv} / {top10_md}")
print(f"[Best by class] {best_tsv}")
PY
}

# Loop principal
for cls in "${CLASSES[@]}"; do
  [[ -z "${CLASS_PAIRS_MAP[$cls]+x}" ]] && continue
  safe_cls="$(sanitize "$cls")"

  echo ""
  echo "############################################################"
  echo "Classe: $cls"
  echo "############################################################"

  while IFS= read -r pair_line; do
    [[ -z "$pair_line" ]] && continue
    e1_npz="${pair_line%%|*}"
    e3_npz="${pair_line##*|}"
    e1_stem="$(basename "$e1_npz" .npz)"
    e3_stem="$(basename "$e3_npz" .npz)"

    cls_npz_dir="$NPZ_ROOT/$safe_cls"
    cls_report_dir="$REPORT_ROOT/$safe_cls"
    cls_log_dir="$LOG_ROOT/$safe_cls"
    cls_status_dir="$STATUS_ROOT/$safe_cls"
    mkdir -p "$cls_npz_dir" "$cls_report_dir" "$cls_log_dir" "$cls_status_dir"

    for prompt_mode in "${PROMPT_MODES[@]}"; do
      for n_iter in "${N_ITERS[@]}"; do
        for sam_model in "${SAM_MODELS[@]}"; do
          COUNT=$((COUNT + 1))

          model_short="$(sanitize_model "$sam_model")"
          STEM="${e3_stem}__pm-${prompt_mode}__ni-${n_iter}__m-${model_short}"
          FLAT_NPZ="$cls_npz_dir/${STEM}.npz"
          LOG="$cls_log_dir/${STEM}.log"
          REPORT="$cls_report_dir/${STEM}.analysis.txt"
          MARKER="$cls_status_dir/${STEM}.done"

          echo ""
          echo "[$COUNT/$TOTAL_RUNS] $cls :: mode=$prompt_mode iter=$n_iter model=$model_short"
          echo "  E1: $(basename "$e1_npz")"
          echo "  E3: $(basename "$e3_npz")"

          RUN_START_TS=$(date +%s)
          RUN_STATUS="ok"

          EXISTING_VALID=0
          if [[ "$RESUME" -eq 1 && "$FORCE" -eq 0 && -f "$FLAT_NPZ" ]]; then
            if npz_is_valid_e4 "$FLAT_NPZ"; then
              EXISTING_VALID=1
            else
              echo "[Aviso] .npz inválido. Removendo: $FLAT_NPZ" >&2
              rm -f "$FLAT_NPZ" "$MARKER"
            fi
          fi

          if [[ "$EXISTING_VALID" -eq 1 ]]; then
            echo "[Resume] $(basename "$FLAT_NPZ")"; RUN_STATUS="resumed_done"
          else
            rm -f "${FLAT_NPZ}.tmp."*
            TMP_DIR="$cls_npz_dir/tmp_${STEM}"
            mkdir -p "$TMP_DIR"

            cmd=("$PYTHON_BIN" "$STEP4_SCRIPT"
              --features     "$e1_npz"
              --cosaliency   "$e3_npz"
              --output       "$TMP_DIR"
              --model        "$sam_model"
              --prompt_mode  "$prompt_mode"
              --n_iter       "$n_iter")
            [[ "$VIZ" -eq 1 ]]    && cmd+=(--visualize --viz_max "$VIZ_MAX")
            [[ "$EXPORT" -eq 1 ]] && cmd+=(--export)

            echo "[Run] ${cmd[*]}"
            set +e
            "${cmd[@]}" > "$LOG" 2>&1
            status=$?
            set -e

            if [[ "$status" -ne 0 ]]; then
              echo "[ERRO] Etapa 4 falhou. Veja: $LOG" >&2
              RUN_STATUS="failed"
            elif [[ ! -f "$TMP_DIR/refined_masks.npz" ]]; then
              echo "[ERRO] refined_masks.npz não encontrado em $TMP_DIR" >&2
              RUN_STATUS="failed_missing_npz"
            else
              cp "$TMP_DIR/refined_masks.npz" "$FLAT_NPZ"
              rm -rf "$TMP_DIR"
              echo "[OK] $(basename "$FLAT_NPZ")"
            fi
          fi

          if [[ "$RUN_STATUS" == "failed"* ]]; then
            RUN_END_TS=$(date +%s); RUN_ELAPSED=$((RUN_END_TS-RUN_START_TS))
            SWEEP_ELAPSED=$((RUN_END_TS-SWEEP_START_TS))
            AVG_SEC=$(awk -v e="$SWEEP_ELAPSED" -v c="$COUNT" 'BEGIN{if(c>0){printf "%.2f",e/c}else{printf "0.00"}}')
            REM=$((TOTAL_RUNS-COUNT))
            EST=$(awk -v a="$AVG_SEC" -v r="$REM" 'BEGIN{printf "%d",a*r}')
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
              "$COUNT" "$TOTAL_RUNS" "$cls" "$e1_stem" "$e3_stem" \
              "$prompt_mode" "$n_iter" "$model_short" \
              "$RUN_STATUS" "$(now_human)" "$(now_human)" "$RUN_ELAPSED" \
              "$(format_duration "$RUN_ELAPSED")" "$AVG_SEC" "$REM" "$EST" \
              "$(format_duration "$EST")" "$(eta_human "$EST")" \
              "$FLAT_NPZ" "$LOG" "$REPORT" >> "$TIMING_TSV"
            continue
          fi

          run_analyzer "$FLAT_NPZ" "$e3_npz" "$REPORT"
          compute_step4_metrics \
            "$cls" "$e1_npz" "$e3_npz" \
            "$prompt_mode" "$n_iter" "$sam_model" \
            "$FLAT_NPZ" "$REPORT" >> "$METRICS_TSV"
          write_done_marker "$MARKER" "$FLAT_NPZ" "$RUN_STATUS"

          RUN_END_TS=$(date +%s)
          RUN_ELAPSED=$((RUN_END_TS-RUN_START_TS))
          SWEEP_ELAPSED=$((RUN_END_TS-SWEEP_START_TS))
          AVG_SEC=$(awk -v e="$SWEEP_ELAPSED" -v c="$COUNT" 'BEGIN{if(c>0){printf "%.2f",e/c}else{printf "0.00"}}')
          REM=$((TOTAL_RUNS-COUNT))
          EST=$(awk -v a="$AVG_SEC" -v r="$REM" 'BEGIN{printf "%d",a*r}')
          PCT=$(awk -v c="$COUNT" -v t="$TOTAL_RUNS" 'BEGIN{if(t>0){printf "%.2f",100*c/t}else{printf "0.00"}}')

          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$COUNT" "$TOTAL_RUNS" "$cls" "$e1_stem" "$e3_stem" \
            "$prompt_mode" "$n_iter" "$model_short" \
            "$RUN_STATUS" "$(now_human)" "$(now_human)" "$RUN_ELAPSED" \
            "$(format_duration "$RUN_ELAPSED")" "$AVG_SEC" "$REM" "$EST" \
            "$(format_duration "$EST")" "$(eta_human "$EST")" \
            "$FLAT_NPZ" "$LOG" "$REPORT" >> "$TIMING_TSV"

          echo "[Tempo] $(format_duration "$RUN_ELAPSED") | $COUNT/$TOTAL_RUNS (${PCT}%) | ETA: $(eta_human "$EST")"
        done
      done
    done
  done <<< "${CLASS_PAIRS_MAP[$cls]}"

  make_aggregate "$TOTAL_CLASSES" "$MIN_CLASSES_FOR_AGGREGATE"
done

make_aggregate "$TOTAL_CLASSES" "$MIN_CLASSES_FOR_AGGREGATE"

cat <<DONE

============================================================
Concluído.
.npz Etapa 4             : $NPZ_ROOT
Relatórios analisador    : $REPORT_ROOT
Logs                     : $LOG_ROOT
Métricas completas       : $METRICS_TSV
Agregado por config      : $AGG_TSV
Top 10 TSV               : $TOP10_TSV
Top 10 Markdown          : $TOP10_MD
Melhor config por classe : $BEST_BY_CLASS_TSV
Tempos                   : $TIMING_TSV
============================================================
DONE