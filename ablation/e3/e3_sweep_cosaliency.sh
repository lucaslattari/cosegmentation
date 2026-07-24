#!/usr/bin/env bash
set -Eeuo pipefail
export LC_NUMERIC=C

# =============================================================================
# Sweep da Etapa 3 — Co-Saliency via Spectral Clustering
#
# Itera sobre pares (features.npz E1, correspondence.npz E2) já existentes
# e roda step3_cosaliency.py + analyze_step3_cosaliency.py para cada
# combinação de (par_e1_e2, n_clusters, k_neighbors, fg_threshold, saliency_alpha).
#
# Estrutura esperada de entrada:
#   <step1_npz_root>/<classe>/<stem_e1>.npz
#   <step2_npz_root>/<classe>/<stem_e1>__topk-<K>__temp-<T>.npz
#
# O script casa automaticamente o features.npz da E1 com os correspondence.npz
# da E2 que foram gerados a partir dele (mesmo stem_e1 no nome).
#
# Uso básico:
#   chmod +x run_step3_cosaliency_sweep.sh
#   dos2unix run_step3_cosaliency_sweep.sh
#   ./run_step3_cosaliency_sweep.sh \
#     -1 ./features_sweep_selected/npz \
#     -2 ./correspondence_sweep/npz \
#     -s ./step3_cosaliency.py \
#     -a ./analyze_step3_cosaliency.py \
#     -o ./cosaliency_sweep \
#     -g ./gt_index.json
#
# Variáveis de ambiente úteis:
#   PYTHON_BIN=python3
#   N_CLUSTERS_STR="4 5 6 8"
#   K_NEIGHBORS_STR="8 10 15"
#   FG_THRESHOLD_STR="0.4 0.5 0.6"
#   SALIENCY_ALPHA_STR="0.0 0.25 0.5 0.75 1.0"
#     # peso do prior espacial no score final [0=só E2, 1=só espacial].
#     # Default "0.5" se não definido (igual ao default do próprio
#     # step3_cosaliency.py) — comportamento antigo preservado se você não
#     # setar essa variável. Vale MUITO a pena variar: um probe manual
#     # mostrou que o ótimo é fortemente dependente da classe (objetos
#     # pequenos/descentralizados como pipas pioram com alpha alto; objetos
#     # centrais/compactos como figuras humanas melhoram).
#   STEP1_CONFIG_FILTER=""   # substring para filtrar .npz da E1
#   STEP2_CONFIG_FILTER=""   # substring para filtrar .npz da E2
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

# Defaults
STEP1_NPZ_ROOT="./features_sweep_selected/npz"
STEP2_NPZ_ROOT="./correspondence_sweep/npz"
STEP3_SCRIPT="./step3_cosaliency.py"
ANALYZER_SCRIPT="./analyze_step3_cosaliency.py"
OUT_ROOT="./cosaliency_sweep"
GT_INDEX="${GT_INDEX:-}"
NO_PADDING_MASK="${NO_PADDING_MASK:-0}"
MAX_CLASSES=""
CLASS_REGEX=""
STEP1_CONFIG_FILTER="${STEP1_CONFIG_FILTER:-}"
STEP2_CONFIG_FILTER="${STEP2_CONFIG_FILTER:-}"
FORCE=0
VIZ=0
VIZ_MAX=8
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ANALYZER="${RUN_ANALYZER:-1}"
RESUME="${RESUME:-1}"
VALIDATE_EXISTING="${VALIDATE_EXISTING:-1}"
REUSE_ANALYZER_REPORTS="${REUSE_ANALYZER_REPORTS:-1}"

usage() {
  cat <<USAGE
Uso:
  $0 -1 <npz_e1> -2 <npz_e2> [opções]

Opções:
  -1  Pasta raiz dos .npz da Etapa 1. Default: ./features_sweep_selected/npz
  -2  Pasta raiz dos .npz da Etapa 2. Default: ./correspondence_sweep/npz
  -s  Script da etapa 3. Default: ./step3_cosaliency.py
  -a  Script analisador. Default: ./analyze_step3_cosaliency.py
  -o  Pasta raiz de saída. Default: ./cosaliency_sweep
  -g  Caminho para gt_index.json.
  -n  Máximo de classes (teste rápido). Ex.: -n 2
  -k  Regex para filtrar classes. Ex.: -k 'Panda|Bear'
  -f  Força reprocessamento
  -v  Gera visualizações (--visualize)
  -h  Ajuda

Variáveis de ambiente úteis:
  PYTHON_BIN=python3
  N_CLUSTERS_STR="4 5 6 8"
  K_NEIGHBORS_STR="8 10 15"
  FG_THRESHOLD_STR="0.4 0.5 0.6"
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2"
  STEP2_CONFIG_FILTER="topk-10__temp-0.05"
  GT_INDEX=/caminho/gt_index.json
  NO_PADDING_MASK=0   # 1 desativa a exclusão de padding nas métricas do analisador
  RUN_ANALYZER=1
  RESUME=1

Exemplos:
  # Teste rápido: 2 classes, configs fixas da E1/E2, 1 combinação E3
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2" \\
  STEP2_CONFIG_FILTER="topk-10__temp-0.05" \\
  N_CLUSTERS_STR="5" K_NEIGHBORS_STR="10" FG_THRESHOLD_STR="0.5" SALIENCY_ALPHA_STR="0.5" \\
  $0 -1 ./features_sweep_selected/npz -2 ./correspondence_sweep/npz \\
     -g ./gt_index.json -n 2

  # Sweep completo, incluindo saliency_alpha
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2" \\
  STEP2_CONFIG_FILTER="topk-10__temp-0.05" \\
  N_CLUSTERS_STR="4 5 6 8" K_NEIGHBORS_STR="8 10 15" FG_THRESHOLD_STR="0.4 0.5 0.6" \\
  SALIENCY_ALPHA_STR="0.0 0.25 0.5 0.75 1.0" \\
  $0 -1 ./features_sweep_selected/npz -2 ./correspondence_sweep/npz -g ./gt_index.json
USAGE
}

while getopts ":1:2:s:a:o:g:n:k:fvh" opt; do
  case "$opt" in
    1) STEP1_NPZ_ROOT="$OPTARG" ;;
    2) STEP2_NPZ_ROOT="$OPTARG" ;;
    s) STEP3_SCRIPT="$OPTARG" ;;
    a) ANALYZER_SCRIPT="$OPTARG" ;;
    o) OUT_ROOT="$OPTARG" ;;
    g) GT_INDEX="$OPTARG" ;;
    n) MAX_CLASSES="$OPTARG" ;;
    k) CLASS_REGEX="$OPTARG" ;;
    f) FORCE=1 ;;
    v) VIZ=1 ;;
    h) usage; exit 0 ;;
    :) echo "[ERRO] Opção -$OPTARG requer argumento." >&2; usage; exit 2 ;;
    \?) echo "[ERRO] Opção inválida: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

for d in "$STEP1_NPZ_ROOT" "$STEP2_NPZ_ROOT"; do
  [[ -d "$d" ]] || { echo "[ERRO] Pasta não encontrada: $d" >&2; exit 1; }
done
[[ -f "$STEP3_SCRIPT" ]] || { echo "[ERRO] Script E3 não encontrado: $STEP3_SCRIPT" >&2; exit 1; }

abs() { "$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }
STEP1_NPZ_ROOT="$(abs "$STEP1_NPZ_ROOT")"
STEP2_NPZ_ROOT="$(abs "$STEP2_NPZ_ROOT")"
STEP3_SCRIPT="$(abs "$STEP3_SCRIPT")"
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

# Grade experimental E3
read -r -a N_CLUSTERS_LIST   <<< "${N_CLUSTERS_STR:-4 5 6 8}"
read -r -a K_NEIGHBORS_LIST  <<< "${K_NEIGHBORS_STR:-8 10 15}"
read -r -a FG_THRESHOLD_LIST <<< "${FG_THRESHOLD_STR:-0.4 0.5 0.6}"
read -r -a SALIENCY_ALPHA_LIST <<< "${SALIENCY_ALPHA_STR:-0.5}"

# fg_decision_mode="e2_border_topk" (experimental) não usa saliency_alpha —
# colapsa o grid de alpha pra um único valor-placeholder, evitando rodar a
# mesma config várias vezes à toa (o valor é ignorado nesse modo).
FG_DECISION_MODE="${FG_DECISION_MODE:-saliency}"
E2_BORDER_WEIGHT="${E2_BORDER_WEIGHT:-0.3}"
TOPK_FRACTION="${TOPK_FRACTION:-0.30}"
if [[ "$FG_DECISION_MODE" != "saliency" ]]; then
  SALIENCY_ALPHA_LIST=(0.5)
fi

# Utilitários
sanitize() {
  local s="$1"
  s="${s//\\//_}"; s="${s//\//_}"; s="${s// /_}"
  printf '%s' "$s" | tr -cd '[:alnum:]_.-'
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

npz_is_valid_e3() {
  local npz="$1"
  [[ -f "$npz" ]] || return 1
  [[ "$VALIDATE_EXISTING" -eq 0 ]] && return 0
  "$PYTHON_BIN" - "$npz" <<'PY' >/dev/null 2>&1
import sys, numpy as np
data = np.load(sys.argv[1], allow_pickle=True)
for k in ["fg_mask_patch","fg_mask_coarse","cluster_labels","cluster_scores",
          "cluster_is_fg","n_clusters_used","image_paths","image_sizes"]:
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

printf "class_name\tstep1_stem\tstep2_stem\tn_clusters\tk_neighbors\tfg_threshold\tsaliency_alpha\tN\tP\tnc_used\tnc_fg\tnc_bg\tfg_frac_mean\tspatial_coherence\tcluster_margin\tn_ambiguous\tpatch_iou_refined\tpatch_iou_coarse\trefine_delta\tpixel_iou\toracle_global_iou\toracle_imagewise_iou\tmapping_accuracy\tauc_roc\tproxy_score\tstep3_npz\tanalyzer_report\n" > "$METRICS_TSV"
printf "run_index\ttotal_runs\tclass_name\tstep1_stem\tstep2_stem\tn_clusters\tk_neighbors\tfg_threshold\tsaliency_alpha\tstatus\tstarted_at\tfinished_at\telapsed_sec\telapsed_hms\tavg_sec\tremaining_runs\test_sec\test_hms\test_finish\tstep3_npz\tlog\tanalyzer_report\n" > "$TIMING_TSV"

# Descobre pares (e1_npz, e2_npz) por classe
declare -A CLASS_PAIRS_MAP

for cls in "${CLASSES[@]}"; do
  safe_cls="$(sanitize "$cls")"
  e1_dir="$STEP1_NPZ_ROOT/$safe_cls"
  e2_dir="$STEP2_NPZ_ROOT/$safe_cls"
  [[ -d "$e1_dir" && -d "$e2_dir" ]] || continue

  pairs=()
  while IFS= read -r -d '' e1_npz; do
    e1_base="$(basename "$e1_npz" .npz)"
    [[ -n "$STEP1_CONFIG_FILTER" ]] && ! echo "$e1_base" | grep -q "$STEP1_CONFIG_FILTER" && continue
    while IFS= read -r -d '' e2_npz; do
      e2_base="$(basename "$e2_npz" .npz)"
      [[ -n "$STEP2_CONFIG_FILTER" ]] && ! echo "$e2_base" | grep -q "$STEP2_CONFIG_FILTER" && continue
      # Verifica que o e2 foi gerado a partir deste e1 (stem_e1 é prefixo do stem_e2)
      [[ "$e2_base" == "${e1_base}"* ]] || continue
      pairs+=("${e1_npz}|${e2_npz}")
    done < <(find "$e2_dir" -maxdepth 1 -name "*.npz" -print0 | sort -z)
  done < <(find "$e1_dir" -maxdepth 1 -name "*.npz" -print0 | sort -z)

  if [[ "${#pairs[@]}" -eq 0 ]]; then
    echo "[Aviso] Nenhum par E1/E2 para '$cls'" >&2; continue
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
  TOTAL_RUNS=$(( TOTAL_RUNS + n_pairs * ${#N_CLUSTERS_LIST[@]} * ${#K_NEIGHBORS_LIST[@]} * ${#FG_THRESHOLD_LIST[@]} * ${#SALIENCY_ALPHA_LIST[@]} ))
done

TOTAL_CLASSES="${#CLASSES[@]}"
MIN_CLASSES_FOR_AGGREGATE="${MIN_CLASSES_FOR_AGGREGATE:-all}"
COUNT=0
SWEEP_START_TS=$(date +%s)

cat <<INFO
============================================================
Sweep Etapa 3 — Co-Saliency via Spectral Clustering
Pasta .npz Etapa 1     : $STEP1_NPZ_ROOT
Pasta .npz Etapa 2     : $STEP2_NPZ_ROOT
Ground truth (GT)      : ${GT_INDEX:-não fornecido}
Filtro E1              : ${STEP1_CONFIG_FILTER:-nenhum}
Filtro E2              : ${STEP2_CONFIG_FILTER:-nenhum}
Script etapa 3         : $STEP3_SCRIPT
Analyzer               : $ANALYZER_SCRIPT
Saída                  : $OUT_ROOT
Classes                : $TOTAL_CLASSES
Total execuções        : $TOTAL_RUNS
n_clusters             : ${N_CLUSTERS_LIST[*]}
k_neighbors            : ${K_NEIGHBORS_LIST[*]}
fg_threshold           : ${FG_THRESHOLD_LIST[*]}
saliency_alpha         : ${SALIENCY_ALPHA_LIST[*]}
Início                 : $(now_human)
Resume                 : $RESUME
============================================================
INFO

# Cálculo de métricas proxy
compute_step3_metrics() {
  local class_name="$1" step1_npz="$2" step2_npz="$3"
  local n_clusters="$4" k_neighbors="$5" fg_threshold="$6" saliency_alpha="$7"
  local step3_npz="$8" analyzer_report="$9"

  "$PYTHON_BIN" - \
    "$class_name" "$step1_npz" "$step2_npz" \
    "$n_clusters" "$k_neighbors" "$fg_threshold" "$saliency_alpha" \
    "$step3_npz" "$analyzer_report" <<'PY'
import sys, json, math, itertools
from pathlib import Path
import numpy as np

(class_name, step1_npz, step2_npz, n_clusters, k_neighbors,
 fg_threshold, saliency_alpha, step3_npz, analyzer_report) = sys.argv[1:]
n_clusters = int(n_clusters)
fg_threshold = float(fg_threshold)
saliency_alpha = float(saliency_alpha)

def safe(v):
    try:
        v = float(v)
        return f"{v:.6f}" if not (math.isnan(v) or math.isinf(v)) else "nan"
    except Exception:
        return "nan"

def iou(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return float(inter/union) if union > 0 else 0.0

try:
    data = np.load(step3_npz, allow_pickle=True)
    fg_patch  = data["fg_mask_patch"].astype(np.float32)
    fg_coarse = data["fg_mask_coarse"].astype(np.float32)
    cl_labels = data["cluster_labels"].astype(np.int32)
    cl_scores = data["cluster_scores"].astype(np.float32)
    cl_is_fg  = data["cluster_is_fg"].astype(bool)
    nc_used   = int(data["n_clusters_used"])

    N = fg_patch.shape[0]
    P = fg_patch.shape[1] * fg_patch.shape[2]

    # Fração FG
    fracs = fg_patch.reshape(N, -1).mean(axis=1)
    fg_frac_mean = float(np.mean(fracs))

    # Coerência espacial (maior componente)
    spatial_coh = "nan"
    try:
        from scipy.ndimage import label as ndimage_label
        lc_fracs = []
        for i in range(N):
            mask = (fg_patch[i] > 0.5).astype(np.uint8)
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

    # Separação FG/BG nos clusters
    fg_sc = cl_scores[cl_is_fg]
    bg_sc = cl_scores[~cl_is_fg]
    margin = safe(float(fg_sc.mean() - bg_sc.mean())) if len(fg_sc) > 0 and len(bg_sc) > 0 else "nan"
    n_ambiguous = int(np.sum((cl_scores > 0.4) & (cl_scores < 0.6)))

    nc_fg = int(cl_is_fg.sum())
    nc_bg = int((~cl_is_fg).sum())

    # Tenta ler métricas GT do json do analisador
    patch_iou_r = patch_iou_c = refine_delta = "nan"
    pixel_iou = oracle_g = oracle_iw = mapping_acc = auc_roc = "nan"
    stem = Path(analyzer_report).stem.replace(".analysis", "")
    json_path = Path(analyzer_report).parent / f"{stem}__quality" / "step3_quality.json"
    if json_path.exists():
        try:
            d = json.loads(json_path.read_text(encoding="utf-8"))
            m = d.get("metrics", {})
            patch_iou_r  = safe(m.get("patch_iou_refined_mean"))
            patch_iou_c  = safe(m.get("patch_iou_coarse_mean"))
            refine_delta = safe(m.get("patch_iou_delta_refined_minus_coarse"))
            pixel_iou    = safe(m.get("pixel_iou_mean"))
            oracle_g     = safe(m.get("oracle_best_global_iou"))
            oracle_iw    = safe(m.get("oracle_imagewise_iou_mean"))
            mapping_acc  = safe(m.get("mapping_accuracy"))
            auc_roc      = safe(m.get("fg_score_auc_roc"))
        except Exception:
            pass

    # Proxy score não supervisionado
    s_frac   = 1.0 - abs(fg_frac_mean - 0.35) / 0.35  # ideal=0.35
    s_frac   = max(0.0, min(1.0, s_frac))
    s_margin = max(0.0, min(1.0, float(margin) if margin != "nan" else 0.0))
    s_coh    = max(0.0, min(1.0, float(spatial_coh) if spatial_coh != "nan" else 0.0))
    s_amb    = max(0.0, 1.0 - n_ambiguous / max(nc_used, 1))
    proxy = 0.30 * s_coh + 0.30 * s_margin + 0.25 * s_frac + 0.15 * s_amb

    step1_stem = Path(step1_npz).stem
    step2_stem = Path(step2_npz).stem

    values = [
        class_name, step1_stem, step2_stem,
        str(n_clusters), k_neighbors, str(fg_threshold), str(saliency_alpha),
        str(N), str(P), str(nc_used), str(nc_fg), str(nc_bg),
        safe(fg_frac_mean), spatial_coh, margin, str(n_ambiguous),
        patch_iou_r, patch_iou_c, refine_delta,
        pixel_iou, oracle_g, oracle_iw, mapping_acc, auc_roc,
        safe(proxy), str(Path(step3_npz)), str(Path(analyzer_report)),
    ]
    print("\t".join(values))
except Exception as e:
    print(f"[ERRO_METRICS] {step3_npz}: {e}", file=sys.stderr)
    raise
PY
}

# Analisador
run_analyzer() {
  local step3_npz="$1" step1_npz="$2" report_file="$3"

  if [[ "$REUSE_ANALYZER_REPORTS" -eq 1 && "$FORCE" -eq 0 && -s "$report_file" ]]; then
    echo "[Analyzer] Reaproveitando: $report_file"; return 0
  fi
  if [[ "$RUN_ANALYZER" -eq 0 ]]; then
    echo "[Analyzer] Desativado." > "$report_file"; return 0
  fi
  if [[ ! -f "$ANALYZER_SCRIPT" ]]; then
    echo "[Analyzer] Script não encontrado." > "$report_file"; return 0
  fi

  local stem
  stem="$(basename "$report_file" .analysis.txt)"
  local out_dir
  out_dir="$(dirname "$report_file")/${stem}__quality"

  local analyzer_args=(
    --cosaliency "$step3_npz"
    --output     "$out_dir"
  )
  [[ -n "$step1_npz" && -f "$step1_npz" ]] && analyzer_args+=(--features "$step1_npz")
  [[ -n "$GT_INDEX" ]]                      && analyzer_args+=(--gt_index "$GT_INDEX")
  [[ "$NO_PADDING_MASK" == "1" ]]           && analyzer_args+=(--no_padding_mask)

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

# Melhor por classe
best_by_cls = {}
for r in rows:
    c = r["class_name"]
    if c not in best_by_cls or float(r["proxy_score"]) > float(best_by_cls[c]["proxy_score"]):
        best_by_cls[c] = r

best_fields = [
    "class_name","step1_stem","step2_stem","n_clusters","k_neighbors","fg_threshold","saliency_alpha",
    "N","P","nc_fg","nc_bg","fg_frac_mean","spatial_coherence","cluster_margin",
    "patch_iou_refined","pixel_iou","oracle_global_iou","mapping_accuracy",
    "auc_roc","proxy_score","step3_npz",
]
with best_tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, delimiter="\t", fieldnames=best_fields)
    w.writeheader()
    for c in sorted(best_by_cls):
        r = best_by_cls[c]
        w.writerow({k: r.get(k,"") for k in best_fields})

# Agrega por configuração E3 (ignora step1/step2 stem — médias sobre todos os pares)
groups = defaultdict(list)
for r in rows:
    key = (r["n_clusters"], r["k_neighbors"], r["fg_threshold"], r["saliency_alpha"])
    groups[key].append(r)

def mean(xs): return sum(xs)/len(xs) if xs else math.nan
def stdev(xs): return stats.pstdev(xs) if len(xs)>1 else 0.0

agg_rows = []
for (nc, kn, fgt, alpha), gr in groups.items():
    classes = sorted({r["class_name"] for r in gr})
    def col(k): return [float(r[k]) for r in gr if safe(r.get(k)) is not None]

    agg_rows.append({
        "n_clusters": nc, "k_neighbors": kn, "fg_threshold": fgt, "saliency_alpha": alpha,
        "classes_done": len(classes), "classes_total": total_cls,
        "coverage": len(classes)/total_cls if total_cls else 0.0,
        "mean_proxy":           mean(col("proxy_score")),
        "std_proxy":            stdev(col("proxy_score")),
        "mean_spatial_coh":     mean(col("spatial_coherence")),
        "mean_cluster_margin":  mean(col("cluster_margin")),
        "mean_fg_frac":         mean(col("fg_frac_mean")),
        "mean_patch_iou":       mean(col("patch_iou_refined")),
        "mean_pixel_iou":       mean(col("pixel_iou")),
        "mean_oracle_global":   mean(col("oracle_global_iou")),
        "mean_oracle_imagewise":mean(col("oracle_imagewise_iou")),
        "mean_mapping_acc":     mean(col("mapping_accuracy")),
        "mean_auc_roc":         mean(col("auc_roc")),
    })

min_cls = total_cls if min_cls_arg == "all" else int(min_cls_arg)
agg_all = sorted(agg_rows, key=lambda r: (r["classes_done"], r["mean_proxy"]), reverse=True)
qualified = [r for r in agg_rows if int(r["classes_done"]) >= min_cls] or agg_rows
qualified.sort(key=lambda r: float(r["mean_proxy"]), reverse=True)
top10 = qualified[:10]

agg_fields = [
    "n_clusters","k_neighbors","fg_threshold","saliency_alpha",
    "classes_done","classes_total","coverage",
    "mean_proxy","std_proxy",
    "mean_spatial_coh","mean_cluster_margin","mean_fg_frac",
    "mean_patch_iou","mean_pixel_iou",
    "mean_oracle_global","mean_oracle_imagewise","mean_mapping_acc","mean_auc_roc",
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
    f.write("# Top 10 agregado — configurações da Etapa 3\n\n")
    f.write("> Ranking pela média de `proxy_score` entre classes.\n\n")
    f.write(f"Classes avaliadas: **{total_cls}**. Mínimo para ranking: **{min_cls}**.\n\n")
    f.write("| # | nc | kn | fgt | alpha | classes | cov | mean_proxy | mean_spatial | mean_margin | mean_patch_iou | mean_pixel_iou | mean_oracle | mean_mapping | mean_auc |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(top10, 1):
        f.write(f"| {i} | {r['n_clusters']} | {r['k_neighbors']} | {r['fg_threshold']} | {r['saliency_alpha']} "
                f"| {r['classes_done']}/{r['classes_total']} | {r['coverage']:.1%} "
                f"| {s(r,'mean_proxy')} | {s(r,'mean_spatial_coh')} | {s(r,'mean_cluster_margin')} "
                f"| {s(r,'mean_patch_iou')} | {s(r,'mean_pixel_iou')} "
                f"| {s(r,'mean_oracle_global')} | {s(r,'mean_mapping_acc')} | {s(r,'mean_auc_roc')} |\n")

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
    e2_npz="${pair_line##*|}"
    e1_stem="$(basename "$e1_npz" .npz)"
    e2_stem="$(basename "$e2_npz" .npz)"

    cls_npz_dir="$NPZ_ROOT/$safe_cls"
    cls_report_dir="$REPORT_ROOT/$safe_cls"
    cls_log_dir="$LOG_ROOT/$safe_cls"
    cls_status_dir="$STATUS_ROOT/$safe_cls"
    mkdir -p "$cls_npz_dir" "$cls_report_dir" "$cls_log_dir" "$cls_status_dir"

    for nc in "${N_CLUSTERS_LIST[@]}"; do
      for kn in "${K_NEIGHBORS_LIST[@]}"; do
        for fgt in "${FG_THRESHOLD_LIST[@]}"; do
        for alpha in "${SALIENCY_ALPHA_LIST[@]}"; do
          COUNT=$((COUNT + 1))

          STEM="${e2_stem}__nc-${nc}__kn-${kn}__fgt-${fgt}__alpha-${alpha}"
          if [[ "$FG_DECISION_MODE" != "saliency" ]]; then
            STEM="${STEM}__fgmode-${FG_DECISION_MODE}__ebw-${E2_BORDER_WEIGHT}__tkf-${TOPK_FRACTION}"
          fi
          FLAT_NPZ="$cls_npz_dir/${STEM}.npz"
          LOG="$cls_log_dir/${STEM}.log"
          REPORT="$cls_report_dir/${STEM}.analysis.txt"
          MARKER="$cls_status_dir/${STEM}.done"

          echo ""
          echo "[$COUNT/$TOTAL_RUNS] $cls :: nc=$nc kn=$kn fgt=$fgt alpha=$alpha"
          echo "  E1: $(basename "$e1_npz")"
          echo "  E2: $(basename "$e2_npz")"

          RUN_START_TS=$(date +%s)
          RUN_STATUS="ok"

          EXISTING_VALID=0
          if [[ "$RESUME" -eq 1 && "$FORCE" -eq 0 && -f "$FLAT_NPZ" ]]; then
            if npz_is_valid_e3 "$FLAT_NPZ"; then
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

            cmd=("$PYTHON_BIN" "$STEP3_SCRIPT"
              --features       "$e1_npz"
              --correspondence "$e2_npz"
              --output         "$TMP_DIR"
              --n_clusters     "$nc"
              --k_neighbors    "$kn"
              --fg_threshold   "$fgt"
              --saliency_alpha "$alpha"
              --fg_decision_mode "$FG_DECISION_MODE")
            if [[ "$FG_DECISION_MODE" == "e2_border_topk" ]]; then
              cmd+=(--e2_border_weight "$E2_BORDER_WEIGHT" --topk_fraction "$TOPK_FRACTION")
            fi
            [[ "$VIZ" -eq 1 ]] && cmd+=(--visualize --viz_max "$VIZ_MAX")

            echo "[Run] ${cmd[*]}"
            set +e
            "${cmd[@]}" > "$LOG" 2>&1
            status=$?
            set -e

            if [[ "$status" -ne 0 ]]; then
              echo "[ERRO] Etapa 3 falhou. Veja: $LOG" >&2
              RUN_STATUS="failed"
            elif [[ ! -f "$TMP_DIR/cosaliency.npz" ]]; then
              echo "[ERRO] cosaliency.npz não encontrado em $TMP_DIR" >&2
              RUN_STATUS="failed_missing_npz"
            else
              cp "$TMP_DIR/cosaliency.npz" "$FLAT_NPZ"
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
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
              "$COUNT" "$TOTAL_RUNS" "$cls" "$e1_stem" "$e2_stem" "$nc" "$kn" "$fgt" "$alpha" \
              "$RUN_STATUS" "$(now_human)" "$(now_human)" "$RUN_ELAPSED" \
              "$(format_duration "$RUN_ELAPSED")" "$AVG_SEC" "$REM" "$EST" \
              "$(format_duration "$EST")" "$(eta_human "$EST")" \
              "$FLAT_NPZ" "$LOG" "$REPORT" >> "$TIMING_TSV"
            continue
          fi

          run_analyzer "$FLAT_NPZ" "$e1_npz" "$REPORT"
          compute_step3_metrics \
            "$cls" "$e1_npz" "$e2_npz" \
            "$nc" "$kn" "$fgt" "$alpha" \
            "$FLAT_NPZ" "$REPORT" >> "$METRICS_TSV"
          write_done_marker "$MARKER" "$FLAT_NPZ" "$RUN_STATUS"

          RUN_END_TS=$(date +%s)
          RUN_ELAPSED=$((RUN_END_TS-RUN_START_TS))
          SWEEP_ELAPSED=$((RUN_END_TS-SWEEP_START_TS))
          AVG_SEC=$(awk -v e="$SWEEP_ELAPSED" -v c="$COUNT" 'BEGIN{if(c>0){printf "%.2f",e/c}else{printf "0.00"}}')
          REM=$((TOTAL_RUNS-COUNT))
          EST=$(awk -v a="$AVG_SEC" -v r="$REM" 'BEGIN{printf "%d",a*r}')
          PCT=$(awk -v c="$COUNT" -v t="$TOTAL_RUNS" 'BEGIN{if(t>0){printf "%.2f",100*c/t}else{printf "0.00"}}')

          printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$COUNT" "$TOTAL_RUNS" "$cls" "$e1_stem" "$e2_stem" "$nc" "$kn" "$fgt" "$alpha" \
            "$RUN_STATUS" "$(now_human)" "$(now_human)" "$RUN_ELAPSED" \
            "$(format_duration "$RUN_ELAPSED")" "$AVG_SEC" "$REM" "$EST" \
            "$(format_duration "$EST")" "$(eta_human "$EST")" \
            "$FLAT_NPZ" "$LOG" "$REPORT" >> "$TIMING_TSV"

          echo "[Tempo] $(format_duration "$RUN_ELAPSED") | $COUNT/$TOTAL_RUNS (${PCT}%) | ETA: $(eta_human "$EST")"
        done
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
.npz Etapa 3             : $NPZ_ROOT
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