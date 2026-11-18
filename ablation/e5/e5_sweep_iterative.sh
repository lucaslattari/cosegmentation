#!/usr/bin/env bash
set -Eeuo pipefail
export LC_NUMERIC=C

# =============================================================================
# Sweep da Etapa 5 — Refinamento Iterativo (SAM 2)
#
# Itera sobre pares (features.npz E1, refined_masks.npz E4 — já na config
# vencedora fixada) e roda step5_iterative.py + analyze_step5_iterative.py
# para cada combinação de (par_e1_e4, n_fg_pts, n_bg_pts).
#
# IMPORTANTE — diferente dos sweeps anteriores: max_iter e converge_thresh
# NÃO são varridos aqui. Cada execução já roda com max_iter alto e
# converge_thresh alto (quase nunca para cedo), salvando mask_history (uma
# máscara por iteração — requer step5_iterative.py com save_mask_history,
# o default atual). O analyze_step5_iterative.py então calcula, a partir
# dessa ÚNICA execução, o IoU contra GT em cada iteração e reporta a melhor
# — o que equivale a ter testado qualquer max_iter/converge_thresh menor,
# sem rodar o SAM de novo. Só n_fg_pts/n_bg_pts precisam de execuções
# separadas de verdade, porque mudam os prompts em si, não só quando parar.
#
# Estrutura esperada de entrada:
#   <step1_npz_root>/<classe>/<stem_e1>.npz
#   <step4_npz_root>/<classe>/<stem_e4>.npz   (refined_masks.npz da Etapa 4,
#                                               já filtrado para a config
#                                               vencedora via STEP4_CONFIG_FILTER)
#
# Uso básico:
#   chmod +x run_step5_iterative_sweep.sh
#   ./run_step5_iterative_sweep.sh \
#     -1 ./features_sweep_6classes/npz \
#     -4 ./sam_sweep_6classes_stage2/npz \
#     -s ./pipeline/step5_iterative.py \
#     -a ./ablation/analyze_step5_iterative.py \
#     -o ./step5_sweep \
#     -g ./gt_index.json \
#     --step4_filter "pm-box_points_mask__ni-2__m-sam2.1-hiera-small"
#
# Variáveis de ambiente úteis:
#   PYTHON_BIN=python3
#   N_FG_PTS_STR="3 5 8"
#   N_BG_PTS_STR="3 5 8"
#   MAX_ITER=6                 # teto de iterações por execução (fixo, generoso)
#   CONVERGE_THRESH=0.99       # alto o suficiente pra quase nunca parar cedo
#   PROMPT_MODE_STR="box_points_mask"   # deve bater com o vencedor da Etapa 4
#   SAM_MODEL_STR="facebook/sam2.1-hiera-small"  # idem — deve bater com o vencedor real da E4
#   STEP1_CONFIG_FILTER=""     # substring para filtrar .npz da E1
#   STEP4_CONFIG_FILTER=""     # substring para filtrar .npz da E4
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
STEP4_NPZ_ROOT="./sam_sweep/npz"
STEP5_SCRIPT="./step5_iterative.py"
ANALYZER_SCRIPT="./analyze_step5_iterative.py"
OUT_ROOT="./step5_sweep"
GT_INDEX="${GT_INDEX:-}"
MAX_CLASSES=""
CLASS_REGEX=""
STEP1_CONFIG_FILTER="${STEP1_CONFIG_FILTER:-}"
STEP4_CONFIG_FILTER="${STEP4_CONFIG_FILTER:-}"
FORCE=0
VIZ=0
VIZ_MAX=8
EXPORT=0
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ANALYZER="${RUN_ANALYZER:-1}"
RESUME="${RESUME:-1}"
VALIDATE_EXISTING="${VALIDATE_EXISTING:-1}"
REUSE_ANALYZER_REPORTS="${REUSE_ANALYZER_REPORTS:-1}"
MAX_ITER="${MAX_ITER:-6}"
CONVERGE_THRESH="${CONVERGE_THRESH:-0.99}"

usage() {
  cat <<USAGE
Uso:
  $0 -1 <npz_e1> -4 <npz_e4> [opções]

Opções:
  -1  Pasta raiz dos .npz da Etapa 1. Default: ./features_sweep_selected/npz
  -4  Pasta raiz dos .npz da Etapa 4 (refined_masks.npz). Default: ./sam_sweep/npz
  -s  Script da etapa 5. Default: ./step5_iterative.py
  -a  Script analisador. Default: ./analyze_step5_iterative.py
  -o  Pasta raiz de saída. Default: ./step5_sweep
  -g  Caminho para gt_index.json (OBRIGATÓRIO — analyze_step5_iterative.py
      não roda sem GT, essa etapa só faz sentido com métrica real).
  -n  Máximo de classes (teste rápido). Ex.: -n 2
  -k  Regex para filtrar classes. Ex.: -k 'Panda|Bear'
  -f  Força reprocessamento
  -v  Gera visualizações (--visualize)
  -e  Exporta imagens segmentadas individuais (--export)
  --step1_filter STR   Equivalente a STEP1_CONFIG_FILTER
  --step4_filter STR   Equivalente a STEP4_CONFIG_FILTER
  -h  Ajuda

Variáveis de ambiente úteis:
  PYTHON_BIN=python3
  N_FG_PTS_STR="3 5 8"
  N_BG_PTS_STR="3 5 8"
  MAX_ITER=6
  CONVERGE_THRESH=0.99
  PROMPT_MODE_STR="box_points_mask"
  SAM_MODEL_STR="facebook/sam2.1-hiera-small"
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2"
  STEP4_CONFIG_FILTER="pm-box_points_mask__ni-2__m-sam2.1-hiera-small"
  GT_INDEX=/caminho/gt_index.json
  RUN_ANALYZER=1
  RESUME=1

Exemplo:
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2" \\
  STEP4_CONFIG_FILTER="pm-box_points_mask__ni-2__m-sam2.1-hiera-small" \\
  N_FG_PTS_STR="3 5 8" N_BG_PTS_STR="3 5 8" \\
  $0 -1 ./features_sweep_6classes/npz -4 ./sam_sweep_6classes_stage2/npz \\
     -g ./gt_index.json
USAGE
}

while getopts ":1:4:s:a:o:g:n:k:fveh-:" opt; do
  case "$opt" in
    1) STEP1_NPZ_ROOT="$OPTARG" ;;
    4) STEP4_NPZ_ROOT="$OPTARG" ;;
    s) STEP5_SCRIPT="$OPTARG" ;;
    a) ANALYZER_SCRIPT="$OPTARG" ;;
    o) OUT_ROOT="$OPTARG" ;;
    g) GT_INDEX="$OPTARG" ;;
    n) MAX_CLASSES="$OPTARG" ;;
    k) CLASS_REGEX="$OPTARG" ;;
    f) FORCE=1 ;;
    v) VIZ=1 ;;
    e) EXPORT=1 ;;
    h) usage; exit 0 ;;
    -)
      case "$OPTARG" in
        step1_filter) STEP1_CONFIG_FILTER="${!OPTIND}"; OPTIND=$((OPTIND+1)) ;;
        step4_filter) STEP4_CONFIG_FILTER="${!OPTIND}"; OPTIND=$((OPTIND+1)) ;;
        *) echo "[ERRO] Opção longa inválida: --$OPTARG" >&2; usage; exit 2 ;;
      esac
      ;;
    :) echo "[ERRO] Opção -$OPTARG requer argumento." >&2; usage; exit 2 ;;
    \?) echo "[ERRO] Opção inválida: -$OPTARG" >&2; usage; exit 2 ;;
  esac
done

for d in "$STEP1_NPZ_ROOT" "$STEP4_NPZ_ROOT"; do
  [[ -d "$d" ]] || { echo "[ERRO] Pasta não encontrada: $d" >&2; exit 1; }
done
[[ -f "$STEP5_SCRIPT" ]] || { echo "[ERRO] Script E5 não encontrado: $STEP5_SCRIPT" >&2; exit 1; }

if [[ -z "$GT_INDEX" ]]; then
  echo "[ERRO] -g/GT_INDEX é obrigatório para o sweep da Etapa 5." >&2
  echo "       analyze_step5_iterative.py não roda sem GT — sem ele não há" >&2
  echo "       o que decidir entre configs." >&2
  exit 1
fi

abs() { "$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"; }
STEP1_NPZ_ROOT="$(abs "$STEP1_NPZ_ROOT")"
STEP4_NPZ_ROOT="$(abs "$STEP4_NPZ_ROOT")"
STEP5_SCRIPT="$(abs "$STEP5_SCRIPT")"
ANALYZER_SCRIPT="$(abs "$ANALYZER_SCRIPT")"
mkdir -p "$OUT_ROOT"
OUT_ROOT="$(abs "$OUT_ROOT")"

[[ -f "$GT_INDEX" ]] || { echo "[ERRO] gt_index.json não encontrado: $GT_INDEX" >&2; exit 1; }
GT_INDEX="$(abs "$GT_INDEX")"
echo "[GT] Usando ground truth: $GT_INDEX"

# Filtra classes
CLASSES=()
for cls in "${SELECTED_CLASSES[@]}"; do
  [[ -n "$CLASS_REGEX" ]] && ! echo "$cls" | grep -qiE "$CLASS_REGEX" && continue
  CLASSES+=("$cls")
done
[[ -n "$MAX_CLASSES" ]] && CLASSES=("${CLASSES[@]:0:$MAX_CLASSES}")
[[ "${#CLASSES[@]}" -eq 0 ]] && { echo "[ERRO] Nenhuma classe após filtros." >&2; exit 1; }

# Grade experimental E5 (só n_fg_pts/n_bg_pts — ver nota no topo do arquivo)
read -r -a N_FG_PTS_LIST <<< "${N_FG_PTS_STR:-3 5 8}"
read -r -a N_BG_PTS_LIST <<< "${N_BG_PTS_STR:-3 5 8}"
read -r -a PROMPT_MODES  <<< "${PROMPT_MODE_STR:-box_points_mask}"
read -r -a SAM_MODELS    <<< "${SAM_MODEL_STR:-facebook/sam2.1-hiera-small}"

# Utilitários
sanitize() {
  local s="$1"
  s="${s//\\//_}"; s="${s//\//_}"; s="${s// /_}"
  printf '%s' "$s" | tr -cd '[:alnum:]_.-'
}
sanitize_model() {
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

npz_is_valid_e5() {
  local npz="$1"
  [[ -f "$npz" ]] || return 1
  [[ "$VALIDATE_EXISTING" -eq 0 ]] && return 0
  "$PYTHON_BIN" - "$npz" <<'PY' >/dev/null 2>&1
import sys, numpy as np
data = np.load(sys.argv[1], allow_pickle=True)
for k in ["final_masks","iou_scores","n_iters_run","converged","image_paths","image_sizes"]:
    if k not in data: raise ValueError(f"missing {k}")
if "mask_history" not in data:
    raise ValueError("missing mask_history — rodado com versão antiga do step5_iterative.py")
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

printf "class_name\tstep1_stem\tstep4_stem\tn_fg_pts\tn_bg_pts\tN\tn_iters_available\tbest_iter\tbaseline_gt_iou\tbest_gt_iou\tgain_over_baseline\tconverged\tstep5_npz\tanalyzer_report\n" > "$METRICS_TSV"
printf "run_index\ttotal_runs\tclass_name\tstep1_stem\tstep4_stem\tn_fg_pts\tn_bg_pts\tstatus\tstarted_at\tfinished_at\telapsed_sec\telapsed_hms\tavg_sec\tremaining_runs\test_sec\test_hms\test_finish\tstep5_npz\tlog\tanalyzer_report\n" > "$TIMING_TSV"

# Descobre pares (e1_npz, e4_npz) por classe — mesma convenção das etapas
# anteriores: o e4_base deve começar com o e1_base como prefixo.
declare -A CLASS_PAIRS_MAP

for cls in "${CLASSES[@]}"; do
  safe_cls="$(sanitize "$cls")"
  e1_dir="$STEP1_NPZ_ROOT/$safe_cls"
  e4_dir="$STEP4_NPZ_ROOT/$safe_cls"
  [[ -d "$e1_dir" && -d "$e4_dir" ]] || continue

  pairs=()
  while IFS= read -r -d '' e1_npz; do
    e1_base="$(basename "$e1_npz" .npz)"
    [[ -n "$STEP1_CONFIG_FILTER" ]] && ! echo "$e1_base" | grep -q "$STEP1_CONFIG_FILTER" && continue
    # Mesmo ajuste feito no sweep da Etapa 4: o E1 pode ter o prefixo
    # "<Classe>__" no stem (redundante, já que o arquivo está na pasta da
    # classe) enquanto o E4 não repete esse prefixo (STEM da E4 vem do
    # e3_stem, que também não tem prefixo de classe). Tira o prefixo de
    # ambos antes de comparar, em vez de exigir e1_base inteiro como
    # prefixo literal do e4_base.
    e1_match="$e1_base"
    [[ "$e1_match" == "${safe_cls}__"* ]] && e1_match="${e1_match#${safe_cls}__}"
    while IFS= read -r -d '' e4_npz; do
      e4_base="$(basename "$e4_npz" .npz)"
      [[ -n "$STEP4_CONFIG_FILTER" ]] && ! echo "$e4_base" | grep -q "$STEP4_CONFIG_FILTER" && continue
      e4_match="$e4_base"
      [[ "$e4_match" == "${safe_cls}__"* ]] && e4_match="${e4_match#${safe_cls}__}"
      [[ "$e4_match" == "${e1_match}"* ]] || continue
      pairs+=("${e1_npz}|${e4_npz}")
    done < <(find "$e4_dir" -maxdepth 1 -name "*.npz" -print0 | sort -z)
  done < <(find "$e1_dir" -maxdepth 1 -name "*.npz" -print0 | sort -z)

  if [[ "${#pairs[@]}" -eq 0 ]]; then
    echo "[Aviso] Nenhum par E1/E4 para '$cls'" >&2; continue
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
  TOTAL_RUNS=$(( TOTAL_RUNS + n_pairs * ${#N_FG_PTS_LIST[@]} * ${#N_BG_PTS_LIST[@]} * ${#PROMPT_MODES[@]} * ${#SAM_MODELS[@]} ))
done

TOTAL_CLASSES="${#CLASSES[@]}"
MIN_CLASSES_FOR_AGGREGATE="${MIN_CLASSES_FOR_AGGREGATE:-all}"
COUNT=0
SWEEP_START_TS=$(date +%s)

cat <<INFO
============================================================
Sweep Etapa 5 — Refinamento Iterativo
Pasta .npz Etapa 1     : $STEP1_NPZ_ROOT
Pasta .npz Etapa 4     : $STEP4_NPZ_ROOT
Ground truth (GT)      : $GT_INDEX
Filtro E1              : ${STEP1_CONFIG_FILTER:-nenhum}
Filtro E4              : ${STEP4_CONFIG_FILTER:-nenhum}
Script etapa 5         : $STEP5_SCRIPT
Analyzer               : $ANALYZER_SCRIPT
Saída                  : $OUT_ROOT
Classes                : $TOTAL_CLASSES
Total execuções        : $TOTAL_RUNS
n_fg_pts               : ${N_FG_PTS_LIST[*]}
n_bg_pts               : ${N_BG_PTS_LIST[*]}
prompt_mode (fixo)     : ${PROMPT_MODES[*]}
modelo SAM (fixo)      : ${SAM_MODELS[*]}
max_iter (fixo, alto)  : $MAX_ITER
converge_thresh (fixo) : $CONVERGE_THRESH
Início                 : $(now_human)
Resume                 : $RESUME
============================================================
INFO

# Métricas: lê o step5_quality.json que o analisador já escreveu (não
# recalcula nada aqui — diferente do padrão de proxy_score não-supervisionado
# das etapas anteriores, essa etapa exige GT desde o início, então usamos
# a métrica real (best_gt_iou) direto como critério de seleção, sem
# intermediário não-supervisionado que a gente já viu falhar várias vezes.
compute_step5_metrics() {
  local class_name="$1" step1_npz="$2" step4_npz="$3"
  local n_fg="$4" n_bg="$5" step5_npz="$6" analyzer_report="$7"

  "$PYTHON_BIN" - \
    "$class_name" "$step1_npz" "$step4_npz" "$n_fg" "$n_bg" \
    "$step5_npz" "$analyzer_report" <<'PY'
import sys, json
from pathlib import Path
import numpy as np

class_name, step1_npz, step4_npz, n_fg, n_bg, step5_npz, analyzer_report = sys.argv[1:]

def safe(v):
    try:
        v = float(v)
        import math
        return f"{v:.6f}" if not (math.isnan(v) or math.isinf(v)) else "nan"
    except Exception:
        return "nan"

try:
    data = np.load(step5_npz, allow_pickle=True)
    N = len(data["image_paths"])
    n_iters_available = int(data["n_iters_run"])
    converged = bool(data["converged"])

    stem = Path(analyzer_report).stem.replace(".analysis", "")
    json_path = Path(analyzer_report).parent / f"{stem}__quality" / "step5_quality.json"

    best_iter = "nan"; baseline = "nan"; best_gt = "nan"; gain = "nan"
    if json_path.exists():
        d = json.loads(json_path.read_text(encoding="utf-8"))
        best_iter = str(d.get("best_iter", "nan"))
        baseline  = safe(d.get("baseline_gt_iou"))
        best_gt   = safe(d.get("best_gt_iou"))
        gain      = safe(d.get("gain_over_baseline"))

    step1_stem = Path(step1_npz).stem
    step4_stem = Path(step4_npz).stem

    values = [
        class_name, step1_stem, step4_stem, str(n_fg), str(n_bg),
        str(N), str(n_iters_available), best_iter, baseline, best_gt, gain,
        str(converged), str(Path(step5_npz)), str(Path(analyzer_report)),
    ]
    print("\t".join(values))
except Exception as e:
    print(f"[ERRO_METRICS] {step5_npz}: {e}", file=sys.stderr)
    raise
PY
}

# Analisador (escrita atômica — mesma proteção contra queda de energia das
# outras etapas)
run_analyzer() {
  local step5_npz="$1" report_file="$2"

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
    --iterative  "$step5_npz"
    --output     "$out_dir"
    --gt_index   "$GT_INDEX"
  )

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
rows = [r for r in rows if safe(r.get("best_gt_iou")) is not None]

# Melhor por classe — critério é a métrica REAL (best_gt_iou), não um proxy
# não-supervisionado. Essa etapa só roda com GT disponível, então não há
# motivo pra passar por um proxy intermediário que a gente já viu errar
# repetidas vezes nas etapas anteriores.
best_by_cls = {}
for r in rows:
    c = r["class_name"]
    if c not in best_by_cls or float(r["best_gt_iou"]) > float(best_by_cls[c]["best_gt_iou"]):
        best_by_cls[c] = r

best_fields = [
    "class_name","step1_stem","step4_stem","n_fg_pts","n_bg_pts",
    "N","n_iters_available","best_iter","baseline_gt_iou","best_gt_iou",
    "gain_over_baseline","converged","step5_npz",
]
with best_tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, delimiter="\t", fieldnames=best_fields)
    w.writeheader()
    for c in sorted(best_by_cls):
        r = best_by_cls[c]
        w.writerow({k: r.get(k,"") for k in best_fields})

# Agrega por (n_fg_pts, n_bg_pts)
groups = defaultdict(list)
for r in rows:
    key = (r["n_fg_pts"], r["n_bg_pts"])
    groups[key].append(r)

def mean(xs): return sum(xs)/len(xs) if xs else math.nan
def stdev(xs): return stats.pstdev(xs) if len(xs)>1 else 0.0

agg_rows = []
for (fg, bg), gr in groups.items():
    classes = sorted({r["class_name"] for r in gr})
    def col(k): return [float(r[k]) for r in gr if safe(r.get(k)) is not None]

    agg_rows.append({
        "n_fg_pts": fg, "n_bg_pts": bg,
        "classes_done": len(classes), "classes_total": total_cls,
        "coverage": len(classes)/total_cls if total_cls else 0.0,
        "mean_best_gt_iou":    mean(col("best_gt_iou")),
        "std_best_gt_iou":     stdev(col("best_gt_iou")),
        "min_best_gt_iou":     min(col("best_gt_iou")) if col("best_gt_iou") else math.nan,
        "mean_gain":           mean(col("gain_over_baseline")),
        "mean_best_iter":      mean([float(r["best_iter"]) for r in gr
                                     if r.get("best_iter") not in (None,"","nan")]),
    })

min_cls = total_cls if min_cls_arg == "all" else int(min_cls_arg)
agg_all = sorted(agg_rows, key=lambda r: (r["classes_done"], r["mean_best_gt_iou"]), reverse=True)
qualified = [r for r in agg_rows if int(r["classes_done"]) >= min_cls] or agg_rows
qualified.sort(key=lambda r: float(r["mean_best_gt_iou"]), reverse=True)
top10 = qualified[:10]

agg_fields = [
    "n_fg_pts","n_bg_pts",
    "classes_done","classes_total","coverage",
    "mean_best_gt_iou","std_best_gt_iou","min_best_gt_iou",
    "mean_gain","mean_best_iter",
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
    f.write("# Top 10 agregado — configurações da Etapa 5 (Refinamento Iterativo)\n\n")
    f.write("> Ranking pela MÉDIA de `best_gt_iou` (métrica real, com GT) entre as classes. "
            "`best_iter` médio indica quantas iterações de fato compensam.\n\n")
    f.write(f"Classes avaliadas: **{total_cls}**. Mínimo para ranking: **{min_cls}**.\n\n")
    f.write("| # | n_fg_pts | n_bg_pts | classes | cov | mean_gt_iou | min_gt_iou | mean_gain | mean_best_iter |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(top10, 1):
        f.write(f"| {i} | {r['n_fg_pts']} | {r['n_bg_pts']} "
                f"| {r['classes_done']}/{r['classes_total']} | {r['coverage']:.1%} "
                f"| {s(r,'mean_best_gt_iou')} | {s(r,'min_best_gt_iou')} "
                f"| {s(r,'mean_gain')} | {s(r,'mean_best_iter')} |\n")

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
    e4_npz="${pair_line##*|}"
    e1_stem="$(basename "$e1_npz" .npz)"
    e4_stem="$(basename "$e4_npz" .npz)"

    cls_npz_dir="$NPZ_ROOT/$safe_cls"
    cls_report_dir="$REPORT_ROOT/$safe_cls"
    cls_log_dir="$LOG_ROOT/$safe_cls"
    cls_status_dir="$STATUS_ROOT/$safe_cls"
    mkdir -p "$cls_npz_dir" "$cls_report_dir" "$cls_log_dir" "$cls_status_dir"

    for prompt_mode in "${PROMPT_MODES[@]}"; do
      for sam_model in "${SAM_MODELS[@]}"; do
        model_short="$(sanitize_model "$sam_model")"
        for n_fg in "${N_FG_PTS_LIST[@]}"; do
          for n_bg in "${N_BG_PTS_LIST[@]}"; do
            COUNT=$((COUNT + 1))

            STEM="${e4_stem}__fgpts-${n_fg}__bgpts-${n_bg}"
            FLAT_NPZ="$cls_npz_dir/${STEM}.npz"
            LOG="$cls_log_dir/${STEM}.log"
            REPORT="$cls_report_dir/${STEM}.analysis.txt"
            MARKER="$cls_status_dir/${STEM}.done"

            echo ""
            echo "[$COUNT/$TOTAL_RUNS] $cls :: n_fg_pts=$n_fg n_bg_pts=$n_bg"
            echo "  E1: $(basename "$e1_npz")"
            echo "  E4: $(basename "$e4_npz")"

            RUN_START_TS=$(date +%s)
            RUN_STATUS="ok"

            EXISTING_VALID=0
            if [[ "$RESUME" -eq 1 && "$FORCE" -eq 0 && -f "$FLAT_NPZ" ]]; then
              if npz_is_valid_e5 "$FLAT_NPZ"; then
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

              cmd=("$PYTHON_BIN" "$STEP5_SCRIPT"
                --features       "$e1_npz"
                --refined        "$e4_npz"
                --output         "$TMP_DIR"
                --model          "$sam_model"
                --prompt_mode    "$prompt_mode"
                --max_iter       "$MAX_ITER"
                --converge_thresh "$CONVERGE_THRESH"
                --n_fg_pts       "$n_fg"
                --n_bg_pts       "$n_bg")
              [[ "$VIZ" -eq 1 ]]    && cmd+=(--visualize --viz_max "$VIZ_MAX")
              [[ "$EXPORT" -eq 1 ]] && cmd+=(--export)

              echo "[Run] ${cmd[*]}"
              set +e
              "${cmd[@]}" > "$LOG" 2>&1
              status=$?
              set -e

              if [[ "$status" -ne 0 ]]; then
                echo "[ERRO] Etapa 5 falhou. Veja: $LOG" >&2
                RUN_STATUS="failed"
              elif [[ ! -f "$TMP_DIR/iterative_masks.npz" ]]; then
                echo "[ERRO] iterative_masks.npz não encontrado em $TMP_DIR" >&2
                RUN_STATUS="failed_missing_npz"
              else
                cp "$TMP_DIR/iterative_masks.npz" "$FLAT_NPZ"
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
              printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$COUNT" "$TOTAL_RUNS" "$cls" "$e1_stem" "$e4_stem" "$n_fg" "$n_bg" \
                "$RUN_STATUS" "$(now_human)" "$(now_human)" "$RUN_ELAPSED" \
                "$(format_duration "$RUN_ELAPSED")" "$AVG_SEC" "$REM" "$EST" \
                "$(format_duration "$EST")" "$(eta_human "$EST")" \
                "$FLAT_NPZ" "$LOG" >> "$TIMING_TSV"
              continue
            fi

            run_analyzer "$FLAT_NPZ" "$REPORT"
            compute_step5_metrics \
              "$cls" "$e1_npz" "$e4_npz" "$n_fg" "$n_bg" \
              "$FLAT_NPZ" "$REPORT" >> "$METRICS_TSV"
            write_done_marker "$MARKER" "$FLAT_NPZ" "$RUN_STATUS"

            RUN_END_TS=$(date +%s)
            RUN_ELAPSED=$((RUN_END_TS-RUN_START_TS))
            SWEEP_ELAPSED=$((RUN_END_TS-SWEEP_START_TS))
            AVG_SEC=$(awk -v e="$SWEEP_ELAPSED" -v c="$COUNT" 'BEGIN{if(c>0){printf "%.2f",e/c}else{printf "0.00"}}')
            REM=$((TOTAL_RUNS-COUNT))
            EST=$(awk -v a="$AVG_SEC" -v r="$REM" 'BEGIN{printf "%d",a*r}')
            PCT=$(awk -v c="$COUNT" -v t="$TOTAL_RUNS" 'BEGIN{if(t>0){printf "%.2f",100*c/t}else{printf "0.00"}}')

            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
              "$COUNT" "$TOTAL_RUNS" "$cls" "$e1_stem" "$e4_stem" "$n_fg" "$n_bg" \
              "$RUN_STATUS" "$(now_human)" "$(now_human)" "$RUN_ELAPSED" \
              "$(format_duration "$RUN_ELAPSED")" "$AVG_SEC" "$REM" "$EST" \
              "$(format_duration "$EST")" "$(eta_human "$EST")" \
              "$FLAT_NPZ" "$LOG" >> "$TIMING_TSV"

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
.npz Etapa 5              : $NPZ_ROOT
Relatórios analisador     : $REPORT_ROOT
Logs                      : $LOG_ROOT
Métricas completas        : $METRICS_TSV
Agregado por config       : $AGG_TSV
Top 10 TSV                : $TOP10_TSV
Top 10 Markdown           : $TOP10_MD
Melhor config por classe  : $BEST_BY_CLASS_TSV
Tempos                    : $TIMING_TSV
============================================================
DONE
