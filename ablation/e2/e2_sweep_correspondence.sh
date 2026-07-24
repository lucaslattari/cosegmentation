#!/usr/bin/env bash
set -Eeuo pipefail
# Força separador decimal como ponto em todas as chamadas awk/printf,
# independente do locale do sistema (necessário em pt_BR).
export LC_NUMERIC=C

# =============================================================================
# Sweep da Etapa 2 — Cross-Image Correspondence
#
# Itera sobre os .npz da Etapa 1 (produzidos pelo run_step1_feature_sweep_selected_classes.sh)
# e roda step2_correspondence.py + analyze_step2_correspondence.py para cada
# combinação de (npz_etapa1, top_k, temperature).
#
# A entrada esperada é a pasta de saída do sweep da Etapa 1, especificamente
# os arquivos em <step1_out>/npz/<classe>/*.npz.
# Por padrão, processa as mesmas 16 classes selecionadas.
# Um filtro de configuração (-c) pode restringir quais .npz da Etapa 1 usar.
#
# NOVO: quando GT_INDEX é fornecido, o analisador da Etapa 2 também testa
# heurísticas alternativas de orientação FG/BG (size_based, spatial_based,
# combined) contra a saliência CLS usada em produção. Os resultados entram
# em metrics.tsv (colunas orientation_accuracy_*), no best_by_class.tsv, e
# num relatório dedicado orientation_heuristics_summary.md que agrega, por
# classe, qual heurística venceu e por quanto.
#
# Uso básico:
#   chmod +x run_step2_correspondence_sweep.sh
#   ./run_step2_correspondence_sweep.sh \
#     -i ./features_sweep_selected/npz \
#     -s ./step2_correspondence.py \
#     -a ./analyze_step2_correspondence.py \
#     -o ./correspondence_sweep
#
# Variáveis de ambiente úteis:
#   PYTHON_BIN=python3
#   TOP_K_STR="5 10 15"          # valores de --top_k a testar
#   TEMPERATURE_STR="0.02 0.05 0.1"  # valores de --temperature a testar
#   STEP1_CONFIG_FILTER=""       # substring para filtrar .npz (ex: "vits14__img-518__resize-keep_ar__layer-last__feat-l2")
#   RUN_ANALYZER=1
#   GT_INDEX=/caminho/gt_index.json     # passado ao analisador como --gt_index
#   RESUME=1
#   VALIDATE_EXISTING=1
#   REUSE_ANALYZER_REPORTS=1
#   MIN_CLASSES_FOR_AGGREGATE=all | número
# =============================================================================

# ----------------------------------------------------------------------------
# 16 classes selecionadas — mesma lista do sweep da Etapa 1
# ----------------------------------------------------------------------------
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
# Defaults
# ----------------------------------------------------------------------------
STEP1_NPZ_ROOT="./features_sweep_selected/npz"
STEP2_SCRIPT="./step2_correspondence.py"
ANALYZER_SCRIPT="./analyze_step2_correspondence.py"
OUT_ROOT="./correspondence_sweep"
MAX_CLASSES=""
CLASS_REGEX=""
STEP1_CONFIG_FILTER="${STEP1_CONFIG_FILTER:-}"
FORCE=0
VIZ=0
VIZ_MAX=8
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ANALYZER="${RUN_ANALYZER:-1}"
GT_INDEX="${GT_INDEX:-}"
NO_PADDING_MASK="${NO_PADDING_MASK:-0}"
RESUME="${RESUME:-1}"
VALIDATE_EXISTING="${VALIDATE_EXISTING:-1}"
REUSE_ANALYZER_REPORTS="${REUSE_ANALYZER_REPORTS:-1}"

usage() {
  cat <<USAGE
Uso:
  $0 -i <pasta_npz_etapa1> [opções]

Opções:
  -i  Pasta raiz dos .npz da Etapa 1. Default: ./features_sweep_selected/npz
      Esperado: <pasta>/<classe>/<stem>.npz
  -s  Script da etapa 2. Default: ./step2_correspondence.py
  -a  Script analisador. Default: ./analyze_step2_correspondence.py
  -o  Pasta raiz de saída. Default: ./correspondence_sweep
  -g  Caminho para gt_index.json. Passado ao analisador como --gt_index.
      Também pode ser definido via variável de ambiente GT_INDEX.
      Com GT, o analisador também testa heurísticas alternativas de
      orientação FG/BG (veja orientation_heuristics_summary.md na saída).
  -n  Máximo de classes a processar (teste rápido). Ex.: -n 3
  -k  Regex para filtrar classes. Ex.: -k 'Panda|Bear'
  -f  Força reprocessamento mesmo se o .npz já existir
  -v  Gera visualizações do step2 (--visualize)
  -h  Ajuda

Variáveis de ambiente úteis:
  PYTHON_BIN=python3
  TOP_K_STR="5 10 15"
  TEMPERATURE_STR="0.02 0.05 0.1"
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2"
  GT_INDEX=/caminho/gt_index.json
  NO_PADDING_MASK=0   # 1 desativa a exclusão de padding nas métricas do analisador
  RUN_ANALYZER=1
  RESUME=1
  VALIDATE_EXISTING=1
  REUSE_ANALYZER_REPORTS=1
  MIN_CLASSES_FOR_AGGREGATE=all | número

Exemplos:
  # Teste rápido: 2 classes, uma config E1, top_k=10, temperature=0.05, com GT
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2" \\
  TOP_K_STR="10" TEMPERATURE_STR="0.05" \\
  $0 -i ./features_sweep_selected/npz -g ./gt_index.json -n 2

  # Sweep completo com GT (inclui teste de heurísticas de orientação)
  STEP1_CONFIG_FILTER="vits14__img-518__resize-keep_ar__layer-last__feat-l2" \\
  TOP_K_STR="5 10 15" TEMPERATURE_STR="0.02 0.05 0.1" \\
  $0 -i ./features_sweep_selected/npz -g ./gt_index.json
USAGE
}

while getopts ":i:s:a:o:g:n:k:fvh" opt; do
  case "$opt" in
    i) STEP1_NPZ_ROOT="$OPTARG" ;;
    s) STEP2_SCRIPT="$OPTARG" ;;
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

if [[ ! -d "$STEP1_NPZ_ROOT" ]]; then
  echo "[ERRO] Pasta de .npz da Etapa 1 não encontrada: $STEP1_NPZ_ROOT" >&2
  echo "       Rode run_step1_feature_sweep_selected_classes.sh primeiro." >&2
  exit 1
fi
if [[ ! -f "$STEP2_SCRIPT" ]]; then
  echo "[ERRO] Script da etapa 2 não encontrado: $STEP2_SCRIPT" >&2
  exit 1
fi

STEP1_NPZ_ROOT="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$STEP1_NPZ_ROOT")"
STEP2_SCRIPT="$($PYTHON_BIN -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$STEP2_SCRIPT")"
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

# ----------------------------------------------------------------------------
# Filtra lista de classes
# ----------------------------------------------------------------------------
CLASSES=()
for cls in "${SELECTED_CLASSES[@]}"; do
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
  echo "[ERRO] Nenhuma classe restou após os filtros." >&2; exit 1
fi

# ----------------------------------------------------------------------------
# Grade experimental da Etapa 2
# ----------------------------------------------------------------------------
read -r -a TOP_KS      <<< "${TOP_K_STR:-5 10 15}"
read -r -a TEMPERATURES <<< "${TEMPERATURE_STR:-0.02 0.05 0.10}"

# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
sanitize() {
  local s="$1"
  s="${s//\\//_}"; s="${s//\//_}"; s="${s// /_}"
  s="$(printf '%s' "$s" | tr -cd '[:alnum:]_.-')"
  printf '%s' "$s"
}

format_duration() {
  local t="${1:-0}"; [[ "$t" -lt 0 ]] && t=0
  printf "%02d:%02d:%02d" $((t/3600)) $(((t%3600)/60)) $((t%60))
}

now_human()  { date "+%H:%M:%S do dia %d/%m/%Y"; }

eta_human() {
  local rem="${1:-0}"; [[ "$rem" -lt 0 ]] && rem=0
  date -d "@$(( $(date +%s) + rem ))" "+%H:%M do dia %d/%m/%Y"
}

npz_is_valid() {
  local npz="$1"
  [[ -f "$npz" ]] || return 1
  [[ "$VALIDATE_EXISTING" -eq 0 ]] && return 0
  "$PYTHON_BIN" - "$npz" <<'PY' >/dev/null 2>&1
import sys, numpy as np
data = np.load(sys.argv[1], allow_pickle=True)
for k in ["fg_scores","pair_votes","top_pairs","cls_saliency","patch_grid","image_paths"]:
    if k not in data: raise ValueError(f"missing {k}")
PY
}

write_done_marker() {
  local marker="$1" feature_file="$2" status="$3"
  mkdir -p "$(dirname "$marker")"
  { echo "status=$status"; echo "finished_at=$(now_human)"; echo "feature_file=$feature_file"; } \
    > "${marker}.tmp.$$"
  mv -f "${marker}.tmp.$$" "$marker"
}

# ----------------------------------------------------------------------------
# Estrutura de saída
# ----------------------------------------------------------------------------
NPZ_ROOT="$OUT_ROOT/npz"
REPORT_ROOT="$OUT_ROOT/reports"
LOG_ROOT="$OUT_ROOT/logs"
STATUS_ROOT="$OUT_ROOT/status"
mkdir -p "$NPZ_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$STATUS_ROOT"

METRICS_TSV="$OUT_ROOT/metrics.tsv"
AGG_TSV="$OUT_ROOT/aggregate_by_config.tsv"
TOP10_AGG_TSV="$OUT_ROOT/top10_aggregate.tsv"
TOP10_AGG_MD="$OUT_ROOT/top10_aggregate.md"
BEST_BY_CLASS_TSV="$OUT_ROOT/best_by_class.tsv"
TIMING_TSV="$OUT_ROOT/timing.tsv"
ORIENT_SUMMARY_MD="$OUT_ROOT/orientation_heuristics_summary.md"

printf "class_name\tstep1_npz\tstep1_stem\ttop_k\ttemperature\tstep2_npz\tN\tP\tK\tfg_frac_mean\tfg_frac_std\tpolar_fraction_decided\tpolar_bimodality\tpair_agreement_mean\tstability_loo_mae_mean\tcls_corr_mean\tn_neg_corr\tgt_best_iou\tgt_auc_roc\torientation_accuracy\torientation_accuracy_size_based\torientation_accuracy_spatial_based\torientation_accuracy_combined\torientation_best_heuristic\torientation_improvement_over_baseline\tproxy_score\tanalyzer_report\n" > "$METRICS_TSV"
printf "run_index\ttotal_runs\tclass_name\tstep1_stem\ttop_k\ttemperature\tstatus\tstarted_at\tfinished_at\telapsed_sec\telapsed_hms\tavg_sec\tremaining_runs\test_remaining_sec\test_remaining_hms\test_finish_at\tstep2_npz\textract_log\tanalyzer_report\n" > "$TIMING_TSV"

# ----------------------------------------------------------------------------
# Descobre quais .npz da E1 existem para cada classe
# ----------------------------------------------------------------------------
declare -A CLASS_NPZ_MAP   # classe → lista de paths separados por "|"

for cls in "${CLASSES[@]}"; do
  safe_cls="$(sanitize "$cls")"
  cls_npz_dir="$STEP1_NPZ_ROOT/$safe_cls"
  if [[ ! -d "$cls_npz_dir" ]]; then
    echo "[Aviso] Pasta E1 não encontrada para '$cls': $cls_npz_dir" >&2
    continue
  fi

  found_npzs=()
  while IFS= read -r -d '' npz_file; do
    # Aplica filtro de configuração se definido
    if [[ -n "$STEP1_CONFIG_FILTER" ]]; then
      if ! echo "$(basename "$npz_file")" | grep -q "$STEP1_CONFIG_FILTER"; then
        continue
      fi
    fi
    found_npzs+=("$npz_file")
  done < <(find "$cls_npz_dir" -maxdepth 1 -name "*.npz" -print0 | sort -z)

  if [[ "${#found_npzs[@]}" -eq 0 ]]; then
    echo "[Aviso] Nenhum .npz encontrado para '$cls' em $cls_npz_dir" \
         "${STEP1_CONFIG_FILTER:+(filtro: $STEP1_CONFIG_FILTER)}" >&2
    continue
  fi

  CLASS_NPZ_MAP["$cls"]="$(printf '%s|' "${found_npzs[@]}")"
done

# Contagem total de execuções
TOTAL_RUNS=0
for cls in "${CLASSES[@]}"; do
  [[ -z "${CLASS_NPZ_MAP[$cls]+x}" ]] && continue
  IFS='|' read -ra npzs <<< "${CLASS_NPZ_MAP[$cls]}"
  n_npzs=0
  for npz in "${npzs[@]}"; do [[ -n "$npz" ]] && n_npzs=$((n_npzs+1)); done
  TOTAL_RUNS=$(( TOTAL_RUNS + n_npzs * ${#TOP_KS[@]} * ${#TEMPERATURES[@]} ))
done

TOTAL_CLASSES="${#CLASSES[@]}"
MIN_CLASSES_FOR_AGGREGATE="${MIN_CLASSES_FOR_AGGREGATE:-all}"
COUNT=0
SWEEP_START_TS=$(date +%s)
SWEEP_START_HUMAN=$(now_human)

cat <<INFO
============================================================
Sweep Etapa 2 — correspondência cross-image
Pasta .npz Etapa 1     : $STEP1_NPZ_ROOT
Filtro config E1       : ${STEP1_CONFIG_FILTER:-nenhum}
Ground truth (GT)      : ${GT_INDEX:-não fornecido}
Script etapa 2         : $STEP2_SCRIPT
Analyzer               : $ANALYZER_SCRIPT
Saída                  : $OUT_ROOT
Classes a processar    : $TOTAL_CLASSES
Total execuções        : $TOTAL_RUNS
Top-K valores          : ${TOP_KS[*]}
Temperature valores    : ${TEMPERATURES[*]}
Início                 : $SWEEP_START_HUMAN
Resume habilitado      : $RESUME
INFO

# ----------------------------------------------------------------------------
# Extração de métricas do .npz + relatório do analisador
# ----------------------------------------------------------------------------
compute_step2_metrics() {
  local class_name="$1"
  local step1_npz="$2"
  local step1_stem="$3"
  local top_k="$4"
  local temperature="$5"
  local step2_npz="$6"
  local analyzer_report="$7"

  "$PYTHON_BIN" - \
    "$class_name" "$step1_npz" "$step1_stem" "$top_k" "$temperature" \
    "$step2_npz" "$analyzer_report" <<'PY'
import sys, json, math
from pathlib import Path
import numpy as np

class_name, step1_npz, step1_stem, top_k, temperature, step2_npz, analyzer_report = sys.argv[1:]

def safe(v):
    try:
        v = float(v)
        return f"{v:.6f}" if not (math.isnan(v) or math.isinf(v)) else "nan"
    except Exception:
        return "nan"

def iou(pred, gt):
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(inter/union) if union > 0 else 0.0

try:
    data = np.load(step2_npz, allow_pickle=True)
    fg   = data["fg_scores"].astype(np.float32)       # (N, P)
    pv   = data["pair_votes"].astype(np.float32)       # (K, N, P)
    tp   = data["top_pairs"].astype(np.float32)        # (K, 3)
    sal  = data["cls_saliency"].astype(np.float32)     # (N, P)

    N, P = fg.shape
    K    = len(tp)

    # Polarização
    dev = np.abs(fg.flatten() - 0.5)
    polar_decided = float(np.mean(dev > 0.2))
    mean_abs_dev  = float(np.mean(dev))

    flat = fg.flatten()
    mu = float(np.mean(flat)); std = float(np.std(flat))
    if std > 1e-8:
        n = len(flat)
        skew = float(np.mean(((flat-mu)/std)**3))
        kurt = float(np.mean(((flat-mu)/std)**4))
        skew_c = skew * math.sqrt(n*(n-1))/(n-2) if n > 2 else skew
        bimod = (skew_c**2 + 1) / (kurt - 3 + 3 + 3*(n-1)**2/max((n-2)*(n-3),1))
    else:
        bimod = 0.0

    # FG fractions
    frac = fg.mean(axis=1)
    fg_frac_mean = float(np.mean(frac))
    fg_frac_std  = float(np.std(frac))

    # Concordância entre pares
    img_corrs = []
    for i in range(N):
        vecs = []
        for k in range(K):
            pi, pj = int(tp[k,0]), int(tp[k,1])
            if pi == i or pj == i:
                vecs.append(pv[k, i])
        if len(vecs) < 2: continue
        cs = []
        for a in range(len(vecs)):
            for b in range(a+1, len(vecs)):
                sa, sb = float(np.std(vecs[a])), float(np.std(vecs[b]))
                if sa > 1e-6 and sb > 1e-6:
                    cs.append(float(np.corrcoef(vecs[a], vecs[b])[0,1]))
        if cs: img_corrs.append(float(np.mean(cs)))
    pair_agree = float(np.mean(img_corrs)) if img_corrs else float("nan")

    # Estabilidade LOO
    if K >= 3:
        pairs_list = [(int(tp[k,0]),int(tp[k,1]),float(tp[k,2])) for k in range(K)]
        maes = []
        for k_out in range(K):
            rem = [(k,p) for k,p in enumerate(pairs_list) if k != k_out]
            ws  = np.zeros((N,P), dtype=np.float32)
            fw  = np.zeros((N,P), dtype=np.float32)
            for k,(pi,pj,sim) in rem:
                w = max(sim, 0.0)
                fw[pi] += pv[k,pi]*w; fw[pj] += pv[k,pj]*w
                ws[pi] += w;          ws[pj] += w
            fl = np.where(ws>0, fw/ws, 0.5)
            maes.append(float(np.mean(np.abs(fl - fg))))
        stab_mae = float(np.mean(maes))
    else:
        stab_mae = float("nan")

    # Correlação CLS
    corrs = []
    for i in range(N):
        sf, ss = float(np.std(fg[i])), float(np.std(sal[i]))
        if sf > 1e-6 and ss > 1e-6:
            corrs.append(float(np.corrcoef(fg[i], sal[i])[0,1]))
    cls_corr_mean = float(np.mean(corrs)) if corrs else float("nan")
    n_neg_corr    = int(sum(1 for c in corrs if c < -0.1))

    # Proxy score (sem GT): combina sinais não supervisionados
    # Penaliza instabilidade e baixa polarização; premia concordância e correlação CLS
    s_polar = polar_decided                  # [0,1] alto é bom
    s_agree = max(0.0, min(1.0, pair_agree)) if not math.isnan(pair_agree) else 0.0
    s_stab  = max(0.0, 1.0 - stab_mae / 0.2) if not math.isnan(stab_mae) else 0.5
    s_corr  = max(0.0, min(1.0, (cls_corr_mean + 1) / 2)) if not math.isnan(cls_corr_mean) else 0.5
    proxy = 0.35 * s_polar + 0.30 * s_agree + 0.20 * s_stab + 0.15 * s_corr

    # Tenta ler métricas do relatório do analisador
    gt_best_iou = "nan"; gt_auc = "nan"; orient_acc = "nan"
    orient_size = "nan"; orient_spatial = "nan"; orient_combined = "nan"
    orient_best = "n/a"; orient_improve = "nan"
    stem = Path(analyzer_report).stem.replace(".analysis", "")
    json_path = Path(analyzer_report).parent / f"{stem}__quality" / "step2_quality.json"
    # Compatibilidade com execuções antigas que usavam o diretório compartilhado
    for candidate in [json_path,
        Path(analyzer_report).parent.parent / "step2_quality" / "step2_quality.json",
        Path(analyzer_report).parent / "step2_quality" / "step2_quality.json",
    ]:
        if candidate.exists():
            try:
                d = json.loads(candidate.read_text(encoding="utf-8"))
                m = d.get("metrics", {})
                if m.get("gt_best_iou") is not None: gt_best_iou = safe(m["gt_best_iou"])
                if m.get("gt_auc_roc")  is not None: gt_auc      = safe(m["gt_auc_roc"])
                if m.get("orientation_accuracy") is not None: orient_acc = safe(m["orientation_accuracy"])
                if m.get("orientation_accuracy_size_based") is not None:
                    orient_size = safe(m["orientation_accuracy_size_based"])
                if m.get("orientation_accuracy_spatial_based") is not None:
                    orient_spatial = safe(m["orientation_accuracy_spatial_based"])
                if m.get("orientation_accuracy_combined") is not None:
                    orient_combined = safe(m["orientation_accuracy_combined"])
                if m.get("orientation_best_heuristic") is not None:
                    orient_best = str(m["orientation_best_heuristic"])
                if m.get("orientation_improvement_over_baseline") is not None:
                    orient_improve = safe(m["orientation_improvement_over_baseline"])
            except Exception:
                pass
            break

    values = [
        class_name, str(Path(step1_npz)), step1_stem,
        str(top_k), str(temperature), str(Path(step2_npz)),
        str(N), str(P), str(K),
        safe(fg_frac_mean), safe(fg_frac_std),
        safe(polar_decided), safe(bimod),
        safe(pair_agree), safe(stab_mae),
        safe(cls_corr_mean), str(n_neg_corr),
        gt_best_iou, gt_auc, orient_acc,
        orient_size, orient_spatial, orient_combined,
        orient_best, orient_improve,
        safe(proxy), str(Path(analyzer_report)),
    ]
    print("\t".join(values))
except Exception as e:
    print(f"[ERRO_METRICS] {step2_npz}: {e}", file=sys.stderr)
    raise
PY
}

# ----------------------------------------------------------------------------
# Analisador
# ----------------------------------------------------------------------------
run_analyzer() {
  local corr_npz="$1" report_file="$2" step1_npz="${3:-}"

  if [[ "$REUSE_ANALYZER_REPORTS" -eq 1 && "$FORCE" -eq 0 && -s "$report_file" ]]; then
    echo "[Analyzer] Reaproveitando: $report_file"; return 0
  fi
  if [[ "$RUN_ANALYZER" -eq 0 ]]; then
    echo "[Analyzer] Desativado." > "$report_file"; return 0
  fi
  if [[ ! -f "$ANALYZER_SCRIPT" ]]; then
    echo "[Analyzer] Script não encontrado: $ANALYZER_SCRIPT" > "$report_file"; return 0
  fi

  local stem
  stem="$(basename "$report_file" .analysis.txt)"
  local out_dir
  out_dir="$(dirname "$report_file")/${stem}__quality"

  # Monta argumentos do analisador, injetando --gt_index se disponível.
  # Com GT, o analisador também roda o teste de heurísticas alternativas de
  # orientação (comportamento padrão, sem flag extra necessária).
  #
  # --features é ESSENCIAL aqui, não opcional: sem ele o analisador não
  # tem como ler resize_info da Etapa 1, e sem resize_info tanto o GT
  # quanto todas as métricas com padding (polarização, concordância,
  # estabilidade, correlação com CLS, AUC, e principalmente
  # orientation_accuracy) ficam desalinhados ou inflados por padding —
  # ver docstrings de analyze_step2_correspondence.py. Isso NÃO era
  # passado antes, o que quebrava silenciosamente o alinhamento do GT
  # sempre que a Etapa 1 rodou com --resize keep_ar (o default).
  local analyzer_args=(--correspondence "$corr_npz" --output "$out_dir")
  if [[ -n "$step1_npz" ]]; then
    analyzer_args+=(--features "$step1_npz")
  fi
  if [[ -n "$GT_INDEX" ]]; then
    analyzer_args+=(--gt_index "$GT_INDEX")
  fi
  if [[ "$NO_PADDING_MASK" == "1" ]]; then
    analyzer_args+=(--no_padding_mask)
  fi

  # IMPORTANTE (resiliência a queda de energia / interrupção): report_file
  # é o que REUSE_ANALYZER_REPORTS usa para decidir se esta config já foi
  # analisada. Se escrevêssemos diretamente nele durante a execução do
  # Python, uma interrupção no meio da análise deixaria um arquivo
  # não-vazio em disco — no próximo resume, o REUSE_ANALYZER_REPORTS
  # interpretaria isso como "já analisado" e pularia a config
  # silenciosamente, mesmo com o relatório de qualidade incompleto ou
  # ausente. Por isso todo o conteúdo é escrito num arquivo temporário e
  # só promovido (mv atômico) para report_file depois que o processo
  # Python termina — com sucesso ou com falha registrada, nunca no meio
  # do caminho.
  local tmp_report="${report_file}.tmp.$$"
  echo "[Analyzer] $PYTHON_BIN $ANALYZER_SCRIPT ${analyzer_args[*]}" > "$tmp_report"
  set +e
  "$PYTHON_BIN" "$ANALYZER_SCRIPT" "${analyzer_args[@]}" >> "$tmp_report" 2>&1
  local status=$?
  set -e

  mv -f "$tmp_report" "$report_file"

  if [[ "$status" -ne 0 ]]; then
    echo "[AVISO] Analisador falhou para $corr_npz. Veja: $report_file" >&2
  fi
  return 0
}

# ----------------------------------------------------------------------------
# Agregação
# ----------------------------------------------------------------------------
make_aggregate() {
  local total_classes="$1" min_classes="$2"

  "$PYTHON_BIN" - \
    "$METRICS_TSV" "$AGG_TSV" "$TOP10_AGG_TSV" "$TOP10_AGG_MD" \
    "$BEST_BY_CLASS_TSV" "$ORIENT_SUMMARY_MD" "$total_classes" "$min_classes" <<'PY'
import csv, math, statistics as stats, sys
from collections import defaultdict
from pathlib import Path

metrics_tsv, agg_tsv, top10_tsv, top10_md, best_tsv, orient_summary_md = map(Path, sys.argv[1:7])
total_cls   = int(sys.argv[7])
min_cls_arg = sys.argv[8]

def safe(s):
    try:
        v = float(s)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    except Exception:
        return None

with metrics_tsv.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

rows = [r for r in rows if safe(r.get("proxy_score")) is not None]

# Melhor por classe (pela config de maior proxy_score, igual antes)
best_by_cls = {}
for r in rows:
    c = r["class_name"]
    if c not in best_by_cls or float(r["proxy_score"]) > float(best_by_cls[c]["proxy_score"]):
        best_by_cls[c] = r

best_fields = [
    "class_name","step1_stem","top_k","temperature",
    "N","P","K","fg_frac_mean","polar_fraction_decided","pair_agreement_mean",
    "stability_loo_mae_mean","cls_corr_mean","gt_best_iou","gt_auc_roc",
    "orientation_accuracy","orientation_accuracy_size_based",
    "orientation_accuracy_spatial_based","orientation_accuracy_combined",
    "orientation_best_heuristic","orientation_improvement_over_baseline",
    "proxy_score","step2_npz",
]
with best_tsv.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, delimiter="\t", fieldnames=best_fields)
    w.writeheader()
    for c in sorted(best_by_cls):
        r = best_by_cls[c]
        w.writerow({k: r.get(k,"") for k in best_fields})

# Agrega por (step1_stem, top_k, temperature)
groups = defaultdict(list)
for r in rows:
    key = (r["step1_stem"], r["top_k"], r["temperature"])
    groups[key].append(r)

def mean(xs): return sum(xs)/len(xs) if xs else math.nan
def stdev(xs): return stats.pstdev(xs) if len(xs)>1 else 0.0

agg_rows = []
for (stem, tk, temp), gr in groups.items():
    classes = sorted({r["class_name"] for r in gr})
    proxies  = [float(r["proxy_score"]) for r in gr]
    polars   = [float(r["polar_fraction_decided"]) for r in gr if safe(r["polar_fraction_decided"])]
    agrees   = [float(r["pair_agreement_mean"]) for r in gr if safe(r["pair_agreement_mean"])]
    stabs    = [float(r["stability_loo_mae_mean"]) for r in gr if safe(r["stability_loo_mae_mean"])]
    corrs    = [float(r["cls_corr_mean"]) for r in gr if safe(r["cls_corr_mean"])]
    ious     = [float(r["gt_best_iou"]) for r in gr if safe(r["gt_best_iou"])]
    aucs     = [float(r["gt_auc_roc"]) for r in gr if safe(r["gt_auc_roc"])]
    orients  = [float(r["orientation_accuracy"]) for r in gr if safe(r["orientation_accuracy"])]
    orient_improves = [float(r["orientation_improvement_over_baseline"]) for r in gr
                        if safe(r.get("orientation_improvement_over_baseline"))]
    n_alt_wins = sum(
        1 for r in gr
        if r.get("orientation_best_heuristic") not in (None, "", "n/a", "cls_saliency")
        and safe(r.get("orientation_improvement_over_baseline")) is not None
        and safe(r.get("orientation_improvement_over_baseline")) > 0
    )

    agg_rows.append({
        "step1_stem":              stem,
        "top_k":                   tk,
        "temperature":             temp,
        "classes_done":            len(classes),
        "classes_total":           total_cls,
        "coverage":                len(classes)/total_cls if total_cls else 0.0,
        "mean_proxy":              mean(proxies),
        "std_proxy":               stdev(proxies),
        "mean_polar_decided":      mean(polars),
        "mean_pair_agreement":     mean(agrees),
        "mean_stability_loo_mae":  mean(stabs),
        "mean_cls_corr":           mean(corrs),
        "mean_gt_best_iou":        mean(ious) if ious else math.nan,
        "mean_gt_auc_roc":         mean(aucs) if aucs else math.nan,
        "mean_orientation_acc":    mean(orients) if orients else math.nan,
        "mean_orientation_improve": mean(orient_improves) if orient_improves else math.nan,
        "n_alt_wins":              n_alt_wins,
    })

min_cls = total_cls if min_cls_arg == "all" else int(min_cls_arg)
agg_all = sorted(agg_rows, key=lambda r: (r["classes_done"], r["mean_proxy"]), reverse=True)
qualified = [r for r in agg_rows if int(r["classes_done"]) >= min_cls] or agg_rows
qualified.sort(key=lambda r: float(r["mean_proxy"]), reverse=True)
top10 = qualified[:10]

agg_fields = [
    "step1_stem","top_k","temperature","classes_done","classes_total","coverage",
    "mean_proxy","std_proxy","mean_polar_decided","mean_pair_agreement",
    "mean_stability_loo_mae","mean_cls_corr",
    "mean_gt_best_iou","mean_gt_auc_roc","mean_orientation_acc",
    "mean_orientation_improve","n_alt_wins",
]

def fmt(v):
    if isinstance(v, float): return f"{v:.6f}" if not (math.isnan(v) or math.isinf(v)) else "nan"
    return str(v)

for path, data in [(agg_tsv, agg_all), (top10_tsv, top10)]:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=agg_fields)
        w.writeheader()
        for r in data:
            w.writerow({k: fmt(r.get(k,"")) for k in agg_fields})

with top10_md.open("w", encoding="utf-8") as f:
    f.write("# Top 10 agregado — configurações da Etapa 2\n\n")
    f.write("> Ranking pela média de `proxy_score` entre classes.\n\n")
    f.write(f"Classes avaliadas: **{total_cls}**. Mínimo para ranking: **{min_cls}**.\n\n")
    f.write("| # | step1_config | top_k | temp | classes | cov | mean_proxy | mean_polar | mean_agree | mean_stab_mae | mean_cls_corr | mean_gt_iou | mean_auc | mean_orient | mean_orient_ganho | classes_c_alt_venceu |\n")
    f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(top10, 1):
        def g(k): return r.get(k, math.nan)
        def s(k): v=g(k); return f"{v:.4f}" if isinstance(v,float) and not math.isnan(v) else "n/a"
        f.write(
            f"| {i} | `{r['step1_stem']}` | {r['top_k']} | {r['temperature']} "
            f"| {r['classes_done']}/{r['classes_total']} | {g('coverage'):.1%} "
            f"| {s('mean_proxy')} | {s('mean_polar_decided')} | {s('mean_pair_agreement')} "
            f"| {s('mean_stability_loo_mae')} | {s('mean_cls_corr')} "
            f"| {s('mean_gt_best_iou')} | {s('mean_gt_auc_roc')} | {s('mean_orientation_acc')} "
            f"| {s('mean_orientation_improve')} | {r['n_alt_wins']}/{r['classes_done']} |\n"
        )

# ----------------------------------------------------------------------------
# Resumo dedicado das heurísticas de orientação, por classe
# (usa a config de melhor proxy_score de cada classe — mesma escolha do
# best_by_class.tsv)
# ----------------------------------------------------------------------------
rows_for_orient = [best_by_cls[c] for c in sorted(best_by_cls)]

with orient_summary_md.open("w", encoding="utf-8") as f:
    f.write("# Resumo — heurísticas de orientação FG/BG por classe\n\n")
    f.write(
        "> Baseado na configuração de melhor `proxy_score` de cada classe. "
        "Compara a heurística de produção (saliência CLS, usada em "
        "`orient_partition()` no `step2_correspondence.py`) com as alternativas "
        "testadas por `analyze_step2_correspondence.py` "
        "(`size_based`, `spatial_based`, `combined`).\n\n"
    )
    f.write("| Classe | Baseline (CLS) | Melhor heurística | Acc. melhor | Ganho |\n")
    f.write("|---|---:|---|---:|---:|\n")

    n_alt_wins_total = 0
    improvements = []
    for r in rows_for_orient:
        best_h   = r.get("orientation_best_heuristic") or "n/a"
        base_acc = r.get("orientation_accuracy", "nan")
        imp_str  = r.get("orientation_improvement_over_baseline", "nan")
        imp_val  = safe(imp_str)
        if imp_val is not None:
            improvements.append(imp_val)
        if best_h not in ("cls_saliency", "n/a") and imp_val is not None and imp_val > 0:
            n_alt_wins_total += 1
        best_acc_str = {
            "size_based":    r.get("orientation_accuracy_size_based", "nan"),
            "spatial_based": r.get("orientation_accuracy_spatial_based", "nan"),
            "combined":      r.get("orientation_accuracy_combined", "nan"),
            "cls_saliency":  base_acc,
        }.get(best_h, base_acc)
        f.write(f"| {r['class_name']} | {base_acc} | {best_h} | {best_acc_str} | {imp_str} |\n")

    f.write("\n")
    total_classes_summarized = len(rows_for_orient)
    mean_imp = (sum(improvements) / len(improvements)) if improvements else float("nan")
    f.write(
        f"**Classes onde alguma heurística alternativa venceu a produção "
        f"(ganho > 0):** {n_alt_wins_total} / {total_classes_summarized}\n\n"
    )
    f.write(f"**Ganho médio (melhor − produção) entre todas as classes:** {mean_imp:.4f}\n\n")
    f.write(
        "Uso sugerido: se `n_alt_wins_total` for uma fração relevante das classes "
        "(ex: > 30%) e o ganho médio for consistentemente positivo, considere trocar "
        "`orient_partition()` no `step2_correspondence.py` pela heurística mais "
        "frequentemente vencedora (ou pela combinação), em vez de manter só saliência CLS.\n"
    )

print(f"[Aggregate] {agg_tsv}")
print(f"[Top10] {top10_tsv}")
print(f"[Top10 MD] {top10_md}")
print(f"[Best by class] {best_tsv}")
print(f"[Orientação — resumo] {orient_summary_md}")
PY
}

# ----------------------------------------------------------------------------
# Loop principal
# ----------------------------------------------------------------------------
for cls in "${CLASSES[@]}"; do
  [[ -z "${CLASS_NPZ_MAP[$cls]+x}" ]] && continue
  IFS='|' read -ra step1_npzs <<< "${CLASS_NPZ_MAP[$cls]}"
  safe_cls="$(sanitize "$cls")"

  echo ""
  echo "############################################################"
  echo "Classe: $cls"
  echo "############################################################"

  for step1_npz in "${step1_npzs[@]}"; do
    [[ -z "$step1_npz" ]] && continue
    step1_stem="$(basename "$step1_npz" .npz)"

    cls_npz_dir="$NPZ_ROOT/$safe_cls"
    cls_report_dir="$REPORT_ROOT/$safe_cls"
    cls_log_dir="$LOG_ROOT/$safe_cls"
    cls_status_dir="$STATUS_ROOT/$safe_cls"
    mkdir -p "$cls_npz_dir" "$cls_report_dir" "$cls_log_dir" "$cls_status_dir"

    for top_k in "${TOP_KS[@]}"; do
      for temperature in "${TEMPERATURES[@]}"; do
        COUNT=$((COUNT + 1))

        STEM="${step1_stem}__topk-${top_k}__temp-${temperature}"
        FLAT_NPZ="$cls_npz_dir/${STEM}.npz"
        EXTRACT_LOG="$cls_log_dir/${STEM}.log"
        ANALYZER_REPORT="$cls_report_dir/${STEM}.analysis.txt"
        DONE_MARKER="$cls_status_dir/${STEM}.done"

        echo ""
        echo "[$COUNT/$TOTAL_RUNS] $cls :: $(basename "$step1_npz") :: top_k=$top_k temp=$temperature"

        RUN_START_TS=$(date +%s)
        RUN_START_HUMAN=$(now_human)
        RUN_STATUS="ok"

        EXISTING_VALID=0
        if [[ "$RESUME" -eq 1 && "$FORCE" -eq 0 && -f "$FLAT_NPZ" ]]; then
          if npz_is_valid "$FLAT_NPZ"; then
            EXISTING_VALID=1
          else
            echo "[Aviso] .npz inválido. Removendo: $FLAT_NPZ" >&2
            rm -f "$FLAT_NPZ" "$DONE_MARKER"
          fi
        fi

        if [[ "$EXISTING_VALID" -eq 1 ]]; then
          if [[ -f "$DONE_MARKER" ]]; then
            echo "[Resume] Concluído anteriormente. Pulando: $FLAT_NPZ"
            RUN_STATUS="resumed_done"
          else
            echo "[Resume] .npz válido, sem marcador. Reaproveitando: $FLAT_NPZ"
            RUN_STATUS="resumed_npz"
          fi
        else
          rm -f "${FLAT_NPZ}.tmp."*

          cmd=("$PYTHON_BIN" "$STEP2_SCRIPT"
            --features "$step1_npz"
            --output   "$(dirname "$FLAT_NPZ")/tmp_${STEM}"
            --top_k    "$top_k"
            --temperature "$temperature")

          if [[ "$VIZ" -eq 1 ]]; then
            cmd+=(--visualize --viz_max "$VIZ_MAX")
          fi

          echo "[Run] ${cmd[*]}"
          set +e
          "${cmd[@]}" > "$EXTRACT_LOG" 2>&1
          status=$?
          set -e

          if [[ "$status" -ne 0 ]]; then
            echo "[ERRO] Etapa 2 falhou. Veja: $EXTRACT_LOG" >&2
            RUN_STATUS="failed"
          else
            TMP_NPZ="$(dirname "$FLAT_NPZ")/tmp_${STEM}/correspondence.npz"
            if [[ ! -f "$TMP_NPZ" ]]; then
              echo "[ERRO] correspondence.npz não encontrado em $(dirname "$FLAT_NPZ")/tmp_${STEM}" >&2
              RUN_STATUS="failed_missing_npz"
            else
              cp "$TMP_NPZ" "$FLAT_NPZ"
              rm -rf "$(dirname "$FLAT_NPZ")/tmp_${STEM}"
              echo "[OK] $FLAT_NPZ"
            fi
          fi

          if [[ "$RUN_STATUS" == "failed"* ]]; then
            RUN_END_TS=$(date +%s); RUN_ELAPSED=$((RUN_END_TS - RUN_START_TS))
            SWEEP_ELAPSED=$((RUN_END_TS - SWEEP_START_TS))
            AVG_SEC=$(awk -v e="$SWEEP_ELAPSED" -v c="$COUNT" 'BEGIN{if(c>0){printf "%.2f",e/c}else{printf "0.00"}}')
            REM=$((TOTAL_RUNS-COUNT))
            EST=$(awk -v a="$AVG_SEC" -v r="$REM" 'BEGIN{printf "%d", a*r}')
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
              "$COUNT" "$TOTAL_RUNS" "$cls" "$step1_stem" "$top_k" "$temperature" \
              "$RUN_STATUS" "$RUN_START_HUMAN" "$(now_human)" "$RUN_ELAPSED" \
              "$(format_duration "$RUN_ELAPSED")" "$AVG_SEC" "$REM" "$EST" \
              "$(format_duration "$EST")" "$(eta_human "$EST")" \
              "$FLAT_NPZ" "$EXTRACT_LOG" "$ANALYZER_REPORT" >> "$TIMING_TSV"
            continue
          fi
        fi

        run_analyzer "$FLAT_NPZ" "$ANALYZER_REPORT" "$step1_npz"
        compute_step2_metrics \
          "$cls" "$step1_npz" "$step1_stem" "$top_k" "$temperature" \
          "$FLAT_NPZ" "$ANALYZER_REPORT" >> "$METRICS_TSV"
        write_done_marker "$DONE_MARKER" "$FLAT_NPZ" "$RUN_STATUS"

        RUN_END_TS=$(date +%s)
        RUN_ELAPSED=$((RUN_END_TS - RUN_START_TS))
        SWEEP_ELAPSED=$((RUN_END_TS - SWEEP_START_TS))
        AVG_SEC=$(awk -v e="$SWEEP_ELAPSED" -v c="$COUNT" 'BEGIN{if(c>0){printf "%.2f",e/c}else{printf "0.00"}}')
        REM=$((TOTAL_RUNS-COUNT))
        EST=$(awk -v a="$AVG_SEC" -v r="$REM" 'BEGIN{printf "%d", a*r}')
        PCT=$(awk -v c="$COUNT" -v t="$TOTAL_RUNS" 'BEGIN{if(t>0){printf "%.2f",100*c/t}else{printf "0.00"}}')

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$COUNT" "$TOTAL_RUNS" "$cls" "$step1_stem" "$top_k" "$temperature" \
          "$RUN_STATUS" "$RUN_START_HUMAN" "$(now_human)" "$RUN_ELAPSED" \
          "$(format_duration "$RUN_ELAPSED")" "$AVG_SEC" "$REM" "$EST" \
          "$(format_duration "$EST")" "$(eta_human "$EST")" \
          "$FLAT_NPZ" "$EXTRACT_LOG" "$ANALYZER_REPORT" >> "$TIMING_TSV"

        echo "[Tempo] $(format_duration "$RUN_ELAPSED") | progresso: $COUNT/$TOTAL_RUNS (${PCT}%) | ETA: $(eta_human "$EST")"
      done
    done
  done

  make_aggregate "$TOTAL_CLASSES" "$MIN_CLASSES_FOR_AGGREGATE"
done

make_aggregate "$TOTAL_CLASSES" "$MIN_CLASSES_FOR_AGGREGATE"

cat <<DONE

============================================================
Concluído.
.npz Etapa 2                 : $NPZ_ROOT
Relatórios analisador        : $REPORT_ROOT
Logs                         : $LOG_ROOT
Métricas completas           : $METRICS_TSV
Agregado por config          : $AGG_TSV
Top 10 agregado TSV          : $TOP10_AGG_TSV
Top 10 agregado Markdown     : $TOP10_AGG_MD
Melhor config por classe     : $BEST_BY_CLASS_TSV
Resumo heurísticas orientação: $ORIENT_SUMMARY_MD
Tempos                       : $TIMING_TSV
============================================================
DONE