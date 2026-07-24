#!/usr/bin/env bash
# =============================================================================
# run_e3_adaptive_bypass_e45.sh
# =============================================================================
#
# Materializa uma variante E3 adaptive bypass em um experimento separado e
# executa somente E4 -> E5 -> avaliação.
#
# Exemplo PASCAL:
#   bash ablation/run_e3_adaptive_bypass_e45.sh \
#       --set pascal \
#       --source-root features/pascal_coseg_bg_recurrence_l0p50 \
#       --dest-suffix bg_recurrence_l0p50_e3bypass_top20 \
#       --variant adaptive_top20 \
#       --classes person boat bottle bus
#
# Exemplo iCoseg38:
#   bash ablation/run_e3_adaptive_bypass_e45.sh \
#       --set icoseg38 \
#       --source-root features/icoseg38_bg_recurrence_l0p50 \
#       --dest-suffix bg_recurrence_l0p50_e3bypass_top20 \
#       --variant adaptive_top20 \
#       --classes "Pandas" "Agra Taj Mahal" Kite Statue
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PIPELINE="$PROJECT_ROOT/pipeline/run_pipeline.sh"
MATERIALIZER="$SCRIPT_DIR/materialize_e3_adaptive_bypass.py"
PYTHON="${PYTHON:-python3}"

DATASET_SET=""
SOURCE_ROOT=""
DEST_SUFFIX=""
VARIANT=""
TRIGGER_FG_FRAC="0.01"
FORCE="no"
DRY_RUN="no"
CLASSES=()

usage() {
    cat <<'EOF'
USO:
  bash ablation/run_e3_adaptive_bypass_e45.sh [OPÇÕES]

OBRIGATÓRIO:
  --set SET                 pascal, icoseg38, full ou icoseg16
  --source-root PATH        raiz do experimento baseline em features/
  --dest-suffix NOME        sufixo do novo experimento
  --variant NOME            ex.: adaptive_top20, adaptive_abs0p50
  --classes C1 C2 ...       classes até a próxima opção

OPCIONAL:
  --trigger-fg-frac X       default: 0.01
  --force                   sobrescreve cosaliency e reprocessa E4/E5/eval
  --dry-run                 apenas imprime ações
  --help
EOF
}

is_option() {
    [[ "${1:-}" == --* ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --set)
            DATASET_SET="$2"; shift 2
            ;;
        --source-root)
            SOURCE_ROOT="$2"; shift 2
            ;;
        --dest-suffix)
            DEST_SUFFIX="$2"; shift 2
            ;;
        --variant)
            VARIANT="$2"; shift 2
            ;;
        --trigger-fg-frac)
            TRIGGER_FG_FRAC="$2"; shift 2
            ;;
        --classes)
            shift
            CLASSES=()
            while [[ $# -gt 0 ]] && ! is_option "$1"; do
                CLASSES+=("$1")
                shift
            done
            ;;
        --force)
            FORCE="yes"; shift
            ;;
        --dry-run)
            DRY_RUN="yes"; shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "[ERRO] Opção desconhecida: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$DATASET_SET" || -z "$SOURCE_ROOT" || -z "$DEST_SUFFIX" || -z "$VARIANT" ]]; then
    echo "[ERRO] --set, --source-root, --dest-suffix e --variant são obrigatórios."
    usage
    exit 1
fi

if [[ ${#CLASSES[@]} -eq 0 ]]; then
    echo "[ERRO] Informe pelo menos uma classe em --classes."
    exit 1
fi

case "$DATASET_SET" in
    pascal)
        EXP_PREFIX="pascal_coseg"
        ;;
    icoseg38)
        EXP_PREFIX="icoseg38"
        ;;
    full)
        EXP_PREFIX="icoseg_full"
        ;;
    icoseg16)
        EXP_PREFIX="icoseg16"
        ;;
    *)
        echo "[ERRO] Set não suportado: $DATASET_SET"
        exit 1
        ;;
esac

if [[ "$SOURCE_ROOT" != /* ]]; then
    SOURCE_ROOT="$PROJECT_ROOT/$SOURCE_ROOT"
fi

DEST_ROOT="$PROJECT_ROOT/features/${EXP_PREFIX}_${DEST_SUFFIX}"

run() {
    if [[ "$DRY_RUN" == "yes" ]]; then
        printf '[DRY] '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

echo "======================================================================"
echo "E3 ADAPTIVE BYPASS -> E4 -> E5 -> EVAL"
echo "======================================================================"
echo "Set         : $DATASET_SET"
echo "Source root : $SOURCE_ROOT"
echo "Dest root   : $DEST_ROOT"
echo "Dest suffix : $DEST_SUFFIX"
echo "Variant     : $VARIANT"
echo "Trigger     : $TRIGGER_FG_FRAC"
echo "Classes     : ${CLASSES[*]}"

MATERIALIZE_CMD=(
    "$PYTHON" "$MATERIALIZER"
    --source-root "$SOURCE_ROOT"
    --dest-root "$DEST_ROOT"
    --variant "$VARIANT"
    --trigger-fg-frac "$TRIGGER_FG_FRAC"
    --classes "${CLASSES[@]}"
)

if [[ "$FORCE" == "yes" ]]; then
    MATERIALIZE_CMD+=(--force)
fi

run "${MATERIALIZE_CMD[@]}"

PIPE_CMD=(
    bash "$PIPELINE"
    --set "$DATASET_SET"
    --exp-suffix "$DEST_SUFFIX"
    --steps 4,5,eval
    --export
)

if [[ "$FORCE" == "yes" ]]; then
    PIPE_CMD+=(--force)
fi

PIPE_CMD+=("${CLASSES[@]}")

run "${PIPE_CMD[@]}"

echo
echo "======================================================================"
echo "CONCLUÍDO"
echo "======================================================================"
echo "Features: $DEST_ROOT"
echo "Results : $PROJECT_ROOT/results/${EXP_PREFIX}_${DEST_SUFFIX}"
