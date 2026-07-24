#!/usr/bin/env bash
# =============================================================================
# Ablação E2 pair coverage — k vizinhos por imagem
# =============================================================================
#
# Reaproveita features.npz da E1 do experimento baseline e executa:
#
#   E2 experimental (per-image kNN)
#       -> E2B
#       -> E3
#       -> E4
#       -> E5
#       -> avaliação
#
# Não modifica:
#   - pipeline/step2_correspondence.py
#   - features/results do baseline
#
# Defaults:
#   classes: person boat bottle bus
#   k:       2 4 8
#
# Uso:
#   bash ablation/run_e2_pair_coverage_ablation.sh
#
# Exemplos:
#   bash ablation/run_e2_pair_coverage_ablation.sh \
#       --classes person boat bottle bus \
#       --k-values 2 4 8
#
#   bash ablation/run_e2_pair_coverage_ablation.sh \
#       --classes person \
#       --k-values 4
#
#   bash ablation/run_e2_pair_coverage_ablation.sh \
#       --dry-run
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PIPELINE="$PROJECT_ROOT/pipeline/run_pipeline.sh"
E2_EXPERIMENTAL="$SCRIPT_DIR/step2_correspondence_per_image.py"

PYTHON="${PYTHON:-python3}"

BASELINE_SUFFIX="bg_recurrence_l0p50"
DATASET_SET="pascal"

CLASSES=("person" "boat" "bottle" "bus")
K_VALUES=(2 4 8)

TEMPERATURE="0.05"
ORIENTATION="spatial"
DEGREE_ALPHA="0.10"

DRY_RUN="no"
FORCE="no"

usage() {
    cat <<'EOF'
USO:
  bash ablation/run_e2_pair_coverage_ablation.sh [OPÇÕES]

OPÇÕES:
  --classes C1 C2 ...        Classes até a próxima opção
                             (default: person boat bottle bus)
  --k-values K1 K2 ...       Valores de k até a próxima opção
                             (default: 2 4 8)
  --baseline-suffix NOME     Sufixo do experimento que contém a E1
                             (default: bg_recurrence_l0p50)
  --temperature T            E2 temperature (default: 0.05)
  --orientation MODO         spatial ou cls (default: spatial)
  --degree-alpha A           E2 degree alpha (default: 0.10)
  --force                    Recalcula E2 experimental e etapas seguintes
  --dry-run                  Apenas imprime comandos
  --help

EXEMPLO RÁPIDO:
  bash ablation/run_e2_pair_coverage_ablation.sh \
      --classes person boat bottle bus \
      --k-values 2 4 8
EOF
}

is_option() {
    [[ "${1:-}" == --* ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --classes)
            shift
            CLASSES=()
            while [[ $# -gt 0 ]] && ! is_option "$1"; do
                CLASSES+=("$1")
                shift
            done
            ;;
        --k-values)
            shift
            K_VALUES=()
            while [[ $# -gt 0 ]] && ! is_option "$1"; do
                K_VALUES+=("$1")
                shift
            done
            ;;
        --baseline-suffix)
            BASELINE_SUFFIX="$2"; shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"; shift 2
            ;;
        --orientation)
            ORIENTATION="$2"; shift 2
            ;;
        --degree-alpha)
            DEGREE_ALPHA="$2"; shift 2
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

if [[ ! -f "$PIPELINE" ]]; then
    echo "[ERRO] run_pipeline.sh não encontrado: $PIPELINE"
    exit 1
fi

if [[ ! -f "$E2_EXPERIMENTAL" ]]; then
    echo "[ERRO] E2 experimental não encontrada: $E2_EXPERIMENTAL"
    exit 1
fi

if [[ ${#CLASSES[@]} -eq 0 ]]; then
    echo "[ERRO] Nenhuma classe definida."
    exit 1
fi

if [[ ${#K_VALUES[@]} -eq 0 ]]; then
    echo "[ERRO] Nenhum k definido."
    exit 1
fi

BASE_EXP="pascal_coseg_${BASELINE_SUFFIX}"
BASE_FEATURES="$PROJECT_ROOT/features/$BASE_EXP"

echo "======================================================================"
echo "ABLAÇÃO E2 — PAIR COVERAGE / PER-IMAGE KNN"
echo "======================================================================"
echo "Baseline E1 : $BASE_FEATURES"
echo "Classes     : ${CLASSES[*]}"
echo "k values    : ${K_VALUES[*]}"
echo "temperature : $TEMPERATURE"
echo "orientation : $ORIENTATION"
echo "degree alpha: $DEGREE_ALPHA"
echo

run() {
    if [[ "$DRY_RUN" == "yes" ]]; then
        printf '[DRY] '
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

for k in "${K_VALUES[@]}"; do
    SUFFIX="${BASELINE_SUFFIX}_kpi${k}"
    EXP="pascal_coseg_${SUFFIX}"

    echo
    echo "######################################################################"
    echo "# k_per_image = $k"
    echo "# experimento = $EXP"
    echo "######################################################################"

    for class_name in "${CLASSES[@]}"; do
        echo
        echo "----------------------------------------------------------------------"
        echo "Classe: $class_name | k=$k"
        echo "----------------------------------------------------------------------"

        SRC_E1="$BASE_FEATURES/$class_name/features.npz"
        DST_DIR="$PROJECT_ROOT/features/$EXP/$class_name"
        DST_E1="$DST_DIR/features.npz"
        DST_E2="$DST_DIR/correspondence.npz"

        if [[ ! -f "$SRC_E1" ]]; then
            echo "[ERRO] E1 baseline ausente:"
            echo "       $SRC_E1"
            echo "       Rode primeiro o PASCAL baseline para essa classe."
            exit 1
        fi

        run mkdir -p "$DST_DIR"

        # Reaproveita E1 sem duplicar um arquivo grande.
        if [[ ! -e "$DST_E1" ]]; then
            if [[ "$DRY_RUN" == "yes" ]]; then
                echo "[DRY] ln -s '$SRC_E1' '$DST_E1'"
            else
                ln -s "$SRC_E1" "$DST_E1"
            fi
        fi

        # E2 experimental
        if [[ "$FORCE" == "yes" || ! -f "$DST_E2" ]]; then
            run "$PYTHON" "$E2_EXPERIMENTAL" \
                --features "$DST_E1" \
                --output "$DST_DIR" \
                --k-per-image "$k" \
                --temperature "$TEMPERATURE" \
                --orientation "$ORIENTATION" \
                --degree-alpha "$DEGREE_ALPHA"
        else
            echo "[SKIP] E2 experimental já existe: $DST_E2"
        fi

        # E2B -> E3 -> E4 -> E5 -> eval
        CMD=(
            bash "$PIPELINE"
            --set "$DATASET_SET"
            --exp-suffix "$SUFFIX"
            --steps 2b,3,4,5,eval
            --export
        )

        if [[ "$FORCE" == "yes" ]]; then
            CMD+=(--force)
        fi

        CMD+=("$class_name")

        run "${CMD[@]}"
    done
done

echo
echo "======================================================================"
echo "ABLAÇÃO CONCLUÍDA"
echo "======================================================================"
echo "Resultados esperados em:"
for k in "${K_VALUES[@]}"; do
    echo "  results/pascal_coseg_${BASELINE_SUFFIX}_kpi${k}/"
done
echo
echo "Para resumir:"
echo "  python ablation/summarize_e2_pair_coverage_ablation.py \\"
echo "      --baseline-suffix '$BASELINE_SUFFIX' \\"
echo "      --classes ${CLASSES[*]} \\"
echo "      --k-values ${K_VALUES[*]}"
