#!/usr/bin/env bash
# =============================================================================
# run_e3_adaptive_bypass_validation.sh
# =============================================================================
#
# Wrapper para validar o adaptive_top20 em duas fases:
#
#   1) icoseg-controls
#      Materializa adaptive_top20 nos 6 controles iCoseg e roda E4->E5->eval.
#
#   2) pascal-full
#      Materializa adaptive_top20 nas 20 classes PASCAL e roda E4->E5->eval.
#
# Pré-requisitos em ablation/:
#   materialize_e3_adaptive_bypass.py
#   run_e3_adaptive_bypass_e45.sh
#
# Uso:
#   bash ablation/run_e3_adaptive_bypass_validation.sh icoseg-controls
#   bash ablation/run_e3_adaptive_bypass_validation.sh pascal-full
#
# Opções adicionais:
#   --force
#   --dry-run
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUNNER="$SCRIPT_DIR/run_e3_adaptive_bypass_e45.sh"

MODE="${1:-}"
shift || true

EXTRA_ARGS=("$@")

if [[ ! -f "$RUNNER" ]]; then
    echo "[ERRO] Runner não encontrado:"
    echo "       $RUNNER"
    exit 1
fi

usage() {
    cat <<'EOF'
USO:
  bash ablation/run_e3_adaptive_bypass_validation.sh MODE [--force|--dry-run]

MODOS:
  icoseg-controls
      Roda adaptive_top20 nos 6 controles iCoseg já usados na validação E3.

  pascal-full
      Roda adaptive_top20 nas 20 classes do PASCAL CoSeg.

EXEMPLOS:
  bash ablation/run_e3_adaptive_bypass_validation.sh icoseg-controls

  bash ablation/run_e3_adaptive_bypass_validation.sh pascal-full

  bash ablation/run_e3_adaptive_bypass_validation.sh pascal-full --force
EOF
}

case "$MODE" in
    icoseg-controls)
        bash "$RUNNER" \
            --set full \
            --source-root features/icoseg_full_bg_recurrence_l0p50 \
            --dest-suffix bg_recurrence_l0p50_e3bypass_top20 \
            --variant adaptive_top20 \
            --trigger-fg-frac 0.01 \
            --classes \
                "Brown_Bear" \
                "Alaskan_Brown_Bear-Eukaryote_museums_Milwaukee_Zoo_2006-cmlburnett" \
                "Agra_Taj_Mahal-Taj_Mahal-LeszekZadlo" \
                "Hot_Balloons-Skybird_Flight_92708-Tracy_Sasser" \
                "Kite-Brighton_kite_Festival-jadepike4" \
                "Kite-Colt_Park_6-1-08-rreed3" \
            "${EXTRA_ARGS[@]}"
        ;;

    pascal-full)
        bash "$RUNNER" \
            --set pascal \
            --source-root features/pascal_coseg_bg_recurrence_l0p50 \
            --dest-suffix bg_recurrence_l0p50_e3bypass_top20 \
            --variant adaptive_top20 \
            --trigger-fg-frac 0.01 \
            --classes \
                aeroplane \
                bicycle \
                bird \
                boat \
                bottle \
                bus \
                car \
                cat \
                chair \
                cow \
                diningtable \
                dog \
                horse \
                motorbike \
                person \
                pottedplant \
                sheep \
                sofa \
                train \
                tvmonitor \
            "${EXTRA_ARGS[@]}"
        ;;

    --help|-h|"")
        usage
        ;;

    *)
        echo "[ERRO] Modo desconhecido: $MODE"
        echo
        usage
        exit 1
        ;;
esac
