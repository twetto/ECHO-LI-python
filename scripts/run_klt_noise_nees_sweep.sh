#!/usr/bin/env bash
set -euo pipefail

# KLT-inspired NEES depth sweep.
#
# The step count is scaled with depth so the point travels about the same
# fraction of the image across runs:
#
#   n_steps ~= 16 * depth_m
#
# Defaults reproduce the scout sweep used for the paper-guideline notes.

OUT_DIR="${OUT_DIR:-results/klt_noise_nees_sweep}"
N_MC="${N_MC:-20}"
NOISE_MODEL="${NOISE_MODEL:-epipolar-drift}"
SIGMA_CORE="${SIGMA_CORE:-0.35}"
SIGMA_PIXEL="${SIGMA_PIXEL:-0.35}"
TAIL_PROB="${TAIL_PROB:-0.03}"
TAIL_SCALE="${TAIL_SCALE:-3.0}"
DEPTHS="${DEPTHS:-10 12 15 18 20 25 30 40 60 80 100}"
PYTHON="${PYTHON:-venv/bin/python}"

mkdir -p "${OUT_DIR}"

for z in ${DEPTHS}; do
    n_steps="$("${PYTHON}" -c "print(int(round(16.0 * float('${z}'))))")"
    save_path="${OUT_DIR}/klt_noise_nees_${z}m.png"
    echo "=== z=${z}m n_steps=${n_steps} n_mc=${N_MC} ==="
    MPLBACKEND=Agg "${PYTHON}" tests/test_klt_noise_nees.py \
        --z-true "${z}" \
        --n-steps "${n_steps}" \
        --n-mc "${N_MC}" \
        --noise-model "${NOISE_MODEL}" \
        --sigma-core "${SIGMA_CORE}" \
        --sigma-pixel "${SIGMA_PIXEL}" \
        --tail-prob "${TAIL_PROB}" \
        --tail-scale "${TAIL_SCALE}" \
        --save "${save_path}"
done

echo "Saved sweep plots to ${OUT_DIR}"
