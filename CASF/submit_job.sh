#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  SLURM submission script — UGA cluster
#  Code : /scratch/mu59920/MPTLP/
#  Data : /scratch/mu59920/casf_experiments/data/
#
#  Usage (from /scratch/mu59920/MPTLP/):
#    sbatch submit_job.sh <DATASET> <FEATURE_TYPE> [extra CLI options]
#
#  Modes:
#    single              → PH
#    single --spectral   → PTL
#    multi               → MPH
#    multi  --spectral   → MPTL
#
#  Examples:
#    sbatch submit_job.sh 2007 single                             # PH,   GBM
#    sbatch submit_job.sh 2007 single --spectral                  # PTL,  GBM
#    sbatch submit_job.sh 2007 multi                              # MPH,  GBM
#    sbatch submit_job.sh 2007 multi  --spectral                  # MPTL, GBM
#    sbatch submit_job.sh 2007 multi  --model lgbm                # MPH,  LightGBM
#    sbatch submit_job.sh 2007 multi  --spectral --model lgbm     # MPTL, LightGBM
#    sbatch submit_job.sh 2013 multi  --spectral --model xgboost  # MPTL, XGBoost
#    sbatch submit_job.sh 2013 single --model rf --n-seeds 5      # PH,   RF
#    sbatch submit_job.sh 2016 multi  --spectral --model lgbm --n-seeds 50
# ════════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=mptlp
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/mu59920/MPTLP/logs/slurm-%j-%x.out
#SBATCH --error=/scratch/mu59920/MPTLP/logs/slurm-%j-%x.err

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR="/scratch/mu59920/MPTLP"
DATA_ROOT="/scratch/mu59920/casf_experiments/data"
OUTPUT_ROOT="/scratch/mu59920/MPTLP/outputs"

cd "$CODE_DIR"
mkdir -p logs

# ── Thread control ─────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ── Python: use conda/venv if loaded, otherwise fall back to system python ─────
python_executable="${CASF_PYTHON:-python}"

# ── Parse positional arguments ─────────────────────────────────────────────────
dataset="${1:?Usage: sbatch submit_job.sh <DATASET> <FEATURE_TYPE> [options]}"
feature_type="${2:-multi}"
if (( $# >= 2 )); then shift 2; else shift 1; fi

echo "============================================"
echo " Job ID       : ${SLURM_JOB_ID:-local}"
echo " Code dir     : ${CODE_DIR}"
echo " Data root    : ${DATA_ROOT}"
echo " Output root  : ${OUTPUT_ROOT}"
echo " Dataset      : ${dataset}"
echo " Feature type : ${feature_type}"
echo " Extra args   : $*"
echo " Workers      : ${SLURM_CPUS_PER_TASK:-1}"
echo "============================================"

"$python_executable" -u run_experiment.py \
    --dataset      "$dataset"                    \
    --feature-type "$feature_type"               \
    --data-root    "$DATA_ROOT"                  \
    --output-root  "$OUTPUT_ROOT"                \
    --workers      "${SLURM_CPUS_PER_TASK:-1}"   \
    "$@"
