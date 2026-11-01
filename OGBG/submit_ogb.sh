#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
#  SLURM submission script — OGB molecular graph experiments
#  Code : /scratch/mu59920/MPTLP/
#
#  Submits one job per (dataset × mode × filtration × direction) combination.
#  All jobs use MLP by default. Results are saved as JSON files and can be
#  collected into a table with:  python collect_results.py
#
#  Usage:
#    bash submit_ogb.sh          # submit all jobs
#    bash submit_ogb.sh --dry    # print commands without submitting
# ════════════════════════════════════════════════════════════════════════════

CODE_DIR="/scratch/mu59920/OGBG"
RESULTS_DIR="${CODE_DIR}/ogb_results"
LOG_DIR="${CODE_DIR}/logs/ogb"
PYTHON="${CASF_PYTHON:-python}"

DRY_RUN=0
[[ "${1:-}" == "--dry" ]] && DRY_RUN=1

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# ── Experiment grid ───────────────────────────────────────────────────────────
DATASETS=(
    "ogbg-molbace"
    "ogbg-molbbbp"
    "ogbg-molclintox"
    "ogbg-molsider"
    "ogbg-moltoxcast"
    "ogbg-moltox21"
    "ogbg-molhiv"
)

# mode        : PH | PTL | MPH | MPTL
# filtration  : atom_norm | hks     (update after you get results)
# direction   : sub | both
# edge_filt   : bondtype             (fixed; only used for MPH/MPTL)

declare -A NODE_FILT          # dataset → best node filtration from prior runs
NODE_FILT["ogbg-molbace"]="hks"
NODE_FILT["ogbg-molbbbp"]="atom"
NODE_FILT["ogbg-molclintox"]="atom"
NODE_FILT["ogbg-molsider"]="hks"
# Not yet tested — defaulting to atom_norm; update after initial runs
NODE_FILT["ogbg-moltoxcast"]="atom"
NODE_FILT["ogbg-moltox21"]="atom"
NODE_FILT["ogbg-molhiv"]="atom"

MODES=("PH" "PTL" "MPH" "MPTL")
DIRECTIONS=("sub" "both")

# Fixed settings
MODEL="mlp"
EDGE_FILT="en_diff"
N_NODE_THRESH=10
N_EDGE_THRESH=10
EPOCHS=200
RUNS=5
SEED=42
HIDDEN_DIM=128
NUM_LAYERS=3
DROP_RATIO=0.0
LR=0.001

# ── SLURM settings ────────────────────────────────────────────────────────────
PARTITION="batch"
CPUS=4
MEM="16G"
TIME="3-00:00:00"

# ── Submit loop ───────────────────────────────────────────────────────────────
TOTAL=0
for DATASET in "${DATASETS[@]}"; do
    FILT="${NODE_FILT[$DATASET]}"
    for MODE in "${MODES[@]}"; do
        for DIR in "${DIRECTIONS[@]}"; do

            # Job name: short and unique
            DNAME="${DATASET##*-}"          # e.g. molbace
            JNAME="ogb_${DNAME}_${MODE}_${DIR}"

            # Output JSON path
            OUTFILE="${RESULTS_DIR}/${DNAME}_${MODE}_${FILT}_${DIR}.pt"

            CMD="${PYTHON} -u ${CODE_DIR}/train_ogb.py \
                --dataset        ${DATASET}       \
                --mode           ${MODE}          \
                --model          ${MODEL}         \
                --filtration     ${FILT}          \
                --edge_filtration ${EDGE_FILT}    \
                --direction      ${DIR}           \
                --n_node_thresh  ${N_NODE_THRESH} \
                --n_edge_thresh  ${N_EDGE_THRESH} \
                --epochs         ${EPOCHS}        \
                --runs           ${RUNS}          \
                --seed           ${SEED}          \
                --hidden_dim     ${HIDDEN_DIM}    \
                --num_layers     ${NUM_LAYERS}    \
                --drop_ratio     ${DROP_RATIO}    \
                --lr             ${LR}            \
                --filename       ${OUTFILE}"

            if [[ $DRY_RUN -eq 1 ]]; then
                echo "─── DRY RUN: ${JNAME} ───"
                echo "$CMD"
                echo ""
            else
                sbatch \
                    --job-name="${JNAME}" \
                    --partition="${PARTITION}" \
                    --nodes=1 \
                    --ntasks=1 \
                    --cpus-per-task="${CPUS}" \
                    --mem="${MEM}" \
                    --time="${TIME}" \
                    --output="${LOG_DIR}/${JNAME}_%j.out" \
                    --error="${LOG_DIR}/${JNAME}_%j.err" \
                    --wrap="cd ${CODE_DIR} && \
                            export OMP_NUM_THREADS=1 && \
                            export PYTHONUNBUFFERED=1 && \
                            ${CMD}"
                echo "Submitted: ${JNAME}"
            fi

            TOTAL=$((TOTAL + 1))
        done
    done
done

echo ""
echo "Total jobs: ${TOTAL}  (${#DATASETS[@]} datasets × ${#MODES[@]} modes × ${#DIRECTIONS[@]} directions)"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run — nothing submitted. Remove --dry to submit."
fi
