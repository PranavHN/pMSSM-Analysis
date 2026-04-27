#!/bin/bash
# =============================================================================
# Quick CPU-only MCMC convergence test for Run3ModelGen.
#
# Launches N_CHAINS short MCMC chains in parallel on a single node, collects
# the per-chain ROOT ntuples, and runs scripts/mcmc_diagnostics.py to compute
# split-R̂ and ESS_bulk across chains. Designed to validate that the tuned
# `proposal_widths` + `adapt_proposals` config produce reasonable mixing
# (R̂ < ~2, ESS > ~50 per chain) before committing to a long production run.
#
# Wall-time budget: ≲ 90 min on the `interactive` partition.
# Per-chain estimate: ~50 min for 50 post-burn-in samples + 100 burn-in,
# given the reduced pipeline (prep_input + SPheno + micromegas only).
#
# Submit:
#   sbatch slurm/test_mcmc_convergence.sbatch
#
# Outputs land under /ptmp/$USER/mcmc_test_<jobid>/.
# =============================================================================
#SBATCH --job-name=mcmc_convergence_test
#SBATCH --partition=interactive
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1   # SPheno + micromegas are single-threaded; QOS caps total CPUs at 16 on interactive
#SBATCH --mem=24G    # `interactive` caps MaxMemPerNode at 32 GB; 24 GB ≫ enough for 8× SPheno + micromegas
#SBATCH --output=logs/mcmc_test_%j.out
#SBATCH --error=logs/mcmc_test_%j.err

set -euo pipefail

# ---- config -----------------------------------------------------------------
N_CHAINS=8
R3MG_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CONFIG="${R3MG_DIR}/slurm/test_mcmc_convergence.yaml"
OUT_BASE="/ptmp/${USER}/mcmc_test_${SLURM_JOB_ID}"
NTUPLE_DIR="${OUT_BASE}/ntuples"
LOG_DIR="${OUT_BASE}/logs"

mkdir -p "${NTUPLE_DIR}" "${LOG_DIR}" "${R3MG_DIR}/logs"
cd "${R3MG_DIR}"
source build/setup.sh

echo "=========================================="
echo "[mcmc-test] job_id=${SLURM_JOB_ID}"
echo "[mcmc-test] R3MG_DIR=${R3MG_DIR}"
echo "[mcmc-test] CONFIG=${CONFIG}"
echo "[mcmc-test] OUT_BASE=${OUT_BASE}"
echo "[mcmc-test] N_CHAINS=${N_CHAINS}"
echo "[mcmc-test] started: $(date)"
echo "=========================================="

# ---- launch chains in parallel ---------------------------------------------
for seed in $(seq 0 $((N_CHAINS - 1))); do
    chain_dir="${OUT_BASE}/scan.${SLURM_JOB_ID}.${seed}"
    log_file="${LOG_DIR}/chain.${seed}.log"

    echo "[mcmc-test] launching chain seed=${seed} -> ${chain_dir}"
    srun --exclusive \
         --nodes=1 \
         --ntasks=1 \
         --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
         --output="${log_file}" \
         --error="${log_file}" \
         bash -c "
            set -euo pipefail
            cd '${R3MG_DIR}'
            source build/setup.sh
            pixi run genModels.py \
                --scan_dir '${chain_dir}' \
                --seed '${seed}' \
                --config_file '${CONFIG}'
         " &
done

# Wait for all chains
wait
RC=$?
echo "[mcmc-test] all chains finished (rc=${RC}) at $(date)"

# ---- collect ntuples into one dir for the diagnostic ------------------------
collected=0
for seed in $(seq 0 $((N_CHAINS - 1))); do
    src="${OUT_BASE}/scan.${SLURM_JOB_ID}.${seed}/ntuple.${SLURM_JOB_ID}.${seed}.root"
    if [[ -f "${src}" ]]; then
        ln -sf "${src}" "${NTUPLE_DIR}/ntuple.${seed}.root"
        collected=$((collected + 1))
    else
        echo "[warn] missing ntuple for seed=${seed}: ${src}"
    fi
done
echo "[mcmc-test] collected ${collected}/${N_CHAINS} ntuples in ${NTUPLE_DIR}"

# ---- per-chain Metropolis acceptance from logs ------------------------------
echo "=========================================="
echo "[mcmc-test] per-chain Metropolis acceptance from logs:"
for seed in $(seq 0 $((N_CHAINS - 1))); do
    log="${LOG_DIR}/chain.${seed}.log"
    summary=$(grep "\[mcmc-summary\]" "${log}" 2>/dev/null | tail -1 || true)
    if [[ -n "${summary}" ]]; then
        # Extract just the metro_accept_rate field for at-a-glance
        rate=$(echo "${summary}" | grep -oE "metro_accept_rate=[0-9.]+" || true)
        echo "  chain ${seed}: ${rate}"
    else
        echo "  chain ${seed}: (no summary — chain may not have completed)"
    fi
done

# ---- run diagnostics --------------------------------------------------------
echo "=========================================="
echo "[mcmc-test] running mcmc_diagnostics.py on collected ntuples"
echo "=========================================="
# Use --no-file-cut: with very small chains (50 samples) the file-level
# Ω-straddle filter is too aggressive — keep all chains to get a meaningful R̂.
.pixi/envs/default/bin/python source/Run3ModelGen/scripts/mcmc_diagnostics.py \
    --data-dir "${NTUPLE_DIR}" \
    --no-file-cut

echo "=========================================="
echo "[mcmc-test] done. outputs: ${OUT_BASE}"
echo "=========================================="
