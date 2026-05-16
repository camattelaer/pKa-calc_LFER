#!/bin/bash -l
#SBATCH --partition=mapo-all
#SBATCH --ntasks=56
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --job-name=pka_test

module load orca/orca-6.1.0
module load python-envs/orca-python

# ── Scratch setup ─────────────────────────────────────────────────────────────
calc_root=$(pwd)
scratch_dir=/scratch/job.${SLURM_JOB_ID}
mkdir -p "${scratch_dir}"
rsync -ra --exclude="./pka_result/work*/" . "${scratch_dir}/"
cd "${scratch_dir}"

# ── Ensure files are always copied back, even on failure/timeout ───────────────
# The trap runs on EXIT (normal), ERR (any non-zero exit), INT and TERM signals.
# This means rsync happens even if a Python script hangs and SLURM kills the job.
cleanup() {
    echo ">>> Copying results back to ${calc_root} ..."
    rsync -ra --exclude="slurm-*.out" "${scratch_dir}/" "${calc_root}/"
    echo ">>> Cleaning scratch ..."
    rm -rf "${scratch_dir}/"
}
trap cleanup EXIT

# ── GPR predict + population plots ───────────────────────────────────────────
# timeout 300: allow up to 5 min for GPR predict + matplotlib.
# If matplotlib hangs (rare with MPLBACKEND=Agg but possible with font-cache
# rebuilds on first run), this ensures the job proceeds to cleanup/rsync.
python pka_gpr_calibrate.py test_set-new/*.msf --predict \
    --gpr pka_results/gpr_model.json \
    --outdir pka_results_test-new \
    --orca /apps/software/orca/orca-6.1.0/orca \
    --nprocs ${SLURM_NTASKS}

# Non-zero exit from timeout is OK — cleanup trap handles rsync regardless.
exit 0