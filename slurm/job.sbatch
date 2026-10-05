#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=jb_py
#SBATCH --partition=secondary
#SBATCH --mem=64GB
#SBATCH --output=logs/py_%j.out
#SBATCH --error=logs/py_%j.err
# Usage (from the repo root): sbatch [--job-name=NAME] slurm/job.sbatch <command...>
#   e.g. sbatch --job-name=m1 slurm/job.sbatch python -m experiments.m1
# Override any header on the command line (e.g. --time=4:00:00, --gres=gpu:A30:1).

set -euo pipefail
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
conda activate map-stdp
cd "$SLURM_SUBMIT_DIR"
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Is): $*"
"$@"
