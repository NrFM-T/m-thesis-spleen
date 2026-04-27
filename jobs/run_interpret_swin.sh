#!/bin/bash
#SBATCH --job-name=swin_pipe
#SBATCH --output=/home/s2610100/m-thesis/jobs/logs/swin_pipe_%j.log
#SBATCH --error=/home/s2610100/m-thesis/jobs/logs/swin_pipe_%j.err
#SBATCH -p GPU-1A
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --time=03:00:00  # Swinは学習が長引く可能性があるため3時間に設定

cd ${SLURM_SUBMIT_DIR}

# --- 環境の有効化 ---
unset LD_PRELOAD
source ~/miniconda3/etc/profile.d/conda.sh
conda activate m-thesis
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MKL_SERVICE_FORCE_INTEL=1
export INTEL_JIT_PROFILER_DISABLE=1

python /home/s2610100/m-thesis/src/interpret_spleen.py

echo "==================================="
echo " Fin"
echo "==================================="