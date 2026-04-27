#!/bin/bash
#SBATCH --job-name=unet_pipe
#SBATCH --output=/home/s2610100/m-thesis/jobs/logs/unet_pipe_%j.log
#SBATCH --error=/home/s2610100/m-thesis/jobs/logs/unet_pipe_%j.err
#SBATCH -p GPU-1A
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --time=02:00:00  # 学習＋推論のため2時間に拡張

cd ${SLURM_SUBMIT_DIR}

# --- 環境の有効化 ---
unset LD_PRELOAD
source ~/miniconda3/etc/profile.d/conda.sh
conda activate m-thesis
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MKL_SERVICE_FORCE_INTEL=1
export INTEL_JIT_PROFILER_DISABLE=1

# --- 1. 学習の実行 ---
echo "==================================="
echo " UNet Training Started"
echo "==================================="
python /home/s2610100/m-thesis/src/train_spleen.py

# --- 2. 推論と評価の実行 ---
echo "==================================="
echo " UNet Inference & Evaluation Started"
echo "==================================="
python /home/s2610100/m-thesis/src/inference_spleen.py

echo "==================================="
echo " Pipeline Finished Successfully"
echo "==================================="