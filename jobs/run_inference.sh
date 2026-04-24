#!/bin/bash
#SBATCH -p GPU-1A
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -J spleen_inference
# ログの出力先を jobs/logs/ 以下に指定 (%j はジョブID)
#SBATCH -o /home/s2610100/m-thesis/jobs/logs/inference_%j.log
#SBATCH -e /home/s2610100/m-thesis/jobs/logs/inference_%j.err

# 1. 環境のロード
module load cuda/12.1

# 2. 仮想環境の有効化
source ~/miniconda3/etc/profile.d/conda.sh
conda activate m-thesis

# 3. 実行
python /home/s2610100/m-thesis/src/inference_spleen.py