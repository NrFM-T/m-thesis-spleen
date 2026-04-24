#!/bin/bash
#SBATCH -J vgg19_naru
#SBATCH -o output-%j.log
#SBATCH -e output-%j.log
#SBATCH -p GPU-1
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 2

cd ${SLURM_SUBMIT_DIR}

# --- ここを追加 ---
# HAKUSANでcondaを使えるように初期化
source /home/s2610100/.bashrc 

# 自分で作成した仮想環境（例：m-thesis）をアクティベート
# ※環境名が違う場合は、ご自身の環境名に変えてください
conda activate m-thesis 
# ------------------

python /home/s2610100/workshop/260414/mpitest/train_vgg19.py