#!/bin/bash
#SBATCH --job-name=spleen_train
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err
#SBATCH -p VM-GPU-L              # 1. 空きが確認されているパーティションへ変更
#SBATCH --gres=gpu:1             # (通らなければ gpu:a40g:1 も試してください)
#SBATCH -n 1                     # 2. 実績コードに合わせフラグ形式を統一
#SBATCH -c 2                     # 3. CPUコア数を明示
#SBATCH --time=01:00:00

# カレントディレクトリへ移動
cd ${SLURM_SUBMIT_DIR}

# --- 環境の初期化と有効化 ---
unset LD_PRELOAD
source ~/miniconda3/etc/profile.d/conda.sh
conda activate m-thesis

# --- ライブラリパスの優先設定（実績コードより引用） ---
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MKL_SERVICE_FORCE_INTEL=1
export INTEL_JIT_PROFILER_DISABLE=1

# 学習実行
python /home/s2610100/m-thesis/src/train_spleen.py s2610100