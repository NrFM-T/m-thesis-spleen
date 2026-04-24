#!/bin/bash
#SBATCH --partition=GPU-1        # GPUを1枚使用するパーティション
#SBATCH --nodes=1                # ノード数1
#SBATCH --job-name=monai_test    # ジョブの名前
#SBATCH --output=output_%j.log   # 実行ログの出力先

# 環境の有効化
source ~/.bashrc
conda activate m-thesis

# ライブラリ競合回避のための環境変数設定（以前発生したエラー対策）
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# プログラムの実行
python ../src/setup_spleen.py