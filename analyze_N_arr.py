import os
import numpy as np
import matplotlib.pyplot as plt
from config import EXPERIMENT_RESULTS_DIR

# 保存先ディレクトリの設定（存在しなければ作成）
save_dir = os.path.join(EXPERIMENT_RESULTS_DIR, "analyze_N_arr")
os.makedirs(save_dir, exist_ok=True)

# 対象のCSVファイル名リスト
file_names = [
    "N_arr_evolution_500.csv",
    "N_arr_evolution_1000.csv"
]

def compute_entropy(pi):
    """
    与えられた確率分布piに対して自然対数を用いたエントロピーを計算する。
    0の要素は無視する。
    """
    # 0より大きい要素だけで計算
    positive = pi > 0
    return -np.sum(pi[positive] * np.log(pi[positive]))

# tauの値を0.5から1.0まで0.1刻みで処理
for tau in np.arange(0.5, 1.0 + 0.001, 0.1):
    plt.figure(figsize=(8, 6))
    
    # 各ファイルについてエントロピーの時系列を計算・プロット
    for file in file_names:
        try:
            # CSVファイルを読み込み（各行が1ステップ分のN_arr）
            data = np.loadtxt(file, delimiter=',', dtype=int)
            # CSVが1行だけの場合、dataは1次元配列となるので2次元に変換
            if data.ndim == 1:
                data = data[np.newaxis, :]
            
            entropy_list = []
            for row in data:
                # tauによる変換：各要素の1/tau乗を計算し、その後正規化してpiに
                transformed = np.power(row, 1. / tau)
                total = np.sum(transformed)
                # totalが0の場合はエントロピー計算ができないのでスキップ（または0とする）
                if total == 0:
                    pi = np.zeros_like(transformed, dtype=float)
                else:
                    pi = transformed / total

                entropy = compute_entropy(pi)
                entropy_list.append(entropy)
            
            # タイムステップを横軸としてプロット
            plt.plot(entropy_list, label=file)
        
        except Exception as e:
            print(f"{file} の読み込みや解析でエラーが発生しました: {e}")
    
    plt.xlabel("Time Step")
    plt.ylabel("Entropy (nats)")
    plt.title(f"Entropy Time Evolution (tau = {tau:.1f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 保存先ディレクトリ以下にtauがわかる名前で保存
    save_path = os.path.join(save_dir, f"N_arr_entropy_tau_{tau:.1f}.png")
    plt.savefig(save_path)
    plt.close()
