import copy
import os
import json

# 学習設定 large 学習本番
POOL_SIZE = 60000  # 学習のときにサンプルする対象となる試合数
TRAIN_CYCLE = 1000  # 何試合おきに学習を行うか
EVALUATION_CYCLE = 60000  # 何試合おきに評価を行うか
GAME_NUM_IN_H5 = 10  # 一つのh5ファイルに何試合含めるか。AI初期化のオーバーヘッド対策。
SEE_DATA_NUM = 2  # 一つのデータを何度学習に使うか（見込み）
TRAIN_ARRAY_SIZE = 1000  # 学習時にメモリに乗せる試合数。増やすとメモリを圧迫する
TAU_MIN_OPENING = 0.4
TAU_MIN = 0.25
TAU_MAX = 0.6
EVALUATION_TAU = 0.32
MAX_AI_NUM_FOR_EVALUATE = 4
EVALUATE_PLAY_NUM = 400
USE_EVALUATION_RESULT = False
VALID_GAME_NUM = 0.5

# 学習設定 medium
# POOL_SIZE = 200  # 学習のときにサンプルする対象となる試合数
# TRAIN_CYCLE = 50  # 何試合おきに学習を行うか
# EVALUATION_CYCLE = 100  # 何試合おきに評価を行うか
# GAME_NUM_IN_H5 = 2  # 一つのh5ファイルに何試合含めるか
# SEE_DATA_NUM = 2  # 一つのデータを何度学習に使うか（見込み）
# TRAIN_ARRAY_SIZE = 50  # 学習時にメモリに乗せる試合数。増やすとメモリを圧迫する
# TAU_MIN = 0.25
# EVALUATION_TAU = 0.5
# MAX_AI_NUM_FOR_EVALUATE = 2
# EVALUATE_PLAY_NUM = 30
# USE_EVALUATION_RESULT = False
# VALID_GAME_NUM = 1

# 学習設定 small
# POOL_SIZE = 40  # 学習のときにサンプルする対象となる試合数
# TRAIN_CYCLE = 20  # 何試合おきに学習を行うか
# EVALUATION_CYCLE = 40  # 何試合おきに評価を行うか
# GAME_NUM_IN_H5 = 2  # 一つのh5ファイルに何試合含めるか
# SEE_DATA_NUM = 2  # 一つのデータを何度学習に使うか（見込み）
# TRAIN_ARRAY_SIZE = 8  # 学習時にメモリに乗せる試合数。増やすとメモリを圧迫する
# TAU_MIN = 0.25
# EVALUATION_TAU = 0.5
# MAX_AI_NUM_FOR_EVALUATE = 2
# EVALUATE_PLAY_NUM = 10
# USE_EVALUATION_RESULT = False
# VALID_GAME_NUM = 2

H5_NUM = POOL_SIZE // GAME_NUM_IN_H5
TRAIN_H5_NUM = TRAIN_ARRAY_SIZE // GAME_NUM_IN_H5
POOL_EPOCH_NUM = POOL_SIZE // TRAIN_CYCLE  # poolにいくつのepochが入るか（何回学習が回るか）
EPOCH_H5_NUM = TRAIN_CYCLE // GAME_NUM_IN_H5
LEARN_REP_NUM = int(SEE_DATA_NUM * TRAIN_CYCLE / TRAIN_ARRAY_SIZE)  # 一回の学習プロセスあたり何回シャッフル＆パラメータ更新を行うか
TRAIN_FILE_NUM_PER_EPOCH = TRAIN_ARRAY_SIZE / (POOL_EPOCH_NUM * GAME_NUM_IN_H5)  # 小数点を許容し、0.5なら半分しか使わないなどとしてメモリを節約
EVALUATION_EPOCH_NUM = EVALUATION_CYCLE // TRAIN_CYCLE
SAVE_H5_NUM = POOL_EPOCH_NUM * 4

SELFPLAY_SEARCHNODES_MIN = 300
SELFPLAY_SEARCHNODES_MAX = 1500
DEEP_SEARCH_P = 0.15  # 深い探索を実施する確率
DEEP_TH = 0.5  # prev_v, post_vにどれだけ差があれば深い探索にするか
EVALUATION_SEARCHNODES = 1000

PROCESS_NUM = 20

RATE_TH = 0.1  # レートがいくつ以上あがったら新AIとして採用するか
RATE_TH2 = 0.5  # レートがいくつ以上あがったらAI listを更新するか

DATA_DIR = "train_results/data"
TRAIN_LOG_DIR = "train_results/train_log"
PARAMETER_DIR = "train_results/parameter"
KIFU_DIR = "train_results/kifu"
JOSEKI_DIR = "train_results/joseki"

for dir in [DATA_DIR, TRAIN_LOG_DIR, PARAMETER_DIR, KIFU_DIR, JOSEKI_DIR]:
    os.makedirs(dir, exist_ok=True)

# DEFAULT_FILTERS = 96
# DEFAULT_LAYER_NUM = 41
# USE_SELF_ATTENTION = False
# USE_SLIM_HEAD = False

# DEFAULT_FILTERS = 48
# DEFAULT_LAYER_NUM = 9
# USE_SELF_ATTENTION = False
# USE_SLIM_HEAD = False

# self attention
DEFAULT_FILTERS = 128
DEFAULT_LAYER_NUM = 18
USE_SELF_ATTENTION = True
USE_SLIM_HEAD = True
BROAD_CONV_LAYER_NUM = 5
ATTENTION_CYCLE = 3
ATTENTION_VEC_LEN = 32  # DEFAULT_FILTERSの約数にする

# self attentionでの改良をconvに反映させて比較実験
# DEFAULT_FILTERS = 48
# DEFAULT_LAYER_NUM = 9
# USE_SELF_ATTENTION = True
# USE_SLIM_HEAD = True
# BROAD_CONV_LAYER_NUM = 9
# ATTENTION_CYCLE = 3
# ATTENTION_VEC_LEN = 32  # DEFAULT_FILTERSの約数にする

USE_GLOBAL_POOLING = False
USE_VALID = True
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
EPSILON = 1e-7
EMA_DECAY = 3e-4
VALID_EMA_DECAY = EMA_DECAY
BATCH_SIZE = 256
PER_PROCESS_GPU_MEMORY_FRACTION = 0.07
WARMUP_STEPS = 0.1
N_PARALLEL = 16
NOISE = 0.25

# 十分探索していて、十分勝ちに近い手なら、できる限り最短路を選ぶことで試合を早く終わらせる。そのときの十分探索のNの比率と、勝ちに近いとする閾値
SHORTEST_N_RATIO = 0.1
SHORTEST_Q = 0.95

MIMIC_N_RATIO = 0.01  # この割合以下しか読まれていない手は真似をしない
FORCE_MIMIC_TURN = 10  # 最初のこのターンは↑に引っかかっていても真似する
MIMIC_AI_RATIO = 0.02  # 自己対戦においてmimic AIにする割合

FORCE_OPENING_RATE = 0.04  # 探索してほしい定跡をやる割合
FORCE_OPENING_LIST = [  # Official notation
    ["e2", "e8", "e3", "e7", "e4", "e6", "a3h"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "d3h", "c6h", "d5v"],
    ["e2", "e8", "e3", "e7", "e4", "e6", "d3h", "c6h", "e6v"],
    ["e2", "e8", "e3", "e7", "e4", "d4v"]
]

SHORTEST_P_RATIO = 0.15  # 最短路の向きにPを高める割合

V_REGULARIZER = 0.1
P_REGULARIZER = 0.01

# 共通命名
H5_NAME_LIST = ["feature", "pi", "reward", "v_prev", "v_post", "searched_node_num", 
                "dist_diff", "black_walls", "white_walls", "remaining_turn_num", "remaining_black_moves", "remaining_white_moves", 
                "row_wall", "column_wall", "dist_array1", "dist_array2", "B_traversed_arr", "W_traversed_arr", "next_pi"]
FEATURE_NAME_LIST = copy.copy(H5_NAME_LIST)
FEATURE_NAME_LIST.remove("v_prev")
FEATURE_NAME_LIST.remove("v_post")
AUX_NAME_LIST = copy.copy(FEATURE_NAME_LIST)
AUX_NAME_LIST.remove("feature")
AUX_NAME_LIST.remove("pi")
AUX_NAME_LIST.remove("reward")
AUX_NAME_LIST.remove("searched_node_num")

# この値で対応する変数を割ってから二乗誤差を計算する。大体この値ぐらいが最大値だと思うのを入れておく。
REG_SCALE_DICT = {
    "dist_diff": 15,
    "black_walls": 3,
    "white_walls": 3,
    "remaining_turn_num": 20,
    "remaining_black_moves": 15,
    "remaining_white_moves": 15
}
AUX_IMPORTANCE = 0.5
DIST_ARRAY_SCALE = 10
NEXT_PI_IMPORTANCE = 0.5  # next_piのlossは大きめの値を取るので影響を小さくするための係数

APPLICATION_CONFIG_PATH = os.path.join("application_data", "config.json")
def read_application_config():
    with open(APPLICATION_CONFIG_PATH, 'r') as file:
        ret = json.load(file)
    return ret
