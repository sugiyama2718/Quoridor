# coding:utf-8
import numpy as np
import copy
import ctypes
import os

if os.name == "nt":
    lib = ctypes.CDLL('./State_util.dll')
else:
    lib = ctypes.CDLL('./State_util.so')

# Tree構造体の前方宣言
class Tree_c(ctypes.Structure):
    pass

# Tree構造体のフィールドの定義
Tree_c._fields_ = [("N_arr", ctypes.c_int * 137),
                   ("W_arr", ctypes.c_float * 137),
                   ("Q_arr", ctypes.c_float * 137),
                   ("children", ctypes.POINTER(Tree_c) * 137)]


create_tree = lib.createTree
create_tree.restype = ctypes.POINTER(Tree_c)
add_child = lib.addChild
add_child.argtypes = [ctypes.POINTER(Tree_c), ctypes.c_int, ctypes.POINTER(Tree_c)]
delete_tree = lib.deleteTree
delete_tree.argtypes = [ctypes.POINTER(Tree_c)]


class Tree:
    # p is prior probability
    # p, vにはNoneが来ても良い。その場合必要なときに代入するべきことを表す。
    # negate_treeで変数をコピーし忘れないように！
    def __init__(self, s, p=None, v=None, result=0, optimal_action=None):
        action_n = 137
        self.children = {}
        self.s = s
        self.P = p
        self.V = v
        self.result = result  # 1...先手勝利, -1...後手勝利, 0...不明（引き分けは避けて勝敗が必ず定まるとして実装している） rewardとすると0は引き分けなのでresultとした
        self.optimal_action = optimal_action  # 葉ノードもしくは結果が定まっていないときはNone, 定まっているときはどの行動でその結果に至るのか代入すること
        self.is_lose_child_arr = np.zeros((action_n,), dtype=bool)  # 子ノードが負けノードならTrue
        self.P_without_loss = p
        self.dist_diff_arr = 82 * np.ones((action_n,), dtype=int)  # 負けノードについて、歩数差を記録。82は歩数差の上界（升目数+1）
        self.already_certain_path_confirmed = False  # 確定路判定を実行済みならTrue
        self.node_id = None  # graphviz向けの一時変数
        self.state_vec = None
        self.arrays_for_feature_CNN = None

        self.tree_c = create_tree()

    def __del__(self):
        delete_tree(self.tree_c)


    def set_P(self, p):
        self.P = p
        set_p = p * ~self.is_lose_child_arr
        if np.max(set_p) > 0.0:
            self.P_without_loss = set_p
        else:
            self.P_without_loss = p

    def set_is_lose_child_arr(self, action, f):
        self.is_lose_child_arr[action] = f
        set_p = self.P * ~self.is_lose_child_arr
        if np.max(set_p) > 0.0:
            self.P_without_loss = set_p

    def set_is_lose_child_arr_True(self, True_arr):
        self.is_lose_child_arr[True_arr] = True
        set_p = self.P * ~self.is_lose_child_arr
        if np.max(set_p) > 0.0:
            self.P_without_loss = set_p

class OpeningTree:
    # json等で保存できるフォーマットにする。
    def __init__(self, fvec):
        self.children = {}
        self.fvec = fvec
        
        self.score = None
        self.search_nodes = None
        self.epoch = None
        self.search_text = None

        self.visited_num = None
        self.p1_win_num = None
        self.p2_win_num = None
        self.selfplay_epoch = None
        self.game_num = None  # 全体で行われた試合数。ノードによらない値

        self.comment = None
        self.name = None
        self.is_display = False

        self.mcts_result_vec = None  # MCTS探索結果ベクトル (整数ベクトル)
        self.search_count_vec = None  # 各ノードの探索数ベクトル (整数ベクトル)
        # 注意: search_count_vecとvisited_numは意味が重複しています。visited_num = sum(search_count_vec)という関係がある。

    def to_dict(self):
        """
        ノード情報を辞書形式に変換するメソッド。
        """
        ret = {}
        ret["fvec"] = [int(x) for x in self.fvec]  # jsonにするときにリストになっている必要があるため。
        ret["children"] = {}

        # 子ノードの再帰処理
        for k, v in self.children.items():
            if isinstance(v, OpeningTree):
                ret["children"][k] = v.to_dict()
            else:
                ret["children"][k] = [int(x) for x in v]  # 共有ノードの状態ベクトル

        # その他の属性を辞書に追加
        vars_dict = copy.copy(self.__dict__)
        del vars_dict["fvec"]
        del vars_dict["children"]
        for k, v in vars_dict.items():
            if v is not None:
                if isinstance(v, list):
                    ret[k] = [int(x) for x in v]  # 整数ベクトルに変換
                else:
                    ret[k] = v

        return ret

    def __lt__(self, other):  # heap用
        """
        比較メソッド: visited_numで比較する。
        """
        return self.visited_num < other.visited_num


def load_dict_to_opening_tree(json_dict):
    """
    辞書形式のデータからOpeningTreeオブジェクトを復元するメソッド。
    """
    fvec = tuple(json_dict["fvec"])
    ret = OpeningTree(fvec)

    omit_list = ["fvec", "children"]
    for k, v in json_dict.items():
        if k not in omit_list:
            if isinstance(v, list):
                setattr(ret, k, [int(x) for x in v])  # 整数ベクトルに変換して設定
            else:
                setattr(ret, k, v)

    # 子ノードの再帰処理
    for k, v in json_dict["children"].items():
        if isinstance(v, dict):
            ret.children[k] = load_dict_to_opening_tree(v)
        else:
            ret.children[k] = v

    return ret
