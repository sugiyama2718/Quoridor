# coding:utf-8
import numpy as np
import copy
import ctypes
import os
from abc import ABC, abstractmethod

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


class BaseTree(ABC):
    def __init__(self):
        # MCTS_selectで直接アクセスする属性はここで共通的に持つことにする
        self.P = None
        self.tree_c = None
        self.children = {}         # 子ノード(辞書)
        self.P_without_loss = None

    @abstractmethod
    def get_turn(self):
        """ 現在の手番を返す。Treeでは s.turn を返し、OpeningTreeでは後で実装する。 """
        pass

    @abstractmethod
    def move_to_child(self, a):
        """ 子ノードに移動して返す。Tree では self.children[a]、OpeningTree でも同様を予定。 """
        pass


class Tree(BaseTree):
    # p is prior probability
    # p, vにはNoneが来ても良い。その場合必要なときに代入するべきことを表す。
    # negate_treeで変数をコピーし忘れないように！
    def __init__(self, s, p=None, v=None, result=0, optimal_action=None):
        super().__init__()
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

    def get_turn(self):
        return self.s.turn

    def move_to_child(self, a):
        return self.children[a]
    
class OpeningTree(BaseTree):
    # json等で保存できるフォーマットにする。
    def __init__(self, fvec):
        super().__init__()
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
        self.search_count_vec = None  # 各ノードの探索数ベクトル (整数ベクトル)。/2すると実際の探索数で、左右対称局面においては左右対称の手を+1ずつする
        # 注意: 葉ノードを除きvisited_num = sum(search_count_vec) // 2という関係がある。
        self.p1_win_num_vec = None

        self.tree_c = create_tree()

    def __del__(self):
        delete_tree(self.tree_c)

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
                # 共有ノードなど OpeningTree ではない場合（例: 単純なベクトル）
                ret["children"][k] = [int(x) for x in v]  # 状態ベクトルなどを想定

        # その他の属性を辞書に追加
        vars_dict = copy.copy(self.__dict__)
        del vars_dict["fvec"]
        del vars_dict["children"]

        # tree_c も抜き出して別途保存する
        # 今回、children フィールドは保存しないで N_arr, W_arr, Q_arr のみ保存する
        tree_c_dict = None
        if vars_dict["tree_c"] is not None:
            tc = vars_dict["tree_c"].contents
            tree_c_dict = {
                "N_arr": list(tc.N_arr),
                "W_arr": [float(x) for x in tc.W_arr],
                "Q_arr": [float(x) for x in tc.Q_arr],
            }
        # 取り終わったので vars_dict から取り除く
        del vars_dict["tree_c"]

        # もし tree_c が存在したら ret["tree_c"] に登録
        if tree_c_dict is not None:
            ret["tree_c"] = tree_c_dict

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

    def get_turn(self):
        # 後で実装する想定。今は仮に0を返すだけ
        return 0

    def move_to_child(self, a):
        # 後で実装する想定。とりあえず self.children[a] を返すだけ
        return self.children[a]



def load_dict_to_opening_tree(json_dict):
    """
    辞書形式のデータからOpeningTreeオブジェクトを復元するメソッド。
    """
    fvec = tuple(json_dict["fvec"])
    ret = OpeningTree(fvec)

    # childrenは後で再帰的に復元するので除外
    omit_list = ["fvec", "children", "tree_c"]
    for k, v in json_dict.items():
        if k not in omit_list:
            if isinstance(v, list):
                setattr(ret, k, [int(x) for x in v])  # 整数ベクトルに変換して設定
            else:
                setattr(ret, k, v)

    # tree_c の復元
    if "tree_c" in json_dict:
        # OpeningTreeインスタンスに tree_c を新たに用意
        ret.tree_c = Tree_c()
        tcd = json_dict["tree_c"]
        # N_arr (int配列)
        for i, val in enumerate(tcd["N_arr"]):
            ret.tree_c.contents.N_arr[i] = val
        # W_arr, Q_arr は float配列
        for i, val in enumerate(tcd["W_arr"]):
            ret.tree_c.contents.W_arr[i] = val
        for i, val in enumerate(tcd["Q_arr"]):
            ret.tree_c.contents.Q_arr[i] = val

    # 子ノードの再帰処理
    for k, v in json_dict["children"].items():
        if isinstance(v, dict):
            # OpeningTreeなら再帰復元
            ret.children[k] = load_dict_to_opening_tree(v)
        else:
            # 共有ノードの状態ベクトルなど、単なるリストの場合
            ret.children[k] = v

    return ret
