# coding:utf-8
import numpy as np
import copy
import ctypes
import os
from abc import ABC, abstractmethod
from Agent import actionid2str_statevec, actionid2str
from State import State, State_init, accept_action_str, feature_int

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


def get_normalized_action_list(action_list):
    # 左右対称を同一視した行動列を返す。行動の正規化だけでは、行動順序の異なる左右対称で同一局面を同一視できないケースがあるが、無駄な枝を作らなくて良い効果がある。
    mirror_action_list = list(map(mirror_action, action_list))
    if action_list <= mirror_action_list:
        return action_list, False
    else:
        return mirror_action_list, True


def Glendenning2Official(s):
    """
    cf. https://quoridorstrats.wordpress.com/notation/
    """

    n = int(s[1])

    if len(s) == 2:  # move
        ret = s[0] + str(10 - n)
    else:  # wall
        ret = s[0] + str(9 - n) + s[2]
    return ret

def Official2Glendenning(s):
    return Glendenning2Official(s)


def mirror_action(a):
    if len(a) == 2:
        last_letter = "i"
    else:
        last_letter = "h"
    # 文字をUnicodeコードポイントに変換
    code = ord(a[0])

    # 平均値を求め、それを基準に入れ替えを行う
    mid = (ord('a') + ord(last_letter)) / 2
    new_code = int(mid - (code - mid))

    # 新しいコードポイントを文字に戻す
    return chr(new_code) + a[1:]


class BaseTree(ABC):
    def __init__(self):
        # MCTS_selectで直接アクセスする属性はここで共通的に持つことにする
        self.P = None
        self.tree_c = None
        self.children = {}  # Tree: action_id -> Tree, OpeningTree: action_str -> Tree  OpeningTreeでは正規化を行う都合でaction_strで実装した
        self.P_without_loss = None

    @abstractmethod
    def get_turn(self):
        """ 現在の手番を返す """
        pass

    @abstractmethod
    def move_to_child(self, a, actions, nodes):
        """ 子ノードに移動して返す"""
        pass

    @abstractmethod
    def have_child(self, a, actions, nodes):
        """ aを子ノードとして持っているか"""
        pass


class Tree(BaseTree):
    # p is prior probability
    # p, vにはNoneが来ても良い。その場合必要なときに代入するべきことを表す。
    # negate_treeで変数をコピーし忘れないように！
    def __init__(self, s, p=None, v=None, result=0, optimal_action=None):
        super().__init__()
        action_n = 137
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

    def move_to_child(self, a, actions, nodes):
        return self.children[a]

    def have_child(self, a, actions, nodes):
        return a in self.children.keys()
    
class OpeningTree(BaseTree):
    # json等で保存できるフォーマットにする。
    def __init__(self, fvec):
        super().__init__()
        self.fvec = fvec
        self.turn = None
        
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

        # OpeningTreeではw_arrとして先手勝利数*2、QとしてはＮが少ないうちは0に近い値を取るような計算式を採用
        # N_arr: 各ノードの探索数ベクトル (整数ベクトル)。/2すると実際の探索数で、左右対称局面においては左右対称の手を+1ずつする
        self.tree_c = create_tree()  


        self.statevec2node = None  # あるstatevec2nodeへの参照を代入して参照できるようにする

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
        del vars_dict["statevec2node"]  # statevec2nodeは参照用の一時的な変数なので保存しない

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
                elif isinstance(v, np.ndarray):
                    ret[k] = v.astype(float).tolist()  # numpy arrayをfloatのリストに変換
                else:
                    ret[k] = v

        return ret

    def __lt__(self, other):  # heap用
        """
        比較メソッド: visited_numで比較する。
        """
        return self.visited_num < other.visited_num

    def get_turn(self):
        if self.turn is None:
            return 0
        else:
            return self.turn

    def move_to_child(self, a, actions, nodes):
        s = get_normalized_official_s(actions, nodes)
        return move_to_child(self, s, self.statevec2node)

    def have_child(self, a, actions, nodes):
        s = get_normalized_official_s(actions, nodes)
        return s in self.children.keys()


def get_state_from_action_list(action_list):
    state = State()
    State_init(state)
    for a in action_list:
        accept_action_str(state, a)
    return state


def get_normalized_state(action_list):
    """
    Computes the normalized state representation of a given sequence of actions.

    Args:
        action_list (list): A list of actions representing the sequence of moves in the game.

    Returns:
        tuple:
            - state (object): The normalized state representation derived from the action sequence.
            - state_vec (tuple): A tuple representation of the normalized state's feature vector.
            - is_mirrored (bool): A boolean value indicating whether the mirrored state was selected 
              (True if mirrored state was used, False otherwise).
    """
    # Glendenning notation
    mirror_action_list = list(map(mirror_action, action_list))

    state = get_state_from_action_list(action_list)
    mirror_state = get_state_from_action_list(mirror_action_list)

    state_vec = get_state_vec(state)
    mirror_state_vec = get_state_vec(mirror_state)

    if state_vec <= mirror_state_vec:
        return state, state_vec, False
    else:
        return mirror_state, mirror_state_vec, True


def transform_x_to_symmetric(x):
    """
    入力配列xを左右対称に変換します。
    (display_parameterでの表示結果が左右反転になるようにする)
    
    Parameters:
    x (numpy.ndarray): 長さ137の入力配列

    Returns:
    numpy.ndarray: 左右対称に変換された配列
    """
    if x.size != 137:
        raise ValueError("入力配列は長さ137である必要があります。")

    # 配列を分割
    a = x[:64].reshape((8, 8))
    b = x[64:128].reshape((8, 8))
    c = x[128:].reshape((3, 3))

    # a, b は display_parameter 内で「for x in range(8)」という順序で横方向を走るので、
    # axis=0 で flip すれば左右反転が正しく実現できる
    a_flipped = np.flip(a, axis=0)
    b_flipped = np.flip(b, axis=0)

    # c は display_parameter が x ∈ [-1,0,1] ⇒ (2,0,1) の順で横方向を走る特殊ループなので、
    # 単純に np.flip(c, axis=0) すると、表示結果が期待する左右反転にはならない。
    # そこでインデックスを明示的に並べ替える。
    c_flipped = c[[0, 2, 1], :]

    # 変換後の配列を再構築
    x_transformed = np.concatenate([
        a_flipped.flatten(),
        b_flipped.flatten(),
        c_flipped.flatten()
    ])

    return x_transformed


def get_flipped_index(n):
    """
    与えられたインデックスnに対して、左右対称に変換された配列内で1が立っているインデックスを返します。

    Parameters:
    n (int): 0から136の整数

    Returns:
    int: 変換後の配列内で1が立っているインデックス
    """
    if not isinstance(n, int):
        raise TypeError("nは整数である必要があります。")
    if not (0 <= n <= 136):
        raise ValueError("nは0から136の範囲内である必要があります。")

    # 長さ137のゼロ配列を作成し、n番目の要素を1に設定
    x = np.zeros(137, dtype=int)
    x[n] = 1

    # 配列を左右対称に変換
    x_transformed = transform_x_to_symmetric(x)

    # 1が立っているインデックスを取得
    flipped_indices = np.where(x_transformed == 1)[0]

    if flipped_indices.size == 0:
        raise ValueError("変換後の配列に1が見つかりません。")
    elif flipped_indices.size > 1:
        raise ValueError("変換後の配列に複数の1が存在します。期待されるのは単一の1です。")

    return int(flipped_indices[0])


def get_normalized_official_s(actions, nodes):
    s = State()
    mirror_s = State()
    State_init(s)
    State_init(mirror_s)

    is_success = True
    prev_is_mirrored = False

    action_list = []
    mirror_action_list = []
    is_mirror_list = []

    for action in actions:
        if prev_is_mirrored:
            action = get_flipped_index(action)
        action_str = actionid2str(s, action)

        mirror_action = get_flipped_index(action)
        mirror_action_str = actionid2str(mirror_s, mirror_action)

        action_list.append(action_str)
        mirror_action_list.append(mirror_action_str)

        is_success = is_success and accept_action_str(s, action_str)
        is_success = is_success and accept_action_str(mirror_s, mirror_action_str)

        state_vec = get_state_vec(s)
        mirror_state_vec = get_state_vec(mirror_s)

        is_mirrored = (state_vec > mirror_state_vec)
        is_mirror_list.append(is_mirrored)
        prev_is_mirrored = is_mirrored
    
    _, is_normalized = get_normalized_action_list(action_list)
    if len(is_mirror_list) >= 2:
        is_prev_mirrored = is_mirror_list[-2]
    else:
        is_prev_mirrored = False

    #if is_normalized == is_prev_mirrored:  # 2回反転したらもとに戻る。1回だけ反転のときは反転する。
    if is_normalized:
        ret = Glendenning2Official(mirror_action_list[-1])
    else:
        ret = Glendenning2Official(action_list[-1])

    #print(action_list, mirror_action_list, is_normalized, is_prev_mirrored, ret)
    return ret


def move_to_child(node, key, statevec2node):
    if key not in node.children.keys():
        return None
    
    if isinstance(node.children[key], OpeningTree):
        node = node.children[key]
    else:
        state_vec = tuple(node.children[key])
        if state_vec in statevec2node.keys():
            node = statevec2node[tuple(node.children[key])]  # node.children[key]がstate_vecになっている
        else:
            return None
    return node


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
            if k == "P" or k == "P_without_loss":
                setattr(ret, k, np.array(v, dtype=np.float32))
            else:
                setattr(ret, k, v)

    # tree_c の復元
    if "tree_c" in json_dict:
        # OpeningTreeインスタンスに tree_c を新たに用意
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


def get_state_vec(state):
    # stateを固定長タプルにしてdictのkeyにするために使う。state.turnを入れているのは、turnの異なる状態を区別して無限ループを避けるため
    return tuple([state.turn] + list(feature_int(state).flatten()))
