import os
from Tree import OpeningTree
from tqdm import tqdm
from State import State

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


def get_normalized_action_list(action_list):
    # 左右対称を同一視した行動列を返す。行動の正規化だけでは、行動順序の異なる左右対称で同一局面を同一視できないケースがあるが、無駄な枝を作らなくて良い効果がある。
    mirror_action_list = list(map(mirror_action, action_list))
    if action_list <= mirror_action_list:
        return action_list, False
    else:
        return mirror_action_list, True


RECORDS_PATH = "records"
os.makedirs(RECORDS_PATH, exist_ok=True)


def get_opening_node_from_state(state, statevec2node):
    # 既に登録済みの場合はstate_vecを返す
    state_vec = tuple(state.feature_int().flatten())  # MCTSのときと違いターン数を区別しない。
    if state_vec in statevec2node.keys():
        ret = state_vec
    else:
        ret = OpeningTree(state_vec)
        statevec2node[state_vec] = ret
    return ret


def move_to_child(node, key, statevec2node):
    if key not in node.children.keys():
        return None
    
    if isinstance(node.children[key], OpeningTree):
        node = node.children[key]
    else:
        node = statevec2node[tuple(node.children[key])]  # node.children[key]がstate_vecになっている
    return node


def generate_opening_tree(target_epoch, all_kifu_list, max_depth):
    statevec2node = {}
    opening_tree = get_opening_node_from_state(State(), statevec2node)
    opening_tree.visited_num = 0
    opening_tree.p1_win_num = 0
    opening_tree.p2_win_num = 0
    opening_tree.selfplay_epoch = target_epoch
    opening_tree.game_num = len(all_kifu_list)

    # 定跡木の作成
    for action_list in tqdm(all_kifu_list):
        state = State()
        mirror_state = State()

        normalized_action_list, _ = get_normalized_action_list(action_list)
        mirror_action_list = list(map(mirror_action, action_list))

        node = opening_tree
        path = [node]

        for action_str, mirror_action_str, normalized_action_str, depth in zip(action_list, mirror_action_list, normalized_action_list, range(len(action_list))):
            state.accept_action_str(action_str, check_placable=False, calc_placable_array=False, check_movable=False)
            mirror_state.accept_action_str(mirror_action_str, check_placable=False, calc_placable_array=False, check_movable=False)

            state_vec = tuple(state.feature_int().flatten())
            mirror_state_vec = tuple(mirror_state.feature_int().flatten())

            if state_vec <= mirror_state_vec:
                normalized_state = state
            else:
                normalized_state = mirror_state

            if depth <= max_depth:
                key = Glendenning2Official(normalized_action_str)
                if key not in node.children.keys():
                    node.children[key] = get_opening_node_from_state(normalized_state, statevec2node)
                    if isinstance(node.children[key], OpeningTree):
                        node.children[key].visited_num = 0
                        node.children[key].p1_win_num = 0
                        node.children[key].p2_win_num = 0
                        node.children[key].selfplay_epoch = target_epoch
                        node.children[key].game_num = len(all_kifu_list)

                node = move_to_child(node, key, statevec2node)
                path.append(node)

        is_sente_win = 1 - state.turn % 2  # 引き分けは極めて稀なので考慮しない。

        for node in path:
            node.visited_num += 1
            if is_sente_win:
                node.p1_win_num += 1
            else:
                node.p2_win_num += 1
                
    return opening_tree, statevec2node
