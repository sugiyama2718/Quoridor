import os
import graphviz
import math
from Tree import OpeningTree
from tqdm import tqdm
from State import State, State_init, accept_action_str, feature_int
from config import *
from collections import defaultdict
import numpy as np
from Agent import str2actionid

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
    state_vec = tuple(feature_int(state).flatten())  # MCTSのときと違いターン数を区別しない。
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


def get_state_from_action_list(action_list):
    state = State()
    State_init(state)
    for a in action_list:
        accept_action_str(state, a)
    return state


def get_normalized_state(action_list):
    # Glendenning notation
    mirror_action_list = list(map(mirror_action, action_list))

    state = get_state_from_action_list(action_list)
    mirror_state = get_state_from_action_list(mirror_action_list)

    state_vec = tuple(feature_int(state).flatten())
    mirror_state_vec = tuple(feature_int(mirror_state).flatten())

    if state_vec <= mirror_state_vec:
        return state, state_vec, False
    else:
        return mirror_state, mirror_state_vec, True


def generate_opening_tree(all_kifu_list, max_depth, target_epoch=None, disable_tqdm=False):
    statevec2node = {}
    add_state = State()
    State_init(add_state)
    opening_tree = get_opening_node_from_state(add_state, statevec2node)
    opening_tree.visited_num = 0
    opening_tree.p1_win_num = 0
    opening_tree.p2_win_num = 0
    opening_tree.selfplay_epoch = target_epoch
    opening_tree.game_num = len(all_kifu_list)

    # 定跡木の作成
    for action_list in tqdm(all_kifu_list, disable=disable_tqdm):
        state = State()
        State_init(state)
        mirror_state = State()
        State_init(mirror_state)

        normalized_action_list, _ = get_normalized_action_list(action_list)
        mirror_action_list = list(map(mirror_action, action_list))

        node = opening_tree
        path = [node]

        for action_str, mirror_action_str, normalized_action_str, depth in zip(action_list, mirror_action_list, normalized_action_list, range(len(action_list))):
            accept_action_str(state, action_str, check_placable=False, calc_placable_array=False, check_movable=False)
            accept_action_str(mirror_state, mirror_action_str, check_placable=False, calc_placable_array=False, check_movable=False)

            state_vec = tuple(feature_int(state).flatten())
            mirror_state_vec = tuple(feature_int(mirror_state).flatten())

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

        is_sente_win = state.turn % 2  # 引き分けは極めて稀なので考慮しない。

        for node in path:
            node.visited_num += 1
            if is_sente_win:
                node.p1_win_num += 1
            else:
                node.p2_win_num += 1
                
    return opening_tree, statevec2node


def get_epoch_dir_name(epoch):
    floor_epoch = (epoch // EPOCH_DIR_UNIT) * EPOCH_DIR_UNIT
    return "{}_{}".format(floor_epoch, floor_epoch + EPOCH_DIR_UNIT)


visited = None
def build_graph(node, graph, statevec2node, parent_id=None, edge_label=None):
    global visited

    node_id = str(id(node))

    # ノードが既に処理されている場合はスキップ（循環参照対策）
    if id(node) in visited:
        # 親ノードから現在のノードへのエッジを追加
        if parent_id is not None and edge_label is not None:
            graph.edge(parent_id, node_id, label=edge_label)
        return
    visited.add(id(node))

    # ノードのラベルを作成
    visited_num = node.visited_num
    p1_win_num = node.p1_win_num
    p2_win_num = node.p2_win_num

    if visited_num is not None and p1_win_num is not None and p2_win_num is not None:
        p1_percentage = (p1_win_num / visited_num) * 100 if visited_num else 0
        p2_percentage = (p2_win_num / visited_num) * 100 if visited_num else 0
        label = f"{visited_num}\n"
        label += f"{p1_percentage:.1f}%\n"
    else:
        label = "Data missing"

    # ノードをグラフに追加
    graph.node(node_id, label=label)

    # 親ノードから現在のノードへのエッジを追加
    if parent_id is not None and edge_label is not None:
        graph.edge(parent_id, node_id, label=edge_label)

    # 子ノードに対して再帰的に処理
    for child_key in node.children.keys():
        next_node = move_to_child(node, child_key, statevec2node)
        build_graph(next_node, graph, statevec2node, node_id, child_key)

def save_tree_graph(root, statevec2node, path):
    global visited

    graph = graphviz.Digraph(format='png')
    visited = set()
    build_graph(root, graph, statevec2node)
    graph.render(path, view=False)


def compute_contributions(root, statevec2node, total_games, max_depth, is_print=True):
    # Initialize data structures
    nodes_at_depth = {}
    entropies = []
    nodes_seen = set()  # Set to keep track of nodes we've already processed

    # Start with the root node
    nodes_at_depth[0] = [root]
    nodes_seen.add(id(root))  # Use id(root) as a unique identifier

    # Collect nodes at each depth up to max_depth
    for depth in range(max_depth):
        nodes = nodes_at_depth.get(depth, [])
        next_nodes = []
        for node in nodes:
            for child_key in node.children.keys():
                # Use move_to_child function to get the child node
                next_node = move_to_child(node, child_key, statevec2node)
                node_id = id(next_node)
                if node_id not in nodes_seen:
                    nodes_seen.add(node_id)
                    next_nodes.append(next_node)
        if next_nodes:
            nodes_at_depth[depth + 1] = next_nodes

    # Compute entropies at each depth
    for depth in range(max_depth + 1):
        nodes = nodes_at_depth.get(depth, [])
        visited_nums = []
        total_visits = 0
        for node in nodes:
            if node.visited_num is not None:
                visited_nums.append(node.visited_num)
                total_visits += node.visited_num
        if total_visits > 0 and visited_nums:
            probabilities = [vn / total_visits for vn in visited_nums]
            entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        else:
            entropy = 0.0  # No games at this depth
        entropies.append(entropy)

    # Compute contributions for each player
    player1_contrib = 0.0
    player2_contrib = 0.0

    for t in range(len(entropies) - 1):
        delta_entropy = entropies[t + 1] - entropies[t]
        if t % 2 == 0:
            # Player 1 moves at even turns
            player1_contrib += delta_entropy
        else:
            # Player 2 moves at odd turns
            player2_contrib += delta_entropy

    # Compute maximum entropy
    max_entropy = math.log(total_games) if total_games > 0 else 0.0

    # Compute contribution rates
    if abs(player1_contrib - max_entropy) < 1e-10 or abs(player2_contrib - max_entropy) < 1e-10:
        # If one player's contribution equals max_entropy, set both rates to 1
        player1_contrib_rate = 1.0
        player2_contrib_rate = 1.0
    else:
        denominator1 = max_entropy - player2_contrib
        denominator2 = max_entropy - player1_contrib
        player1_contrib_rate = player1_contrib / denominator1 if denominator1 != 0 else 0.0
        player2_contrib_rate = player2_contrib / denominator2 if denominator2 != 0 else 0.0

    # Store the results in variables
    p1_contrib = player1_contrib
    p2_contrib = player2_contrib
    p1_contrib_rate = player1_contrib_rate
    p2_contrib_rate = player2_contrib_rate

    # Print the results
    if is_print:
        print("Maximum Entropy: {:.4f}".format(max_entropy))
        print("Player 1 Contribution: {:.4f}".format(p1_contrib))
        print("Player 1 Contribution Rate: {:.2f}%".format(p1_contrib_rate * 100))
        print("Player 2 Contribution: {:.4f}".format(p2_contrib))
        print("Player 2 Contribution Rate: {:.2f}%".format(p2_contrib_rate * 100))

    return (p1_contrib, p1_contrib_rate), (p2_contrib, p2_contrib_rate)


def get_recent_move_distribution(past_games, action_list):
    """
    現在のaction_listに対応する局面と同じ局面が過去に出現したとき、
    局面が左右対称であれば、次手の元手と左右反転手の両方をaction_count_dictに+1する。
    局面が左右対称でなければ、次手のaction_idを+2する。

    左右対称判定方法:
    mirror_action_listを作り、それぞれから得たstate_vecとmirror_state_vecが等しければ左右対称とみなす。
    """
    # 現在の局面
    state = get_state_from_action_list(action_list)
    state_vec = tuple(feature_int(state).flatten())

    mirror_action_list = list(map(mirror_action, action_list))
    mirror_state = get_state_from_action_list(mirror_action_list)
    mirror_state_vec = tuple(feature_int(mirror_state).flatten())

    # 対称性判定
    symmetrical = (state_vec == mirror_state_vec)

    _, current_state_vec_normalized, current_is_mirrored = get_normalized_state(action_list)

    action_count_dict = defaultdict(int)

    # 過去ゲーム走査
    for game in past_games:
        if len(game) == 0:
            continue

        past_state = State()
        State_init(past_state)
        past_action_list = []

        for i, a in enumerate(game):
            # i手目を指す前の局面（past_action_listまで）
            _, past_state_vec_normalized, past_is_mirrored = get_normalized_state(past_action_list)

            # 一致判定
            if past_state_vec_normalized == current_state_vec_normalized:
                # この直後(i手目)に指された手を取得
                next_action = a

                # is_mirroredが異なる場合、手を左右反転
                if past_is_mirrored != current_is_mirrored:
                    next_action = mirror_action(next_action)

                # action_id取得
                action_id = str2actionid(past_state, next_action)
                if action_id != -1:
                    if symmetrical:
                        # 対象局面なら、action_idとmirror_action_idを両方+1
                        action_count_dict[action_id] += 1
                        mirrored_next_action = mirror_action(next_action)
                        mirrored_action_id = str2actionid(past_state, mirrored_next_action)
                        if mirrored_action_id != -1:
                            action_count_dict[mirrored_action_id] += 1
                    else:
                        # 非対称局面なら+2
                        action_count_dict[action_id] += 2

            # 状態を次手aで更新
            accept_action_str(past_state, a)
            past_action_list.append(a)

    # action_count_dictを137次元のnumpy配列に変換
    n = np.zeros((137,), dtype=int)
    for aid, cnt in action_count_dict.items():
        n[aid] = cnt

    return n


def adaptive_next_sample(p, counts, beta=1.0, random_state=None):
    """
    適応的確率サンプリングによる、次の1サンプルを生成する関数。
    
    Parameters
    ----------
    p : array-like
        目標分布 p = (p_0, p_1, ..., p_{N-1}) 。長さ N の1次元配列。
    counts : array-like
        現在までに観測されている各カテゴリの出現回数。
        長さ N の1次元整数配列。
    beta : float, optional
        補正パラメータ。0に近いほど元の分布 p に従う独立サンプリングに近くなり、
        値が大きいほど経験分布からのずれを強く補正する。
    random_state : int or None, optional
        再現性のための乱数シード。

    Returns
    -------
    x_next : int
        選ばれた次のサンプル（カテゴリのインデックス）。
    """
    p = np.array(p, dtype=float)
    counts = np.array(counts, dtype=int)
    N = len(p)
    rng = np.random.default_rng(random_state)
    
    # これまでのサンプル数 n
    n = counts.sum()
    
    # 経験分布 hat_p の計算
    # n=0 の場合、hat_p = 0 とする
    if n > 0:
        hat_p = counts / n
    else:
        hat_p = np.zeros(N, dtype=float)
    
    # Δ_k = hat_p_k - p_k
    delta = hat_p - p
    
    # q_k = p_k * exp(-beta * Δ_k) を計算し、正規化
    unnormalized_q = p * np.exp(-beta * delta)
    q = unnormalized_q / unnormalized_q.sum()
    
    # qに従って一つサンプル
    x_next = rng.choice(N, p=q)
    
    return x_next


if __name__ == "__main__":
    print(get_epoch_dir_name(0))
    print(get_epoch_dir_name(1))
    print(get_epoch_dir_name(999))
    print(get_epoch_dir_name(1000))
    print(get_epoch_dir_name(1001))


