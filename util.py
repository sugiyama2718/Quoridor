import os
import graphviz
import math
from Tree import OpeningTree, Tree_c, move_to_child, mirror_action, Glendenning2Official, Official2Glendenning, get_normalized_action_list, get_normalized_state, get_state_from_action_list
from tqdm import tqdm
from State import State, State_init, accept_action_str, feature_int
from config import *
from collections import defaultdict
import numpy as np
from Agent import str2actionid, actionid2str_statevec
import ctypes

if os.name == "nt":
    lib = ctypes.CDLL('./State_util.dll')
else:
    lib = ctypes.CDLL('./State_util.so')

select_action = lib.select_action
select_action.argtypes = (ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
                              ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int)
select_action.restype = ctypes.c_int

add_virtual_loss = lib.add_virtual_loss
add_virtual_loss.argtypes = [ctypes.POINTER(Tree_c), ctypes.c_int, ctypes.c_int, ctypes.c_int]
add_virtual_loss.restype = None

subtract_virtual_loss = lib.subtract_virtual_loss
subtract_virtual_loss.argtypes = [ctypes.POINTER(Tree_c), ctypes.c_int, ctypes.c_int, ctypes.c_int]
subtract_virtual_loss.restype = None


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


def calc_Q(N, W):
    N_plus = N + 6
    W_plus = W + 3
    return 2 * (W_plus / N_plus - 0.5)


########################################################
# OpeningTree構築・更新の共通ロジック
########################################################
def _build_opening_tree_core(
    opening_tree,
    statevec2node,
    kifu_list,
    max_depth,
    target_epoch=None,
    disable_tqdm=False,
    pi_lists=None
):
    """
    OpeningTreeとstatevec2nodeに対し、kifu_list(複数ゲーム)を反映させる。
    左右対称局面は同一視する。既存のノードがあれば再利用し、なければ追加する。

    さらに本処理内で tree_c.contents.N_arr も加算する:
      - tree_c.contents.N_arr[i] = 親ノードから「action i」で遷移する子ノードが
        何回訪問されたかを示すカウンタ。
      - 今回は経路を辿るたびに都度+1 or +2する方針。

    Parameters
    ----------
    opening_tree : OpeningTree
        すでに初期化済みのOpeningTreeのrootノード
    statevec2node : dict
        state_vecをキーにOpeningTreeノードを紐づけた辞書
    kifu_list : list of list[str]
        反映させたい棋譜のリスト (アクション文字列のリスト)
    max_depth : int
        何手目まで定跡木に登録するか
    target_epoch : int or None
        optional
    disable_tqdm : bool
        Trueならプログレスバーを表示しない
    pi_lists: list of list[str] or None

    Returns
    -------
    opening_tree, statevec2node (更新後)
    """

    if opening_tree.game_num is None:
        opening_tree.game_num = 0
    opening_tree.game_num += len(kifu_list)
    opening_tree.selfplay_epoch = target_epoch
    opening_tree.statevec2node = statevec2node

    for action_list, pi_list in tqdm(
        zip(kifu_list, pi_lists if pi_lists is not None else [None] * len(kifu_list)),
        disable=disable_tqdm
    ):
        if pi_list is None:
            pi_list = [None] * len(action_list)

        # 1) 状態を用意
        state = State()
        State_init(state)
        mirror_state = State()
        State_init(mirror_state)

        # 2) アクションの左右対称リスト
        normalized_action_list, is_normalized_action_list = get_normalized_action_list(action_list)
        mirror_action_list = list(map(mirror_action, action_list))

        # 経路上のノードを保存 (path_nodes[0] = root)
        node = opening_tree
        path_nodes = [node]

        # 今回の手順で使った (action_id, mirror_action_id, symmetrical, is_normal_state) を保存
        # ただし tree_c.contents.N_arr の更新は「親ノード」ごとに行うため、
        # stepごとに親ノード側を更新するために蓄えておく。
        move_info = []

        for depth, (action_str, mirror_action_str, normalized_act_str, pi) in enumerate(
            zip(action_list[:-1], mirror_action_list[:-1], normalized_action_list[:-1], pi_list[1:])  # piは子ノードに対して割り当てるのでaction_listと一つずらす
        ):
            # action_id, mirror_action_id を計算
            aid = str2actionid(state, action_str)
            maid = str2actionid(mirror_state, mirror_action_str)

            # 局面比較
            prev_state_vec = tuple(feature_int(state).flatten())
            prev_mirror_state_vec = tuple(feature_int(mirror_state).flatten())

            # 左右対称判定
            symmetrical = (prev_state_vec == prev_mirror_state_vec)

            # 実際に手を進める
            accept_action_str(state, action_str, check_placable=False, calc_placable_array=False, check_movable=False)
            accept_action_str(mirror_state, mirror_action_str, check_placable=False, calc_placable_array=False, check_movable=False)

            # 局面比較
            state_vec = tuple(feature_int(state).flatten())
            mirror_state_vec = tuple(feature_int(mirror_state).flatten())

            # normalized_state
            if state_vec <= mirror_state_vec:
                normalized_state = state
                normalized_pi = pi
            else:
                normalized_state = mirror_state
                #normalized_pi = pi  # こちらの方が正解かも
                normalized_pi = transform_x_to_symmetric(pi) if pi is not None else None

            # if is_normalized_action_list:
            #     normalized_pi = pi
            # else:
            #     normalized_pi = transform_x_to_symmetric(pi) if pi is not None else None

            if depth <= max_depth:
                # 公式表記に変換
                key = Glendenning2Official(normalized_act_str)

                # まだ登録されていなければ追加
                if key not in node.children:
                    child_candidate = get_opening_node_from_state(normalized_state, statevec2node)
                    node.children[key] = child_candidate

                    if isinstance(child_candidate, OpeningTree):
                        if child_candidate.visited_num is None:
                            child_candidate.visited_num = 0
                        if child_candidate.p1_win_num is None:
                            child_candidate.p1_win_num = 0
                        if child_candidate.p2_win_num is None:
                            child_candidate.p2_win_num = 0
                        child_candidate.selfplay_epoch = target_epoch
                        if normalized_pi is None:
                            child_candidate.P = None
                            child_candidate.P_without_loss = None
                        else:
                            P = update_array_with_beta(normalized_pi, OPENING_P_ALPHA, OPENING_P_BETA)
                            P = np.power(P, 1. / OPENING_P_TAU)
                            child_candidate.P = np.array(P, dtype=np.float32)
                            child_candidate.P_without_loss = np.array(P, dtype=np.float32)
                        child_candidate.turn = normalized_state.turn
                        child_candidate.statevec2node = statevec2node
                        #print(key, child_candidate.P_without_loss)

                # move_to_child で子ノードに進む
                node = move_to_child(node, key, statevec2node)
                path_nodes.append(node)

                # ここで "move_info" に「この親ノードに対する action_id 情報」を記録する
                # (どちらがnormalized_stateか、symmetricalかを後で使う)
                move_info.append((aid, maid, symmetrical, prev_state_vec <= prev_mirror_state_vec))

        # 手数に応じた勝敗判定（例: 奇数→先手勝ち）
        is_sente_win = 1 if (len(action_list) % 2 == 1) else -1

        # 3) 経路上のノード(=訪れたノード)へ visited_num, p1_win_num/p2_win_num を加算
        for n in path_nodes:
            if n.visited_num is None:
                n.visited_num = 0
            n.visited_num += 1
            if is_sente_win == 1:
                if n.p1_win_num is None:
                    n.p1_win_num = 0
                n.p1_win_num += 1
            else:
                if n.p2_win_num is None:
                    n.p2_win_num = 0
                n.p2_win_num += 1

        # 4) 各ステップで「親ノードの tree_c.contents.N_arr」を更新
        #    move_info[i] は path_nodes[i] → path_nodes[i+1] の手に対応。
        for i, (aid, maid, symmetrical, is_normal) in enumerate(move_info):
            parent_node = path_nodes[i]   # 親ノード

            if symmetrical:
                # 左右対称なら、 action_id, mirror_action_id ともに +1
                if aid != -1:
                    parent_node.tree_c.contents.N_arr[aid] += 1
                    parent_node.tree_c.contents.W_arr[aid] += int(is_sente_win == 1)
                    parent_node.tree_c.contents.Q_arr[aid] = calc_Q(parent_node.tree_c.contents.N_arr[aid], parent_node.tree_c.contents.W_arr[aid])
                if maid != -1:
                    parent_node.tree_c.contents.N_arr[maid] += 1
                    parent_node.tree_c.contents.W_arr[maid] += int(is_sente_win == 1)
                    parent_node.tree_c.contents.Q_arr[maid] = calc_Q(parent_node.tree_c.contents.N_arr[maid], parent_node.tree_c.contents.W_arr[maid])
            else:
                # 非対称
                if not is_normalized_action_list:
                    if aid != -1:
                        parent_node.tree_c.contents.N_arr[aid] += 2
                        parent_node.tree_c.contents.W_arr[aid] += 2 * int(is_sente_win == 1)
                        parent_node.tree_c.contents.Q_arr[aid] = calc_Q(parent_node.tree_c.contents.N_arr[aid], parent_node.tree_c.contents.W_arr[aid])
                else:
                    if maid != -1:
                        parent_node.tree_c.contents.N_arr[maid] += 2
                        parent_node.tree_c.contents.W_arr[maid] += 2 * int(is_sente_win == 1)
                        parent_node.tree_c.contents.Q_arr[maid] = calc_Q(parent_node.tree_c.contents.N_arr[maid], parent_node.tree_c.contents.W_arr[maid])

    return opening_tree, statevec2node


def generate_opening_tree(all_kifu_list, max_depth, target_epoch=None, disable_tqdm=False, pi_lists=None):
    """
    初回など、空のOpeningTreeを作って、all_kifu_listを1からビルドする。
    """
    # 空の statevec2node
    statevec2node = {}

    # ルートとして初期局面用のノードを作成
    init_state = State()
    State_init(init_state)
    root_node = get_opening_node_from_state(init_state, statevec2node)
    # 初期化
    root_node.visited_num = 0
    root_node.p1_win_num = 0
    root_node.p2_win_num = 0
    root_node.game_num = 0
    root_node.selfplay_epoch = target_epoch
    if pi_lists is not None:
        root_node.P = np.array(pi_lists[0][0], dtype=np.float32)
        root_node.P_without_loss = np.array(pi_lists[0][0], dtype=np.float32)

    # まとめて構築
    _build_opening_tree_core(root_node, statevec2node,
                             kifu_list=all_kifu_list,
                             max_depth=max_depth,
                             target_epoch=target_epoch,
                             disable_tqdm=disable_tqdm,
                             pi_lists=pi_lists)

    return root_node, statevec2node


def update_opening_tree_with_new_kifu(opening_tree, statevec2node,
                                      new_kifu_list, max_depth,
                                      target_epoch=None, disable_tqdm=False, pi_lists=None):
    """
    既存のopening_treeとstatevec2nodeに対して、新しい棋譜(new_kifu_list)だけを処理して差分更新する。
    """

    # 初回ではpi_listsを与えられなかったことを想定し、rootだけは改めてPを設定する
    if pi_lists is not None and opening_tree.P is None:
        opening_tree.P = np.array(pi_lists[0][0], dtype=np.float32)
        opening_tree.P_without_loss = np.array(pi_lists[0][0], dtype=np.float32)

    _build_opening_tree_core(opening_tree, statevec2node,
                             kifu_list=new_kifu_list,
                             max_depth=max_depth,
                             target_epoch=target_epoch,
                             disable_tqdm=disable_tqdm,
                             pi_lists=pi_lists)
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


def load_statevec2node(tree, statevec2node=None):
    if statevec2node is None:
        statevec2node = {}
    statevec2node[tree.fvec] = tree
    for child in tree.children.values():
        if isinstance(child, OpeningTree):
            load_statevec2node(child, statevec2node)
    return statevec2node


def display_parameter(x):
    a = x[:64].reshape((8, 8))
    b = x[64:128].reshape((8, 8))
    c = x[128:].reshape((3, 3))
    for y in range(8):
        for x in range(8):
            print("{:5}".format(a[x, y]), end="")
        print("  ", end="")
        for x in range(8):
            print("{:5}".format(b[x, y]), end="")
        print("")
    for y in [-1, 0, 1]:
        for x in [-1, 0, 1]:
            print("{:5}".format(c[x, y]), end="")
        print("")


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


def traverse_opening_tree_and_print(tree, actions):
    """treeにルートノード、actionsに空リストを最初渡す"""

    print(actions)
    print("visited num = {} , p1 win rate = {:.2f}%".format(tree.visited_num, tree.p1_win_num / tree.visited_num * 100))
    if tree.tree_c is not None:
        #display_parameter(np.asarray(tree.tree_c.contents.N_arr, dtype="int32"))
        display_parameter(np.asarray(np.array(tree.tree_c.contents.Q_arr) * 1000, dtype="int32"))
    print()

    for key, node in tree.children.items():
        if isinstance(node, OpeningTree):
            traverse_opening_tree_and_print(node, actions + [key])


def remove_nodes_below_threshold(tree, statevec2node, threshold=1):
    """treeにルートノード、"""

    del_list = []
    for key, v in tree.children.items():
        if isinstance(v, OpeningTree):
            remove_nodes_below_threshold(v, statevec2node, threshold)

        child = move_to_child(tree, key, statevec2node)
        if child.visited_num <= threshold:
            del_list.append(key)

    for key in del_list:
        del tree.children[key]


def MCTS_select(root_tree, C_puct, estimated_V, color):
    t = root_tree
    nodes = []
    actions = []

    while True:
        # t.P が None なら異常終了
        if t.P is None:
            print("!"*200)
            print(actions)
            assert False, "t.P is None is not expected"

        # t.get_turn() で手番を取得し、子ノードには t.move_to_child(a) で移動
        a = select_action(
            t.tree_c.contents.Q_arr,
            t.tree_c.contents.N_arr,
            t.P_without_loss.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            C_puct,
            estimated_V,
            color,
            t.get_turn()
        )

        nodes.append(t)
        actions.append(a)

        # if isinstance(root_tree, OpeningTree):
        #     print(a, actions, t.have_child(a, actions, nodes))

        # 子ノードが無い場合は葉ノードとして処理を終える
        if not t.have_child(a, actions, nodes):
            return t, a, nodes, actions, False
        else:
            t = t.move_to_child(a, actions, nodes)


def select_and_get_nodess_and_actionss(root_tree, C_puct, estimated_V, color, n_parallel, max_node, virtual_loss_n):
    nodess = []
    actionss = []

    for _ in range(min(n_parallel, max_node)):
        _, _, nodes, actions, _ = MCTS_select(root_tree, C_puct, estimated_V, color)
        if nodes is None:
            break
        nodess.append(nodes)
        actionss.append(actions)

        for node, action in zip(nodes, actions):
            if color == node.get_turn() % 2:
                coef = -1
            else:
                coef = 1
            add_virtual_loss(node.tree_c, action, virtual_loss_n, coef)

    # virtual lossを元に戻す
    for nodes, actions in zip(nodess, actionss):
        for node, action in zip(nodes, actions):
            if color == node.get_turn() % 2:
                coef = -1
            else:
                coef = 1
            subtract_virtual_loss(node.tree_c, action, virtual_loss_n, coef)

    return nodess, actionss


def gamma_integer(n):
    """Compute Gamma function for integers (n-1)!."""
    if n <= 0:
        raise ValueError("Gamma function is not defined for non-positive integers.")
    result = 1
    for i in range(1, n):
        result *= i
    return result

def beta_pdf(x, alpha, beta):
    """Beta distribution PDF for integer alpha and beta."""
    # Convert alpha and beta to integers if not already
    alpha = int(alpha)
    beta = int(beta)
    
    # Beta function B(alpha, beta) = Gamma(alpha) * Gamma(beta) / Gamma(alpha + beta)
    B = (gamma_integer(alpha) * gamma_integer(beta)) / gamma_integer(alpha + beta)
    
    # Beta PDF calculation
    return (x**(alpha - 1) * (1 - x)**(beta - 1)) / B

def weighted_by_beta(p, alpha, beta):
    # pは確率分布、shape=(n,), sum(p)=1
    # ベータ分布PDF + pで重み付け。pを足すのはp=1で重み0を回避するため
    w = beta_pdf(p, alpha, beta) + p
    pw = p * w
    p_new = pw / np.sum(pw)
    return p_new


def update_array_with_beta(N2, alpha, beta):
    """
    N2配列をベータ分布に基づいて変換する関数。
    
    Parameters:
        N2 (numpy.ndarray): 入力配列。
        alpha (float): ベータ分布のパラメータα。
        beta (float): ベータ分布のパラメータβ。
    
    Returns:
        numpy.ndarray: 更新されたN2配列。
    """
    # N2の合計
    N2_sum = np.sum(N2)
    
    # 前回のpiを計算し、0で割るリスクを回避
    pi_prev = N2 / N2_sum
    pi_prev = pi_prev * 0.999  # ベータ分布の変換ですべてが0にならないように調整

    # N2を更新
    updated_N2 = N2_sum * weighted_by_beta(pi_prev, alpha, beta)
    
    return updated_N2


if __name__ == "__main__":
    print(get_epoch_dir_name(0))
    print(get_epoch_dir_name(1))
    print(get_epoch_dir_name(999))
    print(get_epoch_dir_name(1000))
    print(get_epoch_dir_name(1001))


