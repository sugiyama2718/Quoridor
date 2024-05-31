# coding:utf-8

BOARD_LEN = 600
WINDOW_WIDTH = 1300

from kivy.config import Config
Config.set('graphics', 'width', str(WINDOW_WIDTH))
Config.set('graphics', 'height', str(BOARD_LEN))
Config.set('graphics', 'resizable', False)
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, StringProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.graphics import Rectangle, Color, Triangle
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView

from Agent import actionid2str
from State import State, CHANNEL, State_init, accept_action_str, display_cui, get_row_wall, get_column_wall
from CNNAI import CNNAI
from BasicAI import state_copy, get_state_vec
import time
import sys
import numpy as np
import gc
import os
from main import normal_play
from multiprocessing import Process
from Agent import Agent
from decimal import Decimal
import random
import json
from util import Glendenning2Official, Official2Glendenning, RECORDS_PATH, mirror_action, get_normalized_action_list, get_opening_node_from_state, move_to_child, get_normalized_state
import math
from Tree import OpeningTree, load_dict_to_opening_tree
import copy
from bs4 import BeautifulSoup
from tqdm import tqdm

touched = False
action = ""

SEARCH_NODE_LIST = [1, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 8000, 10000, 15000, 20000, 30000, 50000, 80000, 100000]
TAU_LIST = [0.16 * i for i in range(5)]
DEFAULT_SEARCH_NODE_INDEX = 6
DEFAULT_TAU_INDEX = 0

AI_WAIT_TIME = 0.1  # AIが考え始めるまでに待機する時間[s]
UNDO_WAIT_TIME = 1.0  # 対AIのときにundoを2⃣回押しやすくするために待つ時間を増やす

TRAINING_HIGH_TIME = 10 * 60.0  # [s]
TRAINING_LOW_TIME = 3 * 60.0  # [s]
TRAINING_HIGH_INCREMENT_TIME = 10
TRAINING_LOW_INCREMENT_TIME = 3
TRAINING_RESULT_HISTORY_LEN_LOW = 10
TRAINING_RESULT_HISTORY_LEN_HIGH = 10
ACHIEVE_TH = 0.6 - 1e-10  # この勝率を上回ったら合格扱い
JOSEKI_WAIT_TIME = 0.3  # 定跡のとき待機する時間。定跡は打っている感じを出すため長めに待機する。

DATA_BASE_DIR = "application_data"
DEBUG_DIR = os.path.join(DATA_BASE_DIR, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)
PARAMETER_PATH = os.path.join(DATA_BASE_DIR, "parameter")
PLAYER_DATA_DIR = os.path.join(DATA_BASE_DIR, "player_data")
JOSEKI_PATH = os.path.join(DATA_BASE_DIR, "joseki", "joseki_epoch1940_2940.txt")
MY_OPENING_PATH = os.path.join(DATA_BASE_DIR, "my_opening")
DEFAULT_OPENING_JSONFILEPATH = os.path.join(MY_OPENING_PATH, "all_opening.json")
DEFAULT_OPENING_HTMLFILEPATH = os.path.join(MY_OPENING_PATH, "all_opening.html")
with open(JOSEKI_PATH, "r") as fin:
    joseki_text = fin.read()
    joseki_list = joseki_text.strip().split("\n")
    joseki_num = len(joseki_list)
os.makedirs(PLAYER_DATA_DIR, exist_ok=True)
os.makedirs(MY_OPENING_PATH, exist_ok=True)

param_files = os.listdir(PARAMETER_PATH)
epoch_list = [0] + sorted(list(set([int(s.split(".")[0][5:]) for s in param_files])))
LEVEL_NUM = len(epoch_list)

#TRAINING_LIST = [(0, 500), (60, 500), (62, 500), (71, 500), (91, 500), (96, 500), (155, 500), (220, 500), (465, 500), (620, 500), (620, 500), (634, 1000)]
TRAINING_LIST = [(0, 500), (1, 500), (2, 500), (3, 500), (4, 500), (5, 500), (6, 500), (7, 500), (8, 500), (9, 500), 
                 (10, 500), (11, 500), (12, 500), (13, 200), (13, 500), (14, 500), (15, 500)]
TRAINING_LEVEL_NUM = len(TRAINING_LIST)

# モードid
ANALYZE_MODE = 0
ANALYZE_TO_THE_END_MODE = 1
OPENING_MODE = 2

GOOD_TH = 0.05


def float2score(x):
    sign = int(x >= 0) - int(x < 0)
    return sign * math.floor(abs(x) * 1000)


def tree2score(tree, turn):
    if int(np.sum(tree.N)) == 0:
        return tree.result * 1000
    score = float2score(np.sum(tree.W) / np.sum(tree.N)) 
    if turn % 2 == 1:
        score *= -1
    return score


def get_description(state, action, tree, is_simple=False):
    #print(action, actionid2str(state, action))
    action_str = Glendenning2Official(actionid2str(state, action))
    #print(action_str)
    #print(action_str, tree.N[action], np.sum(tree.children[action].N))
    if action in tree.children.keys():
        score = tree2score(tree.children[action], state.turn)  # 手番によって符号が変わるが、先読みした先では符号は変わらないのでstate.turnを渡す
        if is_simple:
            return "{:<3} ({:4.1f}%)".format(action_str, 100 * tree.N[action] / np.sum(tree.N))
        else:
            return "{:<3} ({:4.1f}%) {:5}".format(action_str, 100 * tree.N[action] / np.sum(tree.N), score)
    else:
        return None


class GUIHuman(Agent):
    def act(self, state, showNQ=False):
        global touched, action
        if touched:
            touched = False
            return action
        return -1


class FileChooserPopup(Popup):
    def __init__(self, on_select, **kwargs):
        super(FileChooserPopup, self).__init__(**kwargs)
        self.content = FileChooserListView(path="records")
        self.content.bind(on_submit=self.on_file_select)
        self.on_select = on_select

    def on_file_select(self, instance, selection, touch):
        # ファイルが選択された時の処理
        if selection:
            file_path = selection[0]
            self.on_select(file_path)  # コールバック関数を呼び出す
            self.dismiss()  # Popupを閉じる


class Quoridor(Widget):
    turn = NumericProperty(0)
    move_str = StringProperty("player1 move")
    player1wall = NumericProperty(10)
    player2wall = NumericProperty(10)
    Bx = NumericProperty(0)
    By = NumericProperty(0)
    Wx = NumericProperty(0)
    Wy = NumericProperty(0)
    turn0_button = ObjectProperty(None)
    undo_button = ObjectProperty(None)
    redo_button = ObjectProperty(None)
    mode_tab = ObjectProperty(None)
    analyze_tab =ObjectProperty(None)
    opening_tab = ObjectProperty(None)

    graphviz_on = ObjectProperty(None)
    graphviz_off = ObjectProperty(None)
    teban_1p = ObjectProperty(None)
    teban_2p = ObjectProperty(None)
    upside_down = NumericProperty(0)
    analyze_this_turn_button = ObjectProperty(None)
    analyze_to_the_end_button = ObjectProperty(None)
    record_num_text = ObjectProperty(None)
    specify_record_button = ObjectProperty(None)
    next_record_button = ObjectProperty(None)
    analyze_all_turns_button = ObjectProperty(None)
    load_latest_record_button = ObjectProperty(None)
    load_specified_record_button = ObjectProperty(None)
    board_as_text_button = ObjectProperty(None)
    clear_text_button = ObjectProperty(None)

    teban_1p_opening = ObjectProperty(None)
    teban_2p_opening = ObjectProperty(None)

    analyze_this_turn_button_opening = ObjectProperty(None)
    register_button = ObjectProperty(None)
    #load_opening_button = ObjectProperty(None)
    analyze_to_the_end_button_opening = ObjectProperty(None)
    save_as_html_button = ObjectProperty(None)
    analyze_all_nodes_button = ObjectProperty(None)
    name_this_node_button = ObjectProperty(None)
    comment_this_node_button = ObjectProperty(None)
    clear_text_button_opening = ObjectProperty(None)

    user_input_text = ObjectProperty(None)
    main_text = ObjectProperty(None)
    record_text = ObjectProperty(None)

    # Human vs. AI
    search_nodes = SEARCH_NODE_LIST[DEFAULT_SEARCH_NODE_INDEX]
    level = LEVEL_NUM - 1
    tau = TAU_LIST[DEFAULT_TAU_INDEX]

    game_result = StringProperty("")

    def dont_down(self, button):
        if button.state != "down":
            button.state = "down"

    def __init__(self, **kwargs):
        super(Quoridor, self).__init__(**kwargs)
        self.state = State()
        State_init(self.state)
        self.agents = [GUIHuman(0), CNNAI(1, search_nodes=self.search_nodes, tau=0.5)]
        self.analyze_AIs = None

        self.turn0_button.bind(on_release=lambda touch: self.turn0())
        self.undo_button.bind(on_release=lambda touch: self.undo())
        self.redo_button.bind(on_release=lambda touch: self.redo())
        self.graphviz_on.bind(on_press=lambda touch: self.dont_down(self.graphviz_on))
        self.graphviz_off.bind(on_press=lambda touch: self.dont_down(self.graphviz_off))

        self.analyze_tab.bind(on_press=lambda touch: self.analyze_tab_f())
        self.opening_tab.bind(on_press=lambda touch: self.opening_tab_f())

        self.teban_1p.bind(on_press=lambda touch: self.dont_down(self.teban_1p))
        self.teban_2p.bind(on_press=lambda touch: self.dont_down(self.teban_2p))
        self.teban_1p_opening.bind(on_press=lambda touch: self.dont_down(self.teban_1p_opening))
        self.teban_2p_opening.bind(on_press=lambda touch: self.dont_down(self.teban_2p_opening))

        self.analyze_this_turn_button.bind(on_release=lambda touch: self.analyze_this_turn())
        self.analyze_to_the_end_button.bind(on_release=lambda touch: self.analyze_to_the_end())
        self.specify_record_button.bind(on_release=lambda touch: self.specify_record())
        self.next_record_button.bind(on_release=lambda touch: self.next_record())
        self.analyze_all_turns_button.bind(on_release=lambda touch: self.analyze_all_turns())
        self.load_latest_record_button.bind(on_release=lambda touch: self.load_latest_record())
        self.load_specified_record_button.bind(on_release=lambda touch: self.load_specified_record())
        self.board_as_text_button.bind(on_release=lambda touch: self.board_as_text())
        self.clear_text_button.bind(on_release=lambda touch: self.clear_text())

        self.analyze_this_turn_button_opening.bind(on_release=lambda touch: self.analyze_this_turn_opening())
        self.register_button.bind(on_release=lambda touch: self.register())
        #self.load_opening_button.bind(on_release=lambda touch: self.load_opening())
        self.analyze_to_the_end_button_opening.bind(on_release=lambda touch: self.analyze_to_the_end_opening())
        self.save_as_html_button.bind(on_release=lambda touch: self.save_as_html())
        self.analyze_all_nodes_button.bind(on_release=lambda touch: self.analyze_all_nodes())
        self.name_this_node_button.bind(on_release=lambda touch: self.name_this_node())
        self.comment_this_node_button.bind(on_release=lambda touch: self.comment_this_node())
        self.clear_text_button_opening.bind(on_release=lambda touch: self.clear_text())

        self.mode = ANALYZE_MODE
        self.upside_down = 0

        self.use_prev_tree = False  # 途中から実行するなどが多いので、意図しない振る舞いを避けるためにFalseとする
        self.prev_act_time = time.time()
        self.ai_wait_time = AI_WAIT_TIME

        self.row_wall_colors = [Color(0.7, 0.7, 0, 0) for i in range(64)]
        self.column_wall_colors = [Color(0.7, 0.7, 0, 0) for i in range(64)]

        self.state_history = None
        self.action_history = None

        self.target_epoch = None

        self.all_records = []
        self.record_index_now = 0
        self.last_record_path = ""

        self.opening_tree = None
        self.statevec2node = {}

        with self.canvas.before:
            Color(96/255, 32/128, 0, 1)
            Rectangle(pos=(10, 10), size=(BOARD_LEN - 20, BOARD_LEN - 20))
            Color(64/255, 0, 0, 1)
            for i in range(10):
                Rectangle(pos=(int(10 + i / 9 * (BOARD_LEN - 30)), 10), size=(10, BOARD_LEN - 20))
            for i in range(10):
                Rectangle(pos=(10, int(10 + i / 9 * (BOARD_LEN - 30))), size=(BOARD_LEN - 20, 10))

        for i, color in enumerate(self.row_wall_colors):
            self.canvas.add(color)
            x = i % 8
            y = i // 8
            self.canvas.add(Rectangle(pos=(int(20 + x / 9 * (BOARD_LEN - 30)), int(10 + (y + 1) / 9 * (BOARD_LEN - 30))), size=((BOARD_LEN - 30) // 9 * 2 - 10, 10)))
        for i, color in enumerate(self.column_wall_colors):
            self.canvas.add(color)
            x = i % 8
            y = i // 8
            self.canvas.add(Rectangle(pos=(int(10 + (x + 1) / 9 * (BOARD_LEN - 30)), int(20 + y / 9 * (BOARD_LEN - 30))), size=(10, (BOARD_LEN - 30) // 9 * 2 - 10)))

        # 最初からコマを動かせるようにするため
        self.start_game()

        self.load_opening()
        self.register()  # jsonがまだないとき用

    def name_this_node(self):
        print("name_this_node")
        node = self.register(is_display=False)
        node.name = self.user_input_text.text

    def comment_this_node(self):
        print("comment_this_node")
        node = self.register(is_display=False)
        node.comment = self.user_input_text.text
    
    def analyze_all_nodes(self):
        print("analyze_all_nodes")

        def get_target_nodes(tree, action_list):
            ret = []
            if tree.score is None or tree.epoch < self.target_epoch or tree.search_nodes < self.search_nodes:
                ret.append((tree, action_list))
            for action, child in tree.children.items():
                if isinstance(child, OpeningTree):
                    ret.extend(get_target_nodes(child, action_list + [Official2Glendenning(action)]))
            return ret
        
        target_nodes = get_target_nodes(self.opening_tree, [])
        start = time.time()
        for i, (node, action_list) in enumerate(target_nodes):
            if i > 0:
                print("\r{}/{} {:.2f}min".format(i + 1, len(target_nodes), (len(target_nodes) - i) * (time.time() - start) / i / 60), end="")
                
            state, _, _ = get_normalized_state(action_list)

            self.analyze_one_node(node, state)

        self.save_opening_tree()

        print()
        print("done")

    def analyze_to_the_end_opening(self):
        print("analyze_to_the_end_opening")

    def save_as_html(self):
        print("save_as_html")
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quoridor Openings</title>
            <style>
                body {
                    font-family: 'Courier New', Courier, monospace;
                }
                h1 {font-size: 30px;}
                h2 {font-size: 27px;}
                h3 {font-size: 24px;}
                h4 {font-size: 22px;}
                h5 {font-size: 20px;}
                h6 {font-size: 18px;}
                p {
                    white-space: pre-wrap;
                    line-height: 0.5;
                }
            </style>
        </head>
        <body>
        </body>
        </html>
        """

        self.prev_prev_str = ""
        def create_html(node, html, actions_diff, all_actions, depth):
            if depth > 6:  # HTMLの見出しタグはh1からh6までなので、深さ6を超える場合はh6を使用
                depth = 6

            hyphen_len = 50

            if actions_diff is not None:
                prev_actions = all_actions[:-len(actions_diff)]
                prev_str = ", ".join(prev_actions)
                remaining_hyphen_len = max(10, (hyphen_len - len(prev_str)) // 2)
                prev_str = "-" * remaining_hyphen_len + prev_str + "-" * remaining_hyphen_len + os.linesep
            else:
                prev_str = "-" * hyphen_len

            if node.is_display:

                if prev_str != self.prev_prev_str:
                    ptag = soup.new_tag("p")
                    ptag.string = prev_str
                    html.append(ptag)

                # 新しい見出しタグを作成
                h_depth = depth if depth >= 1 else 1  # rootとその下の階層は同じ深さ扱いにしたい
                tag_name = f"h{h_depth}"
                tag = soup.new_tag(tag_name)
                if actions_diff is None:
                    title = "root"
                elif node.name is not None:
                    title = node.name
                else:
                    title = ", ".join(actions_diff)
                tag.string = title

                # 現在のノードを追加
                html.append(tag)

                next_depth = depth + 1
                next_actions_diff = []

                #ptag_string = "display" + os.linesep
                state = State()
                State_init(state)
                for a in all_actions:
                    accept_action_str(state, Official2Glendenning(a))
                ptag_string = display_cui(state, ret_str=True) + os.linesep
            else:
                next_depth = depth
                next_actions_diff = actions_diff

                if prev_str != self.prev_prev_str:
                    ptag_string = prev_str
                else:
                    ptag_string = ""

                if node.name is not None:
                    ptag_string += node.name + os.linesep
                else:
                    ptag_string += ", ".join(actions_diff) + os.linesep

            if node.comment is not None:
                ptag_string += "comment:" + os.linesep
                ptag_string += node.comment + os.linesep + os.linesep

            self.prev_prev_str = prev_str

            if node.search_text is not None:
                _, _, is_mirror = get_normalized_state(list(map(Official2Glendenning, all_actions)))
                if is_mirror:
                    search_text = self.mirror_text(node.search_text)
                else:
                    search_text = node.search_text
                ptag_string += search_text + os.linesep

            ptag = soup.new_tag("p")
            ptag.string = ptag_string
            html.append(ptag)

            # 子ノードを再帰的に処理
            for action, child in node.children.items():
                if isinstance(child, OpeningTree):
                    create_html(child, html, actions_diff=next_actions_diff + [action], all_actions=all_actions + [action], depth=next_depth)

            # ptag = soup.new_tag("p")
            # ptag.string = "-" * 50
            # html.append(ptag)

        soup = BeautifulSoup(html_template, 'html.parser')

        create_html(self.opening_tree, soup.body, actions_diff=None, all_actions=[], depth=0)

        html_string = str(soup)

        with open(DEFAULT_OPENING_HTMLFILEPATH, "w") as fout:
            fout.write(html_string)

    def analyze_one_node(self, node, state):
        MCTS_tree = self.get_MCTS_tree(state)
        root_score, action_cands = self.tree2info(MCTS_tree)

        search_text = self.treeinfo2text(MCTS_tree, root_score, action_cands)

        node.score = root_score
        node.epoch = self.target_epoch
        node.search_nodes = self.search_nodes
        node.search_text = search_text

    
    def analyze_this_turn_opening(self):
        print("analyze_this_turn_opening")

        node = self.register(is_display=False)

        state, _, is_mirror = get_normalized_state(self.action_history[1:self.turn + 1])

        if node.score is None or node.epoch < self.target_epoch or node.search_nodes < self.search_nodes:
            self.analyze_one_node(node, state)
            search_text = node.search_text
            if is_mirror:
                output_text = self.mirror_text(search_text)
            else:
                output_text = search_text

        else:
            if is_mirror:
                output_text = self.mirror_text(node.search_text)
            else:
                output_text = node.search_text

        self.main_text.text += "-" * 20 + os.linesep
        self.main_text.text += output_text
        self.main_text.text += "display = {}".format(node.is_display) + os.linesep

        self.save_opening_tree()

    def mirror_text(self, text):
        text = copy.copy(text)
        lines = text.splitlines()
        new_lines = lines[:2]
        for line in lines[2:]:
            action_texts = line.split(",")[:-1]
            action_texts = [s.strip() for s in action_texts]
            action_texts2 = [s.split(" ")[0] for s in action_texts]
            other_texts = [" ".join(s.split(" ")[1:]) for s in action_texts]
            action_texts2 = [mirror_action(s) for s in action_texts2]
            combined_texts = [" ".join([s1, s2]) for s1, s2 in zip(action_texts2, other_texts)]
            new_lines.append(", ".join(combined_texts))
        return os.linesep.join(new_lines) + os.linesep

    def load_opening(self):
        print("load_opening")

        def load_statevec2node(tree):
            self.statevec2node[tree.fvec] = tree
            for child in tree.children.values():
                if isinstance(child, OpeningTree):
                    load_statevec2node(child)

        if os.path.exists(DEFAULT_OPENING_JSONFILEPATH):
            with open(DEFAULT_OPENING_JSONFILEPATH, "r") as fin:
                json_dict = json.load(fin)
            self.opening_tree = load_dict_to_opening_tree(json_dict)
            load_statevec2node(self.opening_tree)

    def save_opening_tree(self):
        with open(DEFAULT_OPENING_JSONFILEPATH, "w") as fout:
            json.dump(self.opening_tree.to_dict(), fout)


    def register(self, is_display=True):
        print("register")
        
        if self.opening_tree is None:
            add_state = State()
            State_init(add_state)
            self.opening_tree = get_opening_node_from_state(add_state, self.statevec2node)
            self.opening_tree.is_display = True  # rootは常に表示対象とする

        node = self.opening_tree
        action_list, _ = get_normalized_action_list(self.action_history[1:self.turn + 1])
        #action_list = self.action_history[1:self.turn + 1]

        for i, a in enumerate(action_list):
            key = Glendenning2Official(a)

            if key not in node.children.keys():
                state, _, _ = get_normalized_state(action_list[:i+1])  # 毎回normalized_stateを得るとn^2の遅さになりそう
                node.children[key] = get_opening_node_from_state(state, self.statevec2node)

            node = move_to_child(node, key, self.statevec2node)

        if is_display:
            node.is_display = True  # 現局面を表示対象とする

        self.save_opening_tree()

        return node
    
    def specify_record(self):
        print("specify_record")

        s = self.record_num_text.text
        if not s.isdigit():
            self.main_text.text += "must be input as number" + os.linesep
            return
        
        n = int(s)
        if not (0 <= n < len(self.all_records)):
            self.main_text.text += "Input is out of range." + os.linesep
            return
        
        self.load_to_history(self.last_record_path, n)

    def next_record(self):
        print("next_record")
        next_index = (self.record_index_now + 1) % len(self.all_records)
        print(next_index)
        self.load_to_history(self.last_record_path, next_index)
    
    def board_as_text(self):
        print("board_as_text")
        history_text = self.get_record_text(self.action_history[1:self.turn + 1])
        self.main_text.text += history_text
        self.main_text.text += display_cui(self.state, ret_str=True)

    def clear_text(self):
        print("clear_text")
        self.main_text.text = ""

    def analyze_all_turns(self):
        print("analyze_all_turns")
        self.turn0()
        for i in range(len(self.state_history)):
            self.analyze_this_turn()
            self.redo()
            self.update(None)

    def load_to_history(self, target_path, target_record_index=0):
        with open(target_path, "r") as fin:
            text = fin.read()
        self.all_records = text.splitlines()  # 行ごとに棋譜が保存される

        actions = self.all_records[target_record_index].split(",")
        self.record_index_now = target_record_index
        self.record_num_text.text = str(self.record_index_now)
        self.last_record_path = target_path

        onlymoves = [(i, s) for i, s in enumerate(actions) if len(s) == 2]
        i0, s0 = onlymoves[0]
        isOfficial = (i0 % 2 == 0 and int(s0[1]) <= 2) or (i0 % 2 == 1 and int(s0[1]) >= 8)

        self.turn0()
        self.state_history = None
        s = State()
        State_init(s)
        self.add_history(s, None)
        for a in actions:
            if isOfficial:
                a = Official2Glendenning(a)

            if not accept_action_str(self.state, a):
                print(a)
                print("this action is impossible")
                return
            self.turn += 1
            self.add_history(self.state, a)

    def load_latest_record(self):
        print("load_latest_record")
        files = os.listdir(RECORDS_PATH)
        target_filename = sorted(files)[-1]
        self.load_to_history(os.path.join(RECORDS_PATH, target_filename, "record.txt"))

    def load_specified_record(self):
        print("load_specified_record")
        popup = FileChooserPopup(on_select=self.on_file_select)
        popup.open()

    def on_file_select(self, file_path):
        # 選択されたファイルのパスを処理
        print(f"選択されたファイル: {file_path}")
        self.load_to_history(file_path)

    def set_analyze_AIs(self):
        self.analyze_AIs = [self.prepare_AI(0, self.search_nodes, self.tau, self.level, seed=int(time.time())),
                            self.prepare_AI(1, self.search_nodes, self.tau, self.level, seed=int(time.time()))]

    def analyze_to_the_end(self):
        print("analyze_to_the_end")
        self.mode = ANALYZE_TO_THE_END_MODE
        self.set_analyze_AIs()
        # for ai in self.analyze_AIs:
        #     ai.init_prev()
        self.agents = self.analyze_AIs

    def get_multiaction_description_from_tree(self, tree, action, depth, first_call):
        if tree.result != 0:
            return Glendenning2Official(actionid2str(tree.s, tree.optimal_action))

        if depth <= 0 or action not in tree.children.keys():
            return ""
        
        nonzero_num = sum([int(np.sum(tree.children[a].N) >= 1) for a in range(len(tree.N)) if a in tree.children.keys()])
        if nonzero_num == 0:
            return ""
        
        next_tree = tree.children[action]
        next_action = np.argsort(next_tree.N)[-1]
        #print(depth, next_action)
        if first_call:
            text = get_description(tree.s, action, tree, is_simple=False)
        else:
            text = get_description(tree.s, action, tree, is_simple=True)
        return text + ", " + self.get_multiaction_description_from_tree(next_tree, next_action, depth - 1, False)

    def output_tree_to_TextInput(self, tree):
        root_score, action_cands = self.tree2info(tree)

        self.main_text.text += "-" * 20 + os.linesep
        self.main_text.text += self.treeinfo2text(tree, root_score, action_cands)

    def get_MCTS_tree(self, state):
        analyze_AI = self.analyze_AIs[state.turn % 2]
        analyze_AI.init_prev(state)  # 副作用があるかも
        _, MCTS_tree = analyze_AI.MCTS(state, self.search_nodes, analyze_AI.C_puct, self.tau, 
                                                 showNQ=False, noise=0., random_flip=False, use_prev_tree=self.use_prev_tree, opponent_prev_tree=None, return_root_tree=True)
        if self.graphviz_on.state == "down" and not state.pseudo_terminate:
            g = analyze_AI.get_tree_for_graphviz()
            if g is not None:
                g.render(os.path.join("game_trees", "game_tree{}".format(state.turn)))
        return MCTS_tree
    
    def tree2info(self, tree):
        if tree.result == 0:
            nonzero_num = sum([int(np.sum(tree.children[a].N) >= 1) for a in range(len(tree.N)) if a in tree.children.keys()])
            goodmove_num = np.sum(tree.N / np.sum(tree.N) >= GOOD_TH)
            action_cands = np.argsort(tree.N)[::-1][:min(10, max(3, goodmove_num), nonzero_num)]
            root_score = tree2score(tree, tree.s.turn)
        else:
            root_score = tree.result * 1000
            action_cands = [tree.optimal_action]
        return root_score, action_cands
    
    def treeinfo2text(self, tree, root_score, action_cands):
        ret = ""
        #ret += "-" * 20 + os.linesep
        ret += "turn {}".format(tree.s.turn) + os.linesep + "score = {} (nodes = {}, epoch = {})".format(root_score, self.search_nodes, self.target_epoch) + os.linesep
        for action in action_cands:
            #text = get_description(self.state, action, tree)
            text = self.get_multiaction_description_from_tree(tree, action, 3, True)
            if text is not None:
                ret += text + os.linesep
        return ret

    def analyze_this_turn(self):
        print("analyze_this_turn")

        MCTS_tree = self.get_MCTS_tree(self.state)
        self.output_tree_to_TextInput(MCTS_tree)

    def analyze_tab_f(self):
        print("Analyze mode")
        self.mode = ANALYZE_MODE

    def opening_tab_f(self):
        print("Opening mode")
        self.mode = OPENING_MODE

    def get_record_text(self, target_history):
        def f(s):
            return "{:>3}".format(Glendenning2Official(s))
        SHOW_CYCLE = 10
        official_action_history = list(map(f, target_history))
        ret = ""
        for i in range((len(official_action_history) - 1) // SHOW_CYCLE + 1):
            ret += ",".join(official_action_history[i * SHOW_CYCLE: (i + 1) * SHOW_CYCLE]) + os.linesep
        return ret

    def add_history(self, s, a):
        s = state_copy(s)

        if self.state_history is None:  # このときaction_historyもNoneとして実装
            self.state_history = [s]
            self.action_history = [a]
        if self.turn < len(self.state_history):
            self.state_history = self.state_history[:self.turn]
            self.action_history = self.action_history[:self.turn]

        self.state_history.append(s)
        self.action_history.append(a)

        self.record_text.text = self.get_record_text(self.action_history[1:])

    def oneturn(self, color):
        global touched, action

        if isinstance(self.agents[color], CNNAI):
            if time.time() - self.prev_act_time <= self.ai_wait_time:
                return
            self.agents[color].init_prev(self.state)  # analyze this turnと一致させるため。ただし、この場合通常のAI対戦と異なる振る舞いにはなる。
            s, tree = self.agents[color].act(self.state, use_prev_tree=self.use_prev_tree, return_root_tree=True)
            self.output_tree_to_TextInput(tree)
            print("use_prev_tree=" + str(self.use_prev_tree))

            if self.graphviz_on.state == "down" and not self.state.pseudo_terminate:
                g = self.agents[color].get_tree_for_graphviz()
                if g is not None:
                    g.render(os.path.join("game_trees", "game_tree{}".format(self.state.turn)))
        else:
            s = self.agents[color].act(self.state)
        action = ""

        if s == -1 or s == "":
            return

        if isinstance(s, int):
            a = actionid2str(self.state, s)
        else:
            a = s
        if not accept_action_str(self.state, a):
            print(a)
            print("this action is impossible")
            return
        print(Glendenning2Official(a))
        self.agents[1 - color].prev_action = s

        self.turn += 1
        self.add_history(self.state, a)

        # 盤面を動かしたときに、OpeningTreeの対応するノードがあれば情報を表示
        action_list, _ = get_normalized_action_list(self.action_history[1:self.turn + 1])

        node = self.opening_tree
        for a in action_list:
            node = move_to_child(node, Glendenning2Official(a), self.statevec2node)
            if node is None:
                break
        if node is not None and node.search_text is not None:
            _, _, is_mirror = get_normalized_state(self.action_history[1:self.turn + 1])
            if is_mirror:
                output_text = self.mirror_text(node.search_text)
            else:
                output_text = node.search_text
            self.main_text.text += "-" * 20 + os.linesep
            self.main_text.text += output_text
            self.main_text.text += "display = {}".format(node.is_display) + os.linesep

        array_save_path = os.path.join(DEBUG_DIR, "wall_array")
        os.makedirs(array_save_path, exist_ok=True)
        np.savetxt(os.path.join(array_save_path, "{}_r.txt".format(self.turn)), get_row_wall(self.state), fmt='%d')
        np.savetxt(os.path.join(array_save_path, "{}_c.txt".format(self.turn)), get_column_wall(self.state), fmt='%d')
        np.savetxt(os.path.join(array_save_path, "{}_w.txt".format(self.turn)), np.array([self.state.black_walls, self.state.white_walls]), fmt='%d')
        np.savetxt(os.path.join(array_save_path, "{}_pos.txt".format(self.turn)), np.array([self.state.Bx, self.state.By, self.state.Wx, self.state.Wy]), fmt='%d')

        self.prev_act_time = time.time()
        self.ai_wait_time = AI_WAIT_TIME
        touched = False

    def turn0(self):
        if self.turn >= 1:
            print(f"back to turn 0")
            self.turn = 0
            self.state = state_copy(self.state_history[self.turn])
            self.use_prev_tree = False  # undoを使った場合には、以降は差分計算しないことによってAIが意図しない探索をしないようにする
            self.prev_act_time = time.time()
            self.ai_wait_time = UNDO_WAIT_TIME
            #print(self.turn, self.state.turn)

    def undo(self):
        if self.turn >= 1:
            print(f"undo at turn {self.turn}")
            self.turn -= 1
            self.state = state_copy(self.state_history[self.turn])
            self.use_prev_tree = False  # undoを使った場合には、以降は差分計算しないことによってAIが意図しない探索をしないようにする
            self.prev_act_time = time.time()
            self.ai_wait_time = UNDO_WAIT_TIME
            #print(self.turn, self.state.turn)

    def redo(self):
        if self.state_history is not None and self.turn + 1 < len(self.state_history):
            print(f"redo at turn {self.turn}")
            self.turn += 1
            self.state = state_copy(self.state_history[self.turn])
            self.prev_act_time = time.time()
            self.ai_wait_time = UNDO_WAIT_TIME
            #print(self.turn, self.state.turn)

    def set_game_result(self, x):
        if self.game_result == "":
            print(x)
            print("updated")
            self.game_result = x

    def set_default_agents(self):
        agent1 = GUIHuman(0)
        agent2 = GUIHuman(1)
        self.agents = [agent1, agent2]

    def prepare_AI(self, color, search_nodes, tau, level, seed):
        per_process_gpu_memory_fraction = 0.2
        files = os.listdir(PARAMETER_PATH)
        files = [x for x in files if x.startswith("epoch")]
        epochs = [int(x.split(".")[0][5:]) for x in files]
        epochs = list(set(epochs))
        epochs = sorted(epochs)
        print(epochs)
        print(f"seed={seed}")
        if level == 0:
            agent = CNNAI(color, search_nodes=search_nodes, tau=tau, seed=seed,
                            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction, p_is_almost_flat=True, all_parameter_zero=True)
        else:
            agent = CNNAI(color, search_nodes=search_nodes, tau=tau, seed=seed,
                            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
            target_epoch = epochs[level - 1]
            self.target_epoch = target_epoch
            agent.load(os.path.join(PARAMETER_PATH, f"epoch{target_epoch}.ckpt"))
        return agent
    
    def start_game(self, state=None):
        global touched
        
        self.set_default_agents()

        self.set_analyze_AIs()

        self.state = State()
        State_init(self.state)
        self.turn = 0
        self.state_history = None
        self.add_history(self.state, None)
        self.prev_act_time = time.time()
        touched = False
        self.game_result = ""

    def end_game(self):
        if self.mode != ANALYZE_MODE:
            self.mode = ANALYZE_MODE
            self.set_default_agents()
        # win_player = 2 - self.state.turn % 2
        # self.set_game_result(f"{win_player}p win")

    def update(self, dt):

        if not self.state.terminate:
            # 各agentの手番処理
            self.oneturn(self.state.turn % 2)

        # 試合終了処理
        if self.state.terminate:
            self.end_game()

        #self.turn = self.state.turn
        if self.turn % 2 == 0:
            self.move_str = "player1 move"
        else:
            self.move_str = "player2 move"
        self.player1wall = self.state.black_walls
        self.player2wall = self.state.white_walls

        if self.mode == ANALYZE_MODE:
            self.upside_down = int(self.teban_2p.state == "down")
        else:
            self.upside_down = int(self.teban_2p_opening.state == "down")

        if self.upside_down:
            stateBx = 8 - self.state.Bx
            stateBy = 8 - self.state.By
            stateWx = 8 - self.state.Wx
            stateWy = 8 - self.state.Wy
        else:
            stateBx = self.state.Bx
            stateBy = self.state.By
            stateWx = self.state.Wx
            stateWy = self.state.Wy

        self.Bx = int(15 + (stateBx + 0.5) / 9 * (BOARD_LEN - 30))
        self.By = int(15 + (8.5 - stateBy) / 9 * (BOARD_LEN - 30))
        self.Wx = int(15 + (stateWx + 0.5) / 9 * (BOARD_LEN - 30))
        self.Wy = int(15 + (8.5 - stateWy) / 9 * (BOARD_LEN - 30))

        for x in range(8):
            for y in range(8):
                self.row_wall_colors[(7 - y) * 8 + x].a = 0
                self.column_wall_colors[(7 - y) * 8 + x].a = 0

        for x in range(8):
            for y in range(8):
                mouse_x, mouse_y = Window.mouse_pos
                if int(20 + (x + 0.5) / 9 * (BOARD_LEN - 30)) <= mouse_x < int(20 + (x + 1.5) / 9 * (BOARD_LEN - 30)) and int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) <= mouse_y <= int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10:
                    self.row_wall_colors[y * 8 + x].a = 0.5
                if int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) - 10 <= mouse_x <= int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) and int(10 + (y + 0.5) / 9 * (BOARD_LEN - 30)) <= mouse_y < int(10 + (y + 1.5) / 9 * (BOARD_LEN - 30)):
                    self.column_wall_colors[y * 8 + x].a = 0.5
                if int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) - 10 <= mouse_x <= int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) and int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) <= mouse_y <= int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10:
                    self.row_wall_colors[y * 8 + x].a = 0
                    self.column_wall_colors[y * 8 + x].a = 0

        row_wall = get_row_wall(self.state)
        column_wall = get_column_wall(self.state)
        for x in range(8):
            for y in range(8):
                if self.upside_down:
                    disp_x = 7 - x
                    disp_y = 7 - y
                else:
                    disp_x = x
                    disp_y = y

                if row_wall[x, y]:
                    self.row_wall_colors[(7 - disp_y) * 8 + disp_x].a = 1
                if column_wall[x, y]:
                    self.column_wall_colors[(7 - disp_y) * 8 + disp_x].a = 1

        self.search_nodes_label.text = f"search nodes = {self.search_nodes}"
        self.tau_label.text = f"randomness = {self.tau}"
        self.search_nodes_label_opening.text = f"search nodes = {self.search_nodes}"
        self.tau_label_opening.text = f"randomness = {self.tau}"

        #print(Window.mouse_pos)

    def on_touch_down(self, touch):
        global touched, action
        touched = True
        for x in range(9):
            for y in range(9):
                if self.upside_down:
                    wall_x = 7 - x
                    wall_y = 7 - y
                    action_x = 8 - x
                    action_y = 8 - y
                else:
                    wall_x = x
                    wall_y = y
                    action_x = x
                    action_y = y

                if int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) - 10 <= touch.x <= int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) and int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) <= touch.y <= int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10:
                    continue

                if int(20 + x / 9 * (BOARD_LEN - 30)) <= touch.x < int(10 + (x + 1) / 9 * (BOARD_LEN - 30)) and int(20 + y / 9 * (BOARD_LEN - 30)) <= touch.y < int(10 + (y + 1) / 9 * (BOARD_LEN - 30)):
                    action = chr(ord("a") + action_x) + str(9 - action_y)

                # 壁置きではx, yが8以降は無視する必要がある
                if x == 8 or y == 8:
                    continue
                if int(20 + (x + 0.5) / 9 * (BOARD_LEN - 30)) <= touch.x < int(20 + (x + 1.5) / 9 * (BOARD_LEN - 30)) and int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) <= touch.y <= int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10:
                    action = chr(ord("a") + wall_x) + str(8 - wall_y) + "h"
                if int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) - 10 <= touch.x <= int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) and int(10 + (y + 0.5) / 9 * (BOARD_LEN - 30)) <= touch.y < int(10 + (y + 1.5) / 9 * (BOARD_LEN - 30)):
                    action = chr(ord("a") + wall_x) + str(8 - wall_y) + "v"

        #print(touch.x, touch.y)
        super(Quoridor, self).on_touch_down(touch)

    def change_search_nodes(self, *args):
        self.search_nodes = SEARCH_NODE_LIST[int(args[1])]
        for ai in self.analyze_AIs:
            ai.search_nodes = self.search_nodes

    def change_tau(self, *args):
        self.tau = TAU_LIST[int(args[1])]
        for ai in self.analyze_AIs:
            ai.tau = self.tau

class QuoridorApp(App):
    def build(self):
        Builder.load_string(f"""
<Quoridor>:
    turn0_button: turn0_button
    undo_button: undo_button
    redo_button: redo_button
    graphviz_on: graphviz_on
    graphviz_off: graphviz_off

    mode_tab: mode_tab
    analyze_tab: analyze_tab
    opening_tab: opening_tab
    
    teban_1p: teban_1p
    teban_2p: teban_2p
    search_nodes_label: search_nodes_label
    tau_label: tau_label
                            
    analyze_this_turn_button: analyze_this_turn_button
    analyze_all_turns_button: analyze_all_turns_button
    analyze_to_the_end_button: analyze_to_the_end_button
    record_num_text: record_num_text
    specify_record_button: specify_record_button
    next_record_button: next_record_button
    load_latest_record_button: load_latest_record_button
    load_specified_record_button: load_specified_record_button
    board_as_text_button: board_as_text_button
    clear_text_button: clear_text_button
                            
    teban_1p_opening: teban_1p_opening
    teban_2p_opening: teban_2p_opening
    search_nodes_label_opening: search_nodes_label_opening
    tau_label_opening: tau_label_opening
                            
    analyze_this_turn_button_opening: analyze_this_turn_button_opening
    register_button: register_button
    analyze_to_the_end_button_opening: analyze_to_the_end_button_opening
    save_as_html_button: save_as_html_button
    analyze_all_nodes_button: analyze_all_nodes_button
    name_this_node_button: name_this_node_button
    comment_this_node_button: comment_this_node_button
    clear_text_button_opening: clear_text_button_opening
                           
    user_input_text: user_input_text
    record_text: record_text
    main_text: main_text

    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: (0, 0)
            size: ({WINDOW_WIDTH}, 600)
        Color:
            rgba: 1, 0, 0, 1

    canvas:
        Color:
            rgba: 1.0, 1.0, 1.0, 1
        Triangle:
            points: (root.Bx - 20, root.By - 20 + root.upside_down * 40, root.Bx, root.By + 20 - root.upside_down * 40, root.Bx + 20, root.By - 20 + root.upside_down * 40)
        Color:
            rgba: 0, 0, 0, 1
        Triangle:
            points: (root.Wx - 20, root.Wy + 20 - root.upside_down * 40, root.Wx, root.Wy - 20 + root.upside_down * 40, root.Wx + 20, root.Wy + 20 - root.upside_down * 40)

    Label:
        font_size: 30
        pos: (620, 530)
        color: 0, 0, 0, 1
        text: "turn:" + str(root.turn)
    Label:
        font_size: 30
        pos: (680, 500)
        color: 0, 0, 0, 1
        text: root.move_str
    Label:
        font_size: 30
        pos: (680, 470)
        color: 0, 0, 0, 1
        text: "player1 wall:" + str(root.player1wall)
    Label:
        font_size: 30
        pos: (680, 440)
        color: 0, 0, 0, 1
        text: "player2 wall:" + str(root.player2wall)
    Label:
        font_size: 30
        pos: (850, 500)
        color: 0, 0, 0, 1
        text: root.game_result

    Button:
        id: turn0_button
        pos: (595, 100)
        size: (100, 40)
        text: "Back to turn 0"
    Button:
        id: undo_button
        pos: (700, 100)
        size: (80, 40)
        text: "Undo"
    Button:
        id: redo_button
        pos: (790, 100)
        size: (80, 40)
        text: "Redo"

    Label:
        pos: (610, 40)
        size: (80, 40)
        color: 0, 0, 0, 1
        text: 'Graphviz'
    ToggleButton:
        pos: (600, 10)
        size: (50, 40)
        id: graphviz_on
        text: "On"
        group: "graphviz"
    ToggleButton:
        pos: (650, 10)
        size: (50, 40)
        id: graphviz_off
        text: "Off"
        group: "graphviz"
        state: "down"
    TextInput:
        pos: (710, 10)
        size: (200, 80)
        id: user_input_text
        text: ""
        multiline: True

    TabbedPanel:
        id: mode_tab
        do_default_tab: False
        tab_width: self.width // 2
        pos: (600, 150)
        size: (300, 320)
                    
        TabbedPanelItem:
            id: analyze_tab
            text: 'Analyze'
            font_size: 15
            
            BoxLayout:
                orientation: 'vertical'
                spacing: 10
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5
                    padding: 5, 5
                    Label:
                        text: '1p or 2p'
                        size_hint: 0.5, 1.0
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: teban_1p
                        text: "1p"
                        group: "teban"
                        state: "down"
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: teban_2p
                        text: "2p"
                        group: "teban"
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: 18
                        value: {DEFAULT_SEARCH_NODE_INDEX}
                        step: 1
                        cursor_size: 20, 20
                        orientation: "horizontal"
                        on_value: root.change_search_nodes(*args)   
                    Label:
                        id: search_nodes_label
                        font_size: 15
                        text: "search nodes ="
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: 4
                        value: {DEFAULT_TAU_INDEX}
                        step: 1
                        cursor_size: 20, 20
                        orientation: "horizontal"
                        on_value: root.change_tau(*args)   
                    Label:
                        id: tau_label
                        font_size: 15
                        text: "randomness ="
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5
                    Button:
                        size_hint: 0.5, 1.0
                        id: analyze_this_turn_button
                        text: "Analyze this turn"
                    Button:
                        size_hint: 0.5, 1.0
                        id: analyze_all_turns_button
                        text: "analyze all turns"
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5
                    Button:
                        size_hint: 0.5, 1.0
                        id: analyze_to_the_end_button
                        text: "AI moves to the end"
                    BoxLayout:
                        size_hint: 0.5, 1.0
                        orientation: 'horizontal'
                        spacing: 5
                        TextInput:
                            size_hint: 0.5, 1.0
                            id: record_num_text
                        Button:
                            size_hint: 0.5, 1.0
                            id: specify_record_button
                            text: "go"
                        Button:
                            size_hint: 0.5, 1.0
                            id: next_record_button
                            text: "next"
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5
                    Button:
                        size_hint: 0.5, 1.0
                        id: load_latest_record_button
                        text: "Load latest record"
                    Button:
                        size_hint: 0.5, 1.0
                        id: load_specified_record_button
                        text: "Load specified record"
                        font_size: 13

                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5
                    Button:
                        size_hint: 0.5, 1.0
                        id: board_as_text_button
                        text: "Board as text"
                    Button:
                        size_hint: 0.5, 1.0
                        id: clear_text_button
                        text: "Clear text"
                    
        TabbedPanelItem:
            id: opening_tab
            text: 'Opening'
            font_size: 15
            
            BoxLayout:
                orientation: 'vertical'
                spacing: 10
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5
                    padding: 5, 5
                    Label:
                        text: '1p or 2p'
                        size_hint: 0.5, 1.0
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: teban_1p_opening
                        text: "1p"
                        group: "teban"
                        state: "down"
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: teban_2p_opening
                        text: "2p"
                        group: "teban"
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: 18
                        value: {DEFAULT_SEARCH_NODE_INDEX}
                        step: 1
                        cursor_size: 20, 20
                        orientation: "horizontal"
                        on_value: root.change_search_nodes(*args)   
                    Label:
                        id: search_nodes_label_opening
                        font_size: 15
                        text: "search nodes ="
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: 4
                        value: {DEFAULT_TAU_INDEX}
                        step: 1
                        cursor_size: 20, 20
                        orientation: "horizontal"
                        on_value: root.change_tau(*args)   
                    Label:
                        id: tau_label_opening
                        font_size: 15
                        text: "randomness ="
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5

                    Button:
                        size_hint: 0.5, 1.0
                        id: analyze_this_turn_button_opening
                        text: "Analyze this turn"
                    Button:
                        size_hint: 0.5, 1.0
                        id: register_button
                        text: "Register"
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5

                    Button:
                        size_hint: 0.5, 1.0
                        id: analyze_to_the_end_button_opening
                        text: "AI moves to the end"
                    Button:
                        size_hint: 0.5, 1.0
                        id: save_as_html_button
                        text: "Save as html"
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5

                    Button:
                        size_hint: 0.5, 1.0
                        id: analyze_all_nodes_button
                        text: "Analyze all nodes"
                    Button:
                        size_hint: 0.5, 1.0
                        id: name_this_node_button
                        text: "Name this node"
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'horizontal'
                    spacing: 5

                    Button:
                        size_hint: 0.5, 1.0
                        id: comment_this_node_button
                        text: "Comment this node"
                    Button:
                        size_hint: 0.5, 1.0
                        id: clear_text_button_opening
                        text: "Clear text"

    TextInput:
        id: record_text
        pos: (950, 400)
        size: (300, 150)
        multiline: True
        readonly: True
    TextInput:
        id: main_text
        pos: (950, 30)
        size: (300, 350)
        multiline: True
        scroll_y: 0 
        scrollbar_size: 20 
        scrollbars: 'vertical'
        readonly: True
        """)
        #game = Builder.load_file(os.path.join(os.getcwd(), "quoridor.kv"))
        #print(type(game))
        game = Quoridor()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game

# class SecondWindow(App):
#     def build(self):
#         return Button(text='This is the second window')


if __name__ == '__main__':
    QuoridorApp().run()



