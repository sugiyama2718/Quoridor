# coding:utf-8

BOARD_LEN = 600
WINDOW_WIDTH = 1000

from kivy.config import Config
from kivy.metrics import dp
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

from Agent import actionid2str, str2actionid
from State import State, CHANNEL, State_init, accept_action_str, display_cui, get_row_wall, get_column_wall
from State import DRAW_TURN
from CNNAI import CNNAI
from BasicAI import state_copy
import time
import sys
import numpy as np
import gc
import os
from Agent import Agent
import random
import pickle
from util import Glendenning2Official, RECORDS_PATH
from datetime import datetime
from engine_util import prepare_AI

touched = False
action = ""

SEARCH_NODE_LIST = [1, 100, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 100000]
SEARCH_NODE_LIST_LEN = len(SEARCH_NODE_LIST)
TAU_LIST = [0.16 * i for i in range(5)]
DEFAULT_SEARCH_NODE_INDEX = 6
DEFAULT_TAU_INDEX = 2

AI_WAIT_TIME = 0.1  # AIが考え始めるまでに待機する時間[s]
UNDO_WAIT_TIME = 1.0  # 対AIのときにundoを2⃣回押しやすくするために待つ時間を増やす

TRAINING_HIGH_TIME = 10 * 60.0  # [s]
TRAINING_LOW_TIME = 3 * 60.0  # [s]
TRAINING_HIGH_INCREMENT_TIME = 20
TRAINING_LOW_INCREMENT_TIME = 6
TRAINING_RESULT_HISTORY_LEN_LOW = 10
TRAINING_RESULT_HISTORY_LEN_HIGH = 10
ACHIEVE_TH = 0.6 - 1e-10  # この勝率を上回ったら合格扱い
JOSEKI_WAIT_TIME = 0.4  # 定跡のとき待機する時間。定跡は打っている感じを出すため長めに待機する。

DATA_BASE_DIR = "application_data"
PARAMETER_PATH = os.path.join(DATA_BASE_DIR, "parameter")
PLAYER_DATA_DIR = os.path.join(DATA_BASE_DIR, "player_data")
JOSEKI_PATH = os.path.join(DATA_BASE_DIR, "joseki", "joseki_240310.txt")
with open(JOSEKI_PATH, "r") as fin:
    joseki_text = fin.read()
    joseki_list = joseki_text.strip().split("\n")
    joseki_num = len(joseki_list)
os.makedirs(PLAYER_DATA_DIR, exist_ok=True)

param_files = os.listdir(PARAMETER_PATH)
epoch_list = [0] + sorted(list(set([int(s.split(".")[0][5:]) for s in param_files])))
LEVEL_NUM = len(epoch_list)

# TRAINING_LIST = [(0, 500), (1, 500), (2, 500), (3, 500), (4, 500), (5, 500), (6, 500), (7, 500), (8, 500), (9, 500), 
#                  (10, 500), (11, 500), (12, 500), (13, 200), (13, 500), (14, 500), (15, 500), (16, 500), (17, 500), (18, 500), (19, 500), (20, 500), (21, 500)]
TRAINING_LIST = [(i, 500) for i in range(15)]
TRAINING_LEVEL_NUM = len(TRAINING_LIST)

# 対戦モードid
HUMAN_HUMAN_MODE = 0
HUMAN_AI_MODE = 1
AI_AI_MODE = 2
TRAINING_MODE = 3


class GUIHuman(Agent):
    def act(self, state, showNQ=False):
        global touched, action
        if touched:
            touched = False
            return action
        return -1


class Quoridor(Widget):
    turn = NumericProperty(0)
    move_str = StringProperty("player1 move")
    player1wall = NumericProperty(10)
    player2wall = NumericProperty(10)
    Bx = NumericProperty(0)
    By = NumericProperty(0)
    Wx = NumericProperty(0)
    Wy = NumericProperty(0)
    button = ObjectProperty(None)
    turn0_button = ObjectProperty(None)
    undo_button = ObjectProperty(None)
    redo_button = ObjectProperty(None)
    resign_button = ObjectProperty(None)
    mode_tab = ObjectProperty(None)
    human_tab = ObjectProperty(None)
    human_ai_tab =ObjectProperty(None)
    ai_ai_tab = ObjectProperty(None)
    training_tab = ObjectProperty(None)

    flip_on = ObjectProperty(None)
    flip_off = ObjectProperty(None)
    graphviz_on = ObjectProperty(None)
    graphviz_off = ObjectProperty(None)
    teban_1p = ObjectProperty(None)
    teban_2p = ObjectProperty(None)
    view_1p = ObjectProperty(None)
    view_2p = ObjectProperty(None)
    upside_down = NumericProperty(0)

    low_time = ObjectProperty(None)
    high_time = ObjectProperty(None)
    training_level_label = ObjectProperty(None)
    player_level_label = ObjectProperty(None)
    training_info = ObjectProperty(None)
    remaining_time_str = StringProperty("")

    # Human vs. AI
    search_nodes = SEARCH_NODE_LIST[DEFAULT_SEARCH_NODE_INDEX]
    level = LEVEL_NUM - 1
    tau = TAU_LIST[DEFAULT_TAU_INDEX]

    # AI vs. AI
    search_nodes_1p = SEARCH_NODE_LIST[DEFAULT_SEARCH_NODE_INDEX]
    level_1p = LEVEL_NUM - 1
    tau_1p = TAU_LIST[DEFAULT_TAU_INDEX]
    search_nodes_2p = SEARCH_NODE_LIST[DEFAULT_SEARCH_NODE_INDEX]
    level_2p = LEVEL_NUM - 1
    tau_2p = TAU_LIST[DEFAULT_TAU_INDEX]

    # Training
    level_training = TRAINING_LEVEL_NUM - 1

    game_result = StringProperty("")

    def dont_down(self, button):
        self.training_level_label.text = ""
        if button.state != "down":
            button.state = "down"

    def __init__(self, **kwargs):
        super(Quoridor, self).__init__(**kwargs)
        self.state = State()
        State_init(self.state)
        self.agents = [GUIHuman(0), CNNAI(1, search_nodes=self.search_nodes, tau=0.5)]
        self.playing_game = False

        self.button.bind(on_release=lambda touch: self.start_game())
        self.turn0_button.bind(on_release=lambda touch: self.turn0())
        self.undo_button.bind(on_release=lambda touch: self.undo())
        self.redo_button.bind(on_release=lambda touch: self.redo())
        self.resign_button.bind(on_release=lambda touch: self.resign())
        self.graphviz_on.bind(on_press=lambda touch: self.dont_down(self.graphviz_on))
        self.graphviz_off.bind(on_press=lambda touch: self.dont_down(self.graphviz_off))

        self.human_tab.bind(on_press=lambda touch: self.human_tab_f())
        self.human_ai_tab.bind(on_press=lambda touch: self.human_ai_tab_f())
        self.ai_ai_tab.bind(on_press=lambda touch: self.ai_ai_tab_f())
        self.training_tab.bind(on_press=lambda touch: self.training_tab_f())

        self.flip_on.bind(on_press=lambda touch: self.dont_down(self.flip_on))
        self.flip_off.bind(on_press=lambda touch: self.dont_down(self.flip_off))
        self.teban_1p.bind(on_press=lambda touch: self.dont_down(self.teban_1p))
        self.teban_2p.bind(on_press=lambda touch: self.dont_down(self.teban_2p))
        self.view_1p.bind(on_press=lambda touch: self.dont_down(self.view_1p))
        self.view_2p.bind(on_press=lambda touch: self.dont_down(self.view_2p))

        self.low_time.bind(on_press=lambda touch: self.dont_down(self.low_time))
        self.high_time.bind(on_press=lambda touch: self.dont_down(self.high_time))

        self.mode = HUMAN_HUMAN_MODE
        self.upside_down = 0

        self.use_prev_tree = True
        self.prev_act_time = time.time()
        self.ai_wait_time = AI_WAIT_TIME

        self.row_wall_colors = [Color(0.7, 0.7, 0, 0) for i in range(64)]
        self.column_wall_colors = [Color(0.7, 0.7, 0, 0) for i in range(64)]

        self.state_history = None
        self.action_history = None
        self.remaining_time = 0.0

        self.training_color = 0
        self.player_level = -1

        self.is_resign = False

        self.play_count_list = [0] * TRAINING_LEVEL_NUM
        self.play_count_path = os.path.join(PLAYER_DATA_DIR, "play_count.txt")
        if os.path.exists(self.play_count_path):
            self.play_count_list = list(np.asarray(np.loadtxt(self.play_count_path), dtype=int))

        self.record_folder_name = None

        with self.canvas.before:
            Color(96/255, 32/128, 0, 1)
            Rectangle(pos=(dp(10), dp(10)), size=(dp(BOARD_LEN - 20), dp(BOARD_LEN - 20)))
            Color(64/255, 0, 0, 1)
            for i in range(10):
                Rectangle(pos=(int(dp(10 + i / 9 * (BOARD_LEN - 30))), dp(10)), size=(dp(10), dp(BOARD_LEN - 20)))
            for i in range(10):
                Rectangle(pos=(dp(10), dp(int(10 + i / 9 * (BOARD_LEN - 30)))), size=(dp(BOARD_LEN - 20), dp(10)))

        for i, color in enumerate(self.row_wall_colors):
            self.canvas.add(color)
            x = i % 8
            y = i // 8
            self.canvas.add(Rectangle(pos=(dp(int(20 + x / 9 * (BOARD_LEN - 30))), dp(int(10 + (y + 1) / 9 * (BOARD_LEN - 30)))), size=(dp((BOARD_LEN - 30) // 9 * 2 - 10), dp(10))))
        for i, color in enumerate(self.column_wall_colors):
            self.canvas.add(color)
            x = i % 8
            y = i // 8
            self.canvas.add(Rectangle(pos=(dp(int(10 + (x + 1) / 9 * (BOARD_LEN - 30))), dp(int(20 + y / 9 * (BOARD_LEN - 30)))), size=(dp(10), dp((BOARD_LEN - 30) // 9 * 2 - 10))))

        self.virtual_human = GUIHuman(0)  # 対戦前の指し手用

        self.add_history(self.state, None, is_record=False)

    def human_tab_f(self):
        print("Human vs. Human")
        self.mode = HUMAN_HUMAN_MODE

    def human_ai_tab_f(self):
        print("Human vs. AI")
        self.mode = HUMAN_AI_MODE

    def ai_ai_tab_f(self):
        print("AI vs. AI")
        self.mode = AI_AI_MODE

    def training_tab_f(self):
        print("Training")
        self.mode = TRAINING_MODE 

    def add_history(self, s, a, is_record=True):
        s = state_copy(s)

        if self.state_history is None:  # このときaction_historyもNoneとして実装
            self.state_history = [s]
            self.action_history = [a]
        if self.turn < len(self.state_history):
            self.state_history = self.state_history[:self.turn]
            self.action_history = self.action_history[:self.turn]

        self.state_history.append(s)
        self.action_history.append(a)

        if is_record:
            action_strs = map(Glendenning2Official, self.action_history[1:])  # 0はNone
            path = os.path.join(self.record_folder_name, "record.txt")
            with open(path, 'w') as fout:
                fout.write(",".join(action_strs))

    def oneturn_notplaying(self):
        global touched, action

        s = self.virtual_human.act(self.state)
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

        display_cui(self.state)
        self.turn += 1
        self.add_history(self.state, a, is_record=False)

        touched = False

    def oneturn(self, color):
        global touched, action

        if self.mode == TRAINING_MODE and self.turn < len(self.training_joseki):
            self.prev_time = time.time()
            self.prev_remaining_time = self.remaining_time
            if time.time() - self.prev_act_time <= JOSEKI_WAIT_TIME:
                return
            s = self.training_joseki[self.turn]
        elif isinstance(self.agents[color], CNNAI):
            if time.time() - self.prev_act_time <= self.ai_wait_time:
                return
            s, _, _, v_post, _ = self.agents[color].act_and_get_pi(self.state, use_prev_tree=self.use_prev_tree)
            print("score= {}, use_prev_tree={}".format(int(1000 * v_post), self.use_prev_tree))

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
            self.agents[1 - color].prev_action = s
        else:
            a = s
            self.agents[1 - color].prev_action = str2actionid(self.state, s)

        if not accept_action_str(self.state, a):
            print(a)
            print("this action is impossible")
            return
        print(Glendenning2Official(a))

        display_cui(self.state)
        self.turn += 1
        self.add_history(self.state, a)

        self.prev_act_time = time.time()
        self.ai_wait_time = AI_WAIT_TIME
        touched = False

        if self.mode == TRAINING_MODE and isinstance(self.agents[color], CNNAI) and self.turn >= len(self.training_joseki) and self.remaining_time >= 0.0:
            if self.low_time.state == "down":
                self.remaining_time += TRAINING_LOW_INCREMENT_TIME
            elif self.high_time.state == "down":
                self.remaining_time += TRAINING_HIGH_INCREMENT_TIME

        if self.mode == TRAINING_MODE and isinstance(self.agents[color], CNNAI):
            self.prev_time = time.time()
            self.prev_remaining_time = self.remaining_time

    def turn0(self):
        if self.turn >= 1:
            print(f"back to turn {self.turn}")
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

    def resign(self):
        print("resign")
        self.is_resign = True
        self.playing_game = False
        self.end_game()

    def set_remaining_time_str(self):
        if self.remaining_time >= 0.0:
            m = int(self.remaining_time) // 60
            s = self.remaining_time - m * 60
            self.remaining_time_str = "time: {}:{:.1f}".format(m, s)
        else:
            self.remaining_time_str = "time: 0:0.0"

    def set_game_result(self, x):
        if self.game_result == "":
            print(x)
            print("updated")
            self.game_result = x

    def start_game(self):
        global touched

        now = datetime.now()
        # Windowsフォルダ名に使用できる形式にフォーマット
        self.record_folder_name = os.path.join(RECORDS_PATH, now.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.record_folder_name)

        self.remaining_time_str = ""

        if self.mode == HUMAN_HUMAN_MODE:
            agent1 = GUIHuman(0)
            agent2 = GUIHuman(1)
        elif self.mode == HUMAN_AI_MODE:
            if self.teban_1p.state == "down":
                agent1 = GUIHuman(0)
                agent2 = prepare_AI(PARAMETER_PATH, 1, self.search_nodes, self.tau, self.level, seed=int(time.time()))
            else:
                agent1 = prepare_AI(PARAMETER_PATH, 0, self.search_nodes, self.tau, self.level, seed=int(time.time()))
                agent2 = GUIHuman(1)
        elif self.mode == AI_AI_MODE:
            agent1 = prepare_AI(PARAMETER_PATH, 0, self.search_nodes_1p, self.tau_1p, self.level_1p, seed=int(time.time()))
            agent2 = prepare_AI(PARAMETER_PATH, 1, self.search_nodes_2p, self.tau_2p, self.level_2p, seed=int(time.time()))
        elif self.mode == TRAINING_MODE:
            if self.low_time.state == "down":
                self.remaining_time = TRAINING_LOW_TIME
            elif self.high_time.state == "down":
                self.remaining_time = TRAINING_HIGH_TIME
            self.prev_time = time.time()
            self.prev_remaining_time = self.remaining_time
            self.set_remaining_time_str()

            training_index, training_search_nodes = TRAINING_LIST[self.level_training]

            if os.path.exists(self.training_level_dir):
                with open(os.path.join(self.training_level_dir, "game_num.txt"), "r") as fin:
                    self.training_game_num = int(fin.read())

                self.joseki_random_index_list_sente = list(np.loadtxt(os.path.join(self.training_level_dir, "joseki_random_index_sente.txt")))
                self.joseki_random_index_list_sente = [int(x) for x in self.joseki_random_index_list_sente]
                self.joseki_random_index_list_gote = list(np.loadtxt(os.path.join(self.training_level_dir, "joseki_random_index_gote.txt")))
                self.joseki_random_index_list_gote = [int(x) for x in self.joseki_random_index_list_gote]
            else:
                self.training_game_num = 0
                self.joseki_random_index_list_sente = list(range(joseki_num))
                self.joseki_random_index_list_gote = list(range(joseki_num))
                for _ in range(self.level_training):
                    random.random()  # levelごとに別の乱数を使うため
                random.shuffle(self.joseki_random_index_list_sente)
                random.shuffle(self.joseki_random_index_list_gote)

            if self.training_game_num % 2 == 0:
                self.training_color = 0
                agent1 = GUIHuman(0)
                agent2 = prepare_AI(PARAMETER_PATH, 1, training_search_nodes, 0.32, training_index, seed=int(time.time()))
            else:
                self.training_color = 1
                agent1 = prepare_AI(PARAMETER_PATH, 0, training_search_nodes, 0.32, training_index, seed=int(time.time()))
                agent2 = GUIHuman(1)

            training_info_text = "You are "
            random_list_index = (self.training_game_num // 2) % joseki_num
            if self.training_color == 0:
                joseki_index = self.joseki_random_index_list_sente[random_list_index]
                training_info_text += "player1"
            else:
                joseki_index = self.joseki_random_index_list_gote[random_list_index]
                training_info_text += "player2"
            self.training_joseki = joseki_list[joseki_index].strip().split(",")

            joseki_disp = map(Glendenning2Official, self.training_joseki)

            self.training_info.text = training_info_text + os.linesep + "joseki: {}".format(",".join(joseki_disp))
            print(random_list_index, joseki_index)
            print(self.training_joseki)

        self.agents = [agent1, agent2]

        if self.mode == TRAINING_MODE or self.state.terminate:
            self.state = State()
            State_init(self.state)
            self.turn = 0
            self.state_history = None
            self.add_history(self.state, None)
        else:
            # 現局面をそのまま使う
            pass
        self.playing_game = True
        self.use_prev_tree = True
        self.prev_act_time = time.time()
        touched = False
        self.game_result = ""
        self.is_resign = False

    def end_game(self):
        if self.mode == TRAINING_MODE:  # trainingの結果の記録
            if not os.path.exists(self.training_level_dir):
                os.makedirs(self.training_level_dir)

            if self.state.turn % 2 == 1 - self.training_color and not self.is_resign:
                self.set_game_result("You win")
            else:
                self.set_game_result("You lose")

            # 時間切れ含めた勝敗の結果を求める
            if self.game_result == "You win":
                result = 1
            else:
                result = 0

            player_level_path = os.path.join(PLAYER_DATA_DIR, "player_level.txt")
            if os.path.exists(player_level_path):
                self.player_level = int(open(player_level_path, "r").read())

            if self.low_time.state == "down": 
                result_history_path = os.path.join(self.training_level_dir, "result_history_low.pkl")
                achieve_path = os.path.join(self.training_level_dir, "achieve_low.pkl")
                history_len = TRAINING_RESULT_HISTORY_LEN_LOW
            elif self.high_time.state == "down":
                result_history_path = os.path.join(self.training_level_dir, "result_history_high.pkl")
                achieve_path = os.path.join(self.training_level_dir, "achieve_high.pkl")
                history_len = TRAINING_RESULT_HISTORY_LEN_HIGH

            if not os.path.exists(achieve_path):
                achieve = 0
            else:
                achieve = pickle.load(open(achieve_path, "rb"))
            
            if not os.path.exists(result_history_path):
                result_history = [result]
            else:
                result_history = pickle.load(open(result_history_path, "rb"))
                result_history.append(result)
                if len(result_history) > history_len:
                    result_history = result_history[1:]
                if len(result_history) >= history_len and sum(result_history) / len(result_history) >= ACHIEVE_TH:
                    achieve = 1
                    if self.level_training > self.player_level:
                        self.player_level = self.level_training
                        open(player_level_path, "w").write(str(self.player_level))
                    
            print(result_history, achieve)
            pickle.dump(result_history, open(result_history_path, "wb"))
            pickle.dump(achieve, open(achieve_path, "wb"))

            self.training_game_num += 1
            with open(os.path.join(self.training_level_dir, "game_num.txt"), "w") as fout:
                fout.write(str(self.training_game_num))

            np.savetxt(os.path.join(self.training_level_dir, "joseki_random_index_sente.txt"), np.array(self.joseki_random_index_list_sente, dtype=int), fmt='%d')
            np.savetxt(os.path.join(self.training_level_dir, "joseki_random_index_gote.txt"), np.array(self.joseki_random_index_list_gote, dtype=int), fmt='%d')

            self.play_count_list[self.level_training] += 1
            np.savetxt(self.play_count_path, np.array(self.play_count_list, dtype=int), fmt='%d')

            self.training_info.text = ""

            print("training recorded")
        else:
            win_player = 2 - self.state.turn % 2  # resignの有無にかかわらずこの条件で判定可能
            self.set_game_result(f"{win_player}p win")

        self.training_level_label.text = ""

    def update(self, dt):
        self.training_level_dir = os.path.join(PLAYER_DATA_DIR, str(self.level_training))

        if self.playing_game and not self.state.terminate:
            # 各agentの手番処理
            self.oneturn(self.state.turn % 2)

            # 時間計測
            if self.mode == TRAINING_MODE:
                self.remaining_time = self.prev_remaining_time - (time.time() - self.prev_time)
                self.set_remaining_time_str()
                if self.remaining_time <= 0.0:
                    self.set_game_result("You lose")

        elif not self.playing_game:
            self.oneturn_notplaying()

        # 試合終了処理
        if self.state.terminate and self.playing_game:
            self.playing_game = False
            self.end_game()

        #self.turn = self.state.turn
        if self.turn % 2 == 0:
            self.move_str = "player1 move"
        else:
            self.move_str = "player2 move"
        self.player1wall = self.state.black_walls
        self.player2wall = self.state.white_walls

        if self.mode == HUMAN_HUMAN_MODE:
            self.upside_down = int(self.flip_on.state == "down")
        elif self.mode == HUMAN_AI_MODE:
            self.upside_down = int(self.view_2p.state == "down")
        elif self.mode == AI_AI_MODE:
            self.upside_down = 0  # quioridor.kvで三角形の向きを変えるのに使う
        elif self.mode == TRAINING_MODE:
            self.upside_down = self.training_color

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

        self.Bx = dp(int(15 + (stateBx + 0.5) / 9 * (BOARD_LEN - 30)))
        self.By = dp(int(15 + (8.5 - stateBy) / 9 * (BOARD_LEN - 30)))
        self.Wx = dp(int(15 + (stateWx + 0.5) / 9 * (BOARD_LEN - 30)))
        self.Wy = dp(int(15 + (8.5 - stateWy) / 9 * (BOARD_LEN - 30)))

        for x in range(8):
            for y in range(8):
                self.row_wall_colors[(7 - y) * 8 + x].a = 0
                self.column_wall_colors[(7 - y) * 8 + x].a = 0

        for x in range(8):
            for y in range(8):
                mouse_x, mouse_y = Window.mouse_pos
                if int(dp(20 + (x + 0.5) / 9 * (BOARD_LEN - 30))) <= mouse_x < int(dp(20 + (x + 1.5) / 9 * (BOARD_LEN - 30))) and int(dp(10 + (y + 1) / 9 * (BOARD_LEN - 30))) <= mouse_y <= dp(int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10):
                    self.row_wall_colors[y * 8 + x].a = 0.5
                if int(dp(20 + (x + 1) / 9 * (BOARD_LEN - 30))) - 10 <= mouse_x <= int(dp(20 + (x + 1) / 9 * (BOARD_LEN - 30))) and int(dp(10 + (y + 0.5) / 9 * (BOARD_LEN - 30))) <= mouse_y < dp(int(10 + (y + 1.5) / 9 * (BOARD_LEN - 30))):
                    self.column_wall_colors[y * 8 + x].a = 0.5
                if int(dp(20 + (x + 1) / 9 * (BOARD_LEN - 30))) - 10 <= mouse_x <= int(dp(20 + (x + 1) / 9 * (BOARD_LEN - 30))) and int(dp(10 + (y + 1) / 9 * (BOARD_LEN - 30))) <= mouse_y <= dp(int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10):
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
        self.level_label.text = "level = {} (epoch {})".format(self.level, epoch_list[self.level])
        self.tau_label.text = f"randomness = {self.tau}"

        self.search_nodes_label_1p.text = f"1p search nodes = {self.search_nodes_1p}"
        self.level_label_1p.text = "1p level = {} (epoch {})".format(self.level_1p, epoch_list[self.level_1p])
        self.tau_label_1p.text = f"1p randomness = {self.tau_1p}"

        self.search_nodes_label_2p.text = f"2p search nodes = {self.search_nodes_2p}"
        self.level_label_2p.text = "2p level = {} (epoch {})".format(self.level_2p, epoch_list[self.level_2p])
        self.tau_label_2p.text = f"2p randomness = {self.tau_2p}"

        # 戦績の表示。pickle.loadが毎フレーム生じないように工夫する
        if self.training_level_label.text == "":

            if self.low_time.state == "down": 
                result_history_path = os.path.join(self.training_level_dir, "result_history_low.pkl")
                achieve_path = os.path.join(self.training_level_dir, "achieve_low.pkl")
            elif self.high_time.state == "down":
                result_history_path = os.path.join(self.training_level_dir, "result_history_high.pkl")
                achieve_path = os.path.join(self.training_level_dir, "achieve_high.pkl")
            if not os.path.exists(result_history_path):
                result_history = []
            else:
                result_history = pickle.load(open(result_history_path, "rb"))
            if not os.path.exists(achieve_path):
                achieve = 0
            else:
                achieve = pickle.load(open(achieve_path, "rb"))
            training_index, training_search_nodes = TRAINING_LIST[self.level_training]
            self.training_level_label.text = "difficulty = {} (epoch {}, search nodes {})\n history={} {}/{}\n".format(self.level_training, epoch_list[training_index],
                                                                                                                training_search_nodes, result_history[::-1], sum(result_history), len(result_history))
            
            self.training_level_label.text += "play count={}".format(self.play_count_list[self.level_training])
            
            if achieve == 1:
                self.training_level_label.text += " passed this level"

            player_level_path = os.path.join(PLAYER_DATA_DIR, "player_level.txt")
            if os.path.exists(player_level_path):
                self.player_level = int(open(player_level_path, "r").read())
            total_play_count = sum(self.play_count_list)
            self.player_level_label.text = f"Your level={self.player_level}, total play count={total_play_count}"

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

                if dp(int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) - 10) <= touch.x <= dp(int(20 + (x + 1) / 9 * (BOARD_LEN - 30))) and dp(int(10 + (y + 1) / 9 * (BOARD_LEN - 30))) <= touch.y <= dp(int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10):
                    continue

                if dp(int(20 + x / 9 * (BOARD_LEN - 30))) <= touch.x < dp(int(10 + (x + 1) / 9 * (BOARD_LEN - 30))) and dp(int(20 + y / 9 * (BOARD_LEN - 30))) <= touch.y < dp(int(10 + (y + 1) / 9 * (BOARD_LEN - 30))):
                    action = chr(ord("a") + action_x) + str(9 - action_y)

                # 壁置きではx, yが8以降は無視する必要がある
                if x == 8 or y == 8:
                    continue
                if dp(int(20 + (x + 0.5) / 9 * (BOARD_LEN - 30))) <= touch.x < dp(int(20 + (x + 1.5) / 9 * (BOARD_LEN - 30))) and dp(int(10 + (y + 1) / 9 * (BOARD_LEN - 30))) <= touch.y <= dp(int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10):
                    action = chr(ord("a") + wall_x) + str(8 - wall_y) + "h"
                if dp(int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) - 10) <= touch.x <= dp(int(20 + (x + 1) / 9 * (BOARD_LEN - 30))) and dp(int(10 + (y + 0.5) / 9 * (BOARD_LEN - 30))) <= touch.y < dp(int(10 + (y + 1.5) / 9 * (BOARD_LEN - 30))):
                    action = chr(ord("a") + wall_x) + str(8 - wall_y) + "v"

        #print(touch.x, touch.y)
        super(Quoridor, self).on_touch_down(touch)

    def change_search_nodes(self, *args):
        self.search_nodes = SEARCH_NODE_LIST[int(args[1])]

    def change_level(self, *args):
        self.level = int(args[1])

    def change_tau(self, *args):
        self.tau = TAU_LIST[int(args[1])]

    def change_search_nodes_1p(self, *args):
        self.search_nodes_1p = SEARCH_NODE_LIST[int(args[1])]

    def change_level_1p(self, *args):
        self.level_1p = int(args[1])

    def change_tau_1p(self, *args):
        self.tau_1p = TAU_LIST[int(args[1])]

    def change_search_nodes_2p(self, *args):
        self.search_nodes_2p = SEARCH_NODE_LIST[int(args[1])]

    def change_level_2p(self, *args):
        self.level_2p = int(args[1])

    def change_tau_2p(self, *args):
        self.tau_2p = TAU_LIST[int(args[1])]

    def change_level_training(self, *args):
        self.level_training = int(args[1])
        self.training_level_label.text = ""


class QuoridorApp(App):
    def build(self):
        Builder.load_string(f"""
<Quoridor>:
    button: button
    turn0_button: turn0_button
    undo_button: undo_button
    redo_button: redo_button
    resign_button: resign_button
    graphviz_on: graphviz_on
    graphviz_off: graphviz_off

    mode_tab: mode_tab
    human_tab: human_tab
    human_ai_tab: human_ai_tab
    ai_ai_tab: ai_ai_tab
    training_tab: training_tab
    
    flip_on: flip_on
    flip_off: flip_off
    teban_1p: teban_1p
    teban_2p: teban_2p
    view_1p: view_1p
    view_2p: view_2p
    
    search_nodes_label: search_nodes_label
    level_label: level_label
    tau_label: tau_label
    
    search_nodes_label_1p: search_nodes_label_1p
    level_label_1p: level_label_1p
    tau_label_1p: tau_label_1p
    
    search_nodes_label_2p: search_nodes_label_2p
    level_label_2p: level_label_2p
    tau_label_2p: tau_label_2p

    low_time: low_time
    high_time: high_time
    training_level_label: training_level_label
    player_level_label: player_level_label
    training_info: training_info

    canvas.before:
        Color:
            rgba: 1, 1, 1, 1
        Rectangle:
            pos: (0, 0)
            size: ({dp(WINDOW_WIDTH)}, {dp(600)})
        Color:
            rgba: 1, 0, 0, 1

    canvas:
        Color:
            rgba: 1.0, 1.0, 1.0, 1
        Triangle:
            points: (root.Bx - {dp(20)}, root.By - {dp(20)} + root.upside_down * {dp(40)}, root.Bx, root.By + {dp(20)} - root.upside_down * {dp(40)}, root.Bx + {dp(20)}, root.By - {dp(20)} + root.upside_down * {dp(40)})
        Color:
            rgba: 0, 0, 0, 1
        Triangle:
            points: (root.Wx - {dp(20)}, root.Wy + {dp(20)} - root.upside_down * {dp(40)}, root.Wx, root.Wy - {dp(20)} + root.upside_down * {dp(40)}, root.Wx + {dp(20)}, root.Wy + {dp(20)} - root.upside_down * {dp(40)})

    Label:
        font_size: {dp(30)}
        pos: ({dp(620)}, {dp(530)})
        color: 0, 0, 0, 1
        text: "turn:" + str(root.turn)
    Label:
        font_size: {dp(30)}
        pos: ({dp(680)}, {dp(500)})
        color: 0, 0, 0, 1
        text: root.move_str
    Label:
        font_size: {dp(30)}
        pos: ({dp(680)}, {dp(470)})
        color: 0, 0, 0, 1
        text: "player1 wall:" + str(root.player1wall)
    Label:
        font_size: {dp(30)}
        pos: ({dp(680)}, {dp(440)})
        color: 0, 0, 0, 1
        text: "player2 wall:" + str(root.player2wall)
    Label:
        font_size: {dp(30)}
        pos: ({dp(820)}, {dp(530)})
        color: 0, 0, 0, 1
        text: root.remaining_time_str
    Label:
        font_size: {dp(30)}
        pos: ({dp(850)}, {dp(500)})
        color: 0, 0, 0, 1
        text: root.game_result

    Button:
        id: turn0_button
        pos: ({dp(595)}, {dp(100)})
        size: ({dp(100)}, {dp(40)})
        text: "Back to turn 0"
    Button:
        id: undo_button
        pos: ({dp(700)}, {dp(100)})
        size: ({dp(80)}, {dp(40)})
        text: "Undo"
    Button:
        id: redo_button
        pos: ({dp(790)}, {dp(100)})
        size: ({dp(80)}, {dp(40)})
        text: "Redo"
    Button:
        id: resign_button
        pos: ({dp(900)}, {dp(100)})
        size: ({dp(80)}, {dp(40)})
        text: "Resign"

    Label:
        pos: ({dp(610)}, {dp(40)})
        size: ({dp(80)}, {dp(40)})
        color: 0, 0, 0, 1
        text: 'Graphviz'
    ToggleButton:
        pos: ({dp(600)}, {dp(10)})
        size: ({dp(50)}, {dp(40)})
        id: graphviz_on
        text: "On"
        group: "graphviz"
    ToggleButton:
        pos: ({dp(650)}, {dp(10)})
        size: ({dp(50)}, {dp(40)})
        id: graphviz_off
        text: "Off"
        group: "graphviz"
        state: "down"

    Button:
        id: button
        pos: ({dp(700)}, {dp(10)})
        size: ({dp(180)}, {dp(80)})
        text: "Start"

    TabbedPanel:
        id: mode_tab
        do_default_tab: False
        tab_width: self.width // 4
        pos: ({dp(600)}, {dp(150)})
        size: ({dp(380)}, {dp(320)})
        TabbedPanelItem:
            id: human_tab
            font_size: {dp(15)}
            text: 'vs. Human'
           
            BoxLayout:
                orientation: 'vertical'
                spacing: {dp(5)}
                
                BoxLayout:
                    orientation: 'horizontal'
                    size_hint: 1.0, 0.2
                    spacing: {dp(5)}
                    padding: 5, 5
                    Label:
                        size_hint: 0.5, 1.0
                        text: 'Flip board'
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: flip_on
                        text: "On"
                        group: "flip"
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: flip_off
                        text: "Off"
                        group: "flip"
                        state: "down"
                    
                BoxLayout:
                    orientation: 'horizontal'
                    size_hint: 1.0, 0.8
                    
        TabbedPanelItem:
            id: human_ai_tab
            text: 'vs. AI'
            font_size: {dp(15)}
            
            BoxLayout:
                orientation: 'vertical'
                spacing: {dp(25)}
                BoxLayout:
                    size_hint: 1.0, 0.25
                    orientation: 'horizontal'
                    spacing: {dp(5)}
                    padding: {dp(5)}, {dp(5)}
                    Label:
                        text: 'human side'
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
                    Label:
                        text: 'view side'
                        size_hint: 0.5, 1.0
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: view_1p
                        text: "1p"
                        group: "view"
                        state: "down"
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: view_2p
                        text: "2p"
                        group: "view"
                BoxLayout:
                    size_hint: 1.0, 0.25
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: {SEARCH_NODE_LIST_LEN - 1}
                        value: {DEFAULT_SEARCH_NODE_INDEX}
                        step: 1
                        orientation: "horizontal"
                        on_value: root.change_search_nodes(*args)   
                    Label:
                        id: search_nodes_label
                        font_size: {dp(20)}
                        text: "search nodes ="
                BoxLayout:
                    size_hint: 1.0, 0.25
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: {LEVEL_NUM - 1}
                        value: {LEVEL_NUM - 1}
                        step: 1
                        orientation: "horizontal"
                        on_value: root.change_level(*args)   
                    Label:
                        id: level_label
                        font_size: {dp(20)}
                        text: "level ="
                BoxLayout:
                    size_hint: 1.0, 0.25
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: 4
                        value: {DEFAULT_TAU_INDEX}
                        step: 1
                        orientation: "horizontal"
                        on_value: root.change_tau(*args)   
                    Label:
                        id: tau_label
                        font_size: {dp(20)}
                        text: "randomness ="
                    
        TabbedPanelItem:
            id: ai_ai_tab
            text: 'Watch AI game'
            font_size: {dp(13)}
 
            BoxLayout:
                orientation: 'vertical'
                spacing: {dp(15)}
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: {SEARCH_NODE_LIST_LEN - 1}
                        value: {DEFAULT_SEARCH_NODE_INDEX}
                        step: 1
                        cursor_size: {dp(20)}, {dp(20)}
                        orientation: "horizontal"
                        on_value: root.change_search_nodes_1p(*args)   
                    Label:
                        id: search_nodes_label_1p
                        font_size: {dp(15)}
                        text: "1p search nodes ="
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: {LEVEL_NUM - 1}
                        value: {LEVEL_NUM - 1}
                        step: 1
                        cursor_size: {dp(20)},{dp(20)}
                        orientation: "horizontal"
                        on_value: root.change_level_1p(*args)   
                    Label:
                        id: level_label_1p
                        font_size: {dp(15)}
                        text: "1p level ="
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: 4
                        value: {DEFAULT_TAU_INDEX}
                        step: 1
                        cursor_size: {dp(20)},{dp(20)}
                        orientation: "horizontal"
                        on_value: root.change_tau_1p(*args)   
                    Label:
                        id: tau_label_1p
                        font_size: {dp(15)}
                        text: "1p randomness ="                        

                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: {SEARCH_NODE_LIST_LEN - 1}
                        value: {DEFAULT_SEARCH_NODE_INDEX}
                        step: 1
                        cursor_size: {dp(20)},{dp(20)}
                        orientation: "horizontal"
                        on_value: root.change_search_nodes_2p(*args)   
                    Label:
                        id: search_nodes_label_2p
                        font_size: {dp(15)}
                        text: "2p search nodes ="
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: {LEVEL_NUM - 1}
                        value: {LEVEL_NUM - 1}
                        step: 1
                        cursor_size: {dp(20)},{dp(20)}
                        orientation: "horizontal"
                        on_value: root.change_level_2p(*args)   
                    Label:
                        id: level_label_2p
                        font_size: {dp(15)}
                        text: "2p level ="
                BoxLayout:
                    size_hint: 1.0, 0.16
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: 4
                        value: {DEFAULT_TAU_INDEX}
                        step: 1
                        cursor_size: {dp(20)},{dp(20)}
                        orientation: "horizontal"
                        on_value: root.change_tau_2p(*args)   
                    Label:
                        id: tau_label_2p
                        font_size: {dp(15)}
                        text: "2p randomness ="   

        TabbedPanelItem:
            id: training_tab
            font_size: {dp(15)}
            text: 'Training'
           
            BoxLayout:
                orientation: 'vertical'
                spacing: {dp(5)}
                
                BoxLayout:
                    orientation: 'horizontal'
                    size_hint: 1.0, 0.2
                    spacing: {dp(5)}
                    padding: 5, 5
                    Label:
                        size_hint: 0.5, 1.0
                        text: 'Time'
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: low_time
                        text: "3 minutes"
                        group: "time"
                        state: "down"
                    ToggleButton:
                        size_hint: 0.25, 1.0
                        id: high_time
                        text: "10 minutes"
                        group: "time"

                BoxLayout:
                    size_hint: 1.0, 0.35
                    orientation: 'vertical'      
                    Slider:
                        min: 0
                        max: {dp(TRAINING_LEVEL_NUM - 1)}
                        value: {dp(TRAINING_LEVEL_NUM - 1)}
                        step: 1
                        orientation: "horizontal"
                        on_value: root.change_level_training(*args)   
                    Label:
                        id: training_level_label
                        font_size: {dp(18)}
                        text: ""
                Label:
                    size_hint: 1.0, 0.2
                    id: player_level_label
                    font_size: {dp(18)}
                    text: "your level = -1"
                Label:
                    size_hint: 1.0, 0.25
                    id: training_info
                    font_size: {dp(18)}
                    text: "" 
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



