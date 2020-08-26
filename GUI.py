# coding:utf-8

BOARD_LEN = 600

from kivy.config import Config
Config.set('graphics', 'width', '900')
Config.set('graphics', 'height', str(BOARD_LEN))
Config.set('graphics', 'resizable', False)

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, StringProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.graphics import Rectangle, Color, Triangle
from kivy.core.window import Window

from Agent import actionid2str
from State import State, CHANNEL
from State import DRAW_TURN
from CNNAI import CNNAI
from BasicAI import state_copy
import time
import sys
import numpy as np
import gc
import os
from main import normal_play
from multiprocessing import Process
from Agent import Agent

touched = False
action = ""


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
    human_button1 = ObjectProperty(None)
    ai_button1 = ObjectProperty(None)
    human_button2 = ObjectProperty(None)
    ai_button2 = ObjectProperty(None)
    search_nodes = 800

    def dont_down(self, button):
        if button.state != "down":
            button.state = "down"

    def __init__(self, **kwargs):
        super(Quoridor, self).__init__(**kwargs)
        self.state = State()
        self.agents = [GUIHuman(0), CNNAI(1, search_nodes=self.search_nodes, tau=0.5)]
        self.playing_game = False
        self.human_button1.bind(on_press=lambda touch: self.dont_down(self.human_button1))
        self.ai_button1.bind(on_press=lambda touch: self.dont_down(self.ai_button1))
        self.human_button2.bind(on_press=lambda touch: self.dont_down(self.human_button2))
        self.ai_button2.bind(on_press=lambda touch: self.dont_down(self.ai_button2))
        self.button.bind(on_release=lambda touch: self.start_game())

        self.row_wall_colors = [Color(0.7, 0.7, 0, 0) for i in range(64)]
        self.column_wall_colors = [Color(0.7, 0.7, 0, 0) for i in range(64)]

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

    def oneturn(self, color):
        global touched
        s = self.agents[color].act(self.state)
        if isinstance(self.agents[color], CNNAI):
            g = self.agents[color].get_tree_for_graphviz()
            g.render(os.path.join("game_trees", "game_tree{}".format(self.state.turn)))
        if s == -1:
            return

        if isinstance(s, int):
            a = actionid2str(self.state, s)
        else:
            a = s
        if not self.state.accept_action_str(a):
            print(a)
            print("this action is impossible")
            return
        self.agents[1 - color].prev_action = s
        self.state.display_cui()
        print(self.state.get_player_dist_from_goal())
        touched = False

    def start_game(self):
        global touched
        if self.human_button1.state == "down":
            agent1 = GUIHuman(0)
        elif self.ai_button1.state == "down":
            #agent1 = CNNAI(0, search_nodes=self.search_nodes, tau=0.25, v_is_dist=True, p_is_almost_flat=True)
            agent1 = CNNAI(0, search_nodes=self.search_nodes, tau=0.25)
            agent1.load("./parameter/epoch110.ckpt")
        if self.human_button2.state == "down":
            agent2 = GUIHuman(1)
        elif self.ai_button2.state == "down":
            #agent2 = CNNAI(1, search_nodes=self.search_nodes, tau=0.25, v_is_dist=True, p_is_almost_flat=True)
            agent2 = CNNAI(1, search_nodes=self.search_nodes, tau=0.25)
            agent2.load("./parameter/epoch110.ckpt")
        self.agents = [agent1, agent2]
        self.state = State()
        self.playing_game = True
        touched = False

    def update(self, dt):
        if self.playing_game and not self.state.terminate:
            self.oneturn(self.state.turn % 2)
        if self.state.terminate:
            self.playing_game = False

        self.turn = self.state.turn
        if self.turn % 2 == 0:
            self.move_str = "player1 move"
        else:
            self.move_str = "player2 move"
        self.player1wall = self.state.black_walls
        self.player2wall = self.state.white_walls

        self.Bx = int(15 + (self.state.Bx + 0.5) / 9 * (BOARD_LEN - 30))
        self.By = int(15 + (8.5 - self.state.By) / 9 * (BOARD_LEN - 30))
        self.Wx = int(15 + (self.state.Wx + 0.5) / 9 * (BOARD_LEN - 30))
        self.Wy = int(15 + (8.5 - self.state.Wy) / 9 * (BOARD_LEN - 30))

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

        for x in range(8):
            for y in range(8):
                if self.state.row_wall[x, y]:
                    self.row_wall_colors[(7 - y) * 8 + x].a = 1
                if self.state.column_wall[x, y]:
                    self.column_wall_colors[(7 - y) * 8 + x].a = 1

        #print(Window.mouse_pos)

    def on_touch_down(self, touch):
        global touched, action
        touched = True
        for x in range(9):
            for y in range(9):
                if int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) - 10 <= touch.x <= int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) and int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) <= touch.y <= int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10:
                    continue
                if int(20 + (x + 0.5) / 9 * (BOARD_LEN - 30)) <= touch.x < int(20 + (x + 1.5) / 9 * (BOARD_LEN - 30)) and int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) <= touch.y <= int(10 + (y + 1) / 9 * (BOARD_LEN - 30)) + 10:
                    action = chr(ord("a") + x) + str(8 - y) + "h"
                if int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) - 10 <= touch.x <= int(20 + (x + 1) / 9 * (BOARD_LEN - 30)) and int(10 + (y + 0.5) / 9 * (BOARD_LEN - 30)) <= touch.y < int(10 + (y + 1.5) / 9 * (BOARD_LEN - 30)):
                    action = chr(ord("a") + x) + str(8 - y) + "v"
                if int(20 + x / 9 * (BOARD_LEN - 30)) <= touch.x < int(10 + (x + 1) / 9 * (BOARD_LEN - 30)) and int(20 + y / 9 * (BOARD_LEN - 30)) <= touch.y < int(10 + (y + 1) / 9 * (BOARD_LEN - 30)):
                    action = chr(ord("a") + x) + str(9 - y)

        #print(touch.x, touch.y)
        super(Quoridor, self).on_touch_down(touch)


class QuoridorApp(App):
    def build(self):
        game = Quoridor()
        Clock.schedule_interval(game.update, 1.0 / 60.0)
        return game


if __name__ == '__main__':
    QuoridorApp().run()


