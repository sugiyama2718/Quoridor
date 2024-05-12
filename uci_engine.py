import os, sys
from State import State
from Agent import actionid2str, str2actionid
from BasicAI import state_copy
from util import Glendenning2Official, Official2Glendenning

# tensorflowのログを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 標準エラー出力を抑制
#sys.stderr = open(os.devnull, 'w')

from engine_util import prepare_AI
import time

DATA_BASE_DIR = "application_data"
PARAMETER_PATH = os.path.join(DATA_BASE_DIR, "parameter")

class UCIEngine:
    def __init__(self):
        self.AIs = None
        self.state = None
        self.use_prev_trees = [None, None]

        self.bestmoves = [None, None]

    def handle_uci(self):
        print("id name ka Quoridor")
        print("id author Kanta Sugiyama")
        print("uciok")

    def handle_newgame(self):
        self.AIs = [
            prepare_AI(PARAMETER_PATH, 0, search_nodes=1000, tau=0.32, level=-1, seed=int(time.time())),
            prepare_AI(PARAMETER_PATH, 1, search_nodes=1000, tau=0.32, level=-1, seed=int(time.time()))
        ]
        self.state = State()
        self.use_prev_trees = [True, True]
        self.bestmoves = [None, None]

    def handle_go(self):
        if self.AIs is None:
            self.handle_newgame()

        color = self.state.turn % 2
        action_id, _, _, v_post, _ = self.AIs[color].act_and_get_pi(self.state, use_prev_tree=self.use_prev_trees[color])
        #print("score= {}, use_prev_tree={}".format(int(1000 * v_post), self.use_prev_trees[color]))
        action = Glendenning2Official(actionid2str(self.state, action_id))
        print(f"bestmove {action}")
        self.bestmoves[color] = action

    def handle_makemove(self, action):
        # official notation

        if self.AIs is None:
            self.handle_newgame()

        color = self.state.turn % 2
        state_backup = state_copy(self.state)

        if not self.state.accept_action_str(Official2Glendenning(action)):
            print(action)
            print("this action is impossible")
            return
        
        self.AIs[1 - color].prev_action = str2actionid(state_backup, Official2Glendenning(action))
        if self.bestmoves[color] is not None and self.bestmoves[color] == action:
            self.use_prev_trees[color] = True
        else:
            self.use_prev_trees[color] = False

    def handle_quit(self):
        print("Quitting the program")
        return True

    def main(self):
        input_stream = sys.stdin
        #output_stream = sys.stdout

        quit_received = False
        while not quit_received:
            line = input_stream.readline().strip()
            if not line:
                continue
            tokens = line.split()
            cmd = tokens[0]

            if cmd == "uci":
                self.handle_uci()
            elif cmd == "newgame":
                self.handle_newgame()
            elif cmd == "go":
                self.handle_go()
            elif cmd == "makemove":
                self.handle_makemove(tokens[1])
            elif cmd == "quit":
                quit_received = self.handle_quit()

if __name__ == "__main__":
    uciengine = UCIEngine()
    uciengine.main()
