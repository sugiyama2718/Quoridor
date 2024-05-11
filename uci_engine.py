import os, sys
from State import State
from Agent import actionid2str
from util import Glendenning2Official

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

    def handle_go(self):
        if self.AIs is None:
            self.handle_newgame()

        color = self.state.turn & 2
        action_id, _, _, v_post, _ = self.AIs[color].act_and_get_pi(self.state, use_prev_tree=True)
        #print("score= {}, use_prev_tree={}".format(int(1000 * v_post), self.use_prev_tree))
        action = Glendenning2Official(actionid2str(self.state, action_id))
        print(f"bestmove {action}")

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
                pass
            elif cmd == "quit":
                quit_received = self.handle_quit()

if __name__ == "__main__":
    uciengine = UCIEngine()
    uciengine.main()
