import os, sys

# tensorflowのログを抑制
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 標準エラー出力を抑制
sys.stderr = open(os.devnull, 'w')

from engine_util import prepare_AI
import time

DATA_BASE_DIR = "application_data"
PARAMETER_PATH = os.path.join(DATA_BASE_DIR, "parameter")

class UCIEngine:
    def __init__(self):
        self.AIs = None

    def handle_uci(self):
        print("id name ka Quoridor")
        print("id author Kanta Sugiyama")
        print("uciok")

    def handle_newgame(self):
        print("New game started")
        self.AIs = [
            prepare_AI(PARAMETER_PATH, 0, search_nodes=1000, tau=0.32, level=-1, seed=int(time.time())),
            prepare_AI(PARAMETER_PATH, 1, search_nodes=1000, tau=0.32, level=-1, seed=int(time.time()))
        ]
        # TODO: 色はどうやって判断するか？

    def handle_go(self):
        print("Go command received, calculating best move...")

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
