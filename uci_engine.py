import sys

def handle_uci():
    print("id name DummyChess")
    print("id author YourName")
    print("uciok")

def handle_newgame():
    print("New game started")

def handle_go():
    print("Go command received, calculating best move...")

def handle_quit():
    print("Quitting the program")
    return True

def main():
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
            handle_uci()
        elif cmd == "newgame":
            handle_newgame()
        elif cmd == "go":
            handle_go()
        elif cmd == "makemove":
            pass
        elif cmd == "quit":
            quit_received = handle_quit()

if __name__ == "__main__":
    main()
