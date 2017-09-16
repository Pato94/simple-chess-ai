import chess
import chess.uci
import chess.pgn
import glob

def btoi(b):
    if b:
        num = 1
    else:
        num = -1
    return num

values = [".", "p", "r", "n", "b", "q", "k", "P", "R", "N", "B", "Q", "K"]

pgns = glob.glob("games/*")

engine = chess.uci.popen_engine("stockfish/stockfish-8-64")
for file in pgns:
    with open(file) as pgn:
        game = chess.pgn.read_game(pgn)

        info_handler = chess.uci.InfoHandler()
        engine.info_handlers.append(info_handler)
        node = game
    while not node.is_end():
        next_node = node.variations[0]
        engine.position(node.board())
        engine.go(movetime=100)
        score = info_handler.info["score"][1].cp
        normalized_score = btoi(node.board().turn) * (score if score is not None else 999) / 1000.0
        joined_string = " ".join(node.board().__str__().split("\n"))
        node_as_string = "[" + joined_string.replace(" ", ",") + "]"
        for value in values:
            node_as_string = node_as_string.replace(value, (values.index(value) / (len(values) * 1.0)).__str__())
        evaluation_string = "{\"input\":" + node_as_string + ",\"output\":[" + normalized_score.__str__() + "]}"
        print(evaluation_string)
        node = next_node
