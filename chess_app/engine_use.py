from stockfish import Stockfish

class ChessInterface():
    """
    https://pypi.org/project/stockfish/
    Default parametrs
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 1, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    "Ponder": "false",
    "Hash": 16, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    "MultiPV": 1,
    "Skill Level": 20,
    "Move Overhead": 10,
    "Minimum Thinking Time": 20,
    "Slow Mover": 100,
    "UCI_Chess960": "false",
    "UCI_LimitStrength": "false",
    "UCI_Elo": 1350
    """
    def __init__(self) -> None:
        self.stockfish = Stockfish(path="chess_app\src\stockfish_15_x64_avx2.exe")

    def printCurrentPos(self):
        print(self.stockfish.get_board_visual())

    def reset(self):
        clear_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.stockfish.set_fen_position(clear_pos)

    def step(self, pos1, pos2, view_from ='white'):
        self.stockfish.set_position([f'{pos1}{pos2}'])

"""test
def main():
    chs = ChessInterface()
    chs.printCurrentPos()
    chs.step('e2', 'e4')
    chs.printCurrentPos()
    chs.reset()
    chs.printCurrentPos()

main()
"""