from dataclasses import replace
from stockfish import Stockfish
import numpy as np

import chess


class ChessInterface:
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

    def __init__(self, engine_path:str, verbose=False) -> None:
        self.stockfish = Stockfish(path=engine_path)
        self.verbose = verbose

    @staticmethod
    def _strip(x):
        return x.strip()

    def print_board(self, type='base'):
        print(self.stockfish.get_board_visual())

    def get_board_matrix(self):
        board = self.stockfish.get_board_visual().replace('+', '').replace('-', '').replace('|', '')
        np_board = np.array([])
        for line in board.split('\n')[:-2]:
            if line != '' and 'a  b  c' not in line:
                np_board = np.append(np_board, np.array(line[1:-3].split('  ')[:8]))
        np_board = np_board.reshape(8, 8)
        np_board[np_board == ''] = 0
        strp = np.vectorize(self._strip)
        np_board = strp(np_board)

        return np_board

    def get_board_fen(self):
        return self.stockfish.get_fen_position()

    def reset(self):
        clear_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        self.stockfish.set_fen_position(clear_pos)

    def step(self, pos1, pos2, view_from='white'):
        try:
            check = self.stockfish.get_what_is_on_square(pos2)
            eat = False if check == None else True
            self.stockfish.set_position([f'{pos1}{pos2}'])
            return eat
        except ValueError:
            return -1

    def get_top_steps(self, n=5):
        steps = self.stockfish.get_top_moves(n)
        array = np.array([])
        for step in steps:
            array = np.append(array, np.array([step['Move'], step['Centipawn'], step['Mate']]), axis=0)
        array = array.reshape(n, 3)
        array = array[array[:, 1].argsort()][::-1]
        return array

    def player_move(self, move: str):
        if self.stockfish.is_move_correct(move):
            self.set_move_on_board(move)
        else:
            raise ValueError(f'Invalid move: {move}')

    def env_move(self) -> str:
        best_move = self.stockfish.get_best_move()
        self.set_move_on_board(best_move)
        return best_move

    def set_move_on_board(self, move) -> None:
        """
        return:
            1 - фигура съедена
            0 - ход без поедания фигуры
        """
        self.stockfish.make_moves_from_current_position([move])
        # if figure := self.stockfish.get_what_is_on_square(move[2:]):
        #     if figure is not None and self.verbose is True:
        #         print(f'-- {figure} was captured.') # todo: это фигня работает не так, как надо, исправлю потом
