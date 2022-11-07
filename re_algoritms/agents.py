import chess
import random
from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    def __init__(self):
        self.board = chess.Board()

    def set_fen(self, fen: str):
        self.board.set_fen(fen)

    @abstractmethod
    def return_move(self) -> str:
        ...


class RandomAgent(AbstractAgent):
    def return_move(self) -> str:
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            return 'no_move'
        move = legal_moves[random.randint(0, len(legal_moves))-1]
        return str(move)
