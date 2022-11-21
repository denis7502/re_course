import chess
import random
import hashlib
from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    def __init__(self):
        self.board = chess.Board()

    def set_fen(self, fen: str):
        self.board.set_fen(fen)

    @abstractmethod
    def return_move(self, *args, **kwargs) -> str:
        ...


class RandomAgent(AbstractAgent):
    def return_move(self) -> str:
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            return 'no_move'
        move = legal_moves[random.randint(0, len(legal_moves)) - 1]
        return str(move)


class QAgent(AbstractAgent):
    def __init__(self, lr: float, gamma: float):
        super().__init__()
        self.q_table = dict()
        self.lr = lr
        self.gamma = gamma

    def get_legal_moves(self, state):
        self.board.set_fen(state)
        return list(self.board.legal_moves)

    def return_move(self, state, e):
        self.set_fen(state)
        self.update_qtable(state)
        if random.random() < e:
            print(' - exploratory')
            legal_moves = self.get_legal_moves(state)
            move = random.choice(legal_moves)
            return str(move)
        else:
            return str(self.get_move_with_max_qvalue(state))

    def update_qtable(self, state):
        if state not in self.q_table:
            legal_moves = self.get_legal_moves(state)
            base_costs = [0.6 for _ in range(len(legal_moves))]
            new_dict = dict(zip(legal_moves, base_costs))
            self.add_to_qtable(state, new_dict)

    def calculate_qvalue(self, actions, reward):
        max_value = -1

        for state, prev_state in zip(reversed(actions), reversed(actions[:-1])):
            if max_value < 0:
                self.q_table[prev_state][state] = reward
            else:
                self.q_table[prev_state][state] *= (1 - self.lr) + self.lr * self.gamma * max_value
                max_value = max(self.q_table[prev_state][state])

    def add_to_qtable(self, key, value):
        self.q_table[key] = value

    def get_move_with_max_qvalue(self, state):
        moves = self.q_table[state]
        max_q_move = max(moves, key=moves.get)
        return str(max_q_move)
