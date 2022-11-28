import chess
import random
import hashlib
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import namedtuple


# ChessState = namedtuple("ChessState", ["reward", "move"])

@dataclass(order=True)
class ChessState:
    reward: float
    move: str


class AbstractAgent(ABC):
    def __init__(self):
        self.board = chess.Board()
        self.current_state = None

    def set_fen(self, fen: str):
        self.current_state = fen
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
        return [str(move) for move in self.board.legal_moves]

    def get_opponent_moves(self, player_move):
        self.board.push_san(player_move)
        opponent_moves = [str(move) for move in self.board.legal_moves]
        self.board.set_fen(self.current_state)
        return opponent_moves

    def get_next_states(self, player_move, opponent_moves):
        states = list()
        for opp_move in opponent_moves:
            self.board.push_san(player_move)
            self.board.push_san(opp_move)
            states.append(self.board.fen())
            self.board.set_fen(self.current_state)
        return states

    def return_move(self, state, e):
        self.set_fen(state)
        self.update_qtable(state)
        if random.random() < e:
            print(' - exploratory')
            legal_moves = self.get_legal_moves(state)
            move = random.choice(legal_moves)
            return move
        else:
            return str(self.get_move_with_max_qvalue(state))

    def update_qtable(self, state):
        if state not in self.q_table:
            q_item = dict()
            legal_moves = self.get_legal_moves(state)
            for move in legal_moves:
                opponent_moves = self.get_opponent_moves(move)
                next_states = self.get_next_states(move, opponent_moves)
                for next_state in next_states:
                    q_item[next_state] = ChessState(
                        move=move,
                        #reward=0.6
                        reward=random.uniform(0.4, 0.8) # fixme !!
                    )
            self.add_to_qtable(key=state, value=q_item)

    def calculate_qvalue(self, actions, reward):
        max_value = -1

        for state, prev_state in zip(reversed(actions), reversed(actions[:-1])):
            if max_value < 0:
                self.q_table[prev_state][state].reward = reward
                # print(prev_state, state)
            else:
                self.q_table[prev_state][state].reward *= (1 - self.lr) + self.lr * self.gamma * max_value
                # print(prev_state, state)
            #  max_value = max(self.q_table[prev_state][state])
            max_value = self.q_table[prev_state][max(self.q_table[prev_state])].reward  # fixme ????
            # max_q_move = max(self.q_table[prev_state][state], key=self.q_table[prev_state][state].get)
            # max

    def add_to_qtable(self, key, value):
        self.q_table[key] = value

    def get_move_with_max_qvalue(self, state):
        moves = self.q_table[state]
        max_q_move = max(moves, key=moves.get)
        return str(self.q_table[state][max_q_move].move)
