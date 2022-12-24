import torch
import chess
import random
import hashlib
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import namedtuple
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer


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

    def extractPos(self, fen_str):
        desk = np.zeros((8, 8)).astype(int).astype(str)
        arr_str = fen_str.split('/')
        ends = ' '.join(arr_str[-1].split(' ')[1:])
        arr_str[-1] = arr_str[-1].split(' ')[0]
        for i, j in enumerate(arr_str):
            pos = 0
            for k, sym in enumerate(j):
                if not sym.isdigit():
                    desk[i, pos] = sym
                    pos += 1
                else:
                    pos += int(sym)
        return [desk, ends]

    def _checkIsUpper(self, char):
        return char.isupper()

    def insertWhitePos(self, desk1, desk2, ends):
        check = np.vectorize(self._checkIsUpper)
        desk1_chips = np.where(check(desk1) == True)
        desk2_chips = np.where(check(desk2) == True)
        desk = desk1.copy()
        desk[desk1_chips] = '0'
        desk[desk2_chips] = desk2[desk2_chips]

        fen = ''
        for i in range(8):
            chips = np.where(desk[i, :] != '0')[0]
            if len(chips) > 0:
                if len(chips) == 1:
                    fen += f'{chips[0]}{desk[i, chips[0]]}{8 - chips[0] - 1}'
                elif len(chips) > 1:
                    fen += f'{chips[0]}'
                    for j, k in zip([*chips, -1][:-1], [*chips, -1][1:]):
                        if k != -1:
                            v = k - j - 1
                            fen += f'{desk[i, j]}{v if v != 0 else ""}'

                        else:
                            fen += f'{desk[i, j]}{8 - j - 1 if 8 - j - 1 != 0 else ""}'
            else:
                fen += '8'
            fen += '/'

        fen = f'{fen[:-1]} {ends}'

        return fen

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

    def get_legal_moves(self, state, stockfish):
        self.board.set_fen(state)
        lg_move = [str(move) for move in stockfish.get_top_steps()[:, 0]]

        return lg_move

    def get_opponent_moves(self, player_move):
        self.board.push_san(player_move)
        opponent_moves = [str(move) for move in self.board.legal_moves]
        self.board.set_fen(self.current_state)
        return opponent_moves

    def get_next_states(self, player_move, state):
        self.set_fen(state)
        self.board.push_san(player_move)
        # self.board.push_san(opp_move)
        # states.append(self.board.fen())
        # self.board.set_fen(self.current_state)

        return self.board.fen()

    def return_move(self, state, stockfish, e):
        self.set_fen(state)
        # self.update_qtable(state, stockfish)
        if random.random() < e:
            # print(' - exploratory')
            legal_moves = self.get_legal_moves(state, stockfish)
            move = random.choice(legal_moves)
            return move
        else:
            return str(self.get_move_with_max_qvalue(state))

    def update_qtable(self, state, stockfish):
        if state not in self.q_table:
            q_item = dict()
            legal_moves = self.get_legal_moves(state, stockfish)
            for move in legal_moves:
                # opponent_moves = self.get_opponent_moves(move)
                next_state = self.get_next_states(move, state)
                stockfish.stockfish.set_fen_position(next_state)
                try:
                    stockfish.env_move()
                    next_state = self.get_next_states(move, state)
                    # for next_state in next_states:
                    # print(next_state)
                    q_item[next_state] = ChessState(
                        move=move,
                        # reward=0.6
                        reward=random.uniform(0.4, 0.8)  # fixme !!
                    )
                except ValueError:
                    q_item[state] = ChessState(
                        move=move,
                        # reward=0.6
                        reward=1.0  # random.uniform(0.4, 0.8) # fixme !!
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

    def hashFen(self, fen_str):
        desk = self.extractPos(fen_str)
        check = np.vectorize(self._checkIsUpper)
        desk1_chips = np.where(check(desk) == True)
        desk[~desk1_chips] = '0'


class DQNAgent(AbstractAgent):
    def __init__(self, model: torch.nn.Module, gamma: float=0.3, lr: float=3e-4):
        super().__init__()
        self.gamma = gamma
        self.lr = lr
        self.policy_net = model
        self.actions = list()
        self.board_tokenizer = HashingVectorizer(lowercase=False, analyzer='char', n_features=64)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

    @staticmethod
    def move_to_number(move: str) -> float:
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        nums = [str(i + 1) for i in range(len(letters))]
        letters_to_num = dict(zip(letters, nums))

        move = list(move)
        move[0] = letters_to_num[move[0]]
        move[2] = letters_to_num[move[2]]
        move = ''.join(move)
        print(move)

        move = int(move)

        return (move - 1111) / (8888 - 1111)

    def return_move(self, state, good_moves):
        state = torch.tensor(state).to(torch.float32).unsqueeze(0)
        good_moves = torch.tensor([self.move_to_number(move[0]) for move in good_moves]).to(torch.float32).unsqueeze(0)
        print(state.shape, good_moves.shape)

        out = self.policy_net(state, good_moves)
        move_num = torch.argmax(out, dim=1)
        self.actions.append(out)

        return good_moves[move_num]

    def update_policy(self, all_rewards):
        R = 0
        rewards = list()
        for rw in reversed(all_rewards):
            R = rw + self.gamma * R
            rewards.insert(0, R)

        actions = torch.tensor(data=self.actions)
        rewards = torch.tensor(data=rewards)

        loss = 1 - torch.sum( # todo: 1 - ?
            torch.mul(actions, rewards.T).mul(-1), -1   # todo: is it correct?
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


