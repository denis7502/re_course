import chess
from chess_app.environment import ChessInterface
from re_algoritms.agents import RandomAgent, QAgent
from chess_app.dataset.loader import loaderChessPos
import chess.svg
from tqdm import tqdm

def is_checkmate(board, state):
    board.set_fen(state)
    return board.is_checkmate()


e = 0.1
lr = 0.3
gamma = 0.3
n_epochs = 10

player = QAgent(lr, gamma)
loader = loaderChessPos(r'./chess_app/dataset')
board = chess.Board()


bc_win = 0
wh_win = 0
stockfish = ChessInterface(verbose=True, engine_path=r"./chess_app/src/stockfish_15_x64_avx2.exe")
for epoch in range(n_epochs):
    for cnt_to_finish in list(loader.games.keys()):
        for state in tqdm(loader.games[cnt_to_finish]):
            stockfish.stockfish.set_fen_position(state)
            actions = list()

            move_counter = 1
            reward = 0
            black_win = False
            white_win = False
            new_state = state
            states = []
            
            
            while not is_checkmate(board, new_state):
                player.update_qtable(new_state, stockfish)
                stockfish.stockfish.set_fen_position(new_state)
                player_move = player.return_move(new_state, stockfish, e)
                #print('\n',stockfish.get_top_steps(), new_state)
                stockfish.player_move(player_move)
                new_state = stockfish.get_board_fen()
                #states.append(new_state)
                #print(f'\nPlayer move:{player_move}')
            # actions.append(new_state) # ?
                move_counter += 1
                if move_counter > 5:
                    black_win = True
                    bc_win += 1
                    break
                if is_checkmate(board, new_state):
                    wh_win += 1
                    white_win = True
                else:
                    machine_move = stockfish.env_move()
                    new_state = stockfish.get_board_fen()
                    stockfish.stockfish.set_fen_position(new_state)                    
                    #print(f'Machine move:{new_state}')
                actions.append(new_state)

                if is_checkmate(board, new_state) and not white_win:
                    bc_win += 1
                    black_win = True
                #print(actions)
                                    

                #print('---')
                # break

            if black_win is True:
                reward = -1
            else:
                reward = 1
            if len(actions) < 2:
                #print(player.q_table.keys())
                pass
            else:        
                print(actions)
                print(player.q_table[actions[-2]].keys())
                player.calculate_qvalue(actions, reward=reward)
            # break
        print(f'Win: {wh_win}\t Lose: {bc_win}')
