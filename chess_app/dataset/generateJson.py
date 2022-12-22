from loader import loaderChessPos
import chess
import json
import sys
sys.path.append(r'C:\my_prj\re_course')
from chess_app.environment import ChessInterface
from re_algoritms.agents import RandomAgent
from chess_app.dataset.loader import loaderChessPos
from tqdm import tqdm

def gamesToWin(state, stockfish, agent) -> dict:
    board = chess.Board()
    board.set_fen(state)
    agent.set_fen(state)    
    stockfish.stockfish.set_fen_position(state)
    party = {'state':[], 'steps':[]}
    while not board.is_checkmate() or not board.is_stalemate():
        try:
            machine_move = stockfish.env_move()
        except:
            print(state)
            break
        state_machine = stockfish.stockfish.get_fen_position()
        party['steps'].append(machine_move)
        party['state'].append(state_machine)
        fen = stockfish.get_board_fen()
        agent.set_fen(fen)
        board.set_fen(fen)
        
        player_move = player.return_move()
        if player_move == 'no_move':
            break
        stockfish.player_move(player_move)
        fen = stockfish.get_board_fen()
        player.set_fen(fen)
        board.set_fen(fen)

        
    return party


loader = loaderChessPos(r'./chess_app/dataset')

player = RandomAgent()
js = {}
stockfish = ChessInterface(verbose=True, engine_path=r"./chess_app/src/stockfish_15_x64_avx2.exe")
c = 0
for step in loader.games.keys():
    #js[step] = {}
    for state in tqdm(loader.games[step]):
        c += 1
        js[c] = {}
        if state == 'r3k2r/1p1ppp1p/8/b2N1Q1K/8/8/8/8 w KQkq - 0 1' and step == '2step':
            continue
        try:
            js[c][state] = gamesToWin(state, stockfish, player)
        except:
            stockfish = ChessInterface(verbose=True, engine_path=r"./chess_app/src/stockfish_15_x64_avx2.exe")
#del js['2step']['r3k2r/1p1ppp1p/8/b2N1Q1K/8/8/8/8 w KQkq - 0 1']
with open('result.json', 'w', encoding='utf-8') as fp:
    json.dump(js, fp, ensure_ascii=False, indent=4)