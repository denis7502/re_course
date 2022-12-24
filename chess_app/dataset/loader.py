import os
import pandas as pd
import json
import numpy as np

class loaderChessPos():
    def __init__(self, path_to_data) -> None:
        self.PATH = path_to_data
        self.games = self.getAllData()

    def getAllData(self):
        files = [i for i in os.listdir(self.PATH) if i.endswith('.txt')]
        games = {}
        for file in files:
            arr_state = self.readFile(self.PATH + '//' + file)
            games[file[:-4]] = arr_state
        
        return games

    def readFile(self, path):
        #state = []
        with open(path) as f:
            state = [line.rstrip('\n') for line in f]

        return state


class tanstsovVecLoader():
    def __init__(self, path_to_data, path_to_setings) -> None:
        self.PATH = path_to_data
        self.df_propery = path_to_setings
        with open(self.df_propery, 'r') as f:
            settings = json.load(f)            
        self.columns = settings['columns']
        self.swap = settings['chip_swap']
                
    def createDf(self):
        self.df = pd.DataFrame(columns=self.columns)
        
    def extractPos(self, fen_str, id_party, num_step):
        rows = {f'{self.columns[0]}': id_party}
        rows[f'{self.columns[1]}'] = num_step
        desk = np.zeros((8,8)).astype(int).astype(str)
        arr_str = fen_str.split('/')
        swap = arr_str[-1].split(' ')[2]
        arr_str[-1] = arr_str[-1].split(' ')[0]
        for i, j in enumerate(arr_str):
            pos = 0
            for k,sym in enumerate(j):
                if not sym.isdigit():
                    desk[i, pos] = sym
                    pos += 1
                else:
                    pos += int(sym)
        if 'K' in swap:
            rows[f'{self.columns[2]}'] = 1
        else:
            rows[f'{self.columns[2]}'] = 0
        if 'Q' in swap:
            rows[f'{self.columns[3]}'] = 1        
        else:
            rows[f'{self.columns[3]}'] = 0
        if 'k' in swap:
            rows[f'{self.columns[4]}'] = 1        
        else:
            rows[f'{self.columns[4]}'] = 0
        if 'q' in swap:
            rows[f'{self.columns[5]}'] = 1
        else:
            rows[f'{self.columns[5]}'] = 0
            
        for e, i in enumerate(desk.ravel()):
            rows[f'{self.columns[6 + e]}'] = self.swap[i]
            
        return rows
    
    def loadFullData(self):
        with open(self.PATH, 'r') as f:
            data = json.load(f)
        c = 1
        for id_party in data.keys():
            for party in data[id_party].keys():
                self.df.loc[c] = self.extractPos(party, id_party, 0)
                c += 1
                for e, state in enumerate(data[id_party][party]['state']):
                    self.df.loc[c] = self.extractPos(state, id_party, e+1)
                    c += 1
        self.df.to_csv('test.csv', index=False, sep=';')
            
# EXAMPLE
test = tanstsovVecLoader(r'result.json', r'settings.json')
test.createDf()
test.loadFullData()