import os
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

