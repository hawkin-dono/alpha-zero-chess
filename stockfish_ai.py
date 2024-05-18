from stockfish import Stockfish
import chess
from Game import Game

class StockfishAI:
    def __init__(self):
        self.stockfish_parameters = {
            "Debug Log File": "",
            "Contempt": 0,
            "Min Split Depth": 0,
            "Threads": 1, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
            "Ponder": "false",
            "Hash": 16, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
            "MultiPV": 1,
            "Skill Level": 20,
            "Move Overhead": 10,
            "Minimum Thinking Time": 5,
            "Slow Mover": 100,
            "UCI_Chess960": "false",
            "UCI_LimitStrength": "false",
            "UCI_Elo": 2500,
        }
        self.engine = Stockfish(path= 'stockfish/stockfish-windows-x86-64-avx2.exe', parameters= self.stockfish_parameters)
    def get_best_move(self, board: chess.Board):
        self.engine.set_fen_position(board.fen())
        move = self.engine.get_best_move_time(time= 3)
        return move



def main():
    game = Game()
    board = game.getInitBoard()
    player = 1
    step = 0
    stockfish_engine = StockfishAI()
    while True:
        print('----------------------')
        print(f'Step: {step}')
        print(board)
        move = stockfish_engine.get_best_move(board)
        board, player = game.getNextState(board, player, move)
        if game.getGameEnded(board, player):
            break
        step += 1
    # print(get_best_move(board))

if __name__ == "__main__":
   main() 
