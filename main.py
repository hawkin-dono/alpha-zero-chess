import logging
from Coach import Coach
from CoachParallel import CoachParallel
from Game import  Game
from NeuralNet import NeuralNet as nn
from utils import *
from MCTS import MCTS
import numpy as np
import time

log = logging.getLogger(__name__)

args = dotdict({
    'numIters': 20,
    # nums of playing multiple (parallel) games times
    'numEps': 2,  # 20      # number of complete self-play prallel games to simulate during a new iteration.  
    'num_parallel_games': 4,  #nums of parallel games at each episode       => self play numeps * num_parallel_games games
    'tempThreshold': 15,        #
    'updateThreshold': 0.51,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 20000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 800,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 4,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': './model/',
    'load_model': False,
    'load_data': False,
    'load_folder_file': ('model/','checkpoint_12.pth.tar'),
    'load_data_file': ('model/', 'checkpoint_0.pth.tar'),
    'numItersForTrainExamplesHistory': 10,
})

def get_best_move(board, current_player):
    game = Game()
    nnet = nn(game)
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    mcts = MCTS(game, nnet, args)
    
    canonicalBoard = game.getCanonicalForm(board, current_player)
    actionprobs = mcts.getActionProb(canonicalBoard, temp=0) 
    action_idx = np.argmax(actionprobs)
    action = game.policy_idx_to_action(action_idx, board=canonicalBoard)
    action = game.getCanonicalAction(action, current_player)
    
    return action 




def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args, inference= True)
    # c = CoachParallel(g, nnet, args)

    if args.load_data:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn(playwithAI= 'ai')
    # start = time.time()
    # c.executeEpisodewithAI()
    # print("Time taken: ", time.time() - start)

if __name__ == "__main__":
    main()
    
