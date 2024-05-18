import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
from Game import Game
import chess
import ai.chessMain as ai
from stockfish_ai import StockfishAI

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, inference=False):
        self.game: Game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.inference = inference
    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        # board = chess.Board('3k4/2ppp2Q/8/8/8/8/8/R3K3 w - - 0 1')
        self.curPlayer = 1
        episodeStep = 0

        while True:
            # print('---------------------------------')
            # print(board)
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)  #change perspective
            if self.inference:
                temp = int(episodeStep < self.args.tempThreshold)
            else:
                temp = 0
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)          #get the probability of each action (policy)
            sym = self.game.getSymmetries(canonicalBoard, pi)                   #get symmetries of the board and policy
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])
            
            action_idx = np.random.choice(len(pi), p=pi)
            action = self.game.policy_idx_to_action(action_idx, board= canonicalBoard)
            
            if action is None:
                # log.error(f'Action is None. action_idx: {action_idx}, pi: {pi}')
                log.error(f'\nboard: {str(board)}\naction_idx: {action_idx}\naction: {self.game.policy_idx_to_action(action_idx, board=board, require_legal=False)}')
                log.error(f'\nindex of possible move: {np.where(np.array(pi) > 0)}')
                sys.exit(1)
            
            action = self.game.getCanonicalAction(action, self.curPlayer)  #change perspective of move
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                # print('---------------------------------')
                # print(board)
                print(f'reward: {r}')
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]
            
    def executeEpisodewithAI(self, ai_player, play_as_white=True):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        current_turn = play_as_white
        # if play_as_white: print('alpha zero play as white')
        
        if ai_player == 'stockfish':
            stockfish = StockfishAI()
        while True:
            print('---------------------------------')
            print(board)
            episodeStep += 1
            
            if current_turn:
                print('white move')
                canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)  #change perspective
                if self.inference:
                    temp = int(episodeStep < self.args.tempThreshold)
                else:
                    temp = 0
                pi = self.mcts.getActionProb(canonicalBoard, temp=temp)          #get the probability of each action (policy)
                sym = self.game.getSymmetries(canonicalBoard, pi)                   #get symmetries of the board and policy
                for b, p in sym:
                    trainExamples.append([b, self.curPlayer, p, None])
                
                action_idx = np.random.choice(len(pi), p=pi)
                action = self.game.policy_idx_to_action(action_idx, board= canonicalBoard)
                
                if action is None:
                    
                    # log.error(f'Action is None. action_idx: {action_idx}, pi: {pi}')
                    log.error(f'\nboard: {str(board)}\naction_idx: {action_idx}\naction: {self.game.policy_idx_to_action(action_idx, board=board, require_legal=False)}')
                    log.error(f'\nindex of possible move: {np.where(np.array(pi) > 0)}')
                    sys.exit(1)
                
                action = self.game.getCanonicalAction(action, self.curPlayer)  #change perspective of move
                board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
                current_turn = False
            else:
                if ai_player == 'ai':
                    best_move, _ = ai.get_best_move(board, 4)
                else:
                    best_move = stockfish.get_best_move(board)
                best_move = str(best_move)
                
                action = self.game.getCanonicalAction(best_move, self.curPlayer)  #change perspective of move
                canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)  #change perspective
                pi = self.game.action_to_policy(canonicalBoard, action)
                
                sym = self.game.getSymmetries(canonicalBoard, pi)                   #get symmetries of the board and policy
                for b, p in sym:
                    trainExamples.append([b, self.curPlayer, p, None])
                
                board, self.curPlayer = self.game.getNextState(board, self.curPlayer, best_move)
                
                current_turn = True
              

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                print('---------------------------------')
                print(board)
                print(f'reward: {r}')
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self, playwithAI= None):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                if playwithAI is not None:
                    play_as_white = True
                    for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                        self.mcts = MCTS(self.game, self.nnet, self.args)
                        iterationTrainExamples += self.executeEpisodewithAI(play_as_white= play_as_white, ai_player = playwithAI)
                        play_as_white = not play_as_white
                        
                else:
                    for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                        self.mcts = MCTS(self.game, self.nnet, self.args)
                        iterationTrainExamples += self.executeEpisode()
                    
                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            while len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_data_file[0], self.args.load_data_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
