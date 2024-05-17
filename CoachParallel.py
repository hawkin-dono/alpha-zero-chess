import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from NeuralNet import NeuralNet 
from MCTS import MCTS
from Game import Game
import torch
import chess
import concurrent.futures

log = logging.getLogger(__name__)
class SPG(): # self play parallel
    def __init__(self, game: Game):
        self.game = game
        self.memory = []
        self.board = self.game.getInitBoard()

class CoachParallel():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game: Game = game
        self.nnet: NeuralNet = nnet
        self.pnet: NeuralNet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        # self.mcts = MCTSParallel(self.game, self.nnet, self.args)
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        
        # self.loadTrainExamples()

    def executeEpisodeParallel(self):
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
        # board = chess.Board('3k4/2ppp2Q/8/8/8/8/8/R3K3 w - - 0 1')
        self.curPlayer = 1
        episodeStep = 0
        spGames = [SPG(self.game) for _ in range(self.args.num_parallel_games)]

        while len(spGames) > 0:
            episodeStep += 1
            boards = [spGame.board for spGame in spGames]
            canonicalBoards = self.game.getCanonicalForm(boards, self.curPlayer) #change perspective of board
            temp = int(episodeStep < self.args.tempThreshold)
            #Todo: chang getActionProb to parallel
            action_probs = self.mcts.getActionProb(canonicalBoards, temp=temp)          #get the probability of each action (policy)

            for i in range(len(spGames)): 
                spg = spGames[i]
                canonicalBoard = canonicalBoards[i]
                pi = action_probs[i]
                sym = self.game.getSymmetries(canonicalBoards, pi)                   #get symmetries of the board and policy

                for b, p in sym:
                    spg.memory.append([b, self.curPlayer, p, None])
                
                action_idx = np.random.choice(len(pi), p=pi)
                action = self.game.policy_idx_to_action(action_idx, board= canonicalBoard)
                
                if action is None:
                    # log.error(f'Action is None. action_idx: {action_idx}, pi: {pi}')
                    log.error(f'\nboard: {str(spg.board)}\naction_idx: {action_idx}\naction: {self.game.policy_idx_to_action(action_idx, board= canonicalBoard, require_legal=False)}')
                    log.error(f'\nindex of possible move: {np.where(np.array(pi) > 0)}')
                    sys.exit(1)
            
                action = self.game.getCanonicalAction(action, self.curPlayer)  #change perspective of move
                spg.board, self.curPlayer = self.game.getNextState(spg.board, self.curPlayer, action)

                r = self.game.getGameEnded(spg.board, self.curPlayer)

                if r != 0:
                    # print(f'reward: {r}')
                    trainExamples.append([(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in spg.memory])
                    del spGames[i]
        return trainExamples
    
    def executeEpisode(self, mcts: MCTS):
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
            temp = int(episodeStep < self.args.tempThreshold)

            pi = mcts.getActionProb(canonicalBoard, temp=temp)          #get the probability of each action (policy)
            sym = self.game.getSymmetries(canonicalBoard, pi)                   #get symmetries of the board and policy
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])
            # for i in range(len(sym)):
            #     b, p = sym[i]
            #     b = self.game.get
            #     trainExamples.append([sym[i][0], self.curPlayer * (-1)**i, sym[i][1], None])        #the first and third are the same color, second and fourth is opponent

            
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
                print(f'reward: {r}')
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]


    def learn(self):
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

                self.nnet.nnet.eval()
                with torch.inference_mode():
                    for _ in tqdm(range(self.args.numEps), desc="Self Play Episode"):
                        # self.mcts = MCTSParallel(self.game, self.nnet, self.args)           # reset search tree
                        # iterationTrainExamples += self.executeEpisodeParallel()           # collect examples from this episode
                        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.args.num_parallel_games)
                        futures = []
                        for idx in tqdm(range(self.args.num_parallel_games), desc="Self Play Game"):
                            msts = MCTS(self.game, self.nnet, self.args)
                            future = pool.submit(self.executeEpisode, msts)
                            futures.append(future)
                        for future in futures:
                            iterationTrainExamples += future.result()
                        
                        pool.shutdown(wait=True)
                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
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
            
            self.nnet.nnet.train()
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

def main():
    for i in range(5)[::-1]:
        print(i)
        
    
if __name__ == "__main__":
    main()