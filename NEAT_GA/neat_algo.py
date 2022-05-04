import pickle
import numpy as np
import neat
import cv2
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from numpy.lib.function_base import average
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#  Runs in parallel instances the NEAT algorithm on 8 different environements for faster processing
class Worker(object):
    
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.env.reset()

        # Size of image creataed by emulator (x, y, # of colors)
        inx, iny, inc = self.env.observation_space.shape

        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        # Reduces resolution by a scale of 8
        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False
        
        # Creates a Recurrent Neural Network
        net = neat.nn.RecurrentNetwork.create(self.genome, self.config)
        
        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0
        imgarray = []
        
        while not done:
            
            # Downsamples the image for neural network processing
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))

            # Activates network on the downsampled image
            actions = net.activate(imgarray)

            # Matches the neural network output to game actions
            ind = 0
            action = 0
            maxVal = -1
            for val in actions:
                if val > maxVal:
                    maxVal = val
                    action = ind
                ind = ind + 1

            # Steps through the environment
            ob, rew, done, info = self.env.step(action)
            
            xpos = info['x_pos']
            
            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
                fitness += 1
            else:
                counter += 1
            
            # If stuck for 250 frames, kills mario
            if counter > 250:
                done = True
            # If distance is > 3150, game is over and mario won
            if xpos > 3150:
                print("XPOS WINNER AT: ", xpos)
                fitness += 100000
                done = True
                
        #print(fitness)
        self.env.close()
        return fitness

# Evaluates each genome, to be used in parallel
def eval_genomes(genome, config):

    worker = Worker(genome, config)
    return worker.work()

# Default configuration based on parameters in config file
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    "/home/will/Documents/ExpandedMarioProject/Mario-Speedrun-with-Evolutionary-Computation/NEAT_GA/config-recurrent")

# Restores a population based on previous modeling
# p = neat.Checkpointer.restore_checkpoint('rn-config-3o-run_32')

# Creates a new population to run NEAT with
p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10, filename_prefix=""))
pe = neat.ParallelEvaluator(10, eval_genomes)

winner = p.run(pe.evaluate, 1)

# with open('winner.pkl', 'wb') as output:
#     pickle.dump(winner, output, 1)

