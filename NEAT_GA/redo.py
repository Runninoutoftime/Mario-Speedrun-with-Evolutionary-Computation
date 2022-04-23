import pickle
import numpy as np
import data
import neat
import cv2
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros, data
from numpy.lib.function_base import average
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

img_array = []

ac = env.action_space.sample()
inx, iny, inc = env.observation_space.shape

print(inx, iny, inc)

# Iterates through env and records fitness of each genome
# comparing and evolving population
def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        
        ob = env.reset()
        # Random sampled action
        ac = env.action_space.sample()

        # Size of image creataed by emulator (x, y, # of colors)
        inx, iny, inc = env.observation_space.shape

        # Downsampled size of screen by 8
        inx = int(inx / 8)
        iny = int(iny / 8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False

        while not done:
            
            env.render()
            frame = frame + 1

            # Downsampling and grayscale
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            # Flattens nd array
            for x in ob:
                for y in x:
                    img_array.append(y)
            # img_array = ob.flatten()

            nnOutput = net.activate(img_array)
            
            ind = 0
            foundInd = 0
            maxVal = -1
            for val in nnOutput:
                if val > maxVal:
                    maxVal = val
                    foundInd = ind
                ind = ind + 1
            
            # print(len(img_array))
            #print(maxVal)
            # print(foundInd)
            print(nnOutput)


            ob, rew, done, info = env.step(foundInd)

            img_array.clear()


            xpos = info['x_pos']
            
            # When mario goes further right, he gets rewarded with one point
            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos

            # Checks if mario is stuck
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 250:
                done = True
                print(genome_id, fitness_current)

            genome.fitness = fitness_current

class Worker(object):
    
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.env.reset()

        # Size of image creataed by emulator (x, y, # of colors)
        inx, iny, inc = env.observation_space.shape

        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        
        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False
        
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0
        imgarray = []
        
        while not done:
            # self.env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            actions = net.activate(imgarray)

            ind = 0
            action = 0
            maxVal = -1
            for val in actions:
                if val > maxVal:
                    maxVal = val
                    action = ind
                ind = ind + 1
            
            ob, rew, done, info = self.env.step(action)
            
            xpos = info['x_pos']
            
            if xpos > xpos_max:
                xpos_max = xpos
                counter = 0
                fitness += 1
            else:
                counter += 1
                
            if counter > 250:
                done = True
                
            if xpos > 3100:
                fitness += 100000
                done = True
                
        print(fitness)
        return fitness

def eval_genomes(genome, config):

    worker = Worker(genome, config)
    return worker.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    "/home/will/Documents/ExpandedMarioProject/Mario-Speedrun-with-Evolutionary-Computation/NEAT_GA/config")

p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-13')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))
pe = neat.ParallelEvaluator(10, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

