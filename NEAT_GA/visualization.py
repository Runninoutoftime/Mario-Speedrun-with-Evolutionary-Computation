import numpy as np
import neat
import cv2
from nes_py.wrappers import JoypadSpace
from numpy.lib.function_base import average
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import gym_super_mario_bros
# In SIMPLE_MOVEMENT: 0 = no action, 1 = right, 2 = right + A, 3 = right + B, 4 = right + A + B, 5 = A, 6 = left
# Resolution is 256x240, 61440, 1.06666 repeating
# Downsized res is 64x60, 3840, still 1.0666 repeating


# Fitness function will be max of time (since time ticks down starting from 400) and max of distance
# Input will be the raw pixel values of the screen
xor_inputs = []
xor_outputs = [(0), (1), (2)]
config_file = "/home/will/Documents/ExpandedMarioProject/Mario-Speedrun-with-Evolutionary-Computation/NEAT_GA/config"

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
env = JoypadSpace(env, SIMPLE_MOVEMENT)



def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Reset env for each genome
        ob = env.reset()
        # Size of image creataed by emulator (x, y, # of colors)
        inx, iny, inc = env.observation_space.shape

        ob, _, _, _ = env.step(env.action_space.sample())

        inx = int(ob.shape[0]/3)
        iny = int(ob.shape[1]/3)
        done = False
        
        net = neat.nn.RecurrentNetwork.create(genome, config)
        
        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0
        imgarray = []
        
        while not done:
            # ob = env.reset()
            # env.render('rgb_array')

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            cv2.imshow("Test", ob)
            
            imgarray = np.ndarray.flatten(ob)
            # imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            actions = net.activate(imgarray)

            ind = 0
            action = 0
            maxVal = -1
            for val in actions:
                if val > maxVal:
                    maxVal = val
                    action = ind
                ind = ind + 1
            #print(action)

            ob, rew, done, info = env.step(action)
            
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
                
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    "/home/will/Documents/ExpandedMarioProject/Mario-Speedrun-with-Evolutionary-Computation/NEAT_GA/config-recurrent")
    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint('rn-config-3o-run_32')

    winner = p.run(eval_genomes, 1)


run(config_file)