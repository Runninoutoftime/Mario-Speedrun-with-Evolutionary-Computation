import numpy as np
import data
import neat
import cv2
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros, data
from numpy.lib.function_base import average
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


# Intiialization variables
data.pop_size = 5
data.size = 1000
data.furthest = 1
data.time = 500
data.ids = {}
data.times = [0] * data.pop_size

# In SIMPLE_MOVEMENT: 0 = no action, 1 = right, 2 = right + A, 3 = right + B, 4 = right + A + B, 5 = A, 6 = left
# Resolution is 256x240, 61440, 1.06666 repeating
# Downsized res is 64x60, 3840, still 1.0666 repeating


# Fitness function will be max of time (since time ticks down starting from 400) and max of distance
# Input will be the raw pixel values of the screen
xor_inputs = []
xor_outputs = [(0), (1), (2)]
config_file = "/home/will/Documents/ExpandedMarioProject/Mario-Speedrun-with-Evolutionary-Computation/NEAT_GA/config"

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Reset env for each genome
        ob = env.reset()
        
        # Random sampled action
        action = env.action_space.sample()

        # Create NN for genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        frame = 0
        # Mario game loop
        while frame < 3100:
            
            # Downsampling of the pixel array
            pixels = env.render(mode="rgb_array")
            np_array = cv2.resize(pixels, dsize=(64, 60), interpolation=cv2.INTER_CUBIC)
            
            # Flattening of array to pass in the NN
            flattened_array = np.ndarray.flatten(np_array)

            # Output of NN
            output = net.activate(flattened_array)

            print(output)

            frame = frame + 1





def run(config_file):
    config = neat.config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 300)


run(config_file)