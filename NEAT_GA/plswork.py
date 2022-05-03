# import neat
# import visualize
# import numpy as np

# with open('winner.pkl', 'rb') as f:
#     winner = np.load(f, allow_pickle=True)

# config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                 "/home/will/Documents/ExpandedMarioProject/Mario-Speedrun-with-Evolutionary-Computation/NEAT_GA/config-recurrent")

# visualize.draw_net(config, winner, view=False)

import graphviz

graphviz.view("/home/will/Documents/ExpandedMarioProject/Mario-Speedrun-with-Evolutionary-Computation/Digraph.gv")