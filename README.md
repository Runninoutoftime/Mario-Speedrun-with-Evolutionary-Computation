William Gautreaux - ARTI 6550 Final Project Documentation

The algorithms for DSGA1 and DSGA2 are found in the Distance_GA folder under main.py
Uncomment the line for either algorithm to run said algorithm

To run NEAT, simply run neat_algo.py. Visualization, visualize, plswork, and test are all used for testing purposes only.

NEAT can also restore from a previous checkpoint, using the restore_checkpoint function on line 100 in neat_algo.py
To use this, comment out p = neat.Population(config) on line 103 and uncomment line 100

** Note that certain file paths used in the files for all folders included in this project may need to be changed depending
on where the project is downloaded and stored in your own computer. Replacing these file paths should be relatively easy by just
simply deleting them and replacing them with the correct new path to files referenced.

** Every function is commented to explain how they work and their logic

** The Time_GA folder is from a previous project and has no bearing on this project, but can be ran if desired by just running
main.py in the folder   