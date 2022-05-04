import numpy as np
from Distance_GA.MarioOperators import MarioMutationFurthest2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.util.display import Display
from MarioOperators import MarioMutationFurthest, MarioSampling, MarioProblemDistance, MarioTermination
from pymoo.factory import get_crossover, get_visualization, get_termination
import data
import matplotlib.pyplot as plt


# Intiialization variables overriding defaults set i ndata.py
data.pop_size = 5
data.size = 3000
data.furthest = 1
data.time = 500
data.ids = {}
data.times = [0] * data.pop_size



# Total number of different GAs to use
runs = 1
i = 0

# DSGA1
# distAlgorithm = GA(pop_size=data.pop_size, sampling=MarioSampling(), crossover=get_crossover('int_two_point', n_points=2), mutation=MarioMutationFurthest())

# DSGA2
# distAlgorithm = GA(pop_size=data.pop_size, sampling=MarioSampling(), crossover=get_crossover('int_two_point', n_points=3), mutation=MarioMutationFurthest2())

bestPop = []


# Finds the individual with the best distance fitness in the population
def findMinIndiv(res):

    minIndex = np.argmin(res.pop.get("F"))
    finalArray = res.pop.get("X")
    minIndiv = finalArray[minIndex]

    return minIndiv


# Runs different GAs "runs" amount of times then stores the values of the best indivs in each GA to NumpyData.npy
while i < runs:

    # Generates random int for seed
    r = np.random.randint(10000000)
    print("Seed: ", r)

    # Resets data variables for each run
    data.dist = 0
    data.furthest = 1

    # Runs minimization algorithm for GA
    res = minimize(MarioProblemDistance(), distAlgorithm, seed=r, copy_algorithm=True, verbose=True, save_history=True, termination=MarioTermination())
    
    # Finds individual with best fitness for distance
    minIndiv = findMinIndiv(res)

    # Stores all the best individuals from each run into bestPop
    bestPop.append(minIndiv)

    i = i + 1

# Variables with history for visualization purposes
n_evals = np.array([e.evaluator.n_eval for e in res.history])
opt = np.array([e.opt[0].F for e in res.history])

# Visualizes results
plt.title("Convergence")
plt.plot(n_evals, 0 - opt, "--")
plt.yscale("log")
plt.show()

