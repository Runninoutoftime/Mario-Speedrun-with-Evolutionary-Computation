import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.util.display import Display
from MarioOperators import MarioMutationFurthest, MarioSampling, MarioProblemDistance, MarioTermination
from pymoo.factory import get_crossover
import data


# Intiialization variables
data.pop_size = 5
data.size = 1000
data.furthest = 1
data.time = 500
data.ids = {}
data.times = [0] * data.pop_size


# with open('NumpyData.npy', 'rb') as f:
#     print(np.load(f))


# Total number of different GAs to use
runs = 10
i = 0
distAlgorithm = GA(pop_size=data.pop_size, sampling=MarioSampling(), crossover=get_crossover('int_exp', prob=0.9), mutation=MarioMutationFurthest())
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

# Saves bestPop into a file for later use
with open('NumpyData.npy', 'wb') as f:
    np.save(f, bestPop, allow_pickle=True)
