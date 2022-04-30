import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.util.display import Display
from MarioOperators import MarioDisplay, MarioMutationFurthest, MarioSampling, MarioProblem, MarioTermination
from pymoo.factory import get_crossover, get_termination
import data
from pymoo.visualization.scatter import Scatter
from MarioExample import MarioExample
import matplotlib.pyplot as plt




# Intiialization variables
data.pop_size = 2
data.size = 1000
data.furthest = 1
data.time = 500
data.ids = {}
data.times = [0] * data.pop_size


# with open('NumpyData.npy', 'rb') as f:
#     print(np.load(f))


# Total number of different GAs to use
runs = 1
i = 0

NSGA_Algo = NSGA2(pop_size=data.pop_size, sampling=MarioSampling(), crossover=get_crossover('int_exp', prob=0.9), mutation=MarioMutationFurthest())
bestPop = []


# Finds the individual with the best distance fitness in the population
def findMinIndiv(res):

    minIndex = np.argmin(res.pop.get("F"))
    finalArray = res.pop.get("X")
    minIndiv = finalArray[minIndex]

    return minIndiv

try:
    # Runs different GAs "runs" amount of times then stores the values of the best indivs in each GA to NumpyData.npy
    while i < runs:

        # Generates random int for seed
        r = np.random.randint(10000000)
        print("Seed: ", r)

        # Resets data variables for each run
        data.dist = 0
        data.furthest = 1

        # Runs minimization algorithm for GA
        res = minimize(MarioProblem(), NSGA_Algo, seed=r, copy_algorithm=True, verbose=True, save_history=True, termination=get_termination("n_eval", 10), display=MarioDisplay())
        
        # Finds individual with best fitness for distance
        minIndiv = findMinIndiv(res)

        # Stores all the best individuals from each run into bestPop
        bestPop.append(minIndiv)

        i = i + 1
except KeyboardInterrupt:
    pass

F = res.F
xl, xu = MarioProblem().bounds()
plt.figure(figsize=(7, 5))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()

approx_ideal = F.min(axis=0)
approx_nadir = F.max(axis=0)
nF = (F - approx_ideal) / (approx_nadir - approx_ideal)

fl = nF.min(axis=0)
fu = nF.max(axis=0)
print(f"Scale f1: [{fl[0]}, {fu[0]}]")
print(f"Scale f2: [{fl[1]}, {fu[1]}]")

plt.figure(figsize=(7, 5))
plt.scatter(nF[:, 0], nF[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.title("Objective Space")
plt.show()


# plot = Scatter()

# dist, time, death = MarioExample().playGame(minIndiv, True)
# print("Dist: ", dist)
# print("Time: ", time)


# Saves bestPop into a file for later use
with open('./NumpyData.npy', 'wb') as f:
    np.save(f, bestPop, allow_pickle=True)
