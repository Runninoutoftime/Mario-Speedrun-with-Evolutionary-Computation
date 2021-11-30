import numpy as np
from numpy.core.numeric import cross
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.optimize import minimize
from pymoo.util.display import Display
from MarioOperators import MarioMutationFurthest, MarioSampling, MarioProblemDistance, MarioMutation, MarioSelection
from MarioProblem import MarioDisplay
from pymoo.factory import get_crossover, get_mutation, get_sampling, get_selection
from pymoo.interface import crossover
from pymoo.core.population import Population
import data

data.pop_size = 30
data.size = 1000
data.furthest = 1
data.time = 500
data.ids = {}
data.times = [0] * data.pop_size


distAlgorithm = GA(pop_size=data.pop_size, sampling=MarioSampling(), crossover=get_crossover('int_k_point', n_points=3), mutation=MarioMutationFurthest())#mutation=get_mutation('int_pm', prob=1.0))


res = minimize(MarioProblemDistance(), distAlgorithm, ("n_gen", 1), seed=1, copy_algorithm=False, verbose=True, save_history=True)
#res = minimize(MarioProblemMulti(), algorithm, mut)


# with open('NumpyData.npy', 'wb') as f:
#     np.save(f, res.pop.get("X"))

# ISSUE - The program is getting type errors