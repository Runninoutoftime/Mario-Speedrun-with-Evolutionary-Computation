from pymoo.algorithms.soo.nonconvex.ga import GA
import numpy as np
from pymoo.optimize import minimize
from MarioProblem import MarioProblem, MarioMutation
from pymoo.factory import get_crossover, get_sampling
import data


# f = open("data.txt", "w")
# f.write(str(1))
# f.truncate()
data.furthest = 1


sampling = get_sampling('int_random')
crossover = get_crossover("int_sbx")

algorithm = GA(pop_size=10, sampling=sampling, crossover=crossover) #mutation=MarioMutation())



res = minimize(MarioProblem(), algorithm, ("n_gen", 5), seed=2, copy_algorithm=False, verbose=True)

# Working on checkpoints for potential long-runtime ability
# np.save("checkpoint", algorithm)

# checkpoint = np.load("checkpoint.npy", allow_pickle=True).flatten()
# print("Loaded checkpoint: ", checkpoint)

print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
