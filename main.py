from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from MarioProblem import MarioProblem, MarioMutation
from pymoo.factory import get_crossover, get_sampling

sampling = get_sampling('int_random')
crossover = get_crossover("real_two_point")

algorithm = GA(pop_size=10, sampling=sampling, crossover=crossover) #mutation=MarioMutation())



res = minimize(MarioProblem(), algorithm, ("n_gen", 1000), seed=5, verbose=True)

#print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F)
