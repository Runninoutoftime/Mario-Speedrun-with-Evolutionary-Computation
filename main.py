from numpy.core.numeric import cross
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2, binary_tournament
from pymoo.optimize import minimize
from pymoo.util.display import Display
from MarioOperators import MarioCrossover, MarioSampling, MarioProblem, MarioMutation, MarioSelection
from MarioProblem import MarioDisplay
from pymoo.factory import get_crossover, get_sampling, get_selection
from pymoo.interface import crossover
from pymoo.core.population import Population
import data

data.pop_size = 4
data.furthest = 1
data.time = 500
data.ids = {}
data.times = [0] * data.pop_size

#selection = get_selection('tournament', {'pressure' : 2, 'func_comp' : binary_tournament})

#algorithm = GA(pop_size=2, sampling=sampling, crossover=crossover, mutation=MarioMutation())
#algorithm = NSGA2(pop_size=2, mutation=MarioMutation(), crossover=crossover, sampling=MarioSampling())
algorithm = NSGA2(pop_size=data.pop_size, sampling=MarioSampling(), crossover=MarioCrossover(), mutation=MarioMutation()) #selection=MarioSelection())
SingleObjAlgo = GA(pop_size=10, crossover=crossover)

res = minimize(MarioProblem(), algorithm, ("n_gen", 100), seed=1, copy_algorithm=False, verbose=True, display=MarioDisplay())
#res = minimize(MarioProblemMulti(), algorithm, mut)
