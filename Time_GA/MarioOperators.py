import autograd.numpy as anp
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.util.misc import pop_from_sampling
from MarioExample import MarioExample
from pymoo.core.mutation import Mutation
import data
from pymoo.core.termination import Termination




# Creates a 2d array of individuals for the population where each row represents an individual of length data.size
class MarioSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        
        # This initializes a random population
        #X = np.random.randint(3, size=(data.pop_size, data.size))
        
        # This uses a stored numpy array as the initial population
        with open('NumpyData.npy', 'rb') as f:
        	X = np.load(f)
        return X

# Evaluates the individuals in the population based on distance
class MarioProblemDistance(ElementwiseProblem):
    
    def __init__(self):
        super().__init__(n_var=data.size, n_obj=1, xl=0, xu=3, type=anp.integer)

    def _evaluate(self, x, out, *args, **kwargs):

        dist, time, death = MarioExample().playGame(x)

        if dist*-1 < 3100:
        	out["F"] = time * -1
        else:
	        out["F"] = time

class MarioMutationFurthest(Mutation):
    def __init__(self):
        super().__init__()

    # This looks at the furthest the best individual in the population has gone, then mutates with a high rate within 50 genes from where it died (r)
    # Outside this range of genes, a small mutation rate is still applied (s)
    def _do(self, problem, X, **kwargs):

        for i in range(len(X)):
            for x in range(len(X[i])):
                if x > data.furthest - 40 and x < data.furthest:
                    r = np.random.random()

                    # Current mutation chance is below
                    if r < 0.8:
                        X[i,x] = np.random.randint(0,3)
                
                # Still has a (much) smaller chance of mutating genes outside of the "death range"
                else:
                    s = np.random.random()
                    if s < .025:
                        X[i,x] = np.random.randint(0,3)

        return X


# Terminates mario once furthest x_pos distance is 3000 or greater
class MarioTermination(Termination):
    
    def __init__(self):
        super().__init__()
        self.force_termination = False
        self.i = 0

    def do_continue(self, algorithm, **kwargs):

        if self.force_termination:
            return False
        else:
            self.i += 1
            return self._do_continue(algorithm)
    
    def _do_continue(self, algorithm, **kwargs):

        if self.i >= 100:
            return False
        else:
            return True

    def has_terminated(self, algorithm):
        return not self.do_continue(algorithm)
    
