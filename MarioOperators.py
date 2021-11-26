import autograd.numpy as anp
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.selection import Selection
from pymoo.util.display import Display
from MarioExample import MarioExample
from pymoo.core.mutation import Mutation
import data

np.set_printoptions(threshold=300)



# Creates a 2d array of individuals for the population where each row represents an individual of length 3000
class MarioSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.random.randint(7, size=(data.pop_size, 3000))
        stack = np.arange(1, data.pop_size + 1)
        stack = np.reshape(stack, (data.pop_size, 1))
        final = np.hstack((stack, X))

        X = final
        # print(X)
        return final


class MarioProblem(ElementwiseProblem):
    
    def __init__(self):
        super().__init__(n_var=3001, n_obj=2, xl=0, xu=6, type=anp.integer)

    def _evaluate(self, x, out, *args, **kwargs):
        gameplay = np.delete(x, 0)
        id = x[0]

        dist, time, death = MarioExample().playGame(gameplay)

        data.ids[id] = death

        out["F"] = [dist, time]


class MarioMutation(Mutation):

    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        x = 1

        # print("Mutant", X)

        for i in range(len(X)):

            id = X[i, 0]
            dist = data.ids.get(id)
            # print("Id: ", id)
            # print("I: ", i)
            # print("Dist: ", dist)
            # print("Indiv: ", X[i])            

            x = 1
            for x in range(len(X[i])):

                if x > dist - 25 and x <= dist:
                    r = np.random.random()

                    if r < 0.4:
                        X[i, x] = np.random.randint(0, 6)

                s = np.random.random()
                if s < .025:
                    X[i, x] = np.random.randint(0, 6)
            
            # print("New Indiv: ", X[i])
        
        #print("XM", X)
        
        return X

class MarioCrossover(Crossover):
    
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        # print("Cross", X)

        return X

# class MarioSelection(Selection):
    
#     def __init__(self):
#         super().__init__()

#     def _do(self, pop, n_select=5, n_parents=2, **kwargs):
#         print("Pop", pop)
#         return pop