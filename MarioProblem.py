import autograd.numpy as anp
import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.util.display import Display
from MarioExample import MarioExample
from pymoo.core.mutation import Mutation
import data

class MarioProblemMulti(ElementwiseProblem):

    # This function will allow the problem to go through the array one row at a time
    # with each row representing an individual for the evaluate function of the problem.
    def evaluator(self):
        for i in range(data.pop_size):
            if i < data.pop_size:
                if data.ids.count(i) == 0:
                    data.ids.append(i)
                    return i
            if i == data.pop_size:
                if data.ids.count(i) == 0:
                    data.ids.clear()
                    return i



    def __init__(self, array=[1]):
        super().__init__(n_var=1, n_obj=2,type=anp.integer)
        self.array=np.random.randint(7, size=(data.pop_size, 3000))

    def _evaluate(self, x, out, *args, **kwargs):
        print("HUH:", x[0])
        print("MAYBE?", self.array)
        # Finds correct index of array's row to play game with
        index = self.evaluator()
        #print(index)
        #print(self.array[index])
        dist, time, death = MarioExample().playGame(self.array[index])
        data.times[index] = death
        #print(data.times)

        out["F"] = [dist, time]

class MarioCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        #problem.array[]        

        return X


class MarioProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=3000, n_obj=2, n_constr=0, xl=0, xu=6, type=anp.integer, array=[])
        self.array=self.n_var

        # self.r = 0
        # print(self.r in data.ids)
        # while self.r in data.ids:
        #     self.r = self.r + 1
        # data.ids[self.r] = -1    
        


    def _evaluate(self, x, out, *args, **kwargs):
        dist, time = MarioExample().playGame(x)
        #print("DIST:", dist)
        #print("TIME:", time)
        #print(self)
        #print(data.ids)

        out["F"] = [dist, time]
    


class MarioProblemSingle(ElementwiseProblem):

    #Initializes a random list of integers of length 3000, each integer ranging from 0-6 (inclusive)
    def __init__(self):
        super().__init__(n_var=3000, n_obj=1, n_constr=0, xl=0, xu=6, type = anp.integer)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"]= MarioExample().playGame(x)


# TODO
# Looks at the previous x frames before mario died and only mutates within that range
class MarioMutation(Mutation):

    def __init__(self):
        super().__init__()
        self.list = []

    # Use data.furthest here! Look at previous 50 frames maybe?
    def _do(self, problem, X, **kwargs):

        #print(problem.array[0])

        for i in range(data.pop_size):
            for x in range(3000):
                if x > data.times[0] - 25 and x < data.times[0]:
                    r = np.random.random()

                    if r < 0.4:
                        problem.array[i, x] = np.random.randint(0,6)



        # # X is the array of indivs
        # # Row is indiv
        # # Col is genes
        # for i in range(len(X)):
        #     for x in range(len(X[i])):
        #         if x > data.furthest - 25 and x < data.furthest:
        #             r = np.random.random()

        #             # Add duplicate detection
        #             # Current mutation chance is below
        #             if r < 0.4:
        #                 X[i,x] = np.random.randint(0,6)
                
        #         # Still has a (much) smaller chance of mutating genes outside of the "death range"
        #         s = np.random.random()
        #         if s < .025:
        #             X[i,x] = np.random.randint(0,6)

                    
        return X

class MarioDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        self.output.append("mean_dist", np.mean(algorithm.pop.get("F")))
        self.output.append("max_dist", np.max(algorithm.pop.get("F")))