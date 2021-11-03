import autograd.numpy as anp
from pymoo.core.problem import ElementwiseProblem
from MarioExample import MarioExample
from pymoo.core.mutation import Mutation

class MarioProblem(ElementwiseProblem):

    # Initializes a random list of integers of length 3000, each integer ranging from 0-6 (inclusive)
    def __init__(self):
        super().__init__(n_var=3000, n_obj=1, n_constr=0, xl=0, xu=6, type = anp.integer)
    
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"]= MarioExample().playGame(x)

class MarioMutation(Mutation):

    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        print(kwargs.values())
