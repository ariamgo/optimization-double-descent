class Solver:
    """
    Base class for optimization solvers.
    """

    def solve(self, model, objective, X, y):
        raise NotImplementedError
