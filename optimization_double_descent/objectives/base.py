class Objective:
    """
    Base class for optimization objectives.
    """

    def loss(self, X, y, w):
        raise NotImplementedError

    def grad(self, X, y, w):
        raise NotImplementedError
