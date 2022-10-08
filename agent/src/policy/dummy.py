import numpy as np

class Dummy():
    def __init__(self):
        """
        A policy for static obstacle.
        It does nothing
        """
        pass

    def predict(self, state):
        """
        vx = 0
        vy = 0
        return (vx,vy)
        """
        return 0, 0
