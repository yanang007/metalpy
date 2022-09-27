import numpy as np

from .transform import Transform


class Translation(Transform):
    def __init__(self, x, y, z):
        super().__init__()
        self.delta = np.asarray([x, y, z])

    def transform(self, mesh):
        return mesh + self.delta
