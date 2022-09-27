import numpy as np

from scipy.spatial.transform import Rotation as R
from .transform import Transform


class Rotation(Transform):
    def __init__(self, y, a, b, degrees=True, seq='xyz'):
        super().__init__()
        self.params = (y, a, b, degrees, seq)
        self.rot = R.from_euler(seq, [y, a, b], degrees=degrees).as_matrix()

    def transform(self, mesh):
        return mesh.dot(self.rot)

    def clone(self):
        return Rotation(*self.params)
