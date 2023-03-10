from scipy.spatial.transform import Rotation as R

from metalpy.utils.dhash import dhash
from .transform import Transform


class Rotation(Transform):
    def __init__(self, y, a, b, degrees=False, seq='xyz'):
        super().__init__()
        self.params = (y, a, b, degrees, seq)
        # TODO: 解决Rotation精度问题
        self.rot = R.from_euler(seq, [y, a, b], degrees=degrees)

    def transform(self, mesh):
        return self.rot.apply(mesh)

    def inverse_transform(self, mesh):
        return self.rot.apply(mesh, inverse=True)

    def clone(self):
        return Rotation(*self.params)

    def __dhash__(self):
        return dhash(*self.params[:-1], self.params[-1])
