from metalpy.utils.dhash import dhash
from .transform import Transform


class CompositeTransform(Transform):
    def __init__(self):
        super().__init__()
        self.transforms: list[Transform] = []

    def add(self, transform):
        self.transforms.append(transform)

    def transform(self, mesh):
        for trans in self.transforms:
            mesh = trans.transform(mesh)
        return mesh

    def inverse_transform(self, mesh):
        for trans in reversed(self.transforms):
            mesh = trans.inverse_transform(mesh)
        return mesh

    def clone(self):
        ret = CompositeTransform()
        for trans in self.transforms:
            ret.add(trans)

        return ret

    def __dhash__(self):
        return dhash(*self.transforms,)
