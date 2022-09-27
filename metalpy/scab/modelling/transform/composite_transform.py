from .transform import Transform


class CompositeTransform(Transform):
    def __init__(self):
        super().__init__()
        self.transforms = []

    def add(self, transform):
        self.transforms.append(transform)

    def transform(self, mesh):
        for trans in self.transforms:
            mesh = trans.transform(mesh)
        return mesh

    def clone(self):
        ret = CompositeTransform()
        for trans in self.transforms:
            ret.add(trans)

        return ret