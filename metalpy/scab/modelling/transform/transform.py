from abc import ABC


class Transform(ABC):
    def __init__(self):
        pass

    def transform(self, mesh):
        return mesh
