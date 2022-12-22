from abc import ABC, abstractmethod

from metalpy.utils.dhash import dhash


class Transform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, mesh):
        return mesh

    @abstractmethod
    def inverse_transform(self, mesh):
        return mesh

    @abstractmethod
    def clone(self):
        return Transform()

    def __hash__(self):
        return dhash(self).digest()

    @abstractmethod
    def __dhash__(self):
        raise NotImplementedError()
