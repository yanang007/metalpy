from abc import ABC, abstractmethod


class Transform(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, mesh):
        return mesh

    @abstractmethod
    def clone(self):
        return Transform()
