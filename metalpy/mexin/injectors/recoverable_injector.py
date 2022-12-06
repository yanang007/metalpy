from abc import ABC, abstractmethod


class RecoverableInjector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, func):
        pass

    @abstractmethod
    def rollback(self):
        pass
