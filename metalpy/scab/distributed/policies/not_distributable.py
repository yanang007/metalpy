from metalpy.mepa import Worker
from .distributable import Distributable


class NotDistributable(Distributable):
    def __init__(self):
        super().__init__()

    def should_distribute_to(self, worker: Worker):
        return False
