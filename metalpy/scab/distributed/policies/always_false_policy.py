from metalpy.mepa import Worker
from .patch_policy import PatchPolicy


class AlwaysFalse(PatchPolicy):
    def __int__(self):
        pass

    def __call__(self, patch, worker: Worker):
        return False
