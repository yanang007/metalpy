from metalpy.mepa import Worker


class PatchPolicy:
    def __int__(self):
        pass

    def __call__(self, patch, worker: Worker):
        return False
