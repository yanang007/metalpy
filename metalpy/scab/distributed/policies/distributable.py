from metalpy.mepa import Worker


class Distributable:
    def __init__(self):
        pass

    def should_distribute_to(self, worker: Worker):
        return True
