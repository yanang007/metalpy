import numpy as np


class SingleTaskAllocator:
    def __init__(self, n_splits, data):
        self.arrays = np.array_split(data, n_splits)
        self.index = 0

    def slice(self, n_slices=1):
        ret = np.vstack(self.arrays[self.index: self.index + n_slices])
        self.index += n_slices
        return ret

    def assign(self, worker):
        return self.slice(worker.get_weight())


class TaskAllocator:
    def __init__(self, n_splits, *data_list):
        self.allocators = []
        for data in data_list:
            self.allocators.append(SingleTaskAllocator(n_splits, data))

    def assign(self, worker):
        ret = []
        for allocator in self.allocators:
            ret.append(allocator.assign(worker))
        return ret
