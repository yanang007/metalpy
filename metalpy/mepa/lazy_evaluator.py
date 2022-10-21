class UnFinished:
    pass


_unfinished = UnFinished()


class LazyEvaluator:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.rval = _unfinished

    def get(self):
        if isinstance(self.rval, UnFinished):
            self.rval = self.func(*self.args, **self.kwargs)
        return self.rval

    def result(self):
        return self.get()

    def __getitem__(self, key):
        return self.kwargs[key]

    def __setitem__(self, key, value):
        self.kwargs[key] = value
