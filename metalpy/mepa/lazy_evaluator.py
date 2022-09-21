class LazyEvaluator:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.rval = None

    def get(self):
        if self.rval is None:
            self.rval = self.func(*self.args, **self.kwargs)
        return self.rval

    def result(self):
        return self.get()
