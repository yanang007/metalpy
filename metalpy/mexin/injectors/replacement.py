class Replacement:
    def __init__(self, func, orig, executor):
        self.func = func
        self.repl_orig = orig
        self.repl_executor = executor
        self.__name__ = None if orig is None else orig.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
