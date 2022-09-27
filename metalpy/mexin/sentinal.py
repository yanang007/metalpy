class Sentinel:
    def __init__(self):
        pass

    def __getattr__(self, item):
        return lambda *args, **kwargs: None

    def __call__(self, *args, **kwargs):
        pass


NO_MIXIN = Sentinel()
