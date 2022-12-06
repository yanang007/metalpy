
class PropertyReplacement(property):
    def __init__(self, prop: property, orig, executor):
        super().__init__(prop.fget, prop.fset, prop.fdel, prop.__doc__)
        self.repl_orig = orig
        self.repl_executor = executor
        self.__name__ = None if orig is None else orig.__name__
