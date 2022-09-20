class LazyClassFactory:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = list(args)  # 方便删除
        self.kwargs = kwargs

    def find_param(self, predicate, remove=False):
        ret = None
        for i, arg in enumerate(self.args):
            if predicate(i, arg):
                ret = arg
                if remove:
                    del self.args[i]
                break

        if ret is None:
            for key, kwarg in self.kwargs.items():
                if predicate(key, kwarg):
                    ret = kwarg
                    if remove:
                        del self.kwargs[key]
                    break

        return ret

    def find_param_by_type(self, type, remove=False):
        return self.find_param(lambda k, v: isinstance(v, type), remove=remove)

    def find_param_by_name(self, name, remove=False):
        return self.find_param(lambda k, v: k == name, remove=remove)

    def construct(self, **new_kwargs):
        kwargs = self.kwargs.copy()
        kwargs.update(new_kwargs)
        return self.cls(*self.args, **kwargs)

    def clone(self, **new_kwargs):
        kwargs = self.kwargs.copy()
        kwargs.update(new_kwargs)
        return LazyClassFactory(self.cls, *self.args, **kwargs)
