import types


class extends:
    def __init__(self, cls, name):
        self.cls = cls
        self.name = name

    def __call__(self, func):
        cmd = f'self.cls.{self.name} = func'
        exec(cmd)
        return func


class replaces:
    def __init__(self, target, name):
        self.target = target
        self.name = name

    def __call__(self, func):
        orig = getattr(self.target, self.name)
        if isinstance(self.target, type):
            # 对类进行修改不需要显式指定 self 参数
            wrapper = lambda *args, **kwargs: func(*args, **kwargs, orig_fn=orig)
        else:
            # 对对象进行修改需要 types.MethodType包装
            wrapper = lambda *args, **kwargs: func(*args, **kwargs, orig_fn=orig)
            wrapper = types.MethodType(wrapper, self.target)

        cmd = f'self.target.{self.name} = wrapper'
        exec(cmd)

        return wrapper


class before:
    def __init__(self, cls, name):
        self.cls = cls
        self.name = name

    def __call__(self, func):
        def wrapper(self, *args, orig_fn=None, **kwargs):
            func(self, *args, **kwargs)
            return orig_fn(*args, **kwargs)
        return replaces(self.cls, self.name)(wrapper)



class after:
    def __init__(self, cls, name):
        self.cls = cls
        self.name = name

    def __call__(self, func):
        def wrapper(self, *args, orig_fn=None, **kwargs):
            result = orig_fn(*args, **kwargs)
            func(self, *args, **kwargs)
            return result
        return replaces(self.cls, self.name)(wrapper)
