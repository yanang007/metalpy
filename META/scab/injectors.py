from properties.base import PropertyMetaclass


class extends:
    def __init__(self, cls, name):
        self.cls = cls
        self.name = name

    def __call__(self, func):
        cmd = f'self.cls.{self.name} = func'
        exec(cmd)
        return func


class before:
    def __init__(self, cls, name):
        self.cls = cls
        self.name = name

    def __call__(self, func):
        orig = self.cls.__getattribute__(self.name)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            return orig(*args, **kwargs)

        cmd = f'self.cls.{self.name} = wrapper'
        exec(cmd)
        return wrapper
