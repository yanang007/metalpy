from properties.base import PropertyMetaclass


class extends:
    def __init__(self, cls: PropertyMetaclass, name):
        self.cls = cls
        self.name = name

    def __call__(self, func):
        cmd = f'self.cls.{self.name} = func'
        exec(cmd)
        return func
