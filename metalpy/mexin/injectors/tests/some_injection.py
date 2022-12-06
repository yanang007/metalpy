from metalpy.mexin.injectors import replaces, after, before, extends
from some_module import some_func, Clazz, DummyClass


@before(some_func)
def before_some_func(x):
    print('before some_func')


@after(some_func)
def after_some_func(x):
    print('after some_func')


@before(Clazz.__init__)
def before_clazz_ctor(self):
    print('before Clazz.__init__')


@extends(Clazz, 'extended_method')
def Clazz_extended_method(self, x):
    return x


@replaces(DummyClass)
def DummyClass_hijack():
    return 1
