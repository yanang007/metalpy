import pytest

from metalpy.mexin.injectors import after, before, terminate_with, modify_params


def func(x):
    print(x)
    return 'Calling from func'


def test_decorator(capsys):
    @before(func)
    def before_test(x):
        print(f"before {x}")

    @after(func)
    def after_test(x):
        print(f"after {x}")

    assert func('test') == 'Calling from func'
    assert capsys.readouterr().out == 'before test\ntest\nafter test\n'

    @before(func)
    def modify_ret_value_test(x):
        print(f"before before {x}")
        return terminate_with('Calling from modify_ret_value_test')

    assert func('test') == 'Calling from modify_ret_value_test'
    assert capsys.readouterr().out == 'before before test\n'


def test_decorator2(capsys):
    import some_injection
    from some_module import some_func, Clazz, DummyClass

    assert some_func(1) == 1
    assert capsys.readouterr().out == 'before some_func\nafter some_func\n'

    x = Clazz()
    assert capsys.readouterr().out == 'before Clazz.__init__\n'

    assert x.extended_method(1) == 1

    y = DummyClass()
    assert y == 1


def test_mixin(capsys):
    from metalpy.mexin import Mixin
    from metalpy.mexin.mixed import Mixed
    from metalpy.mexin.patch_context import patched
    from some_module import ValueHolder

    class SelfIncrement(Mixin):
        def __init__(self, this: ValueHolder):
            super().__init__(this)

        def increase(self, this):
            this.x += 1

    with patched(Mixed(ValueHolder).mix(SelfIncrement)):
        dummy = ValueHolder(1)

        assert dummy.x == 1
        dummy.increase()
        assert dummy.x == 2

    from metalpy.mexin import Patch

    class DummyPatch(Patch):
        def __init__(self, x):
            super().__init__()
            self.x = x

        def apply(self):
            def reassign(this, x):
                this.x = self.x

            self.add_injector(after(ValueHolder.__init__), reassign)
            self.add_mixin(ValueHolder, SelfIncrement)

    with patched(DummyPatch(x=2)):
        dummy = ValueHolder(x=1)
        dummy.increase()
        assert dummy.x == 3

    dummy = ValueHolder(x=1)
    assert dummy.x == 1
    with pytest.raises(AttributeError):
        dummy.increase()
    with pytest.raises(AttributeError):
        _ = dummy.mixins


class DummyClassWithComplicateFunction:
    def func(self, a, b, c='c', *args, d='d', e='e', **kwargs):
        return a, b, c, d, e, args, kwargs


def test_parameters_update():
    inst = DummyClassWithComplicateFunction()

    @after(inst.func)
    def after_func2(this, a, b, c, *args, d, e, **kwargs):
        return a + 1, b, c, d, e, args, kwargs

    @before(inst.func)
    def before_func2(this, a, b, c, *args, d, e, **kwargs):
        return modify_params({
            0: 0,
            3: 'varargs0',
            4: 'varargs1',
        }, b='new b')

    assert inst.func('a', 'b', 'c', '?', '!', d='d', e='e', f='f') == \
           (0 + 1, 'new b', 'c', 'd', 'e', ('varargs0', 'varargs1'), {'f': 'f'})
