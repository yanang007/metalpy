from metalpy.mexin.injectors import after, before, terminate_with


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

    class Extension(Mixin):
        def __init__(self, this: ValueHolder):
            super().__init__(this)

        def increase(self, this):
            this.x += 1

    with patched(Mixed(ValueHolder).mix(Extension)):
        dummy = ValueHolder(1)

        assert dummy.x == 1
        dummy.increase()
        assert dummy.x == 2
