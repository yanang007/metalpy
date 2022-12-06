Metal Mixin
===========================

Mexin enables non-invasive decoration and replacement of methods and classes,
which is originally developed as patch system of SCAB. 

Installation
------------
Mexin is now a submodule in metalpy, which can be installed using pip:

```console
pip install metalpy
```

Features
--------
1. Decorates or replaces functions and classes with annotations including _replaces, extends, before, after_.
```python
def test(x):
    print(x)

@before(test)
def before_test(x):
    print(f"before {x}")

@after(test)
def after_test(x):
    print(f"after {x}")

test('test')

# before test
# test
# after test
```

2. Extends or overrides instances with **Mixin** system.
```python
class ValueHolder:
    def __init__(self, x):
        self.x = x

class SelfIncrement(Mixin):
    def __init__(self, this: ValueHolder):
        super().__init__(this)

    def value(self, this):
        return this.x  # does not support @property for now

    def increase(self, this):
        this.x += 1


with patched(Mixed(ValueHolder).mix(SelfIncrement)):
    dummy = ValueHolder(x=1)  # x = 1
    dummy.increase())  # x += 1 to be 2
    assert dummy.value() == 2
```

3. Manage global annotations and **Mixin**s with **Patch** system
```python
class ValueHolderPatch(Patch):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def apply(self):
        def reassign(this, x):
            this.x = self.x

        self.add_injector(after(ValueHolder.__init__), reassign)
        self.add_mixin(ValueHolder, SelfIncrement)

with patched(ValueHolderPatch(2)):
    dummy = ValueHolder(x=1)  # x is set to 2 in the injected reassign function
    dummy.increase()  # x += 1 to be 3
    assert dummy.value() == 3
```

4. All global operations are recoverable (except for **Mixin**s that are already mixed into instances）
```python
# Patches are rolled back when existing patched(...) context
dummy = ValueHolder(x=1)  # x = 1
assert dummy.value() == 1
dummy.increase()  # AttributeError: increase does not exist
```