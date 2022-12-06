Metal Mixin
===========================

Mexin允许非侵入式地对原生Python方法和类名进行装饰与劫持，用于支撑SCAB的插件系统。 

安装
------------
Mexin目前是metalpy的一个子模块，你可以使用pip安装它：

```console
pip install metalpy
```

特性
-----
1. 使用_replaces, extends, before, after_等注解来装饰或劫持目标函数和类名
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

2. 使用**Mixin**机制来将扩展目标类的实例或替换实例的成员
```python
class ValueHolder:
    def __init__(self, x):
        self.x = x

class SelfIncrement(Mixin):
    def __init__(self, this: ValueHolder):
        super().__init__(this)

    def value(self, this):
        return this.x  # 暂不支持属性

    def increase(self, this):
        this.x += 1


with patched(Mixed(ValueHolder).mix(SelfIncrement)):
    dummy = ValueHolder(x=1)  # x = 1
    dummy.increase())  # x += 1变为2
    assert dummy.value() == 2
```

3. 使用**Patch**机制来管理全局的注解和**Mixin**
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
    dummy = ValueHolder(x=1)  # 因为after注解，x被设置为了2
    dummy.increase()  # x += 1变为3
    assert dummy.value() == 3
```

4. 所有全局操作均可回滚（已混入实例的Mixin除外）
```python
# 在patched代码块之外影响被回滚
dummy = ValueHolder(x=1)  # x = 1
assert dummy.value() == 1
dummy.increase()  # 错误：不存在increase方法
```