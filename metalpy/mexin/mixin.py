class Mixin:
    """定义一个Mixin类，用于在其它类中添加新的功能
    接受MixinManager的管辖，MixinManager负责管理Mixin的生命周期与信息交换

    Note
    ----
        1. 通过MixinManager的add方法将Mixin添加到目标类中（通常依靠Patch来通知PatchContext执行）
        2. MixinManager会将其中所有非__开头方法的第二个参数会绑定到目标类上，形如
            def xxx(self, this, ...) 其中self是mixin类，this是对应的目标类
            此时在目标类和mixin类中调用的方法都是已绑定了两个类的方法
        3. 通过MixinManager的get方法获取Mixin，例如
            mixinA = mixed_class_instance.mixins.get(MixinA)
    """

    def __init__(self, this):
        pass

    def post_apply(self, this):
        pass
