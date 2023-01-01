class Mixin:
    def __init__(self, this):
        """定义一个Mixin类，用于在其它类中添加新的功能
        接受MixinManager的管辖，MixinManager负责管理Mixin的生命周期与信息交换

        Note
        ----
            1. 通过MixinManager的add方法将Mixin添加到目标类中（通常依靠Patch来通知PatchContext执行）
            2. MixinManager会将其中所有方法的第二个参数绑定到目标类上，此时在目标类和mixin类中调用的方法都是已绑定了两个类的方法，形如

                def xxx(self, this, ...)，其中self是mixin对象实例，this是对应的目标类实例

            3. 可以通过MixinManager的get方法获取Mixin，例如

                mixinA = mixed_class_instance.mixins.get(MixinA)

            同时一些特殊方法会被排除: 1. 私有方法（即名字以__开头） 2. 魔术方法（即名字被__包围） 3. 属性方法
        """
        pass

    def post_apply(self, this):
        pass
