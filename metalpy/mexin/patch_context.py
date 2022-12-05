import threading

from .mixed import Mixed
from .patch import Patch


class PatchContext:
    """用于管理全局Patch的生命周期
    Note
    ----
        进入时会将所有patch按照priority从小到大的顺序应用，退出时按逆序回滚
        由于作用于全局，因此需要加锁来防止多线程环境下造成patch混乱
    """
    lock = threading.Lock()

    def __init__(self, *patches: Patch):
        self.patches = patches
        self.patches = sorted([*self.patches], key=lambda x: x.priority)
        self.mixin_patches = []

    def __enter__(self):
        PatchContext.lock.acquire()  # 防止多线程环境下造成patch混乱

        for patch in self.__get_patches():
            patch.bind_context(self)
            patch.pre_apply()
            patch.apply()
            
        classes_to_be_mixed = []
        for patch in self.patches:
            classes_to_be_mixed.extend(patch.get_mixed_classes())

        mixed_classes = set()
        for clazz in classes_to_be_mixed:
            if clazz not in mixed_classes:
                mixed_classes.add(clazz)

                mixer = Mixed(clazz)
                mixer.bind_context(self)
                mixer.pre_apply()
                mixer.apply()

                self.mixin_patches.append(mixer)

        for patch in self.__get_patches():
            patch.commit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for patch in self.__get_patches(reverse=True):
            patch.rollback()
            patch.post_rollback()
            patch.unbind_context()

        self.lock.release()

    def __get_patches(self, reverse=False) -> list[Patch]:
        if reverse:
            return reversed(self.mixin_patches + self.patches)
        else:
            return self.mixin_patches + self.patches

    def get_patches(self):
        """
        获取所有的patch（不包含因mixin需要而对对应类加的Mixed patch）
        :return: 所有的patch
        """
        return self.patches
