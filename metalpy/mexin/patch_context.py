import threading
from typing import Iterable

from .mixed import Mixed
from .patch import Patch


class PatchContext:
    lock = threading.Lock()

    def __init__(self, *patches: Patch):
        """用于管理全局Patch的生命周期

        Parameters
        ----------
        patches
            待应用的patch

        Note
        ----
            进入时会将所有patch按照priority从小到大的顺序应用，退出时按逆序回滚

            由于作用于全局，因此需要加锁来防止多线程环境下造成patch混乱
        """
        self.patches = patches
        self.patches = sorted([*self.patches], key=lambda x: x.priority)
        self.mixers = []  # Mixed patches

    def __enter__(self):
        self.apply()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rollback()

    def apply(self):
        PatchContext.lock.acquire()  # 防止多线程环境下造成patch混乱

        for patch in self.get_patches():
            patch.bind_context(self)
            patch.pre_apply()
            patch.apply()

        mixers = {}
        mixers_to_bind = []
        for patch in self.patches:
            for clazz, mixin_defs in patch.mixins.items():
                if clazz in mixers:
                    mixer = mixers[clazz]
                else:
                    mixer = Mixed(clazz)
                    mixers[clazz] = mixer
                    mixers_to_bind.append(mixer)

                for mixin_type, args, kwargs in mixin_defs:
                    mixer.mix(mixin_type, *args, **kwargs)

        for mixer in mixers_to_bind:
            mixer.bind_context(self)
            mixer.pre_apply()
            mixer.apply()

            self.mixers.append(mixer)

        for patch in self.get_patches_and_mixers():
            patch.commit()

    def rollback(self):
        for patch in self.get_patches_and_mixers(reverse=True):
            patch.rollback()
            patch.post_rollback()
            patch.unbind_context()

        self.mixers = []

        self.lock.release()

    def get_patches_and_mixers(self, reverse=False) -> Iterable[Patch]:
        """获取所有的patch（包括因mixin需要而对对应类加的Mixed patch）

        Returns
        -------
        ret
            所有的patch
        """
        if reverse:
            return reversed(self.mixers + self.patches)
        else:
            return self.mixers + self.patches

    def get_patches(self):
        """获取所有的patch（不包含因mixin需要而对对应类加的Mixed patch）

        Returns
        -------
        ret
            所有的patch
        """
        return self.patches


def patched(*patches):
    return PatchContext(*patches)
