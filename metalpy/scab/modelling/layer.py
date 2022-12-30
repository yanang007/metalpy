from metalpy.utils.dhash import dhash
from .mix_modes import MixMode, Mixer, dhashable_mixer
from .object import Object


class Layer:
    def __init__(self, mix_mode: Mixer = MixMode.Override):
        self.objects: list[Object] = []
        self.mix_mode = mix_mode

    @property
    def mixer(self):
        return MixMode.dispatch(self.mix_mode)

    def append(self, obj: Object):
        self.objects.append(obj)

    def clear(self):
        self.objects.clear()

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, item):
        return self.objects[item]

    def __iter__(self):
        for obj in self.objects:
            yield obj

    def __dhash__(self):
        return dhash(*self.objects, dhashable_mixer(self.mix_mode))
