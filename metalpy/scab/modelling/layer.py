from .mix_modes import MixMode, Mixer, hash_mixer
from .object import Object


class Layer:
    def __init__(self, mix_mode: Mixer = MixMode.Override):
        self.objects: list[Object] = []
        self.mixer = MixMode.dispatch(mix_mode)

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

    def __hash__(self):
        return hash((*self.objects, hash_mixer(self.mixer)))
