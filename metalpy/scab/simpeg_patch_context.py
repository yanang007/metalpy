from metalpy.mexin import PatchContext


def simpeg_patched(*patches):
    return PatchContext(*patches)
