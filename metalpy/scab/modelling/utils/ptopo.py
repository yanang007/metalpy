import json
from typing import Union

from metalpy.scab.modelling.shapes import Prism, Cuboid


def dump_ptopo(models: list[Prism, Cuboid], f=None) -> list:
    """将棱柱体或立方体模型组导出为ptopo格式

    ptopo格式用于存储空间中竖直棱柱体的分布信息，格式定义为

    [
        [[p0, ...], hmin, hmax],

        [[p0, ...], hmax],  # where hmin is assumed to be 0

        ...
    ]

    Parameters
    ----------
    models
        棱柱或立方体

    f
        导出的目标文件句柄或路径，为None时不导出

    Returns
    -------
    ret
        ptopo定义下的该棱柱场景list
    """
    try:
        if isinstance(f, str):
            f = open(f, 'w')

        ret = []
        for model in models:
            if isinstance(model, Cuboid):
                pdef = [[
                    [model.x0, model.y0],
                    [model.x0, model.y1],
                    [model.x1, model.y1],
                    [model.x1, model.y0],
                ], model.z0, model.z1]
            elif isinstance(model, Prism):
                pdef = [model.pts, model.z0, model.z1]
            else:
                raise ValueError(f'Ptopo format supports only Cuboid and Prism shapes. Got {type(model)} instead.')

            if pdef[1] == 0:  # hmin == 0，则隐藏
                pdef = [pdef[0], pdef[2]]
            ret.append(pdef)

        if f is not None:
            json.dump(ret, f)

        return ret

    finally:
        if f is not None:
            f.close()


def load_ptopo(ptopo: Union[str, list]) -> list[Prism]:
    """导入ptopo格式定义的棱柱体模型组

    Parameters
    ----------
    ptopo
        符合ptopo格式定义的list或定义文件的路径，传入字符串时假定为路径，会从指定文件读取

    Returns
    -------
    ret : list[Prism]
        一个由棱柱组成的列表，包含ptopo文件中定义的所有棱柱
    """
    if isinstance(ptopo, str):
        with open(ptopo, 'r') as f:
            ptopo = json.load(f)

    models = []
    for pdef in ptopo:
        if len(pdef) == 2:
            pts, hmax = pdef
            hmin = 0
        elif len(pdef) == 3:
            pts, hmin, hmax = pdef
        else:
            raise ValueError('Broken prism definition! Expected to be ((pts...) [, hmin], hmax)')
        models.append(Prism(pts, hmin, hmax))

    return models
