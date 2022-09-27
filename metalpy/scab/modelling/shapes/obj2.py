from typing import Union

import numpy as np
import pyvista
import pyvista as pv

from . import Shape3D


def check_density(length, dx, nx, _default):
    if dx is not None:
        return dx
    elif nx is not None:
        return length / nx
    else:
        return _default


def check_scale(length, target_size, target_scale, _default=1):
    if target_scale is not None:
        return target_scale
    elif target_size is not None:
        return target_size / length
    else:
        return _default


def check_is_none(val, _default):
    if val is not None:
        return val
    else:
        return _default


def check_scalar_or_list(val, size, _default):
    if val is None:
        return [_default] * size
    elif not hasattr(val, '__iter__'):
        return [val] * size
    else:
        return list(val)


class Obj2(Shape3D):
    def __init__(self, model: Union[str, pyvista.DataSet],
                 xsize=None, ysize=None, zsize=None, size=None,
                 xscale=None, yscale=None, zscale=None, scale=None
                 ):
        """
        :param model: 模型路径或PyVista (当为None时不主动构造，用于内部的拷贝代码，使用后果自负)
        二选一
        :param nx, ny, nz: 模型三方向网格数
        二选一
        :param xsize, ysize, zsize, size: 模型缩放后的新尺寸
        :param xscale, yscale, zscale, scale: 模型缩放比例
        """
        super().__init__()

        if model is not None:
            should_scale_be_inplace = False
            if isinstance(model, str):
                model = pv.get_reader(model).read()
                should_scale_be_inplace = True

            __default = model.length / 100

            model_size = np.asarray(model.bounds)
            xl, yl, zl = model_size[1::2] - model_size[0::2]

            size = check_scalar_or_list(size, 3, None)

            size[0] = check_is_none(xsize, size[0])
            size[1] = check_is_none(ysize, size[1])
            size[2] = check_is_none(zsize, size[2])
            xsize, ysize, zsize = size

            scale = check_scalar_or_list(scale, 3, 1)

            scale[0] = check_scale(xl, xsize, xscale)
            scale[1] = check_scale(yl, ysize, yscale)
            scale[2] = check_scale(zl, zsize, zscale)

            model = model.scale(scale, inplace=should_scale_be_inplace)
            model.translate(-np.asarray(model.bounds[::2]), inplace=True)
            self.model = model

    def do_place(self, mesh_cell_centers, worker_id):
        model = self.model

        x_min, x_max, y_min, y_max, z_min, z_max = model.bounds
        x = np.arange(x_min, x_max, 1)
        y = np.arange(y_min, y_max, 1)
        z = np.arange(z_min, z_max, 1)
        x, y, z = np.meshgrid(x, y, z)

        poly = pv.PolyData(mesh_cell_centers)
        mesh = pv.UnstructuredGrid(pv.StructuredGrid(*(x[:, None, None] for x in mesh_cell_centers.T)))
        mesh = pv.UnstructuredGrid(poly.cast_to_unstructured_grid())
        selection = mesh.select_enclosed_points(self.model.extract_surface(), tolerance=0.0, check_surface=False)
        indices = selection.point_data['SelectedPoints'].view(np.bool_)

        return indices

    def __hash__(self):
        arr = self.model.cell_centers().points.flatten()
        rand = np.random.RandomState(int(arr[len(arr) // 2]))
        return hash((*rand.choice(arr, min(len(arr), 10), replace=False),))

    def do_clone(self):
        ret = Obj2(None)
        ret.model = self.model.copy(True)
        return ret

    def plot(self, ax, color):
        pass
