import os.path
from typing import Union

import numpy as np
import pyvista as pv

from metalpy.utils.file import make_cache_file
from metalpy.utils.model import split_models_in_memory
from metalpy.utils.time import Timer
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
    def __init__(self, model: Union[str, pv.DataSet],
                 xsize=None, ysize=None, zsize=None, size=None,
                 xscale=None, yscale=None, zscale=None, scale=None,
                 surface_thickness=None, surface_range=None
                 ):
        """
        :param model: 模型路径或PyVista模型对象 (当为None时不主动构造，用于内部的拷贝代码，非必要请勿使用)
        二选一
        :param xsize, ysize, zsize, size: 模型缩放后的新尺寸
        :param xscale, yscale, zscale, scale: 模型缩放比例

        仅保留表面，参数二选一（存在性能影响）
        :param surface_thickness: 表面厚度（向内），即surface_range=[-surface_range, 0]
        :param surface_range: [(float|None), (float|None)] 表面范围（小于0表示模型内，大于0表示模型外，None表示不作判断）
        """
        super().__init__()

        if surface_thickness is not None or surface_range is not None:
            self.surface_only = True
            if surface_range is not None:
                self.surface_range = surface_range
            else:
                if surface_thickness is not None:
                    self.surface_range = [-surface_thickness, 0]
                else:
                    self.surface_range = None
        else:
            self.surface_only = False
            self.surface_range = None

        if model is not None:
            should_scale_be_inplace = False
            if isinstance(model, str):
                model_path = model
                model_name = os.path.basename(model_path)
                cache_path = make_cache_file(model_name + '.vtk')

                if os.path.exists(cache_path):
                    model_path = cache_path

                reader = pv.get_reader(model_path)
                reader.show_progress()

                timer = Timer()
                with timer:
                    model = reader.read()

                if timer.elapsed > 0 and cache_path != model_path:
                    model.save(cache_path, binary=True)

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

            self.models = split_models_in_memory(model)
            self.models[0] = self.models[0].scale([s * 250000 for s in scale], inplace=should_scale_be_inplace)
            self.models[0].translate(-np.asarray(self.models[0].bounds[::2]), inplace=True)
            self.model = model

    def place_impl(self, model, mesh_cell_centers, worker_id):
        poly = pv.PolyData(mesh_cell_centers)
        mesh = pv.UnstructuredGrid(poly.cast_to_unstructured_grid())
        surface = model.extract_surface()

        if self.surface_only:
            surface_distance = mesh.compute_implicit_distance(surface)
            smin, smax = self.surface_range

            indices = surface_distance['implicit_distance']

            if smax is not None:
                indices = indices <= smax

            if smin is not None:
                indices = indices >= smin
        else:
            selection = mesh.select_enclosed_points(surface, tolerance=0.0, check_surface=False)
            indices = selection['SelectedPoints'].view(np.bool_)

        return indices

    def do_place(self, mesh_cell_centers, worker_id):
        indices = None
        for model in self.models:
            ret = self.place_impl(model, mesh_cell_centers, worker_id)
            if indices is None:
                indices = ret
            else:
                indices = np.logical_or(indices, ret)

        return indices

    def __hash__(self):
        arr = self.model.cell_centers().points.flatten()
        rand = np.random.RandomState(int(arr[len(arr) // 2]))
        surface_range = self.surface_range if self.surface_range is not None else (None,)
        return hash((*rand.choice(arr, min(len(arr), 10), replace=False),
                     self.surface_only,
                     *surface_range,
                     ))

    def do_clone(self):
        ret = Obj2(None, surface_range=self.surface_range)
        ret.model = self.model.copy(deep=True)
        return ret

    def plot(self, ax, color):
        pass
