import os.path
from typing import Union

import numpy as np
import pyvista as pv

from metalpy.utils.model import hash_model, split_models_in_memory, load_model_file, load_grouped_file, \
    extract_model_list_bounds
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
    SUBDIVIDE_IN_MEMORY = 'memory'
    SUBDIVIDE_FILE = 'file'
    SUBDIVIDE_NONE = False

    def __init__(self, model: Union[str, pv.DataSet],
                 xsize=None, ysize=None, zsize=None, size=None,
                 xscale=None, yscale=None, zscale=None, scale=None,
                 surface_thickness=None, surface_range=None,
                 subdivide: Union[str, bool, None] = SUBDIVIDE_NONE,
                 verbose=True):
        """
        :param model: 模型路径或PyVista模型对象 (当为None时不主动构造，用于内部的拷贝代码，非必要请勿使用)
        二选一
        :param xsize, ysize, zsize, size: 模型缩放后的新尺寸
        :param xscale, yscale, zscale, scale: 模型缩放比例

        仅保留表面，参数二选一
        :param surface_thickness: 表面厚度（向内），即surface_range=[-surface_range, 0]
        :param surface_range: [(float|None), (float|None)] 表面范围（小于0表示模型内，大于0表示模型外，None表示不作判断）
        :param subdivide: 是否模型细分（情况受限）
            SUBDIVIDE_IN_MEMORY 载入后根据点连接性来模型细分
            SUBDIVIDE_FILE 对模型文件进行组分割（仅限obj文件）再载入

        :param verbose: 是否输出相关信息
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
            models = None
            if isinstance(model, str):
                model_path = model
                should_scale_be_inplace = True  # 从文件导入的model的scale操作可以inplace

                if not subdivide:
                    model = load_model_file(model_path, verbose=verbose)
                    models = [model]
                elif subdivide == Obj2.SUBDIVIDE_IN_MEMORY or subdivide:  # 导入后通过连通性分析分割模型
                    model = load_model_file(model_path, verbose=verbose)
                    models = split_models_in_memory(model, verbose=verbose)
                elif subdivide == Obj2.SUBDIVIDE_FILE:  # 从文件层面切割得到的子模型文件导入到models
                    models = load_grouped_file(model_path, verbose=verbose)
                    if models is None:
                        raise NotImplementedError(
                            f"Model file splitter is not available for {os.path.splitext(model_path)[1]} file. \n"
                            f"Which is {model_path} .")

            bounds = extract_model_list_bounds(models)
            xl, yl, zl = bounds[1::2] - bounds[0::2]

            size = check_scalar_or_list(size, 3, None)

            size[0] = check_is_none(xsize, size[0])
            size[1] = check_is_none(ysize, size[1])
            size[2] = check_is_none(zsize, size[2])
            xsize, ysize, zsize = size

            scale = check_scalar_or_list(scale, 3, 1)

            scale[0] = check_scale(xl, xsize, xscale, scale[0])
            scale[1] = check_scale(yl, ysize, yscale, scale[1])
            scale[2] = check_scale(zl, zsize, zscale, scale[2])

            if np.any(np.asarray(scale) != 1):
                for i in range(len(models)):
                    models[i] = models[i].scale(scale, inplace=should_scale_be_inplace)
                bounds = extract_model_list_bounds(models)

            for m in models:
                m.translate(-bounds[::2], inplace=True)

            self.models = models

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
            selection = mesh.select_enclosed_points(surface, tolerance=0.0, check_surface=True)
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

    def do_hash(self):
        n_samples = np.max((2, 10 // len(self.models)))
        surface_range = self.surface_range if self.surface_range is not None else (None,)
        return hash((*(hash_model(m, n_samples) for m in self.models),
                     self.surface_only,
                     *surface_range,
                     ))

    def do_clone(self):
        ret = Obj2(None, surface_range=self.surface_range)
        ret.models = [m.copy(deep=True) for m in self.models]
        return ret

    def plot(self, ax, color):
        pass

    @property
    def local_bounds(self):
        return extract_model_list_bounds(self.models)

    def to_local_polydata(self):
        return pv.merge(self.models, merge_points=False).copy(deep=True)
