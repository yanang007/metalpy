from typing import Union

import numpy as np
import pyvista as pv

from metalpy.utils.model import extract_model_list_bounds


class SubplotSwitched:
    def __init__(self, plotter: pv.BasePlotter, index_loc=None):
        self.plotter = plotter
        self.backup = plotter.renderers.index_to_loc(plotter.renderers.active_index)
        if index_loc is not None and isinstance(index_loc, int):
            index_loc = [index_loc, None]

        if np.all(index_loc == self.backup):
            self.target = None
        else:
            self.target = index_loc

    def __enter__(self):
        if self.target is not None:
            self.plotter.subplot(*self.target)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.target is not None:
            self.plotter.subplot(*self.backup)


class SliceView:
    def __init__(self, plotter: pv.BasePlotter, index_loc=None):
        """实现一个三维模型的切片视图

        Parameters
        ----------
        plotter
            绑定的绘制设备

        index_loc
            绑定的视图坐标

        Examples
        --------
            >>> plotter = pv.Plotter(shape=(1, 2))
            >>> plotter.subplot(0, 0)
            >>> slice_view = SliceView(plotter, [0, 1])

            >>> voxels = pv.voxelize(pv.Sphere(1, (0, 0, 0)).scale((8, 4, 2), inplace=False), density=0.4)
            >>> plotter.add_mesh(voxels, opacity=1, style='surface', show_edges=True)
            >>> slice_view.add_mesh(voxels, opacity=1, style='surface', show_edges=True)

            >>> plotter.show_grid()
            >>> plotter.enable_cell_picking(callback=slice_view.clip_by_selection, show=False, through=False)

            >>> plotter.show()
        """
        self.mesh_elements = []
        self.actors = []
        self.plotter = plotter
        self.index_loc = index_loc

    def switched(self):
        """将绘制设备的活动视图临时改为绑定的子视图

        Returns
        -------
        ret : SubplotSwitched
            用于with块
        """
        return SubplotSwitched(self.plotter, self.index_loc)

    def _add_mesh(self, mesh, **kwargs):
        ret = self.plotter.add_mesh(mesh, **kwargs)
        self.actors.append(ret)

    def add_mesh(self, mesh, **kwargs):
        """同Plotter.add_mesh

        Parameters
        ----------
        mesh
            待切片的mesh

        kwargs
            其它传给Plotter.add_mesh的参数
        """
        self.mesh_elements.append((mesh, kwargs))

        with self.switched():
            self._add_mesh(mesh, **kwargs)

    def clear(self):
        """清理切片视图中绘制的切片
        """
        with self.switched():
            for t in self.actors:
                self.plotter.remove_actor(t)
            self.actors.clear()

    def clip(self, bounds, refresh=True, update=True):
        """使用指定的边界截取切片

        Parameters
        ----------
        bounds
            切片边界

        refresh
            是否清理已有的切片图形

        update
            是否自动更新视图（如果有其它需要更新的内容可以置为False然后在函数外手动更新）
        """
        with self.switched():
            if refresh:
                self.clear()

            for mesh, kwargs in self.mesh_elements:
                if bounds is not None:
                    mesh = mesh.clip_box(bounds, invert=False)
                if mesh.n_cells > 0:
                    self._add_mesh(mesh, **kwargs)

            if update:
                self.plotter.update(force_redraw=True)

    def clip_by_selection(self, selection: Union[pv.UnstructuredGrid, pv.MultiBlock],
                          enable_ortho_slice=True,
                          refresh=True, update=True):
        """使用选定区域与视角来判断截取切片的范围用于切片

        界面确定方法为:
        1. 视角所朝向的轴为延申轴a
        2. 截取范围中相对场景范围较宽的方向为延申轴b
        3. 截取限定第三轴范围和selection一致下，a-b平面上的切片

        若启用正交切片enable_ortho_slice
        则在第一步后，分别以除a外的两轴进行一次切片，组合展示

        Parameters
        ----------
        selection
            切片边界

        enable_ortho_slice
            是否启用p键单cell选择时展示正交切片

        refresh
            是否清理已有的切片图形

        update
            是否自动更新视图（如果有其它需要更新的内容可以置为False然后在函数外手动更新）
        """
        if refresh:
            self.clear()

        if selection is None:
            print('Bad Selection!')
            return

        bounds = np.asarray(selection.bounds)
        camera_pos = np.asarray(self.plotter.camera_position[0])
        axis = np.argmax(abs(camera_pos))

        scene_bounds = np.asarray(extract_model_list_bounds([a[0] for a in self.mesh_elements]))
        scene_sizes = scene_bounds[1::2] - scene_bounds[::2]
        sizes = bounds[1::2] - bounds[::2]
        sizes[axis] = 0
        axis2 = np.argmax(sizes / scene_sizes)
        print(axis, axis2)

        if axis == axis2:
            print('Bad Selection!')
            return

        bounded_axis = 0 + 1 + 2 - axis - axis2
        clipping_bounds = scene_bounds.copy()
        clipping_bounds[2 * bounded_axis:2 * bounded_axis + 2] = \
            bounds[2 * bounded_axis:2 * bounded_axis + 2]

        self.clip(clipping_bounds, refresh=False, update=False)
        if enable_ortho_slice and \
                isinstance(selection, pv.UnstructuredGrid) and \
                selection.n_cells == 1:
            clipping_bounds = scene_bounds.copy()
            clipping_bounds[2 * axis2:2 * axis2 + 2] = \
                bounds[2 * axis2:2 * axis2 + 2]
            self.clip(clipping_bounds, refresh=False, update=False)

            prev_pos = self.plotter.camera_position
            with self.switched():
                self.plotter.camera_position = prev_pos
        else:
            with self.switched():
                self.plotter.camera_position = ['yz', 'xz', 'xy'][bounded_axis]

        if update:
            self.plotter.update(force_redraw=True)
