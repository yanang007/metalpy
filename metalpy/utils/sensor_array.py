"""
坐标系为右手坐标系，东为+x，地理北为+y，竖直向上为+z
"SimPEG uses a right handed coordinate system, with z being positive upwards"
"""
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

from metalpy.utils.misc import sin2pi, cos2pi, plot_opaque_cube, plot_linear_cube

from colour import Color


def inplaceable(func):
    def wrapper(*args, **kwargs):
        # 规避ide无法识别修饰器中的self参数
        self = args[0]
        args = args[1:]

        # 替换inplace行为
        obj = self.dispatch_inplace_delegate(kwargs.get('inplace'))
        return func(obj, *args, **kwargs)

    return wrapper


class ArraySlices:
    def __init__(self, arrays):
        self.arrays = arrays
        self.index = 0

    def slice(self, n_slices=1):
        ret = np.vstack(self.arrays[self.index: self.index + n_slices])
        self.index += n_slices
        return ret


class SensorArray:
    def __init__(self, pts, inplace=False):
        """
        构造函数
        :param pts: 初始观测点集合
        :param inplace: 指示该实例的所有操作是否为inplace操作
        """
        self.pts = np.asarray(pts, dtype=np.float64)
        self.inplace = inplace

        self.vt = np.zeros(3)
        self.vs = np.zeros(3)
        self.location_func = None
        self.orientation_func = None

    def __len__(self):
        return len(self.pts)

    def clear_movement(self):
        self.vt = np.zeros(3)
        self.vs = np.zeros(3)
        self.location_func = None
        self.orientation_func = None

    def set_default_inplace_behavior(self, b):
        """
        设置默认的inplace行为，True为所有操作均在当前类上修改，False为所有操作都构造新的实例
        :param b:
        :return:
        """
        self.inplace = b
        return self

    def clone(self, clear_movement=False):
        """
        构造一个新实例
        :return:
        """
        ret = SensorArray(self.pts.copy())
        ret.inplace = self.inplace

        if not clear_movement:
            ret.vt = self.vt
            ret.vs = self.vs
            ret.location_func = self.location_func
            ret.orientation_func = self.orientation_func

        return ret

    def dispatch_inplace_delegate(self, inplace):
        if inplace is None:
            inplace = self.inplace
        return self if inplace else self.clone()

    @inplaceable
    def scale(self, f, **kwargs):
        """
        警告，是关于原点放缩，可能不适用大部分情形
        :param f: 比例因子
        :return:
        """
        self.pts *= f
        return self

    @inplaceable
    def rotate(self, y, a, b, rot_tags='xyz', **kwargs):
        rot = R.from_euler(rot_tags, [y, a, b], degrees=True).as_matrix()
        self.pts = self.pts.dot(rot.T)
        return self

    @inplaceable
    def target_rotate(self, y, a, b, rot_tags='xyz', **kwargs):
        rot = R.from_euler(rot_tags, [-y, -a, -b], degrees=True).as_matrix()
        return SensorArray(self.pts.dot(rot.T))

    @inplaceable
    def translate(self, dx, dy, dz, **kwargs):
        translate_mat = np.asarray([[dx, dy, dz]]).repeat(len(self.pts), axis=0)
        self.pts += translate_mat
        return self

    def get_points(self, n_chunks=1, worker_id=0):
        return np.array_split(self.pts, n_chunks)[worker_id]

    def cut(self, n_chunks):
        return ArraySlices(np.array_split(self.pts, n_chunks))

    """
    with_*** 系列函数用于运动状态下观测点生成，采用lazy evaluation，在执行run后才会计算具体的观测点序列
    """

    @inplaceable
    def with_target_speed(self, vt, **kwargs):
        self.vt = vt
        return self

    @inplaceable
    def with_sensor_speed(self, vs, **kwargs):
        self.vs = vs
        return self

    @inplaceable
    def with_trajectory(self, location_func, **kwargs):
        """
        :param location_func:原点位置关于时间的函数, t -> (x, y, z)
        :return:
        """
        self.location_func = location_func
        return self

    @inplaceable
    def with_pose_movement(self, orientation_func, **kwargs):
        """
        :param orientation_func:朝向关于时间的函数, t -> (y, a, b)
        :return:
        """
        self.orientation_func = orientation_func
        return self

    def __default_get_location(self, t, **kwargs):
        relative_velocity = self.vs - self.vt
        delta = relative_velocity * t
        return delta

    def __default_get_orientation(self, t, **kwargs):
        return [0, 0, 0]

    def construct(self, dt=None, n_samples=None, timespan=None, with_timestamps=True):
        """
        参数三选二，采样结果包含0时刻的位置坐标（即初始位置坐标）
        :param dt:采样间隔
        :param n_samples:采样次数
        :param timespan:采样持续时间
        :param with_timestamps:是否返回每个观测点对应的时间刻，True则返回array, times，False则返回array
        :return:依据运动函数得到的一组点序列，作为一个新的SensorArray返回，新实例的运动函数会被重置为无运动
        """

        if dt is None:
            dt = timespan / (n_samples - 1)

        if timespan is None:
            timespan = dt * (n_samples - 1)

        ret = []
        timerange = np.arange(0, timespan + dt, dt)

        location_func = self.location_func
        orientation_func = self.orientation_func

        if location_func is None:
            def location_func(tx): return self.__default_get_location(tx)

        if orientation_func is None:
            def orientation_func(tx): return self.__default_get_orientation(tx)

        for t in timerange:
            delta = location_func(t)
            orientation = orientation_func(t)
            ret.append(self.rotate(*orientation, inplace=False).translate(*delta, inplace=False).get_points())
            # ret.append(self.pts + delta)

        pts = np.vstack(ret)
        ret = self.clone(clear_movement=True)
        ret.pts = pts

        if with_timestamps:
            return ret, timerange.repeat(len(self.pts))
        else:
            return ret

    def to_file(self, fname, **kwargs):
        with open(fname, 'w') as f:
            json.dump(self.pts.tolist(), f)

    def from_file(self, fname):
        with open(fname, 'r') as f:
            self.pts = np.asarray(json.load(f))

    def to_csv(self, fname):
        with open(fname, 'w') as output:
            pd.DataFrame(self.pts).to_csv(output, index=False, header=False, line_terminator='\n')

    @staticmethod
    def concat(s1, s2):
        return SensorArray(np.r_[s1.pts, s2.pts])

    @staticmethod
    def from_csv(fname):
        with open(fname, 'r') as _input:
            pts = pd.read_csv(_input, header=None).to_numpy()
        return SensorArray(pts)


def get_std7pts(distance):
    array = SensorArray([
        [0, 0, 0],
        [distance, 0, 0],
        [0, distance, 0],
        [0, 0, distance],
        [-distance, 0, 0],
        [0, -distance, 0],
        [0, 0, -distance],
    ])

    return array


def get_ext10pts(distance):
    array = SensorArray([
        [0, 0, 0],
        [distance, 0, 0],
        [0, distance, 0],
        [0, 0, distance],
        [-distance, 0, 0],
        [0, -distance, 0],
        [0, 0, -distance],
        [distance, distance, 0],
        [0, distance, distance],
        [distance, 0, distance],
    ])

    return array


def get_grids(distance, nx, ny, nz):
    xs, ys, zs = np.meshgrid(np.linspace(0, (nx - 1) * distance, nx), np.linspace(0, (ny - 1) * distance, ny),
                             np.linspace(0, (nz - 1) * distance, nz))
    xs, ys, zs = xs.flatten(), ys.flatten(), zs.flatten()
    array = SensorArray(np.c_[xs, ys, zs])

    return array

if __name__ == '__main__':

    # 构造网格，间隔10m，xyz方向各行
    array = get_grids(100, 2, 2, 2)

    # 设置array的各个操作默认是否为inplace，即是否会修改自身
    # array.inplace = False

    fig = plt.figure(figsize=(9, 4))

    ax = plt.axes(projection='3d')
    ax.set_zlim(0, 800)

    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')
    b = 512
    plot_opaque_cube(ax, -b, -b, 30, dx=2 * b, dy=2 * b, dz=1, alpha=0.4)
    plot_opaque_cube(ax, -b, -b, 300, dx=2 * b, dy=2 * b, dz=1, alpha=0.4)
    plot_opaque_cube(ax, -b, -b, 600, dx=2 * b, dy=2 * b, dz=1, alpha=0.4)

    a, b, c = 80.0, 10.0, 10.0
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    ax.plot_surface(x, y, z, color='b')

    plt.show()

    ax = plt.axes(projection='3d')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    plot_linear_cube(ax, 0, 0, 0, dx=3.33333, dy=1, dz=1)

    plt.show()

    # 指定目标速度或阵列速度
    linear_movement, ts = array.with_target_speed(np.array([1, 2, 3])).construct(dt=0.2, timespan=5, with_timestamps=True)

    ax = plt.axes(projection='3d')

    red = Color("red")
    colors = list(red.range_to(Color("blue"), len(linear_movement)))
    x, y, z = linear_movement.get_points().T
    ax.scatter(*array.get_points().T)
    # ax.scatter(x, y, z, c=ts, cmap='plasma')
    # ax.quiver(850, 815, -785, -200, 0, 0)
    ax.quiver(50, 50, 50, 200, 0, 0)

    plot_linear_cube(ax, 800, 800, -800, dx=100, dy=30, dz=30)

    plt.show()

    # 指定目标运动轨迹
    circular_movement, ts = get_grids(0, 1, 1, 1).with_target_speed(np.array([20, 0, 0])).with_trajectory(
        location_func=lambda t: 100*np.array([sin2pi(t / 5), cos2pi(t / 5), 0]) - np.array([20,0,0]) * t).construct(dt=0.2, timespan=5,
                                                                                      with_timestamps=True)

    circular_movement = circular_movement.translate(800, 800, 0)
    ax = plt.axes(projection='3d')
    pts = circular_movement.get_points()
    ax.scatter(*pts.T, c=ts, cmap='plasma')

    plot_linear_cube(ax, 800, 800, -800, dx=100, dy=30, dz=30)
    ax.quiver(850, 815, -785, -200, 0, 0)

    # ax.quiver(*pts[-2], *(pts[-1] - pts[-2]), arrow_length_ratio=0.0001)
    plt.show()

    # 通过运动生成初始点阵然后再进行新的运动
    compound_movement, ts = get_grids(0, 1, 1, 1) \
        .with_sensor_speed(np.array([1, 1, 1])) \
        .construct(n_samples=10, timespan=5, with_timestamps=False) \
        .with_trajectory(location_func=lambda t: np.array([sin2pi(t / 5), cos2pi(t / 5), 0])) \
        .construct(dt=0.2, timespan=5, with_timestamps=True)

    ax = plt.axes(projection='3d')
    ax.scatter(*compound_movement.get_points().T, c=ts, cmap='plasma')
    plt.show()



