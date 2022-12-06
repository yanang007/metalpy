SimPEG Commons and Boosters
===========================

SCAB是关于SimPEG的通用工具包，提供许多针对SimPEG的实用功能与扩展，包括: 自动并行化，退磁计算以及其他常用功能。 

SCAB原名**SimPEG Common Auto Batching**，其现在已经成为了SCAB的自动并行化模块。

安装
------------
SCAB目前是metalpy的一个子模块，你可以使用pip安装它，并指定安装scab模块所需的依赖：

```console
pip install "metalpy[scab]"
```

特性
-----
### 一键并行化

只需要添加两行代码，就可以将SimPEG势场正演过程改为基于Taichi并行计算的后端，并支持无缝迁移到GPU上。

<table>
  <tr>
    <th>Before</th>
    <th>After</th>
  </tr>
  <tr>
    <td><pre lang="python">
&nbsp;
&nbsp;
&nbsp;
import ...
import ...
&nbsp;
receiver_list = Point(
&nbsp;&nbsp;&nbsp;&nbsp;receiver_points, 
&nbsp;&nbsp;&nbsp;&nbsp;components=components
)
receiver_list = [receiver_list]
&nbsp;
source_field = sources.SourceField(
&nbsp;&nbsp;&nbsp;&nbsp;receiver_list=receiver_list, parameters=H
)
survey = survey.Survey(source_field)
&nbsp;
simulation = simulation.Simulation3DIntegral(
&nbsp;&nbsp;&nbsp;&nbsp;survey=survey,
&nbsp;&nbsp;&nbsp;&nbsp;mesh=mesh,
&nbsp;&nbsp;&nbsp;&nbsp;model_type="scalar",
&nbsp;&nbsp;&nbsp;&nbsp;chiMap=model_map,
&nbsp;&nbsp;&nbsp;&nbsp;ind_active=active_ind,
&nbsp;&nbsp;&nbsp;&nbsp;store_sensitivities="forward_only",
)
    </pre></td>
    <td><pre lang="python">
import ...
import ...

**_from metalpy.scab import simpeg_patched, Progressed, Tied_**
&nbsp;
#&nbsp;通过patch系统将正演计算后端替换为Taichi，并添加进度条
**_with simpeg_patched(Tied('gpu'), Progressed()):_**
&nbsp;&nbsp;&nbsp;&nbsp;receiver_list = Point(
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;receiver_points, 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;components=components
&nbsp;&nbsp;&nbsp;&nbsp;)
&nbsp;&nbsp;&nbsp;&nbsp;receiver_list = [receiver_list]
&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;source_field = sources.SourceField(
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;receiver_list=receiver_list, parameters=H
&nbsp;&nbsp;&nbsp;&nbsp;)
&nbsp;&nbsp;&nbsp;&nbsp;survey = survey.Survey(source_field)
&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;simulation = simulation.Simulation3DIntegral(
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;survey=survey,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mesh=mesh,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;model_type="scalar",
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;chiMap=model_map,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ind_active=active_ind,
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;store_sensitivities="forward_only",
&nbsp;&nbsp;&nbsp;&nbsp;)
    </pre></td>
  </tr>
</table>

### 分布式计算

只需要再添加一个Patch，就可以实现基于Dask的分布式正演运算。

接收器将被自动分割为多个批次，依靠MEPA分发到各个计算单元上并行运行，最后将结果合并返回。

```python
from metalpy.scab import simpeg_patched, Progressed, Tied, Distributed
from metalpy.mepa import DaskExecutor

executor = DaskExecutor('tcp://scheduler.addr:8786')
with simpeg_patched(Distributed(executor=executor), Tied('gpu'), Progressed()):
    # 正演代码...
```

### 基于层级的建模系统

通过更加人性化的建模系统来操作网格。支持基于层级的建模，可以方便地表达不同的层级的组合关系。 支持导出为PyVista对象。

```python
scene = Scene.of(
    Cuboid([1, 1, 1], size=2),
    Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3),
    Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3).rotated(90, 0, 0),
    Ellipsoid.spheroid(1, 3, 0).translated(0, -2, 2),
    Tunnel([-3, 2, 2], 0.5, 1, 3),
    Obj2('../obj/stl_models/mine.stl', scale=0.03).translated(1, -2, -1),
)

mesh, ind_active = scene.build(cell_size=0.2, cache=True, bounds=Bounded(zmax=2))
grids = scene.mesh_to_polydata(mesh, ind_active)

import pyvista as pv
p = pv.Plotter()
p.add_mesh(scene.to_multiblock(), opacity=0.2)
p.add_mesh(grids.threshold([1, 1]), show_edges=True)
p.show_grid()
p.show()
```

### 注
你其实也可以叫它**Super Cat Annihilates Bugs**（**超级猫猫歼灭Bug**）。
