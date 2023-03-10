SimPEG Commons and Boosters
===========================

SCAB is a collection of tools for SimPEG, 
providing common utilities and extensions including parallelization, demagnitization support and more.

SCAB was previously named **SimPEG Common Auto Batching**, which is now a specific module for parallelization in SCAB.

Installation
------------
SCAB is now a submodule in metalpy, which can be installed using pip,
with extra dependencies required by SCAB:

```console
pip install "metalpy[scab]"
```

Features
--------
### Easy parallelization

Enables your simulation of potential field to run in parallel on Taichi backend with few code changes.
And GPU support is available now.

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
#&nbsp;Switch the backend of forward simulation to Taichi by patch system, 
#&nbsp;and enables progress bar
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

### Distributed Simulation

Distribute the simulation to Dask cluster with another Patch.

Receivers will be divided into batches and be distributed to compute units by MEPA.
The result will be merged as what they should be.  

```python
from metalpy.scab import simpeg_patched, Progressed, Tied, Distributed
from metalpy.mepa import DaskExecutor

executor = DaskExecutor('tcp://scheduler.addr:8786')
with simpeg_patched(Distributed(executor=executor), Tied('gpu'), Progressed()):
    # Do the simulation...
```

### Layer-based Modelling System

Manipulate the mesh in an easier way.

The modelling system supports layer-based modelling, model composition and exporting to PyVista objects.

```python
scene = Scene.of(
    Cuboid([1, 1, 1], size=2),
    Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3),
    Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3).rotated(90, 0, 0, degrees=True),
    Ellipsoid.spheroid(1, 3, 0).translated(0, -2, 2),
    Tunnel([-3, 2, 2], 0.5, 1, 3),
    Obj2('../obj/stl_models/mine.stl', scale=0.03).translated(1, -2, -1),
)

model_mesh = scene.build(cell_size=0.2, cache=True, bounds=Bounded(zmax=2))
grids = model_mesh.to_polydata()

import pyvista as pv
p = pv.Plotter()
p.add_mesh(scene.to_multiblock(), opacity=0.2)
p.add_mesh(grids.threshold([1, 1]), show_edges=True)
p.show_grid()
p.show()
```

### Notes
You may also call it **Super Cat Annihilates Bugs**。
