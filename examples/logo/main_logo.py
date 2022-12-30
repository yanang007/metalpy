import numpy as np
import pyvista as pv
from pyvista.demos import logo

from metalpy.scab.modelling import Scene
from metalpy.scab.modelling.shapes import Obj2, Cuboid, Prism
from metalpy.utils.file import make_cache_file


def main():
    text = logo.text_3d("METALpy", depth=0.3)
    bounds = np.asarray(text.bounds)
    shrink = 0.6
    panel = Cuboid(corner=bounds[::2] + [-2 + shrink, -1 + shrink, -0.5],
                   corner2=bounds[1::2] + [1.2 - shrink, 2 - shrink, -bounds[5]])
    d_edge = 1.2 - shrink
    relief = Prism([
        [panel.x1, panel.y1 - d_edge],
        [panel.x0 + d_edge, panel.y1 - d_edge],
        [panel.x0 + d_edge, panel.y0 + d_edge],
        [panel.x1 - d_edge, panel.y0 + d_edge],
        [panel.x1 - d_edge, panel.y1 - d_edge],
        [panel.x1, panel.y1 - d_edge],
        [panel.x1, panel.y0],
        [panel.x0, panel.y0],
        [panel.x0, panel.y1],
        [panel.x1, panel.y1],
    ], z0=bounds[4], z1=bounds[5])
    cell_size = 0.0332

    scene = Scene.of(Obj2(text, subdivide=True))
    mesh, model = scene.build(cell_size=cell_size)
    nx, ny, nz = mesh.nCx, mesh.nCy, mesh.nCz
    colored_model_layer = np.ones(nx * ny, dtype=np.int64)

    for i in range(nx):
        colored_model_layer[i::nx - 1] = i + 1

    for i in range(ny):
        start = nx + i + (nx - 1) * (i + 1)
        colored_model_layer[start::nx - 1] = nx + i + 1

    colored_model = np.tile(colored_model_layer, nz)
    colored_model[~model] = 0
    mesh_poly = scene.mesh_to_polydata(mesh, colored_model)

    scene2 = Scene.of(panel, relief)
    mesh2, model2 = scene2.build(cell_size=cell_size * 5)
    mesh_poly2 = scene2.mesh_to_polydata(mesh2, model2)

    pv.global_theme.transparent_background = True
    plotter = pv.Plotter(window_size=[300, 100])
    plotter.add_mesh(mesh_poly.threshold(1e-5), show_edges=False, show_scalar_bar=False, colormap='coolwarm')
    plotter.add_mesh(mesh_poly2.threshold(0.5), show_edges=True, show_scalar_bar=False, color='white')
    plotter.view_xy()
    plotter.camera.elevation = -30
    plotter.camera.position = (3.422140121459961, -2.8291177391927698, 6.168548286728033)
    plotter.show(screenshot=make_cache_file('logo.png'))


if __name__ == '__main__':
    main()
