import os
from pathlib import Path

import numpy as np
import pandas as pd
from SimPEG import maps
from SimPEG.potential_fields import magnetics
from matplotlib import pyplot as plt, gridspec

from metalpy.scab import simpeg_patched, Progressed, Tied
from metalpy.scab.modelling import Scene, Bounded
from metalpy.scab.modelling.shapes import Cuboid, Prism, Ellipsoid, Tunnel
from metalpy.scab.utils.misc import define_inducing_field
from metalpy.utils.sensor_array import get_grids_ex
from metalpy.utils.type import ensure_as_iterable


def main():
    scene = Scene.of(
        Cuboid([1, 1, 1], size=2),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3),
        Prism([[0, 0], [-2, 0], [-2, 1], [-1, 1]], 1, 3).rotated(90, 0, 0),
        Ellipsoid.spheroid(1, 3, 0).translated(0, -2, 2),
        Tunnel([-3, 2, 2], 0.5, 1, 3),
        models=10
    )

    model_mesh = scene.build(cell_size=0.2, cache=True)
    model_mesh.to_polydata().threshold(1e-5).plot(
        show_edges=True,
        show_grid=True
    )
    bounds = scene.bounds
    origin, end = bounds[::2], bounds[1::2]
    active_cells = model_mesh.active_cells
    active_model = model_mesh.get_active_model()
    mesh = model_mesh.base_mesh

    with simpeg_patched(Tied(max_cpu_threads=-1), Progressed()):
        source_field = define_inducing_field(50000, 45, 20)
        pts = get_grids_ex(origin=origin, end=end,
                           n=46,  # x, y方向观测点数
                           zs=20   # 指定高度为3m
                           ).pts

        receiver_points = pts

        nC = int(np.sum(active_cells))
        components = ['tmi', 'bx', 'by', 'bz', 'bxx', 'bxy', 'bxz', 'byy', 'byz', 'bzz']
        n_components = len(components)
        receiver_list = magnetics.receivers.Point(receiver_points, components=components)
        receiver_list = [receiver_list]

        inducing_field = magnetics.sources.SourceField(
            receiver_list=receiver_list, parameters=source_field
        )
        survey = magnetics.survey.Survey(inducing_field)
        model_map = maps.IdentityMap(nP=nC)
        simulation = magnetics.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            model_type="scalar",
            chiMap=model_map,
            ind_active=active_cells,
            store_sensitivities="forward_only",
        )

    ret = simulation.dpred(active_model)

    df = pd.DataFrame(pts)
    for i, component in enumerate(components):
        df[component] = ret[i::n_components]

    filename = Path(scene._generate_model_filename(mesh))
    filename = filename.with_suffix('')
    output_path = Path(f'./output/example.{filename}.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    analyze2(output_path, cols=components, plot=True, link_colorbar=False)


def analyze2(paths, cols=None, plot=True, link_colorbar=True):
    from SimPEG.utils import plot2Ddata
    solutions = []
    filenames = []
    titles = []

    if cols is None:
        cols = [3]
    else:
        cols = ensure_as_iterable(cols, excludes=str)

    paths = ensure_as_iterable(paths, excludes=str)
    if len(cols) != len(paths):
        assert np.any(np.r_[1] == [len(cols), len(paths)]), \
            'Length of "paths" and "cols" must be either n, n or 1, n or n, 1'
        if len(cols) == 1:
            cols = cols * len(paths)
        else:
            paths = paths * len(cols)

    n_plot = len(cols)
    plot_width = 15
    spacer = 1
    bar_width = 1
    gs = gridspec.GridSpec(1, (plot_width + bar_width + spacer) * n_plot + 1)

    last_path = None
    for path, col in zip(paths, cols):
        if path != last_path:
            solution_table = pd.read_csv(f'{path}')
        filenames.append(os.path.basename(path).split('.')[0])
        titles.append(f'{filenames[-1]}[{col}]')

        coords = solution_table.iloc[:, 0:3]

        if not isinstance(col, str):
            solution = solution_table.iloc[:, col]
        else:
            solution = solution_table[col]

        solutions.append([coords, solution])

    fig = plt.figure(figsize=(2 + 5 * n_plot, 4))

    ax1 = n_plot * [None]
    ax2 = n_plot * [None]
    cbar = n_plot * [None]
    cplot = n_plot * [None]

    gs_index = 0
    for ii, (coords, solution) in enumerate(solutions):
        ax1[ii] = plt.subplot(gs[0, gs_index:gs_index + plot_width])
        gs_index = gs_index + plot_width

        if link_colorbar:
            clim = (*solutions[0][1].quantile([0.1, 0.99]),)
        else:
            clim = (*solutions[ii][1].quantile([0.1, 0.99]),)

        cplot[ii], _ = plot2Ddata(
            coords.to_numpy(),
            solution,
            ax=ax1[ii],
            ncontour=30,
            clim=clim,
            contourOpts={"cmap": "bwr", "extend": "both"},
        )
        ax1[ii].set_title(titles[ii])
        ax1[ii].set_xlabel("x (m)")
        ax1[ii].set_ylabel("y (m)")

        ax2[ii] = plt.subplot(gs[0, gs_index:gs_index + bar_width])
        gs_index = gs_index + bar_width + spacer

        cbar[ii] = plt.colorbar(cplot[ii], cax=ax2[ii])

    comp_str = '等对比' if len(filenames) != 1 else ''
    fig.savefig(Path(path).parent / f'{filenames[0]}{comp_str}.{cols}.png', dpi=120, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
