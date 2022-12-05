import numpy as np
import taichi as ti
from SimPEG import maps
from SimPEG.potential_fields import magnetics
from SimPEG.utils import plot2Ddata
from discretize.utils import mkvc
import matplotlib as mpl
from matplotlib import pyplot as plt

from forward import setup_cuboids_model
from metalpy.mepa import LinearExecutor
from metalpy.scab import simpeg_patched, Progressed, Tied
from metalpy.scab.demag import Demagnetization
from metalpy.scab.demag.factored_demagnetization import FactoredDemagnetization
from metalpy.scab.demag.utils import get_prolate_spheroid_demag_factor
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.scab.utils.misc import define_inducing_field

from config import get_exec_config
from metalpy.utils.taichi import ti_prepare, ti_config
from metalpy.utils.time import Timer


def main(grid_size, gpu=False):
    if gpu:
        ti_prepare(arch=ti.gpu)

    a, c = 10, 40
    timer = Timer()

    with timer:
        model = Ellipsoid.spheroid(a, c, polar_axis=0)
        mesh, model, model_map, active_cells = \
            setup_cuboids_model(grid_size=grid_size, sus=80, cuboids=[model],
                                xspan=[-c, c], yspan=[-a, a], zspan=[-a, a],
                                executor=LinearExecutor(1),
                                # plot_output=True
                                )

        source_field = define_inducing_field(50000, 45, 20)

    print(f"Modelling: {timer}")

    with timer:
        # analytical demagnetization factor
        N = get_prolate_spheroid_demag_factor(c / a, polar_axis=0)
        demag = FactoredDemagnetization(n=N)
        demaged_model = demag.dpred(model, source_field=source_field)

        # numerical demagnetization factor
        with ti_config(arch=ti.gpu if gpu else ti.cpu):
            demag2 = Demagnetization(
                source_field=source_field,
                mesh=mesh,
                active_ind=active_cells)
            demaged_model2 = demag2.dpred(model)

    print(f"Solving: {timer}")
    print("MagModel MAPE(%): ", (abs(demaged_model2 - demaged_model) / abs(demaged_model)).mean() * 100)

    with simpeg_patched(Tied(), Progressed()):
        obsx = np.arange(-c, c + 1, 1) * 2
        obsy = np.arange(-c, c + 1, 1) * 2
        obsx, obsy = np.meshgrid(obsx, obsy)
        obsx, obsy = mkvc(obsx), mkvc(obsy)
        obsz = np.ones_like(obsy) * 50

        receiver_points = np.c_[obsx, obsy, obsz]

        nC = int(np.sum(active_cells))
        components = ('tmi',)
        receiver_list = magnetics.receivers.Point(receiver_points, components=components)
        receiver_list = [receiver_list]

        inducing_field = magnetics.sources.SourceField(
            receiver_list=receiver_list, parameters=source_field
        )
        survey = magnetics.survey.Survey(inducing_field)
        model_map = maps.IdentityMap(nP=3 * nC)
        simulation = magnetics.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            model_type="vector",
            chiMap=model_map,
            ind_active=active_cells,
            store_sensitivities="ram",
        )

    pred = simulation.dpred(mkvc(demaged_model))
    pred2 = simulation.dpred(mkvc(demaged_model2))

    return demaged_model, pred, demaged_model2, pred2, receiver_points


if __name__ == '__main__':
    executor = get_exec_config()  # LinearExecutor(1)  #

    workers = [w for w in executor.get_workers() if 'large-mem' in w.group]
    if len(workers) == 1:
        f = executor.submit(main, [1.2, 1.2, 0.6], workers=workers)
    else:
        gpu = False
        if executor.is_local():
            gpu = True
        f = executor.submit(main, [2.3, 2.3, 0.9], gpu=gpu)

    demaged_model, pred, demaged_model2, pred2, receiver_points = executor.gather([f])[0]

    print('Model MAPE (%):', (abs(demaged_model2 - demaged_model) / abs(demaged_model)).mean() * 100)
    print('TMI MAPE (%):', (abs(pred - pred2) / abs(pred)).mean() * 100)

    fig = plt.figure(figsize=(17, 4))

    data_array = np.c_[pred, pred2, pred-pred2]
    plot_title = ["Observed", "Predicted", "Absolute Error"]
    plot_units = ["nT", "nT", ""]

    ax1 = 3 * [None]
    ax2 = 3 * [None]
    norm = 3 * [None]
    cbar = 3 * [None]
    cplot = 3 * [None]
    v_lim = [np.max(np.abs(data_array[:, :-1])),
             np.max(np.abs(data_array[:, :-1])),
             np.max(np.abs(data_array[:, -1]))]

    for ii in range(0, 3):
        ax1[ii] = fig.add_axes([0.33 * ii + 0.03, 0.11, 0.25, 0.84])
        cplot[ii] = plot2Ddata(
            receiver_points,
            data_array[:, ii],
            ax=ax1[ii],
            ncontour=30,
            clim=(-v_lim[ii], v_lim[ii]),
            contourOpts={"cmap": "bwr"},
        )
        ax1[ii].set_title(plot_title[ii])
        ax1[ii].set_xlabel("x (m)")
        ax1[ii].set_ylabel("y (m)")

        ax2[ii] = fig.add_axes([0.33 * ii + 0.27, 0.11, 0.01, 0.84])
        norm[ii] = mpl.colors.Normalize(vmin=-v_lim[ii], vmax=v_lim[ii])
        cbar[ii] = mpl.colorbar.ColorbarBase(
            ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.bwr
        )
        cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)
    plt.show()
