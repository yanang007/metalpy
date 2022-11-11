from forward import setup_cuboids_model
from metalpy.mepa import LinearExecutor, ProcessExecutor
from metalpy.scab import simpeg_patched, Distributed, Progressed
from metalpy.scab.demag import Demagnetization
from metalpy.scab.demag.factored_demagnetization import FactoredDemagnetization
from metalpy.scab.demag.utils import get_prolate_spheroid_demag_factor
from metalpy.scab.modelling.shapes import Ellipsoid
from metalpy.scab.utils.misc import define_inducing_field

from config import get_exec_config
from metalpy.utils.time import Timer


def main():
    a, c = 10, 40
    timer = Timer()

    with timer:
        model = Ellipsoid.spheroid(a, c, polar_axis=0)
        mesh, model, model_map, active_cells = \
            setup_cuboids_model(grid_size=[1.7, 1.7, 0.9], sus=80, cuboids=[model],
                                xspan=[-c, c], yspan=[-a, a], zspan=[-a, a],
                                executor=LinearExecutor(3),
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
        with simpeg_patched(Progressed()):
            demag2 = Demagnetization(
                source_field=source_field,
                mesh=mesh,
                active_ind=active_cells)

    with timer:
        demaged_model2 = demag2.dpred(model)

    print(f"Solving: {timer}")

    print((abs(demaged_model2 - demaged_model) / abs(demaged_model)).mean())

    return demaged_model, demaged_model2


if __name__ == '__main__':
    executor = get_exec_config()
    f = executor.submit(main, workers=[w for w in executor.get_workers() if w.group == 'lab'])
    demaged_model, demaged_model2 = executor.gather(f)
    # demaged_model, demaged_model2 = main()
    print((abs(demaged_model2 - demaged_model) / abs(demaged_model)).mean())
    pass
