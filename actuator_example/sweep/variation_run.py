# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from functools import partial

import automembrane.plot_helper as ph
import jax
import matplotlib as mpl
import numpy as np
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.geometry import ClosedPlaneCurveGeometry
from automembrane.integrator import fwd_euler_integrator
from PIL import Image
from scipy.interpolate import splev, splprep
from tqdm.contrib.concurrent import process_map

from actuator_constants import files

jax.config.update("jax_enable_x64", True)
ph.matplotlibStyle(small=10, medium=12, large=14)


def get_dimensional_tension(Ksg_, Kb, coords):
    curvature_scale = np.max(abs(ClosedPlaneCurveGeometry.edge_curvature(coords)))
    return 4 * Kb * curvature_scale**2 * Ksg_


def get_force_density(parameters, coords):
    mem = ClosedPlaneCurveMaterial(**parameters)
    forces = np.array(
        [
            force / ClosedPlaneCurveGeometry.vertex_dual_length(coords)
            for force in mem.force(coords)
        ]
    )
    return forces       


def run_parameter_variation(file, _Ksg_):
    Ksg_force = []
    data = np.load(f"relaxed_coords/{file.stem}.npz")
    relaxed_coords = data["relaxed_coords"]
    for Ksg_ in _Ksg_:
        # Instantiate material properties
        Kb = 0.1
        Ksg = get_dimensional_tension(Ksg_, Kb, relaxed_coords)
        parameters = {
            "Kb": Kb / 4,  # Bending modulus (pN um; original 1e-19 J)
            "Ksg": Ksg,  # Global stretching modulus (pN um/um^2; original 0.05 mN/m)
            "Ksl": 0,
        }
        print(f"dimensional tension for {file.stem}: ", Ksg)
        # Ksg_coords_force.append(np.concatenate(([relaxed_coords], relaxed_forces), axis=0))
        Ksg_force.append(get_force_density(parameters, relaxed_coords))
    Ksg_force = np.asarray(Ksg_force)
    np.savez(f"forces/{file.stem}", _Ksg_=_Ksg_, Ksg_force=Ksg_force)


if __name__ == "__main__":
    f_run = partial(run_parameter_variation, _Ksg_=np.linspace(0, 0.3, 1 + 2**5))
    r = process_map(f_run, files, max_workers=12)
        