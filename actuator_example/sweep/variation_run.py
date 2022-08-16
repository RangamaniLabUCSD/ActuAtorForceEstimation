# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math
import sys

from functools import partial
from collections import defaultdict

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import jax
from PIL import Image
from scipy.interpolate import splev, splprep
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

import automembrane.util as u
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.geometry import ClosedPlaneCurveGeometry
from automembrane.integrator import fwd_euler_integrator
import automembrane.plot_helper as ph

from actuator_constants import files, images

jax.config.update("jax_enable_x64", True)
ph.matplotlibStyle(small=10, medium=12, large=14)

# Plotting settings
padding = 2
cm = mpl.cm.viridis_r

def resample(original_coords, target_edge_length):
    total_length = np.sum(
        np.linalg.norm(
            np.roll(original_coords[:-1], -1, axis=0) - original_coords[:-1], axis=1
        )
    )
    n_vertices = math.floor(total_length / target_edge_length)
    # print(f"  Resampling to {n_vertices} vertices")
    # Periodic cubic B-spline interpolation with no smoothing (s=0)
    tck, _ = splprep([original_coords[:, 0], original_coords[:, 1]], s=0, per=True)

    xi, yi = splev(np.linspace(0, 1, n_vertices), tck)
    coords = np.hstack((xi.reshape(-1, 1), yi.reshape(-1, 1)))
    return coords, tck

def preprocess_mesh(file, ifResample, target_edge_length):
    k = file.stem

    # print("Processing:", k)
    original_coords = np.loadtxt(file)
    original_coords = np.vstack(
        (original_coords, original_coords[0])
    )  # Energy expects last point to equal first

    if ifResample:
        coords, tck = resample(original_coords, target_edge_length)
    else:
        coords = original_coords
    
    return coords, original_coords

def relax_bending(coords, Kb, dt, n_iter):
    # Instantiate material properties
    parameters = {
        "Kb": Kb / 4, 
        "Ksg": 0,
        "Ksl": 10 * Kb,
    }
    mem = ClosedPlaneCurveMaterial(**parameters)
    # Perform energy relaxation
    relaxed_coords = coords
    if n_iter > 0:
        relaxed_coords, energy_log = fwd_euler_integrator(
            relaxed_coords, mem, n_steps=n_iter, dt=dt
        )
        print(
            f"  DELTA E: {energy_log[-1] - energy_log[0]}; E_before: {energy_log[0]}; E_after: {energy_log[-1]}"
        )
    return relaxed_coords   


def get_dimensional_tension(Ksg_, Kb, coords):
    curvature_scale = np.max(ClosedPlaneCurveGeometry.edge_curvature(coords))
    return 4 * Kb * curvature_scale**2 * Ksg_

def run(file, target_edge_length, _Ksg_):
    Ksg_coords_force = []
    coords, original_coords = preprocess_mesh(file, ifResample=True, target_edge_length=target_edge_length)
    relaxed_coords = relax_bending(coords, Kb=0.1, dt=1e-7, n_iter=math.floor(0.5 / 1e-7))
    # relaxed_coords, _ = resample(relaxed_coords, target_edge_length=target_edge_length)
    for Ksg_ in _Ksg_:
        # Instantiate material properties
        Kb = 0.1
        Ksg = get_dimensional_tension(Ksg_, Kb, coords)
        parameters = {
            "Kb": Kb / 4,  # Bending modulus (pN um; original 1e-19 J)
            "Ksg": Ksg,  # Global stretching modulus (pN um/um^2; original 0.05 mN/m)
            "Ksl": 0,
        }
        mem = ClosedPlaneCurveMaterial(**parameters)
        # print("dimensional tension: ", mem.Ksg)

        # Compute force density
        relaxed_forces = np.array([force/ClosedPlaneCurveGeometry.vertex_dual_length(
            relaxed_coords
        ) for force in mem.force(relaxed_coords)])         
        Ksg_coords_force.append(np.concatenate(([relaxed_coords], relaxed_forces), axis=0))
    Ksg_coords_force = np.asarray(Ksg_coords_force)
    np.savez(f"data/{file.stem}", _Ksg_=_Ksg_, original_coords = original_coords, Ksg_coords_force=Ksg_coords_force)


if __name__ == "__main__":
    ## BATCH RENDER
    from actuator_constants import files
    f_run = partial(run, target_edge_length=0.1, _Ksg_ = np.linspace(0,1,1+2**3))
    r = process_map(f_run, files, max_workers=6)
        