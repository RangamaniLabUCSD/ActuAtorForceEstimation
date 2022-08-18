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

def resample(original_coords, n_vertices):
    # total_length = np.sum(
    #     np.linalg.norm(
    #         np.roll(original_coords[:-1], -1, axis=0) - original_coords[:-1], axis=1
    #     )
    # )
    # n_vertices = math.floor(total_length / target_edge_length)
    # print(f"  Resampling to {n_vertices} vertices")
    # Periodic cubic B-spline interpolation with no smoothing (s=0)
    tck, _ = splprep([original_coords[:, 0], original_coords[:, 1]], s=0, per=True)

    xi, yi = splev(np.linspace(0, 1, n_vertices), tck)
    coords = np.hstack((xi.reshape(-1, 1), yi.reshape(-1, 1)))
    return coords, tck

def preprocess_mesh(file, ifResample, n_vertices):
    k = file.stem

    # print("Processing:", k)
    original_coords = np.loadtxt(file)
    original_coords = np.vstack(
        (original_coords, original_coords[0])
    )  # Energy expects last point to equal first

    if ifResample:
        coords, tck = resample(original_coords, n_vertices)
    else:
        coords = original_coords
    
    return coords, original_coords

def relax_bending(coords, Kb, dt_, n_iter):
    # Instantiate material properties
    parameters = {
        "Kb": Kb / 4, 
        "Ksg": 1 * Kb,
        "Ksl": 0.1 * Kb,
    }
    mem = ClosedPlaneCurveMaterial(**parameters)
    # Perform energy relaxation
    relaxed_coords = coords
    if n_iter > 0:
        relaxed_coords, energy_log = fwd_euler_integrator(
            relaxed_coords, mem, n_steps=n_iter, dt=get_dimensional_time_step(dt_, mem.Kb, relaxed_coords)
        )
        print(
            f"  DELTA E: {energy_log[-1] - energy_log[0]}; E_before: {energy_log[0]}; E_after: {energy_log[-1]}"
        )
    return relaxed_coords   

def get_dimensional_time_step(dt_, Kb, coords):
    # curvature_scale = np.max(ClosedPlaneCurveGeometry.edge_curvature(coords))
    # return dt_/(4 * Kb * curvature_scale**2) 
    return dt_

def run(file, n_vertices):
    coords, original_coords = preprocess_mesh(file, ifResample=True, n_vertices=n_vertices)
    relaxed_coords = coords
    if file.stem == "34D-grid2-s2_002_16":
        dt_ = 5e-7
        n_iter = int(1e6)
    if file.stem == "34D-grid2-s3_028_16":
        dt_ = 1e-10
        n_iter = int(1e5)
    if file.stem == "34D-grid2-s5_005_16":
        dt_ = 1e-9
        n_iter = int(1e7)
    else: 
        dt_ = 1e-5
        n_iter = int(1e5)
    relaxed_coords = relax_bending(coords, Kb=1, dt_=dt_, n_iter=n_iter)
    # relaxed_coords, _ = resample(relaxed_coords, n_vertices=n_vertices)
    if file.stem ==  "34D-grid3-ActA1_007_16":
        relaxed_coords = np.flip(relaxed_coords, axis=0)
        original_coords = np.flip(original_coords, axis=0)
    np.savez(f"relaxed_coords/{file.stem}", relaxed_coords = relaxed_coords, original_coords = original_coords)


if __name__ == "__main__":
    ## BATCH RENDER
    from actuator_constants import files
    f_run = partial(run, n_vertices=1000)
    r = process_map(f_run, files, max_workers=12)
        