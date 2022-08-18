# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from functools import partial
from pathlib import Path
from re import A
from typing import Union

import jax
import numpy as np
import numpy.typing as npt
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.integrator import fwd_euler_integrator
from scipy.interpolate import splev, splprep
from tqdm.contrib.concurrent import process_map

from actuator_constants import files

jax.config.update("jax_enable_x64", True)


def resample(
    original_coords: npt.NDArray[np.float64], n_vertices: int = 1000
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Resample discrete plane curve into n_vertices

    Args:
        original_coords (npt.NDArray[np.float64]): Original coordinates
        n_vertices (int, optional): Number of vertices to resample to. Defaults to 1000.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: New coordinates and B-spline parameters
    """
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


def preprocess_mesh(
    file: Union[str, Path], resample_geometry: bool = True, n_vertices: int = 1000
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Preprocess the plane curve geometry

    Args:
        file (Union[str, Path]): Filename to process
        resample_geometry (bool, optional): Flag to resample geometry. Defaults to True.
        n_vertices (int, optional): number of points to resample to. Defaults to 1000.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: reprocessed and original coordinates
    """

    original_coords = np.loadtxt(file)
    original_coords = np.vstack(
        (original_coords, original_coords[0])
    )  # Energy expects last point to equal first

    if resample_geometry:
        coords, _ = resample(original_coords, n_vertices)
    else:
        coords = original_coords
    return coords, original_coords


def relax_bending(coords, Kb, Ksg, Ksl, dt, n_iter):
    # Instantiate material properties
    parameters = {
        "Kb": Kb / 4,
        "Ksg": Ksg,
        "Ksl": Ksl,
    }
    # Perform energy relaxation
    if n_iter > 0:
        coords, _ = fwd_euler_integrator(
            coords,
            ClosedPlaneCurveMaterial(**parameters),
            n_steps=n_iter,
            dt=dt,
        )
        # print(
        #     f"  DELTA E: {energy_log[-1] - energy_log[0]}; E_before: {energy_log[0]}; E_after: {energy_log[-1]}"
        # )
    return coords


def run_relaxation(file: Path, n_vertices: int = 1000):
    coords, original_coords = preprocess_mesh(
        file, resample_geometry=True, n_vertices=n_vertices
    )
    if file.stem == "34D-grid2-s2_002_16":
        dt = 5e-6
        n_iter = int(1e5)
        relaxed_coords = relax_bending(
            coords, Kb=1, Ksg=1, Ksl=0.1, dt=dt, n_iter=n_iter
        )
    elif file.stem == "34D-grid2-s3_028_16":
        dt = 1e-6
        n_iter = int(1e5)
        relaxed_coords = relax_bending(coords, Kb=1, Ksg=1, Ksl=1, dt=dt, n_iter=n_iter)
    elif file.stem == "34D-grid2-s5_005_16":
        dt = 3e-7
        n_iter = int(1e5)
        relaxed_coords = relax_bending(
            coords, Kb=1, Ksg=20, Ksl=1, dt=dt, n_iter=n_iter
        )
        dt = 7e-8
        n_iter = int(5e4)
        relaxed_coords = relax_bending(
            relaxed_coords, Kb=1, Ksg=1, Ksl=0.1, dt=dt, n_iter=n_iter
        )
    else:
        dt = 1e-5
        n_iter = int(1e5)
        relaxed_coords = relax_bending(
            coords, Kb=1, Ksg=1, Ksl=0.1, dt=dt, n_iter=n_iter
        )
    if file.stem == "34D-grid3-ActA1_007_16":
        relaxed_coords = np.flip(relaxed_coords, axis=0)
        original_coords = np.flip(original_coords, axis=0)
    np.savez(
        f"relaxed_coords/{file.stem}",
        relaxed_coords=relaxed_coords,
        original_coords=original_coords,
    )


if __name__ == "__main__":
    f_run = partial(run_relaxation, n_vertices=1000)
    r = process_map(f_run, files, max_workers=12)
