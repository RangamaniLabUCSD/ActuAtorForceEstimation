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
import matplotlib.pyplot as plt
from PIL import Image
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.integrator import fwd_euler_integrator
from scipy.interpolate import splev, splprep
from tqdm.contrib.concurrent import process_map

from actuator_constants import files, images

import automembrane.plot_helper as ph

jax.config.update("jax_enable_x64", True)
ph.matplotlibStyle(small=10, medium=12, large=14)

from make_movie import make_movie


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


def plot_contour(
    fig,
    file_stem,
    original_coords,
    relaxed_coords,
):
    padding = 3
    x_lim = np.array([np.min(relaxed_coords[:, 0]), np.max(relaxed_coords[:, 0])]) + [
        -padding,
        padding,
    ]
    y_lim = np.array([np.min(relaxed_coords[:, 1]), np.max(relaxed_coords[:, 1])]) + [
        -padding,
        padding,
    ]
    ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)

    # flip y-axis
    ax.set_ylim(ax.get_ylim()[::-1])

    # nucleus cell trace
    with Image.open(f"../raw_images/{file_stem}.TIF") as im:
        pixel_scale = images[file_stem]
        x_lim_pix = (x_lim / pixel_scale).round()
        y_lim_pix = (y_lim / pixel_scale).round()

        im = im.crop((x_lim_pix[0], y_lim_pix[0], x_lim_pix[1], y_lim_pix[1]))

        plt.imshow(
            im,
            alpha=0.6,
            extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
            zorder=0,
            cmap=plt.cm.Greys_r,
        )

    (original_line,) = ax.plot(
        original_coords[:, 0],
        original_coords[:, 1],
        "o",
        markersize=0.5,
        color="k",
        label="Original",
    )
    (line,) = ax.plot(
        relaxed_coords[:, 0],
        relaxed_coords[:, 1],
        "--",
        linewidth=0.5,
        color="r",
        label="Relaxed",
    )
    # (line,) = ax.plot(
    #     relaxed_coords[:, 0], relaxed_coords[:, 1], linewidth=1.5, color="r"
    # )

    ax.set_ylabel(r"Y (μm)")
    ax.set_xlabel(r"X (μm)")

    ax.legend(loc="center left")  # , bbox_to_anchor=(1, 0.5))

    # # Shrink current axis
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


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
    return coords


params = {
    "34D-grid2-s2_002_16": {
        "dt": 5e-6,
        "n_iter": int(1e5),
        "Kb": 1,
        "Ksg": 1,
        "Ksl": 0.1,
    },
    "34D-grid2-s3_028_16": {
        "dt": 1e-6,
        "n_iter": int(1e5),
        "Kb": 1,
        "Ksg": 1,
        "Ksl": 1,
    },
    "34D-grid2-s5_005_16": {
        "dt": 7e-8,
        "n_iter": int(5e4),
        "Kb": 1,
        "Ksg": 1,
        "Ksl": 0.1,
    },
    "other": {
        "dt": 1e-5,
        "n_iter": int(1e5),
        "Kb": 1,
        "Ksg": 1,
        "Ksl": 0.1,
    },
}


def run_relaxation(file: Path, n_vertices: int = 1000):
    coords, original_coords = preprocess_mesh(
        file, resample_geometry=True, n_vertices=n_vertices
    )

    if file.stem in params:
        relaxed_coords = relax_bending(coords, **params[file.stem])
    else:
        relaxed_coords = relax_bending(coords, **params["other"])

    # if file.stem == "34D-grid3-ActA1_007_16":
    #     relaxed_coords = np.flip(relaxed_coords, axis=0)
    #     original_coords = np.flip(original_coords, axis=0)

    np.savez(
        f"relaxed_coords/{file.stem}",
        relaxed_coords=relaxed_coords,
        original_coords=original_coords,
    )
    # data = np.load(f"relaxed_coords/{file.stem}.npz")
    # relaxed_coords = data["relaxed_coords"]
    # original_coords = data["original_coords"]

    fig = plt.figure(figsize=(5, 5))
    plot_contour(
        fig,
        file.stem,
        original_coords,
        relaxed_coords,
    )
    fig.set_tight_layout(True)
    plt.savefig("relaxed_coords/" + file.stem + ".png")
    fig.clear()
    plt.close(fig)


def generate_relaxation_movie(file: Path, n_vertices: int = 1000):
    coords, original_coords = preprocess_mesh(
        file, resample_geometry=True, n_vertices=n_vertices
    )

    def get_trajectory(coords, Kb, Ksg, Ksl, dt, n_iter):
        # Instantiate material properties
        parameters = {
            "Kb": Kb / 4,
            "Ksg": Ksg,
            "Ksl": Ksl,
        }
        # Perform energy relaxation
        c, e, f = fwd_euler_integrator(
            coords,
            ClosedPlaneCurveMaterial(**parameters),
            n_steps=n_iter,
            dt=dt,
            save_trajectory=True,
        )
        return c, e, f

    if file.stem in params:
        c, e, f = get_trajectory(coords, **params[file.stem])
    else:
        c, e, f = get_trajectory(coords, **params["other"])

    # if file.stem == "34D-grid3-ActA1_007_16":
    #     relaxed_coords = np.flip(relaxed_coords, axis=0)
    #     original_coords = np.flip(original_coords, axis=0)

    make_movie(c, e, f, original_coords, file, skip=1000)
    del c, e, f


if __name__ == "__main__":
    f_run = partial(run_relaxation, n_vertices=1000)
    r = process_map(f_run, files, max_workers=12)

    # f_run = partial(generate_relaxation_movie, n_vertices=1000)
    # r = process_map(f_run, files, max_workers=1)
