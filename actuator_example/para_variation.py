# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math

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
from automembrane.integrator import fwd_euler_integrator
import automembrane.plot_helper as ph

from actuator_constants import files, images

jax.config.update("jax_enable_x64", True)
ph.matplotlibStyle(small=10, medium=12, large=14)


# Plotting settings
padding = 2
cm = mpl.cm.viridis_r

# Instantiate material properties
parameters = {
    "Kb": 1 / 4,  # Bending modulus (pN um; original 1e-19 J)
    "Ksg": 1,  # Global stretching modulus (pN um/um^2; original 0.05 mN/m)
    "Ksl": 10,
}
mem = ClosedPlaneCurveMaterial(**parameters)

# Discretization settings
target_edge_length = 0.05  # target edge length in um for resampling
# total_time = 0.001
dt_ = 3e-5 # Timestep
n_iter = 20000  # Number of relaxation steps
# n_iter = math.floor(total_time / dt)  # Number of relaxation steps

data = defaultdict(dict)

def resample(original_coords, target_edge_length):
    total_length = np.sum(
        np.linalg.norm(
            np.roll(original_coords[:-1], -1, axis=0) - original_coords[:-1], axis=1
        )
    )
    n_vertices = math.floor(total_length / target_edge_length)
    print(f"  Resampling to {n_vertices} vertices")
    # Periodic cubic B-spline interpolation with no smoothing (s=0)
    tck, _ = splprep([original_coords[:, 0], original_coords[:, 1]], s=0, per=True)

    xi, yi = splev(np.linspace(0, 1, n_vertices), tck)
    coords = np.hstack((xi.reshape(-1, 1), yi.reshape(-1, 1)))
    return coords, tck


def run(material, file, ifResample=False):
    k = file.stem

    print("Processing:", k)
    original_coords = np.loadtxt(file)
    original_coords = np.vstack(
        (original_coords, original_coords[0])
    )  # Energy expects last point to equal first

    if ifResample:
        coords, tck = resample(original_coords, target_edge_length)
        data[k]["spline"] = tck
    else:
        coords = original_coords


    # length_scale = np.sum(material.edge_length(coords)) / 2 / np.pi
    curvature_scale = np.max(material.edge_curvature(coords))
    original_coords = original_coords * curvature_scale
    coords = coords * curvature_scale

    # Perform energy relaxation
    relaxed_coords = coords
    dt = dt_ / parameters["Kb"] / curvature_scale**2
    if n_iter > 0:
        relaxed_coords, energy_log = fwd_euler_integrator(
            relaxed_coords, material, n_steps=n_iter, dt=dt
        )
        print(
            f"  DELTA E: {energy_log[-1] - energy_log[0]}; E_before: {energy_log[0]}; E_after: {energy_log[-1]}"
        )

    # Compute force density
    relaxed_forces = sum(material.force(relaxed_coords)) / material.vertex_dual_length(
        relaxed_coords
    )
    
    
    np.save("foo.npy", relaxed_coords)
    data[k]["original_coords"] = original_coords
    data[k]["relaxed_coords"] = relaxed_coords
    data[k]["relaxed_forces"] = relaxed_forces

    return data, curvature_scale, material


def make_movie(
    file_stem,
    original_coords,
    relaxed_coords,
    relaxed_forces,
    fps: int = 30,
    dpi: int = 100,
    skip: int = 100,
    interactive: bool = False,
):
    x_lim = np.array([np.min(original_coords[:, 0]), np.max(original_coords[:, 0])]) + [
        -padding,
        padding
    ]
    y_lim = np.array([np.min(original_coords[:, 1]), np.max(original_coords[:, 1])]) + [
        -padding,
        padding,
    ]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)

    # flip y-axis
    ax.set_ylim(ax.get_ylim()[::-1])

    # (original_line,) = ax.plot(original_coords[:, 0], original_coords[:, 1], color="k")

    (line,) = ax.plot(relaxed_coords[:, 0], relaxed_coords[:, 1], linewidth=0.2, color="r")
    f_mag = np.linalg.norm(relaxed_forces, axis=1)

    q = ax.quiver(
        relaxed_coords[:, 0],
        relaxed_coords[:, 1],
        relaxed_forces[:, 0],
        relaxed_forces[:, 1],
        f_mag,
        cmap=mpl.cm.viridis_r,
        angles="xy",
        units="xy",
        label="force",
        scale=4e1,
        scale_units="xy",
        width=0.1,
        zorder=10,
    )
    ax.set_ylabel(r"X (μm)")
    ax.set_xlabel(r"Y (μm)")

    time_template = "Iteration = %d"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    with Image.open(f"raw_images/{file_stem}.TIF") as im:
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

    # Shrink current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    cbar = fig.colorbar(
        q,
        ax=ax,
    )
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel("Force Density ($\mathregular{pN/\mu m^2}$)", rotation=270)
    plt.show()


if __name__ == "__main__":
    ## BATCH RENDER
    # f_movie = partial(make_movie, fps=30, dpi=200, skip=100)
    # r = process_map(f_movie, files, max_workers=6)
    from pathlib import Path
    files = list(
        map(
            Path,
            [
                f"coordinates/{i}"
                for i in [
                    "cell1/34D-grid2-s3-acta1_001_16.txt",
                ]
            ],
        )
    )
    for file in files:
        data, curvature_scale, material = run(mem, file, ifResample=True)
        print(curvature_scale)
        make_movie(
            file.stem,
            data[file.stem]["original_coords"] / curvature_scale,
            data[file.stem]["relaxed_coords"] / curvature_scale,
            data[file.stem]["relaxed_forces"] * 0.1 * curvature_scale**3,
            fps=30,
            dpi=200,
            skip=100,
        )
