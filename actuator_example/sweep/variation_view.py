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
from automembrane.geometry import ClosedPlaneCurveGeometry
from automembrane.integrator import fwd_euler_integrator
import automembrane.plot_helper as ph

from actuator_constants import files, images

jax.config.update("jax_enable_x64", True)
ph.matplotlibStyle(small=10, medium=12, large=14)

# Plotting settings
padding = 2
cm = mpl.cm.viridis_r

def plot_force(
    file_stem,
    relaxed_coords,
    relaxed_forces,
    fps: int = 30,
    dpi: int = 100,
    skip: int = 100,
    interactive: bool = False,
):
    x_lim = np.array([np.min(relaxed_coords[:, 0]), np.max(relaxed_coords[:, 0])]) + [
        -padding,
        padding,
    ]
    y_lim = np.array([np.min(relaxed_coords[:, 1]), np.max(relaxed_coords[:, 1])]) + [
        -padding,
        padding,
    ]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)

    # flip y-axis
    ax.set_ylim(ax.get_ylim()[::-1])

    # (original_line,) = ax.plot(relaxed_coords[:, 0], relaxed_coords[:, 1], color="k")

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
        scale=2e-1,
        scale_units="xy",
        width=0.1,
        zorder=10,
    )
    (line,) = ax.plot(
        relaxed_coords[:, 0], relaxed_coords[:, 1], linewidth=1.5, color="r"
    )
    ax.set_ylabel(r"X (μm)")
    ax.set_xlabel(r"Y (μm)")

    time_template = "Iteration = %d"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

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

    # Shrink current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    cbar = fig.colorbar(
        q,
        ax=ax,
    )
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel("Force Density", rotation=270)
    # cbar.ax.set_ylabel("Force Density ($\mathregular{pN/\mu m^2}$)", rotation=270)

def run(file):
    data = np.load(f"data/{file.stem}.npz")
    _Ksg_ = data["_Ksg_"]
    Ksg_coords_force = data["Ksg_coords_force"]
    for Ksg_count, Ksg_ in tqdm(enumerate(_Ksg_), desc="Rendering plots"):
        normalized_force = Ksg_coords_force[Ksg_count][1] / np.max(
            np.linalg.norm(Ksg_coords_force[Ksg_count][1], axis=1)
        )
        plot_force(
            file.stem,
            Ksg_coords_force[Ksg_count][0],
            normalized_force,
            fps=30,
            dpi=200,
            skip=100,
        )
        from variation_run import get_dimensional_tension
        plt.title("$\\bar{\sigma}=$" + f"{Ksg_}" + 
                  "$, \sigma=$" + f"{math.floor(get_dimensional_tension(Ksg_=Ksg_, Kb=1, coords = Ksg_coords_force[Ksg_count][0]))}$\kappa$")
        plt.savefig("figures/" + file.stem + f"_Ksg{Ksg_}" + ".png")
        plt.close()

    
if __name__ == "__main__":
    ## BATCH RENDER
    from actuator_constants import files
    r = process_map(run, files, max_workers=6)
        