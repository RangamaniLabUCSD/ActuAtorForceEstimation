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
    original_coords,
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
    
    # color-coded force
    vertex_normal = ClosedPlaneCurveGeometry.vertex_normal(relaxed_coords)
    signed_f_mag = np.sum(relaxed_forces * vertex_normal, axis = 1)
    points = relaxed_coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
    norm = plt.Normalize(-signed_f_mag.max(), signed_f_mag.max())
    lc = LineCollection(segments, cmap='seismic', norm=norm)
    lc.set_array(signed_f_mag)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)

    # # quiver plot
    # f_mag = np.linalg.norm(relaxed_forces, axis=1)
    # q = ax.quiver(
    #     relaxed_coords[:, 0],
    #     relaxed_coords[:, 1],
    #     relaxed_forces[:, 0],
    #     relaxed_forces[:, 1],
    #     f_mag,
    #     cmap=mpl.cm.viridis_r,
    #     angles="xy",
    #     units="xy",
    #     label="force",
    #     scale=8e-1,
    #     scale_units="xy",
    #     width=0.1,
    #     zorder=10,
    # )
    # cbar = fig.colorbar(
    #     q,
    #     ax=ax,
    # )
    # cbar.ax.get_yaxis().labelpad = 20
    # cbar.ax.set_ylabel("Force Density", rotation=270)
    # # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # # cbar.ax.set_ylabel("Force Density ($\mathregular{pN/\mu m^2}$)", rotation=270)
    
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
    # (original_line,) = ax.plot(original_coords[:, 0], original_coords[:, 1], color="k")
    # (line,) = ax.plot(
    #     relaxed_coords[:, 0], relaxed_coords[:, 1], linewidth=1.5, color="r"
    # )
    # (line,) = ax.plot(
    #     relaxed_coords[:, 0], relaxed_coords[:, 1], linewidth=1.5, color="r"
    # )
    
    ax.set_ylabel(r"X (μm)")
    ax.set_xlabel(r"Y (μm)")

    # Shrink current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

def run(file):
    data = np.load(f"data/{file.stem}.npz")
    _Ksg_ = data["_Ksg_"]
    original_coords = data["original_coords"]
    Ksg_coords_force = data["Ksg_coords_force"]
    for Ksg_count, Ksg_ in tqdm(enumerate(_Ksg_), desc="Rendering plots"):
        coord = Ksg_coords_force[Ksg_count][0]
        forces = Ksg_coords_force[Ksg_count][1:]
        total_force = np.sum(forces, axis=0)
        bending_force = forces[0]
        surface_force = forces[1]
        normalized_total_force = total_force / np.max(
            np.linalg.norm(total_force, axis=1)
        )
        normalized_bending_force = bending_force / np.max(
            np.linalg.norm(bending_force, axis=1)
        )
        normalized_surface_force = surface_force / np.max(
            np.linalg.norm(surface_force, axis=1)
        )
        # plot_force(
        #     file.stem,
        #     original_coords,
        #     coord,
        #     ClosedPlaneCurveGeometry.vertex_normal(coord),
        #     fps=30,
        #     dpi=200,
        #     skip=100,
        # )
        plot_force(
            file.stem,
            original_coords,
            coord,
            normalized_total_force,
            fps=30,
            dpi=200,
            skip=100,
        )
        # plot_force(
        #     file.stem,
        #     original_coords,
        #     coord,
        #     normalized_bending_force,
        #     fps=30,
        #     dpi=200,
        #     skip=100,
        # )
        # plot_force(
        #     file.stem,
        #     original_coords,
        #     coord,
        #     normalized_surface_force,
        #     fps=30,
        #     dpi=200,
        #     skip=100,
        # )
        from variation_run import get_dimensional_tension
        plt.title("$\\bar{\sigma}=$" + f"{Ksg_}" + 
                  "$, \sigma=$" + f"{math.floor(get_dimensional_tension(Ksg_=Ksg_, Kb=1, coords = coord))}$\kappa$")
        plt.savefig("figures/" + file.stem + f"_Ksg{Ksg_}" + ".png")
        plt.close()

    
if __name__ == "__main__":
    ## BATCH RENDER
    from actuator_constants import files
    r = process_map(run, files, max_workers=6)