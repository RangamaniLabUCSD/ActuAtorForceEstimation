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
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from automembrane.geometry import ClosedPlaneCurveGeometry
import automembrane.plot_helper as ph

from actuator_constants import files, images

jax.config.update("jax_enable_x64", True)
ph.matplotlibStyle(small=10, medium=12, large=14)

# Plotting settings
padding = 2
cm = mpl.cm.viridis_r

def plot_force(
    fig,
    file_stem,
    original_coords,
    relaxed_coords,
    relaxed_force,
    _Ksg_,
    Ksg_,
    style="color",
):
    spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 40])

    # bending-tension parameter bar
    ax = fig.add_subplot(
        spec[0], autoscale_on=False, xlim=(np.min(_Ksg_), np.max(_Ksg_)), ylim=(0, 1)
    )
    # ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)
    ax.vlines(Ksg_, 0, 1, linestyles="solid", colors="r", linewidth=6)
    ax.set_xticks([np.min(_Ksg_), 1, np.max(_Ksg_)])
    ax.set_xticklabels(["Bending", "Transition", "Tension"])
    ax.get_yaxis().set_visible(False)
    ax.xaxis.tick_top()

    # nucleus cell trace
    x_lim = np.array([np.min(relaxed_coords[:, 0]), np.max(relaxed_coords[:, 0])]) + [
        -padding,
        padding,
    ]
    y_lim = np.array([np.min(relaxed_coords[:, 1]), np.max(relaxed_coords[:, 1])]) + [
        -padding,
        padding,
    ]
    ax = fig.add_subplot(spec[1], autoscale_on=False, xlim=x_lim, ylim=y_lim)
    # flip y-axis
    ax.set_ylim(ax.get_ylim()[::-1])
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

    # color-coded force
    if style == "color":
        vertex_normal = ClosedPlaneCurveGeometry.vertex_normal(relaxed_coords)
        signed_f_mag = np.sum(relaxed_force * vertex_normal, axis=1)
        signed_f_mag = signed_f_mag / np.max(abs(signed_f_mag))
        points = relaxed_coords.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection

        norm = plt.Normalize(-abs(signed_f_mag).max(), abs(signed_f_mag).max())
        lc = LineCollection(segments, cmap="PRGn", norm=norm)
        lc.set_array(signed_f_mag)
        lc.set_linewidth(4)
        line = ax.add_collection(lc)
        cbar = fig.colorbar(line, ax=ax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(["Pull", "0", "Push"])
        # curvature_scale = np.max(ClosedPlaneCurveGeometry.edge_curvature(relaxed_coords))
        cbar.ax.get_yaxis().labelpad = 20
        # cbar.ax.set_ylabel("Force Density"+f"({round_sig(max_norm * curvature_scale**(-3), 3)}$\kappa$",  rotation=270)
    elif style == "quiver":
        max_norm = np.max(np.linalg.norm(relaxed_force, axis=1))
        normalized_relaxed_force = relaxed_force / max_norm
        f_mag = np.linalg.norm(normalized_relaxed_force, axis=1)
        q = ax.quiver(
            relaxed_coords[:, 0],
            relaxed_coords[:, 1],
            normalized_relaxed_force[:, 0],
            normalized_relaxed_force[:, 1],
            f_mag,
            cmap=mpl.cm.viridis_r,
            angles="xy",
            units="xy",
            label="force",
            scale=8e-1,
            scale_units="xy",
            width=0.1,
            zorder=10,
        )
        cbar = fig.colorbar(
            q,
            ax=ax,
        )
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel("Force Density", rotation=270)
        # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # cbar.ax.set_ylabel("Force Density ($\mathregular{pN/\mu m^2}$)", rotation=270)

    # outline plots
    (original_line,) = ax.plot(
        original_coords[:, 0],
        original_coords[:, 1],
        "-o",
        markersize=0.2,
        linewidth=0.1,
        color="k",
    )
    # (line,) = ax.plot(
    #     relaxed_coords[:, 0], relaxed_coords[:, 1], 'o', linewidth=0.2, color="r"
    # )

    ax.set_ylabel(r"X (μm)")
    ax.set_xlabel(r"Y (μm)")

    # Shrink current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])


def run(file):
    data = np.load(f"forces/{file.stem}.npz")
    _Ksg_ = data["_Ksg_"]
    Ksg_force = data["Ksg_force"]

    data = np.load(f"relaxed_coords/{file.stem}.npz")
    relaxed_coords = data["relaxed_coords"]
    original_coords = data["original_coords"]

    fig = plt.figure(figsize=(5, 5))

    moviewriter = animation.FFMpegWriter(fps=10)
    with moviewriter.saving(fig, f"videos/{file.stem}.mp4", dpi=200):
        for Ksg_count, Ksg_ in tqdm(enumerate(_Ksg_), desc="Rendering plots"):
            forces = Ksg_force[Ksg_count]
            total_force = np.sum(forces, axis=0)
            plot_force(
                fig,
                file.stem,
                original_coords,
                relaxed_coords,
                total_force,
                _Ksg_,
                Ksg_,
                style="quiver",
            )
            # plt.title("$\\bar{\sigma}=$" + f"{Ksg_}" +
            #         "$, \sigma=$" + f"{math.floor(get_dimensional_tension(Ksg_=Ksg_, Kb=1, coords = relaxed_coords))}$\kappa$")
            plt.savefig(
                "figures/" + file.stem + f"_Ksg{math.floor(Ksg_*1000)}" + ".png"
            )
            moviewriter.grab_frame()
            fig.clear()


if __name__ == "__main__":
    ## BATCH RENDER
    from actuator_constants import files

    r = process_map(run, files, max_workers=12)
