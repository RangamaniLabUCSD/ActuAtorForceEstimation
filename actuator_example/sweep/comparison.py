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
from automembrane.geometry import ClosedPlaneCurveGeometry
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.integrator import fwd_euler_integrator
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
):
    spec = fig.add_gridspec(ncols=1, nrows=2,
                          height_ratios=[1,40])
    ax = fig.add_subplot(spec[0], autoscale_on=False, xlim=(np.min(_Ksg_), np.max(_Ksg_)), ylim=(0,1))
    # ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)
    ax.vlines(Ksg_, 0, 1, linestyles="solid", colors ="r", linewidth=6)
    ax.set_xticks([np.min(_Ksg_), 0.25, np.max(_Ksg_)])
    ax.set_xticklabels(["Bending", "", "Tension"])
    ax.get_yaxis().set_visible(False)
    ax.xaxis.tick_top()

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
    
    

    # color-coded force
    # max_norm = np.max(np.linalg.norm(relaxed_force, axis=1))
    # normalized_relaxed_force = relaxed_force / max_norm
    vertex_normal = ClosedPlaneCurveGeometry.vertex_normal(relaxed_coords)
    signed_f_mag = np.sum(relaxed_force * vertex_normal, axis = 1)
    signed_f_mag = signed_f_mag / np.max(abs(signed_f_mag))
    points = relaxed_coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    from matplotlib.collections import LineCollection
    norm = plt.Normalize(-abs(signed_f_mag).max(), abs(signed_f_mag).max())
    lc = LineCollection(segments, cmap='PRGn', norm=norm)
    lc.set_array(signed_f_mag)
    lc.set_linewidth(4)
    line = ax.add_collection(lc)
    cbar = fig.colorbar(line, ax=ax, ticks=[-1, 0, 1], pad=0.01)
    cbar.ax.set_yticklabels(['Pulling', '0', 'Pushing'])
    # curvature_scale = np.max(ClosedPlaneCurveGeometry.edge_curvature(relaxed_coords))
    cbar.ax.get_yaxis().labelpad = 20
    # cbar.ax.set_ylabel("Force Density"+f"({round_sig(max_norm * curvature_scale**(-3), 3)}$\kappa$",  rotation=270)
    
    ax.set_ylabel(r"X (μm)")
    ax.set_xlabel(r"Y (μm)")

    # Shrink current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    


data = np.load("forces/34D-grid2-s2_002_16.npz")
_Ksg_ = data["_Ksg_"]
Ksg_force = data["Ksg_force"]

data = np.load("relaxed_coords/34D-grid2-s2_002_16.npz")
relaxed_coords = data["relaxed_coords"]
original_coords = data["original_coords"]

Ksg_ = 0.2
forces = Ksg_force[3]
total_force = np.sum(forces, axis=0)


fig = plt.figure(figsize=(5, 5))

plot_force(
        fig,
        "34D-grid2-s2_002_16",
        original_coords,
        relaxed_coords,
        total_force,
        _Ksg_,
        Ksg_,
    )

# ax = fig.add_subplot(autoscale_on=False, xlim=(np.min(_Ksg_), np.max(_Ksg_)), ylim=(0,1))
# ax.vlines(Ksg_, 0, 1, linestyles="solid", colors ="k")

plt.savefig("./test.png")