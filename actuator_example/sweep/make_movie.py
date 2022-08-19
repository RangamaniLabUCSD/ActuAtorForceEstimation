# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math
from functools import partial
from pathlib import Path

import automembrane.plot_helper as ph
import automembrane.util as u
import jax
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.integrator import fwd_euler_integrator
from PIL import Image
from scipy.interpolate import splev, splprep
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from actuator_constants import files, images

jax.config.update("jax_enable_x64", True)
ph.matplotlibStyle(small=10, medium=12, large=14)

# Plotting settings
padding = 2
cm = mpl.cm.viridis_r


def make_movie(
    c,
    e,
    f,
    original_coords: npt.NDArray[np.float64],
    file: Path,
    fps: int = 30,
    dpi: int = 100,
    skip: int = 100,
    interactive: bool = False,
):

    n_iter = c.shape[0]

    x_lim = np.array([np.min(original_coords[:, 0]), np.max(original_coords[:, 0])]) + [
        -padding,
        padding,
    ]
    y_lim = np.array([np.min(original_coords[:, 1]), np.max(original_coords[:, 1])]) + [
        -padding,
        padding,
    ]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(autoscale_on=False, xlim=x_lim, ylim=y_lim)

    # flip y-axis
    ax.set_ylim(ax.get_ylim()[::-1])

    (original_line,) = ax.plot(original_coords[:, 0], original_coords[:, 1], color="k")

    (line,) = ax.plot(c[0][:, 0], c[0][:, 1], color="r")
    forces = np.sum(f[0], axis=0)
    f_mag = np.linalg.norm(forces, axis=1)

    q = ax.quiver(
        c[0][:, 0],
        c[0][:, 1],
        forces[:, 0],
        forces[:, 1],
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
    ax.set_ylabel(r"Y (μm)")
    ax.set_xlabel(r"X (μm)")

    time_template = "Iteration = %d"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    with Image.open(f"../raw_images/{file.stem}.TIF") as im:
        pixel_scale = images[file.stem]
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

    cbar = fig.colorbar(
        q,
        ax=ax,
    )
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel("Force Density ($\mathregular{pN/\mu m^2}$)", rotation=270)

    def animate(i):
        line.set_data(c[i][:, 0], c[i][:, 1])
        time_text.set_text(time_template % (i))

        forces = np.sum(f[i], axis=0)
        f_mag = np.linalg.norm(forces, axis=1)
        cmap = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=f_mag.min(), vmax=f_mag.max()), cmap=cm
        )
        cbar.update_normal(cmap)
        q.set_UVC(forces[:, 0], forces[:, 1], f_mag)
        q.set_offsets(c[i])
        q.set_clim(f_mag.min(), f_mag.max())
        return line, time_text, q, cbar

    if interactive:
        ## INTERACTIVE ANIMATION
        ani = animation.FuncAnimation(
            fig, animate, np.arange(0, n_iter, 100), interval=200, blit=True
        )
        plt.show()
    else:
        ## MAKE MOVIE
        moviewriter = animation.FFMpegWriter(fps=fps)
        with moviewriter.saving(fig, f"videos/{file.stem}_relaxation.mp4", dpi=dpi):
            for i in tqdm(np.arange(0, n_iter + 1, skip)):
                animate(i)
                moviewriter.grab_frame()
    ax.clear()
    plt.close(fig)


# if __name__ == "__main__":
#     ## BATCH RENDER
#     f_movie = partial(make_movie, fps=30, dpi=200, skip=100)
#     r = process_map(f_movie, files, max_workers=6)
