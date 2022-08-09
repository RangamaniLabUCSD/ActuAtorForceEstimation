# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import math

from functools import partial

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

from actuator_constants import files, images

jax.config.update("jax_enable_x64", True)
u.matplotlibStyle(small=10, medium=12, large=14)


# Instantiate material properties
parameters = {
    "Kb": 0.1 / 4,  # Bending modulus (pN um; original 1e-19 J)
    "Ksg": 50,  # Global stretching modulus (pN um/um^2; original 0.05 mN/m)
    "Ksl": 1,
}
mem = ClosedPlaneCurveMaterial(**parameters)

# Discretization settings
target_edge_length = 0.05  # target edge length in um for resampling
total_time = 0.1
dt = 5e-6  # Timestep
n_iter = math.floor(total_time / dt)  # Number of relaxation steps

# Plotting settings
padding = 2
cm = mpl.cm.viridis_r


def make_movie(file, fps: int = 30, dpi: int = 100, skip: int = 100):
    original_coords = np.loadtxt(file)
    original_coords = np.vstack(
        (original_coords, original_coords[0])
    )  # Energy expects last point to equal first

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

    x_lim = np.array([np.min(coords[:, 0]), np.max(coords[:, 0])]) + [-padding, padding]
    y_lim = np.array([np.min(coords[:, 1]), np.max(coords[:, 1])]) + [-padding, padding]

    c, e, f = fwd_euler_integrator(
        coords, mem, n_steps=n_iter, dt=dt, save_trajectory=True
    )

    # print([i.shape for i in [c, e, f]])

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

    time_template = "Iteration = %d"
    time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes)

    with Image.open(f"raw_images/{file.stem}.TIF") as im:
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

    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=f_mag.min(), vmax=f_mag.max()), cmap=cm
        ),
        ax=ax,
    )
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel("Force Density ($\mathregular{pN/\mu m^2}$)", rotation=270)

    def animate(i):
        line.set_data(c[i][:, 0], c[i][:, 1])
        time_text.set_text(time_template % (i))

        forces = np.sum(f[i], axis=0)
        f_mag = np.linalg.norm(forces, axis=1)
        q.set_UVC(forces[:, 0], forces[:, 1], f_mag)
        q.set_offsets(c[i])

        cbar.update_normal(
            mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=f_mag.min(), vmax=f_mag.max()), cmap=cm
            )
        )

        return line, time_text, q, cbar

    ## MAKE MOVIE
    moviewriter = animation.FFMpegWriter(fps=fps)
    with moviewriter.saving(fig, f"movies/{file.stem}.mp4", dpi=dpi):
        for i in tqdm(np.arange(0, n_iter + 1, skip)):
            animate(i)
            moviewriter.grab_frame()

    ## INTERACTIVE ANIMATION
    # ani = animation.FuncAnimation(
    #     fig, animate, np.arange(0, n_iter, 100), interval=200, blit=True
    # )
    # plt.show()
    ax.clear()
    plt.close(fig)


if __name__ == "__main__":
    ## BATCH RENDER
    f_movie = partial(make_movie, fps=30, dpi=200, skip=100)
    r = process_map(f_movie, files, max_workers=6)
