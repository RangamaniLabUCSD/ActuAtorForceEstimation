import math
import tempfile
from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
# from jax import jit
# from PIL import Image
from scipy.interpolate import splev, splprep
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

import automembrane.util as u
from automembrane.energy import *

u.matplotlibStyle(small=10, medium=12, large=14)

# List of segmentations as Paths
files = list(
    map(
        Path,
        [
            f"coordinates/{i}"
            for i in [
                "cell1/34D-grid2-s3-acta1_001_16.txt",
                "cell2/34D-grid3-ActA1_007_16.txt",
                "cell2/34D-grid3-ActA1_013_16.txt",
                "cell3/34D-grid2-s2_002_16.txt",
                "cell3/34D-grid2-s5_005_16.txt",
                "cell3/34D-grid3-ActA1_020_16.txt",
                "cell3/34D-grid3-s6_005_16.txt",
                "cell4/34D-grid2-s3_028_16.txt",
                "cell5/34D-grid3-ActA1_001_16.txt",
                "cell5/34D-grid3-ActA1_002_16.txt",
                "cell5/34D-grid3-ActA1_003_16.txt",
                "cell5/34D-grid3-ActA1_004_16.txt",
            ]
        ],
    )
)

# Map of image key to microns per pixel
images = {
    "34D-grid2-s3-acta1_001_16": 0.012723,
    "34D-grid3-ActA1_007_16": 0.015904,
    "34D-grid3-ActA1_013_16": 0.015904,
    "34D-grid2-s2_002_16": 0.015904,
    "34D-grid2-s5_005_16": 0.015904,
    "34D-grid3-ActA1_020_16": 0.015904,
    "34D-grid3-s6_005_16": 0.015904,
    "34D-grid2-s3_028_16": 0.015904,
    "34D-grid3-ActA1_001_16": 0.015904,
    "34D-grid3-ActA1_002_16": 0.015904,
    "34D-grid3-ActA1_003_16": 0.015904,
    "34D-grid3-ActA1_004_16": 0.015904,
}


parameters = {
    "Kb": 0.1 / 4,  # Bending modulus (pN um; original 1e-19 J)
    "Ksg": 50,  # Global stretching modulus (pN um/um^2; original 0.05 mN/m)
}
f_energy = partial(get_energy_2d_closed, **parameters)
f_force = partial(get_force_2d_closed, **parameters)
cm = mpl.cm.viridis_r
target_edge_length = 0.05  # target edge length in um for resampling

total_time = 0.01
dt = 5e-6  # Timestep
n_iter = math.floor(total_time / dt)  # Number of relaxation steps

def make_movie(file):
    energy_log = np.zeros(n_iter)
    data = defaultdict(dict)
    k = file.stem

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

    coords = np.zeros((n_iter + 1, n_vertices, 2))
    forces = np.zeros((n_iter + 1, n_vertices, 2))

    # Periodic cubic B-spline interpolation with no smoothing (s=0)
    tck, _ = splprep([original_coords[:, 0], original_coords[:, 1]], s=0, per=True)
    data[k]["spline"] = tck

    xi, yi = splev(np.linspace(0, 1, n_vertices), tck)
    coords[0] = np.hstack((xi.reshape(-1, 1), yi.reshape(-1, 1)))

    # Perform energy relaxation
    if n_iter > 0:
        for i in tqdm(range(0, n_iter), desc="Energy relaxation"):
            # energy, force = f_value_and_grad(relaxed_coords)
            energy_log[i] = f_energy(coords[i])

            # Compute dual length
            dc = np.roll(coords[i][:-1], -1, axis=0) - coords[i][:-1]
            edgeLengths = np.linalg.norm(dc, axis=1)
            dualLengths = ((edgeLengths + np.roll(edgeLengths, 1)) / 2.0).reshape(-1, 1)
            dualLengths = np.vstack((dualLengths, dualLengths[0]))
            forces[i] = -f_force(coords[i])  # / dualLengths

            coords[i + 1] = np.array(coords[i] - f_force(coords[i]) * dt)
            coords[i + 1][-1] = coords[i + 1][0]
        # print(
        #     f"  DELTA E: {energy_log[-1] - energy_log[0]}; E_before: {energy_log[0]}; E_after: {energy_log[-1]}"
        # )

    rng = range(0, n_iter, 10)
    with tempfile.TemporaryDirectory() as tmp_dir:
        print(tmp_dir)

        for i in tqdm(rng):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            ax.plot(coords[0][:, 0], coords[0][:, 1], color="k")
            ax.plot(coords[i][:, 0], coords[i][:, 1], color="r")

            Q = ax.quiver(
                coords[i][:, 0],
                coords[i][:, 1],
                forces[i][:, 0],
                forces[i][:, 1],
                np.linalg.norm(forces[i], axis=1),
                cmap=cm,
                angles="xy",
                units="xy",
                label="force",
                scale=1,
                scale_units="xy",
                width=0.1,
                zorder=10,
                alpha=0.3,
            )

            ax.set_ylabel(r"X (μm)")
            ax.set_xlabel(r"Y (μm)")

            # pixel_scale = images[k]
            # x_lim = np.array(ax.get_xlim())
            # y_lim = np.array(ax.get_ylim())

            # x_lim_pix = (x_lim / pixel_scale).round()
            # y_lim_pix = (y_lim / pixel_scale).round()

            ax.set_ylim(ax.get_ylim()[::-1])

            fig.savefig(f"{tmp_dir}/{file.stem}_{i}.png")
            plt.close(fig)

        clip = mpy.ImageSequenceClip(
            [f"{tmp_dir}/{file.stem}_{i}.png" for i in rng],
            fps=30,
        )
        clip.write_videofile(f"movies/{file.stem}_test_forces.mp4", fps=30)
        clip.close()


if __name__ == "__main__":
    j = []
    for file in files:
        j.append((file))

    r = process_map(make_movie, j, max_workers=2, maxtasksperchild=1)
