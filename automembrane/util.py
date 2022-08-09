# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. Lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
from typing import Callable
import numpy.typing as npt

import jax
import matplotlib.pyplot as plt


def ellipse(
    nVertex: int = 100, x_scale: float = 1.1, y_scale: float = 0.9, R: float = 1.0
) -> tuple[npt.NDArray[np.float64], bool]:
    """Generate an ellipse

    Args:
        nVertex (int, optional): Number of vertices. Defaults to 100.
        x_scale (float, optional): Factor to scale x radius by. Defaults to 1.1.
        y_scale (float, optional): Factor to scale y radius by. Defaults to 0.9.
        R (float, optional): Radius. Defaults to 1.0.

    Returns:
        tuple[npt.NDArray[np.float64], bool]: ist of vertices, and whether shape is closed
    """
    theta = np.linspace(0, 2 * np.pi, nVertex)
    x = (x_scale * R * np.cos(theta)).reshape(-1, 1)
    y = (y_scale * R * np.sin(theta)).reshape(-1, 1)
    vertexPositions = np.hstack((x, y))
    isClosed = True
    return vertexPositions, isClosed


def half_ellipse(
    nVertex: int = 100, x_scale: float = 1.1, y_scale: float = 0.9, R: float = 1.0
) -> tuple[npt.NDArray[np.float64], bool]:
    """Generate a half ellipse

    Args:
        nVertex (int, optional): Number of vertices. Defaults to 100.
        x_scale (float, optional): Factor to scale x radius by. Defaults to 1.1.
        y_scale (float, optional): Factor to scale y radius by. Defaults to 0.9.
        R (float, optional): Radius. Defaults to 1.0.

    Returns:
        tuple[npt.NDArray[np.float64], bool]: ist of vertices, and whether shape is closed
    """
    theta = np.linspace(-np.pi / 2, np.pi / 2, nVertex + 1)
    x = x_scale * R * np.cos(theta).reshape(-1, 1)
    y = y_scale * R * np.sin(theta).reshape(-1, 1)
    vertexPositions = np.hstack((x, y))
    isClosed = True
    return vertexPositions, isClosed


def cos_curve(nVertex: int):
    amp = 0.5
    x = (np.linspace(0, 2 * np.pi, nVertex)).reshape(-1, 1)
    y = (amp * np.cos(x)).reshape(-1, 1)
    vertexPositions = np.hstack((x, y))
    isClosed = False
    return vertexPositions, isClosed


def matplotlibStyle(small: float = 6, medium: float = 8, large: float = 10):
    """Set matplotlib plotting style

    Args:
        s (int, optional): Small size. Defaults to 6.
        m (int, optional): Medium size. Defaults to 8.
        l (int, optional): Large size. Defaults to 10.
    """
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["savefig.dpi"] = 600
    # mpl.rcParams.update({'font.size': 8})
    plt.rc("font", size=large)  # controls default text sizes
    plt.rc("axes", titlesize=large)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=medium)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=medium)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small, frameon=False)  # legend fontsize
    plt.rc("figure", titlesize=large)  # fontsize of the figure title
    plt.rc("pdf", fonttype=42)


def egrad(g: Callable) -> Callable:
    """Convenience function to generate the elementwise gradient of a function

    Args:
        g (Callable): Function to wrap

    Returns:
        Callable: Wrapped function
    """

    def wrapped(x, *rest, **kwargs):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest, **kwargs), x)
        (x_bar,) = g_vjp(jax.numpy.ones_like(y))
        return x_bar

    return wrapped
