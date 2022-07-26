# Copyright (C) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. lee
#
# ActuAtorForceEstimation is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ActuAtorForceEstimation is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with ActuAtorForceEstimation. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from typing import Callable
import numpy.typing as npt

import jax
import matplotlib.pyplot as plt


def ellipse(nVertex: int = 100) -> tuple[npt.NDArray[np.float64], bool]:
    """Generate an ellipse

    Args:
        nVertex (int, optional): Number of vertices. Defaults to 100.

    Returns:
        tuple[npt.NDArray[np.float64], bool]: List of vertices, and whether shape is closed
    """
    R = 1
    theta = np.linspace(0, 2 * np.pi, nVertex)
    x = (1.1 * R * np.cos(theta)).reshape(-1, 1)
    y = (0.9 * R * np.sin(theta)).reshape(-1, 1)
    vertexPositions = np.hstack((x, y))
    isClosed = True
    return vertexPositions, isClosed


def getGeometry2(nVertex: int):
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
    """Compute the elementwise gradient of a function

    Args:
        g (Callable): Function to wrap

    Returns:
        Callable: Wrapped function
    """

    def wrapped(x, *rest, **kwargs):
        y, g_vjp = jax.vjp(lambda x: g(x, *rest, **kwargs), x)
        (x_bar,) = g_vjp(np.ones_like(y))
        return x_bar

    return wrapped
