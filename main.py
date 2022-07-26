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

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import automembrane.util as u
from automembrane.energy import getEnergy2DClosed, getEnergy2DOpen

if __name__ == "__main__":
    u.matplotlibStyle(medium=10)
    fig, ax = plt.subplots(2)

    parameters = {
        "Kb": 1,  # Bending modulus
        "Kbc": 0,  # Constant of bending modulus vs protein density
        "Ksg": 0,  # Global stretching modulus
        "At": 0,  # Preferred area
        "epsilon": 0,  # Binding energy per protein
        "Kv": 0,  # pressure-volume modulus
        "Vt": 0,  # Volume target
    }

    nVertex = 12  # number of vertices

    n = 0
    vertexPositions, isClosed = u.ellipse(nVertex)

    f_energy = partial(getEnergy2DClosed, **parameters)

    energy = f_energy(vertexPositions)
    print("Energy is ", energy)

    forces = -u.egrad(f_energy)(vertexPositions)
    ax[n].scatter(vertexPositions[:, 0], vertexPositions[:, 1], label="membrane")

    ax[n].scatter(vertexPositions[:2, 0], vertexPositions[:2, 1], label="membrane")
    ax[n].quiver(
        vertexPositions[:, 0],
        vertexPositions[:, 1],
        forces[:, 0],
        forces[:, 1],
        label="force",
    )
    ax[n].legend()
    ax[n].set_aspect("equal")
    n = n + 1

    # Rebind energy for open curves
    f_energy = partial(getEnergy2DOpen, **parameters)

    vertexPositions, isClosed = u.getGeometry2(nVertex)
    energy = f_energy(vertexPositions)

    print("Energy is ", energy)
    forces = np.array(-u.egrad(f_energy)(vertexPositions))

    forces[:3] = 0
    forces[-3:] = 0

    ax[n].scatter(vertexPositions[:, 0], vertexPositions[:, 1], label="membrane")
    ax[n].quiver(
        vertexPositions[:, 0],
        vertexPositions[:, 1],
        forces[:, 0],
        forces[:, 1],
        label="force",
    )
    ax[n].legend()
    ax[n].set_aspect("equal")
    n = n + 1

    plt.savefig("2d_new.pdf")
    # plt.show()
