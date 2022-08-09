# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. Lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import matplotlib.pyplot as plt
import numpy as np

import automembrane.util as u
from automembrane.energy import ClosedPlaneCurveMaterial, OpenPlaneCurveMaterial

if __name__ == "__main__":
    u.matplotlibStyle(medium=10)
    fig, ax = plt.subplots(2)

    parameters = {
        "Kb": 1,  # Bending modulus
        "Ksg": 0,  # Global stretching modulus
        "Ksl": 0,
    }

    nVertex = 9  # number of vertices

    n = 0
    vertexPositions, isClosed = u.ellipse(nVertex)

    m = ClosedPlaneCurveMaterial(**parameters)

    energy = m.energy(vertexPositions)
    print("Energy is ", energy)

    forces = np.sum(m.force(vertexPositions), axis=0)

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

    m = OpenPlaneCurveMaterial(**parameters)
    vertexPositions, isClosed = u.cos_curve(nVertex)
    energy = m.energy(vertexPositions)

    print("Energy is ", energy)
    forces = np.array(np.sum(m.force(vertexPositions), axis=0))

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
