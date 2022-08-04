# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. Lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import value_and_grad
import pymem3dg.visual as dg_vis
import matplotlib.pyplot as plt


def getGeometry1(nVertex):
    R = 1
    theta = np.linspace(-np.pi / 2, np.pi / 2, nVertex + 1)
    x = 1.1 * R * np.cos(theta)
    y = 0.9 * R * np.sin(theta)
    x[-1] = x[0]
    y[-1] = y[0]
    vertexPositions = np.append(x[:-1], y[:-1])
    isClosed = True
    return vertexPositions, isClosed


def removeBoundaryForces(forces):
    nVertex = int(np.shape(forces)[0] / 2)
    forces[:3] = 0
    forces[nVertex - 3 : nVertex] = 0
    forces[nVertex : nVertex + 3] = 0
    forces[-3:] = 0
    return forces


def getEnergy(vertexPositions, isClosed):
    nVertex = int(np.shape(vertexPositions)[0] / 2)

    vertexRadii = vertexPositions[:nVertex]
    vertexHeights = vertexPositions[nVertex:]

    edgeRadii = (vertexRadii[0 : nVertex - 1] + vertexRadii[1:nVertex]) * 0.5

    dr = np.diff(vertexRadii)
    dh = np.diff(vertexHeights)
    ds = np.sqrt(dr**2 + dh**2)
    s1 = np.append(ds, 0.0)
    s2 = np.append(0.0, ds)
    vertexArclengths = (s1 + s2) * 0.5
    vertexAreas = 2 * np.pi * vertexRadii * vertexArclengths
    vertexVolumes = np.pi * edgeRadii**2 * dh

    edgeAbsoluteAngles = np.arctan2(dh, dr)
    vertexTurningAngles = np.diff(edgeAbsoluteAngles) % (2 * np.pi)
    vertexTurningAngles = (vertexTurningAngles + np.pi) % (2 * np.pi) - np.pi
    vertexTurningAngles = np.append(0.0, vertexTurningAngles)
    vertexTurningAngles = np.append(vertexTurningAngles, 0.0)
    vertexAxialCurvatures = vertexTurningAngles / vertexArclengths

    edgeRadialCurvatures = np.sin(edgeAbsoluteAngles) / edgeRadii * ds
    rc1 = np.append(edgeRadialCurvatures, 0.0)
    rc2 = np.append(0.0, edgeRadialCurvatures)
    vertexRadialCurvatures = (rc1 + rc2) * 0.5 / vertexArclengths

    vertexMeanCurvatures = 0.5 * (vertexAxialCurvatures + vertexRadialCurvatures)

    bendingEnergy = Kb * np.sum(vertexMeanCurvatures**2 * vertexAreas)
    surfaceEnergy = Ksg * np.sum(vertexAreas)
    pressureEnergy = Kv * np.sum(vertexVolumes)

    return bendingEnergy + surfaceEnergy


if __name__ == "__main__":
    dg_vis.matplotlibStyle(m=10)
    fig, ax = plt.subplots(1)

    Kb = 1
    Kbc = 0
    Ksg = 0
    At = 0
    epsilon = 0
    Kv = 0
    Vt = 0
    nVertex = 100

    nSubplot = 0

    vertexPositions, isClosed = getGeometry1(nVertex)
    energy = getEnergy(vertexPositions, isClosed)
    print("Energy is ", energy)
    forces = -egrad(getEnergy)(vertexPositions, isClosed)
    forces = removeBoundaryForces(forces)
    ax.scatter(vertexPositions[:nVertex], vertexPositions[nVertex:], label="membrane")
    ax.quiver(
        vertexPositions[:nVertex],
        vertexPositions[nVertex:],
        forces[:nVertex],
        forces[nVertex:],
        label="force",
    )
    ax.scatter(-vertexPositions[:nVertex], vertexPositions[nVertex:])
    ax.quiver(
        -vertexPositions[:nVertex],
        vertexPositions[nVertex:],
        -forces[:nVertex],
        forces[nVertex:],
    )
    ax.legend()
    ax.set_aspect("equal")
    nSubplot = nSubplot + 1
    # vertexPositions, isClosed = getGeometry1(nVertex)
    # energy = getEnergy(vertexPositions, isClosed)
    # print("Energy is ", energy)
    # forces = -egrad(getEnergy)(vertexPositions, isClosed)
    # forces = removeBoundaryForces(forces)
    # ax[nSubplot].scatter(vertexPositions[:nVertex],
    #                      vertexPositions[nVertex:], label="membrane")
    # ax[nSubplot].quiver(vertexPositions[:nVertex], vertexPositions[nVertex:],
    #                     forces[:nVertex], forces[nVertex:], label="force")
    # ax[nSubplot].legend()
    # ax[nSubplot].set_aspect('equal')
    # nSubplot = nSubplot+1

    plt.savefig("axi.pdf")
    plt.show()
