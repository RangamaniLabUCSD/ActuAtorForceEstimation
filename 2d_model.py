import autograd.numpy as np
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import value_and_grad
import pymem3dg.visual as dg_vis
import matplotlib.pyplot as plt


def getGeometry1(nVertex):
    R = 1
    theta = np.linspace(0, 2 * np.pi, nVertex + 1)
    x = 1.1 * R * np.cos(theta)
    y = 0.9 * R * np.sin(theta)
    x[-1] = x[0]
    y[-1] = y[0]
    vertexPositions = np.append(x[:-1], y[:-1])
    isClosed = True
    return vertexPositions, isClosed


def getGeometry2(nVertex):
    amp = 0.5
    x = np.linspace(0, 2 * np.pi, nVertex + 1)
    y = amp * np.cos(x)
    vertexPositions = np.append(x[:-1], y[:-1])
    isClosed = False
    return vertexPositions, isClosed


def removeBoundaryForces(forces):
    nVertex = int(np.shape(forces)[0]/2)
    forces[:3] = 0
    forces[nVertex - 3: nVertex] = 0
    forces[nVertex: nVertex + 3] = 0
    forces[-3:] = 0
    return forces


def getEnergy(vertexPositions, isClosed):
    nVertex = int(np.shape(vertexPositions)[0]/2)
    x = vertexPositions[:nVertex]
    y = vertexPositions[nVertex:]

    if isClosed:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        dx = np.diff(x)  # n+1-1
        dy = np.diff(y)

        edgeLengths = np.sqrt(dx ** 2 + dy ** 2)

        edgeAbsoluteAngles = np.arctan2(dy, dx)
        vertexTurningAngles = np.diff(
            np.append(edgeAbsoluteAngles, edgeAbsoluteAngles[0])
        ) % (2 * np.pi)
        vertexTurningAngles = (vertexTurningAngles +
                               np.pi) % (2 * np.pi) - np.pi
        vertexTurningAngles = np.append(
            vertexTurningAngles, vertexTurningAngles[0])

        edgeCurvatures = (
            np.tan(vertexTurningAngles[:-1] / 2) +
            np.tan(vertexTurningAngles[1:] / 2)
        ) / edgeLengths
        bendingEnergy = Kb * \
            np.sum(edgeCurvatures * edgeCurvatures * edgeLengths)

        surfaceEnergy = Ksg * np.sum(edgeLengths)

        return bendingEnergy + surfaceEnergy
    else:
        dx = np.diff(x)
        dy = np.diff(y)
        edgeLengths = np.sqrt(dx ** 2 + dy ** 2)
        edgeAbsoluteAngles = np.arctan2(dy, dx)
        vertexTurningAngles = np.diff(edgeAbsoluteAngles) % (2 * np.pi)
        vertexTurningAngles = (vertexTurningAngles +
                               np.pi) % (2 * np.pi) - np.pi
        vertexTurningAngles = np.append(
            vertexTurningAngles, vertexTurningAngles[-1])
        vertexTurningAngles = np.append(vertexTurningAngles[0],
                                        vertexTurningAngles)
        edgeCurvatures = (
            np.tan(vertexTurningAngles[:-1] / 2) +
            np.tan(vertexTurningAngles[1:] / 2)
        ) / edgeLengths

        bendingEnergy = Kb * \
            np.sum(edgeCurvatures * edgeCurvatures * edgeLengths)

        surfaceEnergy = Ksg * np.sum(edgeLengths)
        return bendingEnergy + surfaceEnergy


if __name__ == "__main__":
    dg_vis.matplotlibStyle(m=10)
    fig, ax = plt.subplots(2)

    Kb = 1
    Kbc = 0
    Ksg = 0
    At = 0
    epsilon = 0
    Kv = 0
    Vt = 0
    nVertex = 150

    n = 0

    vertexPositions, isClosed = getGeometry1(nVertex)
    energy = getEnergy(vertexPositions, isClosed)
    print("Energy is ", energy)
    forces = -egrad(getEnergy)(vertexPositions, isClosed)
    ax[n].scatter(vertexPositions[:nVertex],
                  vertexPositions[nVertex:], label="membrane")
    ax[n].quiver(vertexPositions[:nVertex], vertexPositions[nVertex:],
                 forces[:nVertex], forces[nVertex:], label="force")
    ax[n].legend()
    ax[n].set_aspect('equal')
    n = n+1

    vertexPositions, isClosed = getGeometry2(nVertex)
    energy = getEnergy(vertexPositions, isClosed)
    print("Energy is ", energy)
    forces = -egrad(getEnergy)(vertexPositions, isClosed)
    forces = removeBoundaryForces(forces)
    ax[n].scatter(vertexPositions[:nVertex],
                  vertexPositions[nVertex:], label="membrane")
    ax[n].quiver(vertexPositions[:nVertex], vertexPositions[nVertex:],
                 forces[:nVertex], forces[nVertex:], label="force")
    ax[n].legend()
    ax[n].set_aspect('equal')
    n = n+1

    plt.savefig("2d.pdf")
    # plt.show()
