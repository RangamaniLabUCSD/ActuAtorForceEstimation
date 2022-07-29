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

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


def getEnergy2DClosed(
    vertexPositions: npt.NDArray[np.float64],
    Kb: float = 1,
    Kbc: float = 0,
    Ksg: float = 0,
    At: float = 0,
    epsilon: float = 0,
    Kv: float = 0,
    Vt: float = 0,
) -> float:
    """Compute the energy of a 2D discrete closed polygon

    Args:
        vertexPositions (npt.NDArray[np.float64]): Coordinates
        Kb (float, optional): Bending modulus. Defaults to 1.
        Kbc (float, optional): Constant relating bending modulus and protein density. Defaults to 0.
        Ksg (float, optional): Global stretching modulus. Defaults to 0.
        At (float, optional): Area target. Defaults to 0.
        epsilon (float, optional): Protein binding energy. Defaults to 0.
        Kv (float, optional): Pressure-volume modulus. Defaults to 0.
        Vt (float, optional): Volume target. Defaults to 0.

    Returns:
        float: Energy of the system
    """
    if not jnp.allclose(vertexPositions[-1], vertexPositions[0]):
        raise RuntimeError(f"First ({vertexPositions[0]}) and last ({vertexPositions[-1]}) points are expected to be the same.")
        
    x = vertexPositions[:-1, 0]
    y = vertexPositions[:-1, 1]
    dx = jnp.roll(x, -1) - x
    dy = jnp.roll(y, -1) - y
    edgeLengths = jnp.sqrt(dx**2 + dy**2)
    edgeAbsoluteAngles = jnp.arctan2(dy, dx)

    vertexTurningAngles = (jnp.roll(edgeAbsoluteAngles, -1) - edgeAbsoluteAngles) % (
        2 * jnp.pi
    )
    vertexTurningAngles = (vertexTurningAngles + jnp.pi) % (2 * jnp.pi) - jnp.pi

    tan_vertex_turning_angles = jnp.tan(vertexTurningAngles / 2)

    edgeCurvatures = (
        tan_vertex_turning_angles + jnp.roll(tan_vertex_turning_angles, 1)
    ) / edgeLengths

    bendingEnergy = Kb * jnp.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
    surfaceEnergy = Ksg * jnp.sum(edgeLengths)
    return bendingEnergy + surfaceEnergy


def getEnergy2DClosed_notrace(
    vertexPositions: npt.NDArray[np.float64],
    Kb: float = 1,
    Kbc: float = 0,
    Ksg: float = 0,
    At: float = 0,
    epsilon: float = 0,
    Kv: float = 0,
    Vt: float = 0,
) -> float:
    """Compute the energy of a 2D discrete closed polygon

    Args:
        vertexPositions (npt.NDArray[np.float64]): Coordinates
        Kb (float, optional): Bending modulus. Defaults to 1.
        Kbc (float, optional): Constant relating bending modulus and protein density. Defaults to 0.
        Ksg (float, optional): Global stretching modulus. Defaults to 0.
        At (float, optional): Area target. Defaults to 0.
        epsilon (float, optional): Protein binding energy. Defaults to 0.
        Kv (float, optional): Pressure-volume modulus. Defaults to 0.
        Vt (float, optional): Volume target. Defaults to 0.

    Returns:
        float: Energy of the system
    """
    if not np.allclose(vertexPositions[-1], vertexPositions[0]):
        raise RuntimeError(f"First ({vertexPositions[0]}) and last ({vertexPositions[-1]}) points are expected to be the same.")
        
    x = vertexPositions[:-1, 0]
    y = vertexPositions[:-1, 1]
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    edgeLengths = np.sqrt(dx**2 + dy**2)
    edgeAbsoluteAngles = np.arctan2(dy, dx)

    vertexTurningAngles = (np.roll(edgeAbsoluteAngles, -1) - edgeAbsoluteAngles) % (
        2 * np.pi
    )
    vertexTurningAngles = (vertexTurningAngles + np.pi) % (2 * np.pi) - np.pi

    tan_vertex_turning_angles = np.tan(vertexTurningAngles / 2)

    edgeCurvatures = (
        tan_vertex_turning_angles + np.roll(tan_vertex_turning_angles, 1)
    ) / edgeLengths

    bendingEnergy = Kb * np.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
    surfaceEnergy = Ksg * np.sum(edgeLengths)

    print(bendingEnergy, surfaceEnergy)
    return bendingEnergy + surfaceEnergy


def getEnergy2DOpen(
    vertexPositions: npt.NDArray[np.float64],
    Kb: float = 1,
    Kbc: float = 0,
    Ksg: float = 0,
    At: float = 0,
    epsilon: float = 0,
    Kv: float = 0,
    Vt: float = 0,
) -> float:
    """Compute the energy of a 2D discrete open polygon

    Args:
        vertexPositions (npt.NDArray[np.float64]): Coordinates
        Kb (float, optional): Bending modulus. Defaults to 1.
        Kbc (float, optional): Constant relating bending modulus and protein density. Defaults to 0.
        Ksg (float, optional): Global stretching modulus. Defaults to 0.
        At (float, optional): Area target. Defaults to 0.
        epsilon (float, optional): Protein binding energy. Defaults to 0.
        Kv (float, optional): Pressure-volume modulus. Defaults to 0.
        Vt (float, optional): Volume target. Defaults to 0.

    Returns:
        float: Energy of the system
    """
    x = vertexPositions[:, 0]
    y = vertexPositions[:, 1]
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    edgeLengths = jnp.sqrt(dx**2 + dy**2)
    edgeAbsoluteAngles = jnp.arctan2(dy, dx)

    vertexTurningAngles = (jnp.diff(edgeAbsoluteAngles)) % (2 * jnp.pi)
    vertexTurningAngles = (vertexTurningAngles + jnp.pi) % (2 * jnp.pi) - jnp.pi

    vertexTurningAngles = jnp.append(vertexTurningAngles, vertexTurningAngles[-1])
    vertexTurningAngles = jnp.append(vertexTurningAngles[0], vertexTurningAngles)

    edgeCurvatures = (
        jnp.tan(vertexTurningAngles[:-1] / 2) + jnp.tan(vertexTurningAngles[1:] / 2)
    ) / edgeLengths

    bendingEnergy = Kb * jnp.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
    surfaceEnergy = Ksg * jnp.sum(edgeLengths)

    return bendingEnergy + surfaceEnergy
