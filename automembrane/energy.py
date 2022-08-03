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

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

# from . import util as u

@partial(jax.jit, static_argnames=["Kb", "Ksg"])
def _get_energy_2d_closed(
    vertex_positions: npt.NDArray[np.float64],
    Kb: float = 0.1,
    Ksg: float = 50,
) -> float:
    """Compute the energy of a 2D discrete closed polygon.

    Note that this function assumes that the coordinates of the last point are the same as the first point.

    Args:
        vertex_positions (npt.NDArray[np.float64]): Coordinates
        Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
        Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.

    Returns:
        float: Energy of the system
    """
    d_pos = jnp.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
    edgeLengths = jnp.linalg.norm(d_pos, axis=1)
    edgeAbsoluteAngles = jnp.arctan2(d_pos[:, 1], d_pos[:, 0])

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


def get_energy_2d_closed(
    vertex_positions: npt.NDArray[np.float64],
    Kb: float = 0.1,
    Ksg: float = 50,
) -> float:
    """Compute the energy of a 2D discrete closed polygon.

    Note that this function assumes that the coordinates of the last point are the same as the first point.

    Args:
        vertex_positions (npt.NDArray[np.float64]): Coordinates
        Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
        Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.

    Returns:
        float: Energy of the system
    """
    if not jnp.allclose(vertex_positions[-1], vertex_positions[0]):
        raise RuntimeError(
            f"First ({vertex_positions[0]}) and last ({vertex_positions[-1]}) points are expected to be the same."
        )
    return _get_energy_2d_closed(vertex_positions, Kb, Ksg)


@partial(jax.jit, static_argnames=["Kb", "Ksg"])
def _get_force_2d_closed(
    vertex_positions: npt.NDArray[np.float64],
    Kb: float = 0.1,
    Ksg: float = 50,
) -> float:
    f_energy = partial(_get_energy_2d_closed, Kb=Kb, Ksg=Ksg)
    return jax.grad(f_energy)(vertex_positions)
    

def get_force_2d_closed(
    vertex_positions: npt.NDArray[np.float64],
    Kb: float = 0.1,
    Ksg: float = 50,
) -> float:
    """Compute the force of a 2D discrete closed polygon.

    Note that this function assumes that the coordinates of the last point are the same as the first point.

    Args:
        vertex_positions (npt.NDArray[np.float64]): Coordinates
        Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
        Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.

    Returns:
        float: Energy of the system
    """
    if not jnp.allclose(vertex_positions[-1], vertex_positions[0]):
        raise RuntimeError(
            f"First ({vertex_positions[0]}) and last ({vertex_positions[-1]}) points are expected to be the same."
        )
    return _get_force_2d_closed(vertex_positions, Kb, Ksg)


def get_energy_2d_closed_notrace(
    vertex_positions: npt.NDArray[np.float64],
    Kb: float = 1,
    Ksg: float = 0,
) -> tuple[float, float]:
    """Compute the energy of a 2D discrete closed polygon without jax tracing.

    Args:
        vertex_positions (npt.NDArray[np.float64]): Coordinates
        Kb (float, optional): Bending modulus. Defaults to 1.
        Ksg (float, optional): Global stretching modulus. Defaults to 0.

    Returns:
        tuple(float, float): Tuple of bending energy and surface energy
    """
    if not np.allclose(vertex_positions[-1], vertex_positions[0]):
        raise RuntimeError(
            f"First ({vertex_positions[0]}) and last ({vertex_positions[-1]}) points are expected to be the same."
        )

    d_pos = np.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
    edgeLengths = np.linalg.norm(d_pos, axis=1)
    edgeAbsoluteAngles = np.arctan2(d_pos[:, 1], d_pos[:, 0])

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

    return bendingEnergy, surfaceEnergy


@partial(jax.jit, static_argnames=["Kb", "Ksg"])
def get_energy_2d_open(
    vertex_positions: npt.NDArray[np.float64],
    Kb: float = 0.1,
    Ksg: float = 50,
) -> float:
    """Compute the energy of a 2D discrete open polygon

    Args:
        vertex_positions (npt.NDArray[np.float64]): Coordinates
        Kb (float, optional): Bending modulus. Defaults to 1.
        Ksg (float, optional): Global stretching modulus. Defaults to 0.

    Returns:
        float: Energy of the system
    """
    x = vertex_positions[:, 0]
    y = vertex_positions[:, 1]
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


def get_energy_2d_open_notrace(
    vertex_positions: npt.NDArray[np.float64],
    Kb: float = 0.1,
    Ksg: float = 50,
) -> float:
    """Compute the energy of a 2D discrete open polygon without jax tracing

    Args:
        vertex_positions (npt.NDArray[np.float64]): Coordinates
        Kb (float, optional): Bending modulus. Defaults to 1.
        Ksg (float, optional): Global stretching modulus. Defaults to 0.

    Returns:
         tuple(float, float): Tuple of bending energy and surface energy
    """
    x = vertex_positions[:, 0]
    y = vertex_positions[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    edgeLengths = np.sqrt(dx**2 + dy**2)
    edgeAbsoluteAngles = np.arctan2(dy, dx)

    vertexTurningAngles = (np.diff(edgeAbsoluteAngles)) % (2 * np.pi)
    vertexTurningAngles = (vertexTurningAngles + np.pi) % (2 * np.pi) - np.pi

    vertexTurningAngles = np.append(vertexTurningAngles, vertexTurningAngles[-1])
    vertexTurningAngles = np.append(vertexTurningAngles[0], vertexTurningAngles)

    edgeCurvatures = (
        np.tan(vertexTurningAngles[:-1] / 2) + np.tan(vertexTurningAngles[1:] / 2)
    ) / edgeLengths

    bendingEnergy = Kb * np.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
    surfaceEnergy = Ksg * np.sum(edgeLengths)

    return bendingEnergy, surfaceEnergy
