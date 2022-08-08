# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. Lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

# @partial(jax.jit, static_argnames=["Kb", "Ksg", "Ksl"])
# def _get_energy_2d_closed(
#     vertex_positions: npt.NDArray[np.float64],
#     Kb: float = 0.1,
#     Ksg: float = 50,
#     Ksl: float = 1,
# ) -> npt.NDArray[np.float64]:
#     """Compute the energy of a 2D discrete closed polygon.

#     Note that this function assumes that the coordinates of the last point are the same as the first point.

#     Args:
#         vertex_positions (npt.NDArray[np.float64]): Coordinates
#         Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
#         Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.

#     Returns:
#         npt.NDArray[np.float64]: Componentwise energies of the system
#     """
#     d_pos = jnp.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
#     edgeLengths = jnp.linalg.norm(d_pos, axis=1)
#     referenceEdgeLength = jnp.mean(edgeLengths)

#     edgeAbsoluteAngles = jnp.arctan2(d_pos[:, 1], d_pos[:, 0])

#     vertexTurningAngles = (jnp.roll(edgeAbsoluteAngles, -1) - edgeAbsoluteAngles) % (
#         2 * jnp.pi
#     )
#     vertexTurningAngles = (vertexTurningAngles + jnp.pi) % (2 * jnp.pi) - jnp.pi

#     tan_vertex_turning_angles = jnp.tan(vertexTurningAngles / 2)

#     edgeCurvatures = (
#         tan_vertex_turning_angles + jnp.roll(tan_vertex_turning_angles, 1)
#     ) / edgeLengths

#     bendingEnergy = Kb * jnp.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
#     surfaceEnergy = Ksg * jnp.sum(edgeLengths)
#     regularizationEnergy = Ksl * jnp.sum(
#         ((edgeLengths - referenceEdgeLength) / referenceEdgeLength) ** 2
#     )
#     return jnp.array([bendingEnergy, surfaceEnergy, regularizationEnergy])


# def get_energy_2d_closed(
#     vertex_positions: npt.NDArray[np.float64],
#     Kb: float = 0.1,
#     Ksg: float = 50,
#     Ksl: float = 1,
# ) -> npt.NDArray[np.float64]:
#     """Compute the energy of a 2D discrete closed polygon.

#     Note that this function assumes that the coordinates of the last point are the same as the first point.

#     Args:
#         vertex_positions (npt.NDArray[np.float64]): Coordinates
#         Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
#         Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.

#     Returns:
#         float: Energy of the system
#     """
#     if not jnp.allclose(vertex_positions[-1], vertex_positions[0]):
#         raise RuntimeError(
#             f"First ({vertex_positions[0]}) and last ({vertex_positions[-1]}) points are expected to be the same."
#         )
#     return _get_energy_2d_closed(vertex_positions, Kb, Ksg, Ksl)


# @partial(jax.jit, static_argnames=["Kb", "Ksg"])
# def _get_force_2d_closed(
#     vertex_positions: npt.NDArray[np.float64],
#     Kb: float = 0.1,
#     Ksg: float = 50,
# ) -> npt.NDArray[np.float64]:
#     f_energy = partial(_get_energy_2d_closed, Kb=Kb, Ksg=Ksg)
#     # return jax.grad(f_energy)(vertex_positions)
#     return jax.jacrev(f_energy)(vertex_positions)


# def get_force_2d_closed(
#     vertex_positions: npt.NDArray[np.float64],
#     Kb: float = 0.1,
#     Ksg: float = 50,
# ) -> npt.NDArray[np.float64]:
#     """Compute the force of a 2D discrete closed polygon.

#     Note that this function assumes that the coordinates of the last point are the same as the first point.

#     Args:
#         vertex_positions (npt.NDArray[np.float64]): Coordinates
#         Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
#         Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.

#     Returns:
#         float: Energy of the system
#     """
#     if not jnp.allclose(vertex_positions[-1], vertex_positions[0]):
#         raise RuntimeError(
#             f"First ({vertex_positions[0]}) and last ({vertex_positions[-1]}) points are expected to be the same."
#         )
#     return _get_force_2d_closed(vertex_positions, Kb, Ksg)


# @partial(jax.jit, static_argnames=["Kb", "Ksg", "Ksl"])
# def _get_energy_force_2d_closed(
#     self,
#     vertex_positions: npt.NDArray[np.float64],
#     Kb: float = 0.1,
#     Ksg: float = 50,
#     Ksl: float = 1,
# ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     energy, vjp = jax.vjp(_get_energy_2d_closed, vertex_positions, Kb, Ksg, Ksl)
#     force = jax.vmap(vjp, in_axes=0)(jnp.ones_like(energy))
#     return energy, force


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


class Material(ABC):
    @abstractmethod
    def energy(
        self, vertex_positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def force(
        self, vertex_positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def energy_force(
        self, vertex_positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        pass


class ClosedPlaneCurveMaterial(Material):
    def __init__(
        self,
        Kb: float = 0.1,
        Ksg: float = 50,
        Ksl: float = 1,
    ):
        """_summary_

        Args:
            Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
            Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.
            Ksl (float, optional): Regularization modulus. Defaults to 1.
        """
        self.Kb = Kb
        self.Ksg = Ksg
        self.Ksl = Ksl

    def _check_valid(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> bool:
        """_summary_

        Args:
            vertex_positions (npt.NDArray[np.float64]): _description_

        Raises:
            RuntimeError: _description_

        Returns:
            bool: _description_
        """
        if not np.allclose(vertex_positions[-1], vertex_positions[0]):
            raise RuntimeError(
                f"First ({vertex_positions[0]}) and last ({vertex_positions[-1]}) points are expected to be the same."
            )
        return True

    @partial(jax.jit, static_argnums=0)
    def _energy(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the energy of a 2D discrete closed polygon.

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: Componentwise energies of the system
        """
        d_pos = jnp.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
        edgeLengths = jnp.linalg.norm(d_pos, axis=1)
        referenceEdgeLength = jnp.mean(edgeLengths)

        edgeAbsoluteAngles = jnp.arctan2(d_pos[:, 1], d_pos[:, 0])

        vertexTurningAngles = (
            jnp.roll(edgeAbsoluteAngles, -1) - edgeAbsoluteAngles
        ) % (2 * jnp.pi)
        vertexTurningAngles = (vertexTurningAngles + jnp.pi) % (2 * jnp.pi) - jnp.pi

        tan_vertex_turning_angles = jnp.tan(vertexTurningAngles / 2)

        edgeCurvatures = (
            tan_vertex_turning_angles + jnp.roll(tan_vertex_turning_angles, 1)
        ) / edgeLengths

        bendingEnergy = self.Kb * jnp.sum(edgeCurvatures * edgeCurvatures * edgeLengths)
        surfaceEnergy = self.Ksg * jnp.sum(edgeLengths)
        regularizationEnergy = self.Ksl * jnp.sum(
            ((edgeLengths - referenceEdgeLength) / referenceEdgeLength) ** 2
        )
        return jnp.array([bendingEnergy, surfaceEnergy, regularizationEnergy])

    def energy(
        self, vertex_positions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute the energy of a 2D discrete closed polygon.

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates
            Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
            Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.

        Returns:
            float: Energy of the system
        """
        self._check_valid(vertex_positions)
        return self._energy(vertex_positions)

    @partial(jax.jit, static_argnums=0)
    def _force(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        # return jax.grad(f_energy)(vertex_positions)
        return jax.jacrev(self._energy)(vertex_positions)

    def force(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the force of a 2D discrete closed polygon.

        Note that this function assumes that the coordinates of the last point are the same as the first point.

        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates
            Kb (float, optional): Bending modulus in units of pN um.Defaults to 1.
            Ksg (float, optional): Global stretching modulus in units of PN um/um^2. Defaults to 0.

        Returns:
            float: Energy of the system
        """
        self._check_valid(vertex_positions)
        return self._force(vertex_positions)

    @partial(jax.jit, static_argnums=0)
    def _energy_force(
        self,
        vertex_positions: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        energy, vjp = jax.vjp(self._energy, vertex_positions)
        (force,) = jax.vmap(vjp, in_axes=0)(jnp.eye(len(energy)))
        return energy, force

    def energy_force(
        self, vertex_positions: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self._check_valid(vertex_positions)
        return self._energy_force(vertex_positions)
