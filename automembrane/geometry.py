# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. Lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

class ClosedPlaneCurveGeometry:
    @staticmethod
    # @jax.jit
    def edge_curvature(
            vertex_positions: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            """Compute edge curvature
            Args:
                vertex_positions (npt.NDArray[np.float64]): Coordinates

            Returns:
                npt.NDArray[np.float64]: edge curvature
            """
            d_pos = jnp.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
            edgeLengths = jnp.linalg.norm(d_pos, axis=1)

            edgeAbsoluteAngles = jnp.arctan2(d_pos[:, 1], d_pos[:, 0])
            vertexTurningAngles =    (
                jnp.roll(edgeAbsoluteAngles, -1) - edgeAbsoluteAngles
            ) % (2 * jnp.pi)
            vertexTurningAngles = (vertexTurningAngles + jnp.pi) % (2 * jnp.pi) - jnp.pi

            tan_vertex_turning_angles = jnp.tan(vertexTurningAngles / 2)
            edgeCurvatures = (
                tan_vertex_turning_angles + jnp.roll(tan_vertex_turning_angles, 1)
            ) / edgeLengths
            return edgeCurvatures
        
        
    @staticmethod
    # @jax.jit
    def vertex_dual_length(
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute dual edge length
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: vertex dual length
        """
        dc = np.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
        edgeLengths = np.linalg.norm(dc, axis=1)
        dualLengths = ((edgeLengths + np.roll(edgeLengths, 1)) / 2.0).reshape(-1, 1)
        dualLengths = np.vstack((dualLengths, dualLengths[0]))

        return dualLengths

    @staticmethod
    # @jax.jit
    def edge_length(
        vertex_positions: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute edge length
        Args:
            vertex_positions (npt.NDArray[np.float64]): Coordinates

        Returns:
            npt.NDArray[np.float64]: edge length
        """
        dc = np.roll(vertex_positions[:-1], -1, axis=0) - vertex_positions[:-1]
        edgeLengths = np.linalg.norm(dc, axis=1)
        return edgeLengths