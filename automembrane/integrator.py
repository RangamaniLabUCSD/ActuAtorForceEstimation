# Copyright (c) 2022 Eleanor Jung, Cuncheng Zhu, and Christopher T. Lee
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from functools import partial
import jax

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from .energy import Material


# @partial(jax.jit, static_argnums=[1], static_argnames=["n_steps", "dt"])
def fwd_euler_integrator(
    coords: npt.NDArray[np.float64],
    mat: Material,
    n_steps: int = int(1e5),
    dt: float = 5e-6,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Perform forward Euler integration

    Args:
        coords (npt.NDArray[np.float64]): Vertex coordinate position
        mat (Material): Membrane material holding parameters
        n_steps (int, optional): Number of steps to take. Defaults to int(1e5).
        dt (float, optional): Time step. Defaults to 5e-6.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: _description_
    """
    energy_log = np.zeros((n_steps, *mat._energy(coords).shape))

    for i in tqdm(range(0, n_steps), desc="Energy relaxation"):
        energy_log[i], force = mat._energy_force(coords)
        # c_t+1 = c_t - force * dt
        coords = np.array(coords - np.sum(force, axis=0) * dt)
        coords[-1] = coords[0]
    # print(
    #     f"  DELTA E: {energy_log[-1] - energy_log[0]}; E_before: {energy_log[0]}; E_after: {energy_log[-1]}"
    # )
    return coords, energy_log
