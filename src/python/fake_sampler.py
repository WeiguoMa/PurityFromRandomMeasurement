import os
import sys
from itertools import product
from typing import Union, List, Optional

import numpy as np
from qutip import Qobj

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dependency')))


class FakeSampler:
    def __init__(self, system_size: int, basis: Optional[List[str]] = None, circuit: bool = False):
        """
        Initialize the FakeSampler.

        Args:
            system_size (int): The size of the system.
            basis (Optional[List[str]]): The basis states.
            circuit (bool): Flag to indicate if a circuit is used.
        """
        self.basis = [''.join(state) for state in product('01', repeat=system_size)] if basis is None else basis
        from fakeSampler_backend import FakeSampler_backend
        self.backend = FakeSampler_backend(proj_basis=self.basis)

    def fake_sampling_dm(self, dm: Union[np.ndarray, Qobj],
                         measure_times: int,
                         measurement_orientation: Optional[List[int]] = None) -> List[List[int]]:
        """
        Perform fake sampling on a density matrix.

        Args:
            dm (Union[np.ndarray, Qobj]): The density matrix.
            measure_times: K times of measurement.
            measurement_orientation (Optional[List[str]]): The measurement orientations.

        Returns:
            Tuple[List[str], List[int]]: The measurement orientations and their corresponding eigenvalues.

        Raises:
            TypeError: If the density matrix is not of type np.ndarray or qutip.Qobj.
            ValueError: If the length of measurement_orientation does not match the system size.
        """
        system_size = int(np.log2(dm.shape[0]))

        if isinstance(dm, Qobj):
            dm = dm.full()
        elif not isinstance(dm, np.ndarray):
            raise TypeError("The type of Density Matrix must be np.ndarray or qutip.Qobj.")

        if measurement_orientation is None:
            measurement_orientation = [2] * system_size
        elif len(measurement_orientation) != system_size:
            raise ValueError("The length of measurement_orientation must be equal to the system size.")

        state01 = self.backend.fakeSampling_dm(dm_array=dm,
                                               measure_times=measure_times,
                                               measurement_orientation=measurement_orientation)

        return state01


def random_measurementScheme(qnumber: int, amount: int) -> List[List[int]]:
    """
    Generate random measurement scheme for a given qnumber.

    Args:
        qnumber: The number of qubits.
        amount: M. The amount of measurement schemes to generate

    Returns:
        List[str]: The measurement orientations.
    """
    return [np.random.choice([0, 1, 2], qnumber).tolist() for _ in range(amount)]
