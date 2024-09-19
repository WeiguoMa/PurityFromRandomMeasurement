import os
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dependency')))


class RenyiEntropy:
    def __init__(self, measurementScheme: List[List[int]], measurementResults: List[List[List[int]]]):
        from RenyiEntropy_backend import RenyiEntropy_backend
        self.backend = RenyiEntropy_backend(measurementScheme, measurementResults)

    def calculateRenyiEntropy(self, classical_shadow: bool = False) -> float:
        """
        Calculate the Renyi entropy using the measurement outcomes.

        Args:
            classical_shadow (bool): Flag to indicate if the classical shadow is used.

        Returns:
            List[float]: A list of Renyi entropy values.
        """
        renyi_entropy = self.backend.calculateRenyiEntropy(classical_shadow)
        return renyi_entropy
