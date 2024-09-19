import os
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../dependency')))


class ShadowState:
    def __init__(self, qnumber: int):
        from ShadowState_backend import ShadowState_backend
        self.backend = ShadowState_backend(qnumber)

    def stateEstimation(self, measureOperations: List[List[int]], measureResults: List[List[List[int]]]):
        """
        Perform state estimation using the measurement outcomes.

        Args:
            measureOperations (List[List[int]]): The measurement operations.
            measureResults (List[List[List[int]]): The measurement outcomes.

        Returns:
            List[List[float]]: The estimated states.
        """
        estimated_states = self.backend.stateEstimation(measureOperations, measureResults)
        return estimated_states
