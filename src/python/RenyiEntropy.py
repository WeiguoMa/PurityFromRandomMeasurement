import os
import sys
from typing import List, Dict
import numpy as np
import itertools

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


class RenyiEntropyWithDifferentRhos:
    def __init__(self, qnumber: int):
        self.qn = qnumber
        self.hamming_dict = self.all_hamming_distances()

    def all_hamming_distances(self):
        def hamming_distance(a, b):
            return sum(el1 != el2 for el1, el2 in zip(a, b))

        bit_combinations = list(itertools.product([0, 1], repeat=self.qn))

        distance_dict = {}
        for i in range(len(bit_combinations)):
            for j in range(i, len(bit_combinations)):
                A = bit_combinations[i]
                B = bit_combinations[j]
                key = (tuple(A), tuple(B)) if A <= B else (tuple(B), tuple(A))
                distance_dict[key] = hamming_distance(A, B)

        return distance_dict

    def quick_hamming_query(self, A, B):
        key = (tuple(A), tuple(B)) if A <= B else (tuple(B), tuple(A))
        return self.hamming_dict.get(key, None)

    def calculatePurity(self, measureOutcomesA: List[List[List[int]]], measureOutcomesB: List[List[List[int]]]):
        M, K = len(measureOutcomesA), len(measureOutcomesA[0])
        coefficient = 2 ** self.qn / (M * K * (K - 1))

        sumValue = 0.0
        for m in range(M):
            for k in range(K):
                outcomesA_K = measureOutcomesA[m][k]
                for k_prime in range(k + 1, K):
                    hamming_distance = self.quick_hamming_query(outcomesA_K, measureOutcomesB[m][k_prime])
                    sumValue += (-2) ** (-hamming_distance)

        return 2 * coefficient * sumValue
