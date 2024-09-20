import time
from typing import List
from tqdm import tqdm

import numpy as np

from src.python.Physics.generate_TEST_DM import state_qutip_evolution, pseudo_random_DM
from src.python.RenyiEntropy import RenyiEntropy
from src.python.fake_sampler import FakeSampler, random_measurementScheme

K = 100
M = 1000
QNUMBER = 4

# TIME_LIST = [0, 5, 8, 10, 14, 20]
# TEST_DM = state_qutip_evolution(QNUMBER, TIME_LIST)

TEST_DM = pseudo_random_DM(QNUMBER, numPure=1, numMixed=3)

FAKESAMPLER = FakeSampler(QNUMBER)
RANDOM_MEASUREMENT_SCHEME = random_measurementScheme(QNUMBER, amount=M)


# print("Random Measurement Scheme: ", RANDOM_MEASUREMENT_SCHEME)

def calculate_renyi2_IDEAL(dm):
    return -np.log2(np.trace(dm @ dm).real)


STANDARD_PURITY = [calculate_renyi2_IDEAL(dm) for dm in TEST_DM]
STANDARD_TRACE = [np.trace(dm).real for dm in TEST_DM]


measurementDMs: List[List[List[List[int]]]] = []
"""
Samples -> List:4DMs -> List:4MeasurementOrientationsM -> List:4SamplesK -> List:4EigenvaluesN
"""
for density_matrix in TEST_DM:
    measurementDMs.append(
        [FAKESAMPLER.fake_sampling_dm(
            density_matrix, measure_times=K, measurement_orientation=measurement_orientation
        ) for measurement_orientation in RANDOM_MEASUREMENT_SCHEME]
    )

# print("Measurement DMs: ", measurementDMs[0])

total_time_cs = 0.0
total_time_random = 0.0

renyiEntropy_CS = []
renyiEntropy_randomMeasurement = []

for measurementDM in tqdm(measurementDMs):
    renyiCalculator = RenyiEntropy(RANDOM_MEASUREMENT_SCHEME, measurementDM)

    start_time = time.time()
    renyiEntropy_CS.append(renyiCalculator.calculateRenyiEntropy(classical_shadow=True))
    total_time_cs += time.time() - start_time

    start_time = time.time()
    renyiEntropy_randomMeasurement.append(renyiCalculator.calculateRenyiEntropy(classical_shadow=False))
    total_time_random += time.time() - start_time

print("Standard Trace: ", STANDARD_TRACE)
print("Standard Purity: ", STANDARD_PURITY)
print(f'------------------- CURRENT M: {M}, K: {K} -------------------')
print(f"Total time for renyiEntropy_CS: {total_time_cs:.6f} seconds")
print("CS: ", renyiEntropy_CS)
print(f"Total time for renyiEntropy_randomMeasurement: {total_time_random:.6f} seconds")
print("Random Measurement: ", renyiEntropy_randomMeasurement)  # Seems to be correct with M=100, K=1000
