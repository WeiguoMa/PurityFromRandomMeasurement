from typing import List
from tqdm import tqdm

import numpy as np

from src.python.Physics.generate_TEST_DM import main
from src.python.RenyiEntropy import RenyiEntropy
from src.python.fake_sampler import FakeSampler, random_measurementScheme

K = 10000
M = 100
QNUMBER = 4
TIME_LIST = [0, 1, 2, 3, 4, 5]
TEST_DM = main(QNUMBER, TIME_LIST)
FAKESAMPLER = FakeSampler(QNUMBER)
RANDOM_MEASUREMENT_SCHEME = random_measurementScheme(QNUMBER, amount=M)

# print("Random Measurement Scheme: ", RANDOM_MEASUREMENT_SCHEME)

def calculate_renyi2_IDEAL(dm):
    return -np.log2(np.trace(dm @ dm).real)


STANDARD_PURITY = [calculate_renyi2_IDEAL(dm) for dm in TEST_DM]
print("Standard Purity: ", STANDARD_PURITY)

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

renyiEntropy_CS = []
renyiEntropy_randomMeasurement = []
for measurementDM in tqdm(measurementDMs):
    renyiCalculator = RenyiEntropy(RANDOM_MEASUREMENT_SCHEME, measurementDM)
    renyiEntropy_randomMeasurement.append(renyiCalculator.calculateRenyiEntropy(classical_shadow=False))
    renyiEntropy_CS.append(renyiCalculator.calculateRenyiEntropy(classical_shadow=True))

print(f'------------------- CURRENT M: {M}, K: {K} -------------------')
print("CS: ", renyiEntropy_CS)
print("Random Measurement: ", renyiEntropy_randomMeasurement)       # Seems to be correct with M=100, K=1000
