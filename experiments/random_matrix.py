import os
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.python.Physics.generate_TEST_DM import pseudo_random_DM
from src.python.RenyiEntropy import RenyiEntropy
from src.python.fake_sampler import FakeSampler, random_measurementScheme

K = 1000
M = 1000
QNUMBER = 5
n_repeats = 5

TEST_DM = pseudo_random_DM(QNUMBER, numPure=1, numMixed=7)

FAKESAMPLER = FakeSampler(QNUMBER)


def calculate_renyi2_IDEAL(dm):
    return -np.log2(np.trace(dm @ dm).real)


STANDARD_PURITY = [calculate_renyi2_IDEAL(dm) for dm in TEST_DM]
STANDARD_TRACE = [np.trace(dm).real for dm in TEST_DM]

renyiEntropy_CS_repeats = []
renyiEntropy_randomMeasurement_repeats = []
time_cs_repeats = []
time_random_repeats = []

for _ in tqdm(range(n_repeats), desc="Repeating Calculations"):
    RANDOM_MEASUREMENT_SCHEME = random_measurementScheme(QNUMBER, amount=M)
    measurementDMs: List[List[List[List[int]]]] = []
    for density_matrix in TEST_DM:
        measurementDMs.append(
            [FAKESAMPLER.fake_sampling_dm(
                density_matrix, measure_times=K, measurement_orientation=measurement_orientation
            ) for measurement_orientation in RANDOM_MEASUREMENT_SCHEME]
        )

    renyiEntropy_CS = []
    renyiEntropy_randomMeasurement = []
    total_time_cs = 0.0
    total_time_random = 0.0

    for measurementDM in measurementDMs:
        renyiCalculator = RenyiEntropy(RANDOM_MEASUREMENT_SCHEME, measurementDM)

        start_time = time.time()
        renyiEntropy_CS.append(renyiCalculator.calculateRenyiEntropy(classical_shadow=True))
        total_time_cs += time.time() - start_time

        start_time = time.time()
        renyiEntropy_randomMeasurement.append(renyiCalculator.calculateRenyiEntropy(classical_shadow=False))
        total_time_random += time.time() - start_time

    renyiEntropy_CS_repeats.append(renyiEntropy_CS)
    renyiEntropy_randomMeasurement_repeats.append(renyiEntropy_randomMeasurement)
    time_cs_repeats.append(total_time_cs)
    time_random_repeats.append(total_time_random)

renyiEntropy_CS_mean = np.mean(renyiEntropy_CS_repeats, axis=0)
renyiEntropy_CS_std = np.std(renyiEntropy_CS_repeats, axis=0)

renyiEntropy_randomMeasurement_mean = np.mean(renyiEntropy_randomMeasurement_repeats, axis=0)
renyiEntropy_randomMeasurement_std = np.std(renyiEntropy_randomMeasurement_repeats, axis=0)

time_cs_mean = np.mean(time_cs_repeats)
time_random_mean = np.mean(time_random_repeats)

print("Standard Trace: ", STANDARD_TRACE)
print("Standard Purity: ", STANDARD_PURITY)

print(f'------------------- CURRENT M: {M}, K: {K} -------------------')
print(f"Average time for renyiEntropy_CS: {time_cs_mean:.6f} seconds")
print(f"Average time for renyiEntropy_randomMeasurement: {time_random_mean:.6f} seconds")

x_labels = [f'DM {i + 1}' for i in range(len(TEST_DM))]

plt.figure(figsize=(10, 6))
for i, purity in enumerate(STANDARD_PURITY):
    plt.hlines(y=purity, xmin=i - 0.2, xmax=i + 0.2, colors='green', linestyles='dashed',
               label='Standard Purity' if i == 0 else "")
plt.errorbar(x_labels, renyiEntropy_CS_mean, yerr=renyiEntropy_CS_std, fmt='o', label='Renyi Entropy Shadow', capsize=5)
plt.errorbar(x_labels, renyiEntropy_randomMeasurement_mean, yerr=renyiEntropy_randomMeasurement_std, fmt='o',
             label='Renyi Entropy Hamming', capsize=5)
plt.ylim(-0.05, 1.05)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Density Matrices', fontsize=20)
plt.ylabel('Renyi Entropy', fontsize=20)
plt.title('Renyi Entropy with Error Bars', fontsize=22)
plt.legend(fontsize=17)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../figures/random_matrix_M_{M}_K_{K}_qn_{QNUMBER}.pdf")
