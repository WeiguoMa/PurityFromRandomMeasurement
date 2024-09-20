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

K = 100
M = 1000
QNUMBER = 4
n_repeats = 10

TEST_DM = pseudo_random_DM(QNUMBER, numPure=1, numMixed=7)

FAKESAMPLER = FakeSampler(QNUMBER)
RANDOM_MEASUREMENT_SCHEME = random_measurementScheme(QNUMBER, amount=M)


def calculate_renyi2_IDEAL(dm):
    return -np.log2(np.trace(dm @ dm).real)


STANDARD_PURITY = [calculate_renyi2_IDEAL(dm) for dm in TEST_DM]
STANDARD_TRACE = [np.trace(dm).real for dm in TEST_DM]

renyiEntropy_CS_repeats = []
renyiEntropy_randomMeasurement_repeats = []
time_cs_repeats = []
time_random_repeats = []

for _ in tqdm(range(n_repeats), desc="Repeating Calculations"):
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
plt.xlabel('Density Matrices')
plt.ylabel('Renyi Entropy')
plt.title('Renyi Entropy with Error Bars')
plt.legend()
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
bar_width = 0.35
index = np.arange(2)
time_means = [time_cs_mean, time_random_mean]

ax.bar(index, time_means, bar_width, color=['blue', 'green'], label=['Shadow Time', 'Hamming Time'])

ax.set_xlabel('Entropy Calculation Type')
ax.set_ylabel('Time (seconds)')
ax.set_title('Time Consumption for Renyi Entropy Calculations')
ax.set_xticks(index)
ax.set_xticklabels(['Shadow', 'Hamming'])
plt.show()
