from typing import List

import numpy as np

from src.python.Physics.generate_TEST_DM import main
from src.python.fake_sampler import FakeSampler, random_measurementScheme
from src.python.ShadowState import ShadowState

K = 1000
M = 1000
QNUMBER = 2
TIME_LIST = [0, 1]
TEST_DM = main(QNUMBER, TIME_LIST)

print("Standard DM:")
print(TEST_DM[-1])

FAKESAMPLER = FakeSampler(QNUMBER)
RANDOM_MEASUREMENT_SCHEME = random_measurementScheme(QNUMBER, amount=M)

shadowState = ShadowState(QNUMBER)

measurementResults_DMs = []
for dm in TEST_DM:
    measurementResults_DMs.append(
        [
            FAKESAMPLER.fake_sampling_dm(
                dm, measure_times=K, measurement_orientation=measurement_orientation
            ) for measurement_orientation in RANDOM_MEASUREMENT_SCHEME
        ]
    )

estimated_states = [
    shadowState.stateEstimation(RANDOM_MEASUREMENT_SCHEME, measurementResults)
    for measurementResults in measurementResults_DMs
]

print("Estimated DM:")
print(estimated_states[-1])
