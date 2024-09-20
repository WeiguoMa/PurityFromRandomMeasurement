import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.python.Physics.generate_TEST_DM import pseudo_random_DM
from src.python.fake_sampler import FakeSampler, random_measurementScheme
from src.python.ShadowState import ShadowState

K = 1000
M = 1000
QNUMBER = 2
TEST_DM = pseudo_random_DM(QNUMBER, numPure=1, numMixed=1)

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
print("TRACE:", np.trace(estimated_states[-1]))
print("TRACE2:", np.trace(np.matmul(estimated_states[-1], estimated_states[-1])))
print("RENYI2:", -np.log2(np.trace(np.matmul(estimated_states[-1], estimated_states[-1]))))
