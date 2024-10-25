import numpy as np
from typing import List
from tqdm import tqdm

from qutip import tensor, sigmaz, qeye, Qobj, ptrace
import matplotlib.pyplot as plt
from src.python.Physics.generate_TEST_DM import ferromagnet_state
from src.python.fake_sampler import FakeSampler, random_measurementScheme
from src.python.RenyiEntropy import RenyiEntropy


def generate_Q(qnumber: int):
    Q = None
    for j in range(qnumber):
        op_list = [sigmaz() if i == j else qeye(2) for i in range(qnumber)]
        if Q is None:
            Q = tensor(op_list)
        else:
            Q += tensor(op_list)
    return Q / 2


def rho4Test(qnumber: int, theta: float):
    return ferromagnet_state(qnumber, theta)


def measurementOutcomes(sampler: FakeSampler, rhoA: Qobj, measurementScheme: List, K: int):
    outcomes = [
        sampler.fake_sampling_dm(rhoA, measure_times=K, measurement_orientation=orientation)
        for orientation in measurementScheme
    ]
    return outcomes


def evolve_state(rhoA, QA, _alpha):
    _rhoAp = (-1j * _alpha * QA).expm() * rhoA * (1j * _alpha * QA).expm()
    return _rhoAp


def approx_rhoAQ(rhoA, QA, L):
    alphaList = np.linspace(-np.pi, np.pi, num=L)
    sumRho = 0
    for alpha in tqdm(alphaList):
        sumRho += evolve_state(rhoA, QA, alpha)
    return sumRho / L


def calculateRenyi2(dm: Qobj):
    return -np.log2((dm * dm).tr())


def calculatePurity(dm: Qobj):
    return (dm * dm).tr()


if __name__ == '__main__':
    qn = 4
    M, K = 100, 100
    fakesampler = FakeSampler(system_size=qn)
    MEASURE_SCHEME = random_measurementScheme(qn, amount=M)

    L = 50
    Q = generate_Q(qn)

    thetaList = np.linspace(start=0.0, stop=np.pi, num=100)
    REList = []
    REIdealList = []
    for theta in thetaList:
        rhoA = rho4Test(qn, theta)
        MEASURE_OUTCOMES_RhoA = measurementOutcomes(fakesampler, rhoA=rhoA, measurementScheme=MEASURE_SCHEME, K=K)
        renyi_rhoQ = RenyiEntropy(measurementScheme=MEASURE_SCHEME, measurementResults=MEASURE_OUTCOMES_RhoA).calculateRenyiEntropy()

        rhoAQ = approx_rhoAQ(rhoA, Q, L)
        MEASURE_OUTCOMES_RhoAQ = measurementOutcomes(fakesampler, rhoA=rhoAQ, measurementScheme=MEASURE_SCHEME, K=K)
        renyi_rhoAQ = RenyiEntropy(measurementScheme=MEASURE_SCHEME, measurementResults=MEASURE_OUTCOMES_RhoAQ).calculateRenyiEntropy()

        REList.append(renyi_rhoAQ - renyi_rhoQ)
        REIdealList.append(calculateRenyi2(rhoAQ) - calculateRenyi2(rhoA))

    plt.plot(thetaList, REList, label='Renyi Entropy with Hamming Distance')
    plt.plot(thetaList, REIdealList, label='Renyi Entropy Ideal')
    plt.xticks([0, np.pi / 2, np.pi], ['0', r'$\pi/2$', r'$\pi$'])
    plt.show()