import sys
import os
import numpy as np
from typing import List
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from qutip import tensor, sigmaz, qeye, Qobj, sigmax, sigmay, basis, ket2dm
import matplotlib.pyplot as plt
from src.python.Physics.generate_TEST_DM import ferromagnet_state
from src.python.fake_sampler import FakeSampler, random_measurementScheme
from src.python.RenyiEntropy import RenyiEntropy

__ket0__, __ket1__ = basis(2, 0), basis(2, 1)
__ketDict__ = {
    'X0': ket2dm(__ket0__ + __ket1__),
    'X1': ket2dm(__ket0__ - __ket1__),
    'Y0': ket2dm(__ket0__ + 1j * __ket1__),
    'Y1': ket2dm(__ket0__ - 1j * __ket1__),
    'Z0': ket2dm(__ket0__),
    'Z1': ket2dm(__ket1__)
}


def measure_projector(schemes: List[int], outcomes: List[int]):
    def single_projector(scheme: int, outcome: int):
        return (qeye(2) + (-1) ** outcome * [sigmax(), sigmay(), sigmaz()][scheme]) / 2

    return tensor([single_projector(scheme, outcome) for scheme, outcome in zip(schemes, outcomes)])


def measured_state(rho, measureScheme: List[int], measureOutcome: List[int]):
    _projector = measure_projector(measureScheme, measureOutcome)
    _state = _projector * rho * _projector.dag()
    return _state / _state.tr()


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
    for alpha in alphaList:
        sumRho += evolve_state(rhoA, QA, alpha)
    return sumRho / L


def MIPTBasedMeasureOutcomes(rhoA: Qobj, QA: Qobj, L: int, sampler: FakeSampler,
                             measureSchemes: List[List[int]], measureOutcomesRhoA: List[List[List[int]]]) -> List[
    List[List[int]]]:
    measureOutcomesRhoAQ = []
    for idx, measureScheme in enumerate(measureSchemes):
        outcomeRhoAQ = []
        for outcomeRhoA in measureOutcomesRhoA[idx]:
            _state4Evolve = measured_state(rhoA, measureScheme, outcomeRhoA)
            _evolvedState = approx_rhoAQ(_state4Evolve, QA, L)
            outcomeRhoAQ.append(
                sampler.fake_sampling_dm(_evolvedState, measure_times=1, measurement_orientation=measureScheme)[0]
            )
        measureOutcomesRhoAQ.append(outcomeRhoAQ)
    return measureOutcomesRhoAQ


def calculateRenyi2(dm: Qobj):
    return -np.log2((dm * dm).tr())


def calculatePurity(dm: Qobj):
    return (dm * dm).tr()


if __name__ == '__main__':
    qn = 4
    M, K = 100, 100
    fakesampler = FakeSampler(system_size=qn)
    MEASURE_SCHEME = random_measurementScheme(qn, amount=M)

    L = 15
    Q = generate_Q(qn)

    thetaList = np.linspace(start=0.0, stop=np.pi, num=20)
    REList = []
    REIdealList = []
    for theta in tqdm(thetaList):
        rhoA = rho4Test(qn, theta)
        MEASURE_OUTCOMES_RhoA = measurementOutcomes(fakesampler, rhoA=rhoA, measurementScheme=MEASURE_SCHEME, K=K)
        renyi_rhoQ = RenyiEntropy(measurementScheme=MEASURE_SCHEME,
                                  measurementResults=MEASURE_OUTCOMES_RhoA).calculateRenyiEntropy()

        rhoAQ = approx_rhoAQ(rhoA, Q, L)
        MEASURE_OUTCOMES_RhoAQ = MIPTBasedMeasureOutcomes(rhoA, Q, L, fakesampler, MEASURE_SCHEME,
                                                          MEASURE_OUTCOMES_RhoA)
        renyi_rhoAQ = RenyiEntropy(measurementScheme=MEASURE_SCHEME,
                                   measurementResults=MEASURE_OUTCOMES_RhoAQ).calculateRenyiEntropy()

        REList.append(renyi_rhoAQ - renyi_rhoQ)
        REIdealList.append(calculateRenyi2(rhoAQ) - calculateRenyi2(rhoA))

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(thetaList, REList, label='Renyi Entropy with Global dePhasing')
    plt.plot(thetaList, REIdealList, label='Renyi Entropy Ideal')

    plt.xticks([0, np.pi / 2, np.pi], ['0', r'$\pi/2$', r'$\pi$'], fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel(r'$\theta$', fontsize=20)
    plt.ylabel(r'$\Delta S$', fontsize=20)
    plt.title(f"Case for L={L}, M={M}, K={K}", fontsize=20)

    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig('../figures/evolutionRhoAQ_sequential.pdf')
