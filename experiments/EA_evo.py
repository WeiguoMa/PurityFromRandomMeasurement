import os
import sys
import numpy as np
from typing import List
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from qutip import tensor, sigmaz, qeye, Qobj
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
    for alpha in alphaList:
        sumRho += evolve_state(rhoA, QA, alpha)
    return sumRho / L


def calculateRenyi2(dm: Qobj):
    return -np.log2((dm * dm).tr())


def calculatePurity(dm: Qobj):
    return (dm * dm).tr()


def thetaRun(qn: int, thetaNum: int, M: int, K: int, L: int, epoches: int = 1):
    """
    Return:
        renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS, renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS
        shape -> (epoches, thetaNum, 6)
    """
    Q = generate_Q(qn)
    fakesampler = FakeSampler(system_size=qn)
    MEASURE_SCHEME = random_measurementScheme(qn, amount=M)

    _theta_values = np.linspace(0.0, np.pi, thetaNum)
    _results_epoches = np.empty((epoches, thetaNum, 6))

    for epoch in tqdm(range(epoches)):
        _results = np.empty((thetaNum, 6))

        for i, theta in enumerate(_theta_values):
            rhoA = rho4Test(qn, theta)
            MEASURE_OUTCOMES_RhoA = measurementOutcomes(fakesampler, rhoA=rhoA, measurementScheme=MEASURE_SCHEME, K=K)
            renyi_rhoA_ideal = calculateRenyi2(rhoA)

            renyi_rhoA = RenyiEntropy(measurementScheme=MEASURE_SCHEME, measurementResults=MEASURE_OUTCOMES_RhoA)
            renyi_rhoA_hamming = renyi_rhoA.calculateRenyiEntropy()
            renyi_rhoA_CS = renyi_rhoA.calculateRenyiEntropy(classical_shadow=True)

            rhoAQ = approx_rhoAQ(rhoA, Q, L)
            MEASURE_OUTCOMES_RhoAQ = measurementOutcomes(fakesampler, rhoA=rhoAQ, measurementScheme=MEASURE_SCHEME, K=K)
            renyi_rhoAQ_ideal = calculateRenyi2(rhoAQ)

            renyi_rhoAQ = RenyiEntropy(measurementScheme=MEASURE_SCHEME, measurementResults=MEASURE_OUTCOMES_RhoAQ)
            renyi_rhoAQ_hamming = renyi_rhoAQ.calculateRenyiEntropy()
            renyi_rhoAQ_CS = renyi_rhoAQ.calculateRenyiEntropy(classical_shadow=True)

            _results[i] = [
                renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS,
                renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS
            ]

        _results_epoches[epoch] = _results

    return _theta_values, _results_epoches


def analyze_results(results_epoches, theta_values, fig1_loc: str, fig2_loc: str):
    epoches = results_epoches.shape[0]

    renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS = results_epoches[:, :, 0], results_epoches[:, :, 1], results_epoches[:, :, 2]
    renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS = results_epoches[:, :, 3], results_epoches[:, :, 4], results_epoches[:, :, 5]

    v1 = np.mean(renyi_rhoAQ_ideal - renyi_rhoA_ideal, axis=0)
    v6 = np.mean(renyi_rhoAQ_hamming - renyi_rhoA_hamming, axis=0)
    v7 = np.mean(renyi_rhoAQ_CS - renyi_rhoA_CS, axis=0)

    v2 = np.mean(renyi_rhoA_CS - renyi_rhoA_ideal, axis=0)
    v3 = np.mean(renyi_rhoA_hamming - renyi_rhoA_ideal, axis=0)
    v4 = np.mean(renyi_rhoAQ_CS - renyi_rhoAQ_ideal, axis=0)
    v5 = np.mean(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal, axis=0)

    v2_std = np.std(renyi_rhoA_CS - renyi_rhoA_ideal, axis=0) / np.sqrt(epoches)
    v3_std = np.std(renyi_rhoA_hamming - renyi_rhoA_ideal, axis=0) / np.sqrt(epoches)
    v4_std = np.std(renyi_rhoAQ_CS - renyi_rhoAQ_ideal, axis=0) / np.sqrt(epoches)
    v5_std = np.std(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal, axis=0) / np.sqrt(epoches)
    v6_std = np.std(renyi_rhoAQ_hamming - renyi_rhoA_hamming, axis=0) / np.sqrt(epoches)
    v7_std = np.std(renyi_rhoAQ_CS - renyi_rhoA_CS, axis=0) / np.sqrt(epoches)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.errorbar(theta_values, v1, yerr=0, label='Ideal', fmt='o', capsize=1, linestyle='-')
    plt.errorbar(theta_values, v6, yerr=v6_std, label='Hamming', fmt='^', capsize=1, linestyle='-')
    plt.errorbar(theta_values, v7, yerr=v7_std, label='Classical Shadow', fmt='*', capsize=1, linestyle='-')
    plt.xlabel(r"$\theta$", fontsize=20)
    plt.ylabel(r"$\Delta S$", fontsize=20)
    plt.xticks([0, np.pi / 2, np.pi], ['0', r'$\pi/2$', r'$\pi$'], fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Entanglement Asymmetry with Methods", fontsize=22)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig1_loc)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.errorbar(theta_values, v2, yerr=v2_std, label='A - Classical Shadow', fmt='o', capsize=1, linestyle='-')
    plt.errorbar(theta_values, v3, yerr=v3_std, label='A - Hamming', fmt='^', capsize=1, linestyle='-')
    plt.errorbar(theta_values, v4, yerr=v4_std, label='AQ - Classical Shadow', fmt='*', capsize=1, linestyle='--')
    plt.errorbar(theta_values, v5, yerr=v5_std, label='AQ - Hamming', fmt='+', capsize=1, linestyle='--')
    plt.xlabel(r"$\theta$", fontsize=20)
    plt.ylabel(r"Difference $\Delta S$", fontsize=20)
    plt.xticks([0, np.pi / 2, np.pi], ['0', r'$\pi/2$', r'$\pi$'], fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Difference of Entanglement Asymmetry with Methods", fontsize=22)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig2_loc)


if __name__ == '__main__':
    qn = 4
    M, K = 100, 100

    T = 50
    L = 100

    thetaSlice = 100

    thetaList, results = thetaRun(qn=qn, thetaNum=thetaSlice, M=M, K=K, L=L, epoches=T)
    print(results.shape)

    analyze_results(results, thetaList, '../figures/evolutionRhoAQ_v1.pdf', '../figures/evolutionRhoAQ_v2.pdf')

