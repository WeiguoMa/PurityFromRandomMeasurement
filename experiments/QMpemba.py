import os
import sys
import numpy as np
from typing import List, Union, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from qutip import tensor, sigmaz, qeye, Qobj, ket2dm, sigmay, sigmax, basis, ptrace, sesolve, mesolve
from src.python.fake_sampler import FakeSampler, random_measurementScheme
from src.python.RenyiEntropy import RenyiEntropy


class EntanglementAsymmetry:
    def __init__(self, qNumberTotal: int, subA: List[int], L: int, Q: Qobj):
        self.qNumberTotal = qNumberTotal
        self.qNumber = len(subA)
        self.subA = subA

        self.sampler = FakeSampler(system_size=self.qNumber)
        self.L = L

        self.Q = Q
        self.alphaList = np.linspace(-np.pi, np.pi, num=self.L)
        self.alphaU = self.pre_alphaU(self.Q)

    @staticmethod
    def idealRenyi2(dm: Qobj):
        return -np.log2((dm * dm).tr())

    def pre_alphaU(self, Q):
        return [((-1j * alpha * Q).expm(), (1j * alpha * Q).expm()) for alpha in self.alphaList]

    def approx_rhoAQ(self, rhoA):
        sumRho: Qobj = 0
        for minus, plus in self.alphaU:
            sumRho += minus * rhoA * plus
        return sumRho / self.L

    def measurementOutcomes(self,
                            K: int,
                            rhoA: Qobj,
                            measurementScheme: List[List[int]]) -> List[List[List[int]]]:
        outcomes = [
            self.sampler.fake_sampling_dm(rhoA, measure_times=K, measurement_orientation=orientation)
            for orientation in measurementScheme
        ]
        return outcomes

    def timeRun(self, rhos: List[Qobj], M: int, K: int, epoches: int = 1):
        """
        Args:
            rhos: List of density matrices of Full system.
            M: Number of measurement schemes.
            K: Number of measurements for each scheme.
            epoches: Number of epoches to run.
        Return:
            renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS, renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS
            shape -> (epoches, thetaNum, 6)
        """
        num_rhoAs = len(rhos)
        _results_epoches = np.empty((epoches, num_rhoAs, 6))

        rhoAs = [ptrace(rho, self.subA) for rho in rhos]

        for epoch in tqdm(range(epoches)):
            _results = np.empty((num_rhoAs, 6))
            for i, rhoA in enumerate(rhoAs):
                MEASURE_SCHEME = random_measurementScheme(self.qNumber, amount=M)

                MEASURE_OUTCOMES_RhoA = self.measurementOutcomes(K=K, rhoA=rhoA, measurementScheme=MEASURE_SCHEME)
                renyi_rhoA_ideal = self.idealRenyi2(rhoA)

                renyi_rhoA = RenyiEntropy(measurementScheme=MEASURE_SCHEME, measurementResults=MEASURE_OUTCOMES_RhoA)
                renyi_rhoA_hamming = renyi_rhoA.calculateRenyiEntropy()
                renyi_rhoA_CS = renyi_rhoA.calculateRenyiEntropy(classical_shadow=True)

                rhoAQ = self.approx_rhoAQ(rhoA)
                MEASURE_OUTCOMES_RhoAQ = self.measurementOutcomes(K=K, rhoA=rhoAQ, measurementScheme=MEASURE_SCHEME)
                renyi_rhoAQ_ideal = self.idealRenyi2(rhoAQ)

                renyi_rhoAQ = RenyiEntropy(measurementScheme=MEASURE_SCHEME, measurementResults=MEASURE_OUTCOMES_RhoAQ)
                renyi_rhoAQ_hamming = renyi_rhoAQ.calculateRenyiEntropy()
                renyi_rhoAQ_CS = renyi_rhoAQ.calculateRenyiEntropy(classical_shadow=True)

                _results[i] = [
                    renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS,
                    renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS
                ]

            _results_epoches[epoch] = _results

        return _results_epoches


class SimpleModel:
    def __init__(self,
                 qNumber: int,
                 subA: List[int],
                 thetas: Union[List[float], np.ndarray],
                 zzCoefficient: float = 0,
                 superposition: bool = False,
                 quench: bool = False,
                 timeList: Union[List[float], np.ndarray] = None):
        self.qNumber = qNumber
        self.qNumber_subA = len(subA)
        self.Q = self.generate_Q()

        if superposition:
            self.ferroSuperpositionState = self.ferromagnet_state_superposition(thetas)
            self.ferroSuperpositionRho = [ket2dm(state) for state in self.ferroSuperpositionState]
        else:
            self.ferroState = self.ferromagnet_state(thetas)
            self.ferroRho = [ket2dm(state) for state in self.ferroState]

        if quench:
            self.quenchHamiltonian = self.quench_hamiltonian(zzCoefficient)
            if timeList is None:
                raise ValueError('timeList is required for quench model')
            else:
                self.timeList = timeList
            self.states_with_time_ferroSuperpositionState = self.evolveTime(self.ferroSuperpositionState, self.timeList)
            self.rhos_with_time_ferroSuperpositionRho = [
                [ket2dm(stateTime) for stateTime in stateTheta]
                for stateTheta in self.states_with_time_ferroSuperpositionState
            ]

    def generate_Q(self) -> Qobj:
        Q: Qobj = 0
        for j in range(self.qNumber_subA):
            Q += tensor([sigmaz() if i == j else qeye(2) for i in range(self.qNumber_subA)])
        return Q / 2

    def ferroHamiltonian(self):
        sy_sum: Qobj = 0
        for j in range(self.qNumber):
            sy_sum += tensor([sigmay() if idx == j else qeye(2) for idx in range(self.qNumber)])
        return sy_sum

    def ferromagnet_state(self, thetas: Union[List[float], np.ndarray]):
        initialState = tensor([basis(2, 0) for _ in range(self.qNumber)])
        return [(-1j * theta / 2 * self.ferroHamiltonian()).expm() * initialState for theta in thetas]

    def ferromagnet_state_superposition(self, thetas: Union[List[float], np.ndarray]):
        initialState = tensor([basis(2, 0) for _ in range(self.qNumber)])
        sy_sum = self.ferroHamiltonian()

        return [
            1 / np.sqrt(2) *
            ((-1j * theta / 2 * sy_sum).expm() * initialState - (1j * theta / 2 * sy_sum).expm() * initialState)
            for theta in thetas
        ]

    def quench_hamiltonian(self, zzCoefficient: float = 0.0):
        H: Qobj = 0
        for j in range(self.qNumber - 1):
            sx_j_sx_j1 = tensor(
                [sigmax() if k == j or k == j + 1 else qeye(2) for k in range(self.qNumber)]
            )

            sy_j_sy_j1 = tensor(
                [sigmay() if k == j or k == j + 1 else qeye(2) for k in range(self.qNumber)]
            )

            sz_j_sz_j1 = tensor(
                [sigmaz() if k == j or k == j + 1 else qeye(2) for k in range(self.qNumber)]
            )

            H += - 1 / 4 * (sx_j_sx_j1 + sy_j_sy_j1 + zzCoefficient * sz_j_sz_j1)

        return H

    def evolveTime(self, initial_states: List[Qobj], timeList: List[float]) -> List[List[Qobj]]:
        """
        Return:
            Outer-[] -> For thetas;
            Inner-[] -> For timeList;
        """
        if initial_states[0].isket:
            return self.evolveTime_seSolve(initial_states, timeList)
        else:
            return self.evolveTime_meSolve(initial_states, timeList)

    def evolveTime_seSolve(self, initial_states: List[Qobj], timeList: List[float]) -> List[List[Qobj]]:
        return [sesolve(self.quenchHamiltonian, initial_state, timeList).states for initial_state in initial_states]

    def evolveTime_meSolve(self, initial_states: List[Qobj], timeList: List[float], c_ops: Optional = None):
        return [mesolve(self.quenchHamiltonian, initial_state, timeList, c_ops=c_ops).states for initial_state in
                initial_states]


def analyze_results(results_epoches, time_list, fig1_loc: str, fig2_loc: str):
    epoches = results_epoches.shape[0]

    renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS = results_epoches[:, :, 0], results_epoches[:, :, 1], results_epoches[:, :, 2]
    renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS = results_epoches[:, :, 3], results_epoches[:, :, 4], results_epoches[:, :, 5]

    v1 = np.mean(renyi_rhoAQ_ideal - renyi_rhoA_ideal, axis=0)
    v6 = np.mean(renyi_rhoAQ_hamming - renyi_rhoA_hamming, axis=0)
    v7 = np.mean(renyi_rhoAQ_CS - renyi_rhoA_CS, axis=0)

    v2 = np.mean(np.abs(renyi_rhoA_CS - renyi_rhoA_ideal), axis=0)
    v3 = np.mean(np.abs(renyi_rhoA_hamming - renyi_rhoA_ideal), axis=0)
    v4 = np.mean(np.abs(renyi_rhoAQ_CS - renyi_rhoAQ_ideal), axis=0)
    v5 = np.mean(np.abs(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal), axis=0)

    v2_std = np.std(np.abs(renyi_rhoA_CS - renyi_rhoA_ideal), axis=0) / np.sqrt(epoches)
    v3_std = np.std(np.abs(renyi_rhoA_hamming - renyi_rhoA_ideal), axis=0) / np.sqrt(epoches)
    v4_std = np.std(np.abs(renyi_rhoAQ_CS - renyi_rhoAQ_ideal), axis=0) / np.sqrt(epoches)
    v5_std = np.std(np.abs(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal), axis=0) / np.sqrt(epoches)
    v6_std = np.std(np.abs(renyi_rhoAQ_hamming - renyi_rhoA_hamming), axis=0) / np.sqrt(epoches)
    v7_std = np.std(np.abs(renyi_rhoAQ_CS - renyi_rhoA_CS), axis=0) / np.sqrt(epoches)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.errorbar(time_list, v1, yerr=0, label='Ideal', fmt='o', capsize=1, linestyle='-')
    plt.errorbar(time_list, v6, yerr=v6_std, label='Hamming', fmt='^', capsize=1, linestyle='-')
    plt.errorbar(time_list, v7, yerr=v7_std, label='Classical Shadow', fmt='*', capsize=1, linestyle='-')
    plt.xlabel(r"Time $t$", fontsize=20)
    plt.ylabel(r"$\Delta S$", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"Entanglement Asymmetry for M={M}, K={K}, L={10}", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig1_loc)

    plt.figure(figsize=(10, 6), dpi=300)
    plt.errorbar(time_list, v2, yerr=v2_std, label='A - Classical Shadow', fmt='o', capsize=1, linestyle='-')
    plt.errorbar(time_list, v3, yerr=v3_std, label='A - Hamming', fmt='^', capsize=1, linestyle='-')
    plt.errorbar(time_list, v4, yerr=v4_std, label='AQ - Classical Shadow', fmt='*', capsize=1, linestyle='--')
    plt.errorbar(time_list, v5, yerr=v5_std, label='AQ - Hamming', fmt='+', capsize=1, linestyle='--')
    plt.xlabel(r"Time $t$", fontsize=20)
    plt.ylabel(r"Difference of $S$", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"Difference of R\'enyi Entropy for M={M}, K={K}, L={10}", fontsize=18)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(fig2_loc)


if __name__ == '__main__':
    QNUMBER_TOTAL, SUBSYSTEM_A = 10, [0, 1]

    # THETA_SLICES = 100

    INT_SLICES = 10
    TIME_MAX, TIME_SLICES = 5, 50

    M, K = 100, 100

    THETA_LIST = [np.pi/4, np.pi / 2]
    TIME_LIST = np.linspace(0, TIME_MAX, TIME_SLICES)

    model = SimpleModel(QNUMBER_TOTAL, SUBSYSTEM_A, THETA_LIST, superposition=True, quench=True, timeList=TIME_LIST)

    eaCalculator = EntanglementAsymmetry(QNUMBER_TOTAL, subA=SUBSYSTEM_A, L=INT_SLICES, Q=model.Q)

    results0 = eaCalculator.timeRun(model.rhos_with_time_ferroSuperpositionRho[0], M, K, epoches=10)
    analyze_results(results0,
                    TIME_LIST,
                    '../figures/evolutionRhoAQ_time_v1_theta0.pdf',
                    '../figures/evolutionRhoAQ_time_v2_theta0.pdf')

    results1 = eaCalculator.timeRun(model.rhos_with_time_ferroSuperpositionRho[1], M, K, epoches=10)
    analyze_results(results1,
                    TIME_LIST,
                    '../figures/evolutionRhoAQ_time_v1_theta1.pdf',
                    '../figures/evolutionRhoAQ_time_v2_theta1.pdf')
