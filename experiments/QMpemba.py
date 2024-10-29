import os
import sys
import numpy as np
from typing import List, Union, Optional, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from qutip_qip.operations import ry
import pickle

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

    def slicesRun(self, rhos: List[Qobj], M: int, K: int, epoches: int = 1):
        """
        Args:
            rhos: List of density matrices of Full system.
            M: Number of measurement schemes.
            K: Number of measurements for each scheme.
            epoches: Number of epoches to run.
        Return:
            renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS, renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS
            shape -> (epoches, slicesNum, 6)
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

    def theta_timeRun(self, rhos: List[List[Qobj]], M: int, K: int, epoches: int = 1):
        """
        Args:
            rhos: List of density matrices of Full system. Outer-[] for thetas; Inner-[] for timeList;
            M: Number of measurement schemes.
            K: Number of measurements for each scheme.
            epoches: Number of epoches to run.
        """
        _results_thetas_epoches_times = np.array([self.slicesRun(rho, M, K, epoches) for rho in rhos])
        return _results_thetas_epoches_times.transpose(1, 0, 2, 3)      # result_epoches_thetas_times


class SimpleModel:
    def __init__(self,
                 qNumber: int,
                 subA: List[int],
                 thetas: Union[List[float], np.ndarray],
                 coupling_strength: float = 1,
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
            self.quenchHamiltonian = self.quench_hamiltonian(coupling_strength, zzCoefficient)
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

    def ferromagnet_state(self, thetas: Union[List[float], np.ndarray]):
        return [tensor([ry(theta) * basis(2, 0) for _ in range(self.qNumber)]) for theta in thetas]

    def ferromagnet_state_superposition(self, thetas: Union[List[float], np.ndarray]):
        return [
            1 / np.sqrt(2) *
            (tensor([ry(theta) * basis(2, 0) for _ in range(self.qNumber)])
                - tensor([ry(-theta) * basis(2, 0)for _ in range(self.qNumber)]))
            for theta in thetas
        ]

    def quench_hamiltonian(self, coupling_strength: float = 1.0, zzCoefficient: float = 0.0):
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

        return coupling_strength * H

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


def data_preparation(results: np.ndarray, **kwargs) -> Dict:
    """
    Args:
        results: Shape - (epoches, thetas, times, 6);
    """
    _RESULT = {
        key: value for key, value in kwargs.items()
    }
    epoches = results.shape[0]
    renyi_rhoA_ideal, renyi_rhoA_hamming, renyi_rhoA_CS = results[:, :, :, 0], results[:, :, :, 1], results[:, :, :, 2]
    renyi_rhoAQ_ideal, renyi_rhoAQ_hamming, renyi_rhoAQ_CS = results[:, :, :, 3], results[:, :, :, 4], results[:, :, :, 5]

    _RESULT['AQ_A_IDEAL'] = np.mean(renyi_rhoAQ_ideal - renyi_rhoA_ideal, axis=0)

    _RESULT['AQ_A_HAMMING'] = np.mean(renyi_rhoAQ_hamming - renyi_rhoA_hamming, axis=0)
    _RESULT['AQ_A_CS'] = np.mean(renyi_rhoAQ_CS - renyi_rhoA_CS, axis=0)
    _RESULT['AQ_A_HAMMING_ERRORBAR'] = np.std(np.abs(renyi_rhoAQ_hamming - renyi_rhoA_hamming), axis=0) / np.sqrt(epoches)
    _RESULT['AQ_A_CS_ERRORBAR'] = np.std(np.abs(renyi_rhoAQ_CS - renyi_rhoA_CS), axis=0) / np.sqrt(epoches)

    _RESULT['A_CS_IDEAL'] = np.mean(np.abs(renyi_rhoA_CS - renyi_rhoA_ideal), axis=0)
    _RESULT['A_HAMMING_IDEAL'] = np.mean(np.abs(renyi_rhoA_hamming - renyi_rhoA_ideal), axis=0)
    _RESULT['AQ_CS_IDEAL'] = np.mean(np.abs(renyi_rhoAQ_CS - renyi_rhoAQ_ideal), axis=0)
    _RESULT['AQ_HAMMING_IDEAL'] = np.mean(np.abs(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal), axis=0)
    _RESULT['A_CS_IDEAL_ERRORBAR'] = np.std(np.abs(renyi_rhoA_CS - renyi_rhoA_ideal), axis=0) / np.sqrt(epoches)
    _RESULT['A_HAMMING_IDEAL_ERRORBAR'] = np.std(np.abs(renyi_rhoA_hamming - renyi_rhoA_ideal), axis=0) / np.sqrt(epoches)
    _RESULT['AQ_CS_IDEAL_ERRORBAR'] = np.std(np.abs(renyi_rhoAQ_CS - renyi_rhoAQ_ideal), axis=0) / np.sqrt(epoches)
    _RESULT['AQ_HAMMING_IDEAL_ERRORBAR'] = np.std(np.abs(renyi_rhoAQ_hamming - renyi_rhoAQ_ideal), axis=0) / np.sqrt(epoches)

    return _RESULT


def plot_EA_Hamming_CS_THETAS(result, save: bool = False):
    plt.figure(figsize=(10, 6), dpi=300)

    theta_list, time_list = result.get('THETA_LIST'), [t * 1000 for t in result.get('TIME_LIST')]
    aq_a_ideal, aq_a_hamming, aq_a_cs = result.get('AQ_A_IDEAL'), result.get('AQ_A_HAMMING'), result.get('AQ_A_CS')
    aq_a_hamming_error, aq_a_cs_error = result.get('AQ_A_HAMMING_ERRORBAR'), result.get('AQ_A_CS_ERRORBAR')

    for i, theta in enumerate(theta_list):
        plt.errorbar(time_list, aq_a_ideal[i], label=result["THETA_LABELS"][i], fmt='o', capsize=1, linestyle='-')

        plt.errorbar(time_list, aq_a_hamming[i], yerr=aq_a_hamming_error[i], fmt='^', capsize=1, linestyle='--', alpha=0.5)
        plt.errorbar(time_list, aq_a_cs[i], yerr=aq_a_cs_error[i], fmt='*', capsize=1, linestyle='-.', alpha=0.5)

    _custom_lines = [
        Line2D([0], [0], linestyle='--', marker='^', label='Hamming'),
        Line2D([0], [0], linestyle='-.', marker='*', label='Classical Shadow')
    ]

    plt.xlabel("Time ($ns$)", fontsize=20)
    plt.ylabel("$\\Delta S$", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f"EA for N{result.get('QNUMBER_TOTAL')}-A{result.get('SUBSYSTEM_A')}, M={result['M']}, K={result['K']}", fontsize=18)
    plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + _custom_lines, fontsize=14)
    plt.tight_layout()

    if save:
        save_location = f"../figures/QMpemba_N{result.get('QNUMBER_TOTAL')}_A{result.get('SUBSYSTEM_A')}_M{result['M']}_N{result['K']}.pdf"
        plt.savefig(save_location)


if __name__ == '__main__':
    QNUMBER_TOTAL, SUBSYSTEM_A = 10, [3, 4]
    INT_SLICES = 20

    ZZ_COEFFICIENT = 0
    COUPLING_STRENGTH = 50      # MHz

    TIME_MAX, TIME_SLICES = 50e-3, 50      # \mu s

    M, K = 100, 100
    EPOCHES = 50

    THETA_LABELS = [r'$4/5$', r'$\pi/3$', r'$3/2$']
    THETA_LIST = [4/5, np.pi/3, 3 / 2]
    TIME_LIST = np.linspace(0, TIME_MAX, TIME_SLICES)

    print(f'QNUMBER_TOTAL: {QNUMBER_TOTAL}, SUBSYSTEM_A: {SUBSYSTEM_A},'
          f' M: {M}, K: {K}, INT_SLICES: {INT_SLICES}, TIME_MAX: {TIME_MAX}, THETA_LABELS: {THETA_LABELS}')

    model = SimpleModel(
        QNUMBER_TOTAL, SUBSYSTEM_A, THETA_LIST,
        coupling_strength=COUPLING_STRENGTH, zzCoefficient=ZZ_COEFFICIENT,
        superposition=True, quench=True, timeList=TIME_LIST
    )

    eaCalculator = EntanglementAsymmetry(QNUMBER_TOTAL, subA=SUBSYSTEM_A, L=INT_SLICES, Q=model.Q)

    results = eaCalculator.theta_timeRun(model.rhos_with_time_ferroSuperpositionRho, M, K, epoches=EPOCHES)
    format_results = data_preparation(results=results,
                                      M=M, K=K, INT_SLICES=INT_SLICES,
                                      THETA_LIST=THETA_LIST, THETA_LABELS=THETA_LABELS,
                                      QNUMBER_TOTAL=QNUMBER_TOTAL, SUBSYSTEM_A=SUBSYSTEM_A,
                                      TIME_SLICES=TIME_SLICES, TIME_MAX=TIME_MAX, TIME_LIST=TIME_LIST,
                                      COUPLING_STRENGTH=COUPLING_STRENGTH, EPOCHES=EPOCHES)

    with open(f'../data/QMpemba_M{M}_K{K}_N{QNUMBER_TOTAL}_A{SUBSYSTEM_A}.pkl', 'wb') as f:
        pickle.dump(format_results, f)

    plot_EA_Hamming_CS_THETAS(format_results, save=True)
