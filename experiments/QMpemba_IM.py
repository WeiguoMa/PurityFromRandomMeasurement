import os
import pickle
import sys
from typing import List, Union, Optional
from multiprocessing import Pool

import numpy as np
from qutip_qip.operations import ry
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from qutip import tensor, sigmaz, qeye, Qobj, ket2dm, sigmay, sigmax, basis, ptrace, sesolve, mesolve
from src.python.fake_sampler import FakeSampler, random_measurementScheme
from src.python.RenyiEntropy import RenyiEntropy
from QMpemba_DataAnalysis_plot import data_preparation_EA


__ket0__, __ket1__ = basis(2, 0), basis(2, 1)
__ketDict__ = {
    '00': 1 / np.sqrt(2) * (__ket0__ + __ket1__),
    '01': 1 / np.sqrt(2) * (__ket0__ - __ket1__),
    '10': 1 / np.sqrt(2) * (__ket0__ + 1j * __ket1__),
    '11': 1 / np.sqrt(2) * (__ket0__ - 1j * __ket1__),
    '20': __ket0__,
    '21': __ket1__
}


class EntanglementAsymmetry:
    def __init__(self,
                 qNumberTotal: int,
                 subA: List[int],
                 L: int, Q: Qobj,
                 classical_shadow: bool = False,
                 intermediate_measure: bool = False):
        self.qNumberTotal = qNumberTotal
        self.qNumber = len(subA)
        self.subA = subA

        self.sampler = FakeSampler(system_size=self.qNumber)
        self.L = L

        self.Q = Q
        self.alphaList = np.linspace(-np.pi, np.pi, num=self.L)
        self.alphaU = self.pre_alphaU(self.Q)

        self.classicalShadow = classical_shadow
        self.intermediateMeasure = intermediate_measure
        self.measuredEvolvedState = {}

    @staticmethod
    def idealRenyi2(dm: Qobj):
        return -np.log2((dm * dm).tr())

    @staticmethod
    def measured_state(measureScheme: List[int], measureOutcome: List[int]):
        _state = tensor(
            [__ketDict__[f'{scheme}{outcome}'] for scheme, outcome in zip(measureScheme, measureOutcome)]
        )
        return _state

    def pre_alphaU(self, Q):
        return [((-1j * alpha * Q).expm(), (1j * alpha * Q).expm()) for alpha in self.alphaList]

    def approx_rhoAQ(self, rhoA):
            sumRho: Qobj = 0
            for minus, plus in self.alphaU:
                sumRho += minus * rhoA * plus
            return sumRho / self.L

    def approx_stateAQ(self, stateA):
        sumState: Qobj = 0
        for minus, _ in self.alphaU:
            sumState += minus * stateA
        return sumState / self.L

    def approx_AQ(self, AQ: Qobj):
        if AQ.isket:
            return ket2dm(self.approx_stateAQ(AQ))
        else:
            return self.approx_rhoAQ(AQ)

    def measurementOutcomes(self, K: int, rhoA: Qobj,
                            measurementScheme: List[List[int]]) -> List[List[List[int]]]:
        outcomes = [
            self.sampler.fake_sampling_dm(rhoA, measure_times=K, measurement_orientation=orientation)
            for orientation in measurementScheme
        ]
        return outcomes

    def IMBasedMeasureOutcomes(self, measureSchemes: List[List[int]],
                               measureOutcomesRhoA: List[List[List[int]]]) -> List[List[List[int]]]:
        measureOutcomesRhoAQ = [
            [
                self.sampler.fake_sampling_dm(
                    self.measuredEvolvedState.setdefault(
                        f'{measureScheme}{outcomeRhoA}', self.approx_AQ(self.measured_state(measureScheme, outcomeRhoA))
                    ),
                    measure_times=1,
                    measurement_orientation=measureScheme
                )[0]    # sampler.fake_sampling_dm() returns List[List[int]] -> List[int] for K=1
                for outcomeRhoA in measureOutcomesRhoA[idx]
            ]
            for idx, measureScheme in enumerate(measureSchemes)
        ]
        return measureOutcomesRhoAQ

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

                renyi_rhoA_CS = renyi_rhoA.calculateRenyiEntropy(classical_shadow=True) if self.classicalShadow else -1

                if not self.intermediateMeasure:
                    rhoAQ = self.approx_AQ(rhoA)
                    MEASURE_OUTCOMES_RhoAQ = self.measurementOutcomes(K=K, rhoA=rhoAQ, measurementScheme=MEASURE_SCHEME)
                    renyi_rhoAQ_ideal = self.idealRenyi2(rhoAQ)
                else:
                    renyi_rhoAQ_ideal = None
                    MEASURE_OUTCOMES_RhoAQ = self.IMBasedMeasureOutcomes(MEASURE_SCHEME, MEASURE_OUTCOMES_RhoA)

                renyi_rhoAQ = RenyiEntropy(measurementScheme=MEASURE_SCHEME, measurementResults=MEASURE_OUTCOMES_RhoAQ)
                renyi_rhoAQ_hamming = renyi_rhoAQ.calculateRenyiEntropy()
                renyi_rhoAQ_CS = renyi_rhoAQ.calculateRenyiEntropy(classical_shadow=True) if self.classicalShadow else -1

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
        return _results_thetas_epoches_times.transpose(1, 0, 2, 3)  # result_epoches_thetas_times


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
             - tensor([ry(-theta) * basis(2, 0) for _ in range(self.qNumber)]))
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


if __name__ == '__main__':
    MLIST = [10, 50, 100, 200, 500, 1000]
    KLIST = [50, 100, 200, 500, 1000]

    for M in MLIST:
        print(f'========== Executing M:{M} ==========')
        for K in KLIST:
            print(f'----- Executing K:{K} -----')
            INT_SLICES = 20
            QNUMBER_TOTAL, SUBSYSTEM_A = 10, [2, 3, 4]

            ZZ_COEFFICIENT = 0
            COUPLING_STRENGTH = 50  # MHz

            TIME_MAX, TIME_SLICES = 50e-3, 50  # \mu s

            # M, K = 100, 100
            EPOCHES = 50

            THETA_LABELS = [r'$4/5$', r'$\pi/3$', r'$3/2$']
            THETA_LIST = [4 / 5, np.pi / 3, 3 / 2]
            TIME_LIST = np.linspace(0, TIME_MAX, TIME_SLICES)

            print(f'QNUMBER_TOTAL: {QNUMBER_TOTAL}, SUBSYSTEM_A: {SUBSYSTEM_A},'
                  f' M: {M}, K: {K}, INT_SLICES: {INT_SLICES}, TIME_MAX: {TIME_MAX}, THETA_LABELS: {THETA_LABELS}')

            model = SimpleModel(
                QNUMBER_TOTAL, SUBSYSTEM_A, THETA_LIST,
                coupling_strength=COUPLING_STRENGTH, zzCoefficient=ZZ_COEFFICIENT,
                superposition=True, quench=True, timeList=TIME_LIST
            )

            eaCalculator = EntanglementAsymmetry(QNUMBER_TOTAL, subA=SUBSYSTEM_A,
                                                 L=INT_SLICES, Q=model.Q, intermediate_measure=True)

            results = eaCalculator.theta_timeRun(model.rhos_with_time_ferroSuperpositionRho, M, K, epoches=EPOCHES)
            format_results = data_preparation_EA(results=results,
                                                 M=M, K=K, INT_SLICES=INT_SLICES,
                                                 THETA_LIST=THETA_LIST, THETA_LABELS=THETA_LABELS,
                                                 QNUMBER_TOTAL=QNUMBER_TOTAL, SUBSYSTEM_A=SUBSYSTEM_A,
                                                 TIME_SLICES=TIME_SLICES, TIME_MAX=TIME_MAX, TIME_LIST=TIME_LIST,
                                                 COUPLING_STRENGTH=COUPLING_STRENGTH, EPOCHES=EPOCHES)

            with open(f'../data/QMpemba/QMpemba_M{M}_K{K}_N{QNUMBER_TOTAL}_A{SUBSYSTEM_A}_EP{EPOCHES}.pkl', 'wb') as f:
                pickle.dump(format_results, f)
