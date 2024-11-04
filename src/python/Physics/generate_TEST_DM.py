"""
Author: weiguo_ma
Time: 09.19.2024
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Union, List

import numpy as np
from qutip import Qobj, tensor, sigmax, sigmay, sigmaz, qeye, mesolve, basis, ket2dm


def create_operator(i, N):
    opr = sum(
        tensor([qeye(2) if _i != i else element for _i in range(N)]) *
        tensor([qeye(2) if _i != i + 1 else element for _i in range(N)])
        for element in [sigmax(), sigmay(), sigmaz()]
    )
    return opr


def ham_in_manuscript(qnumber, J, Bx, location: Union[List[int], int] = 0) -> Qobj:
    coupling_terms = sum(-J * create_operator(i, qnumber) for i in range(qnumber - 1))
    if isinstance(location, int):
        field_term = Bx * tensor([sigmax() if _i == location else qeye(2) for _i in range(qnumber)])
    elif isinstance(location, list):
        field_term = Bx * tensor([sigmax() if _i in location else qeye(2) for _i in range(qnumber)])
    else:
        raise ValueError("Location must be an int or a list of ints")

    ham = coupling_terms + field_term
    return ham


def time_evolution(initial_state: Qobj, hamiltonian: Qobj, time_list: List):
    return mesolve(hamiltonian, initial_state, tlist=time_list).states


def state_qutip_evolution(qnumber: int, time_list: List):
    J = 1.
    Bx = 0.1 * J
    initial_state = ket2dm(tensor([basis(2, 0) for _ in range(qnumber)]))
    hamiltonian = ham_in_manuscript(qnumber, J, Bx)
    evolution_states = time_evolution(initial_state, hamiltonian, time_list)
    return [evolution_states[i].full() for i in range(len(evolution_states))]


def random_pure_state(dim):
    psi = np.random.rand(dim) + 1j * np.random.rand(dim)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, np.conj(psi))


def random_mixed_state(dim):
    rho = np.zeros((dim, dim), dtype=complex)
    for _ in range(dim):
        psi = np.random.rand(dim) + 1j * np.random.rand(dim)
        psi = psi / np.linalg.norm(psi)
        rho += np.outer(psi, np.conj(psi)) * np.random.rand()
    rho = rho / np.trace(rho)
    return rho


def pseudo_random_DM(qnumber: int, numPure: int, numMixed: int):
    pureList = [random_pure_state(2 ** qnumber) for _ in range(numPure)]
    mixedList = [random_mixed_state(2 ** qnumber) for _ in range(numMixed)]
    return pureList + mixedList


def ferromagnet_state(qnumber: int, theta: float):
    initialState = tensor([basis(2, 0) for _ in range(qnumber)])
    sy_sum = None
    for j in range(qnumber):
        op_list = [sigmay() if idx == j else qeye(2) for idx in range(qnumber)]
        if sy_sum is None:
            sy_sum = tensor(op_list)
        else:
            sy_sum += tensor(op_list)

    U = (-1j * theta / 2 * sy_sum).expm()
    return ket2dm(U * initialState)


def ferromagnet_state_superposition(qnumber: int, theta: float):
    initialState = tensor([basis(2, 0) for _ in range(qnumber)])
    sy_sum = None
    for j in range(qnumber):
        op_list = [sigmay() if idx == j else qeye(2) for idx in range(qnumber)]
        if sy_sum is None:
            sy_sum = tensor(op_list)
        else:
            sy_sum += tensor(op_list)

    U_minus = (-1j * theta / 2 * sy_sum).expm()
    U_plus = (1j * theta / 2 * sy_sum).expm()
    return ket2dm(1 / np.sqrt(2) * (U_minus * initialState - U_plus * initialState))


def quench_hamiltonian(qnumber: int):
    H = 0

    for j in range(qnumber - 1):
        sx_j_sx_j1 = tensor(
            [sigmax() if k == j or k == j + 1 else qeye(2) for k in range(qnumber)]
        )

        sy_j_sy_j1 = tensor(
            [sigmay() if k == j or k == j + 1 else qeye(2) for k in range(qnumber)]
        )

        sz_j_sz_j1 = tensor(
            [sigmaz() if k == j or k == j + 1 else qeye(2) for k in range(qnumber)]
        )

        H += - 1 / 4 * (sx_j_sx_j1 + sy_j_sy_j1 + 0.4 * sz_j_sz_j1)

    return H
