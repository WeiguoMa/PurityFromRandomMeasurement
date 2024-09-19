"""
Author: weiguo_ma
Time: 09.19.2024
Contact: weiguo.m@iphy.ac.cn
"""
from typing import Union, List

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


def main(qnumber: int, time_list: List):
    J = 1.
    Bx = 0.1 * J
    initial_state = ket2dm(tensor([basis(2, 0) for _ in range(qnumber)]))
    hamiltonian = ham_in_manuscript(qnumber, J, Bx)
    evolution_states = time_evolution(initial_state, hamiltonian, time_list)
    return [evolution_states[i].full() for i in range(len(evolution_states))]