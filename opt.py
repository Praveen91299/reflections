import numpy as np
from openfermion.linalg import expectation
from scipy.linalg import expm

def construct_configuration_wf(so_occupations):
    for idx, occ in enumerate(so_occupations):
        if idx == 0:
            if occ == 0:
                state_vector = np.array([[1],[0]])
            else:
                state_vector = np.array([[0],[1]])
        else:
            if occ == 0:
                state_vector = np.kron(state_vector, np.array([[1],[0]]))
            else:
                state_vector = np.kron(state_vector, np.array([[0],[1]]))
    return state_vector

def construct_trial_wf(parameters, ansatz_elements, reference_wf):
    #this function is where your trial wavefunction statevector will be constructed,
    #so this function will depend on your ansatz.

    #as an example, I'll do exp(E-E'), where E is the excitation in `ansatz_elements`,
    #and implement the exponential through brute-force matrix exponentiation.
    E = ansatz_elements[0]
    U = expm( parameters[0] * ( E - E.conjugate().T) )

    return U @ reference_wf

def evaluate_trial_energy(parameters,
                          hamiltonian,
                          ansatz_elements,
                          reference_wf):

    trial_wf = construct_trial_wf(parameters, ansatz_elements, reference_wf)
    return np.real(expectation(hamiltonian, trial_wf))
