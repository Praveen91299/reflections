from openfermion.ops import FermionOperator, QubitOperator
from openfermion.linalg import expectation
from openfermion import load_operator, get_sparse_operator, FermionOperator, count_qubits, expectation
from openfermion.transforms import normal_ordered, qubit_operator_to_pauli_sum
from openfermion.utils import hermitian_conjugated, commutator
from scipy.linalg import expm
import numpy as np
from sympy import Matrix
from scipy.optimize import minimize
from opt import evaluate_trial_energy, construct_configuration_wf
import matplotlib.pyplot as plt
import json, os
from scipy.sparse import csc_matrix
from math import ceil, floor, log10
from openfermion.hamiltonians import s_squared_operator
from scipy.sparse import csc_matrix
from math import ceil, floor, log10
from scipy.optimize import minimize
from opt import evaluate_trial_energy, construct_configuration_wf
import pathos.multiprocessing as multiprocessing
from eos.utils.qcc.iqcc import iQCC_fixed_mf
from openfermion.transforms import jordan_wigner

iEj = lambda i, j: str(i)+'^ '+str(j)+' '
ijE = lambda i, j: str(i)+'^ '+str(j)+'^ '
Eij = lambda i, j: str(i)+' '+str(j)+' '
ijEkl = lambda i, j, k, l: ijE(i, j) + Eij(k, l)

n0 = FermionOperator(iEj(0, 0))
n1 = FermionOperator(iEj(1, 1))
n2 = FermionOperator(iEj(2, 2))
n3 = FermionOperator(iEj(3, 3))

#polynomial classes
P = {
    'p11' : 1. - 2. * n0,
    'p21' : 1. - 2. * n0 * n1,
    'p22' : 1. - 2. * n0 + 2. * n0 * n1,
    'p23' : 1. - 2. * n0 - 2. * n1 + 2. *  n0 * n1,
    'p24' : 1. - 2. * n0 - 2. * n1 + 4. *  n0 * n1,
    'p25' : 1. - 2. * n0 - 2. * n1 * n2 + 2. *  n0 * n2,
    'p26' : 1. - 2. * n0 - 2. * n1 * n2 + 2. *  n0 * n2 + 2. *  n0 * n1,
    'p27' : 1. - 2. * n0 - 2. * n1 + 2. *  n0 * n2 + 2. *  n0 * n1,
    'p28' : 1. - 2. * n0 - 2. * n1 - 2. * n2 + 2. *  n0 * n1 + 2. *  n1 * n2 + 2. *  n0 * n2,
    'p29' : 1. - 2. * n0 - 2. * n1 - 2. *  n2* n3 + 2. *  n0 * n2 + 2. *  n1 * n3 + 2. * n0 * n1
}

poly_list = ['p11', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29']

iter_count = 0


def evaluate_energy_UCCSD(params, hamiltonian, rotations, reference_wf):
    U = expm(sum([a*b for a , b in zip(params[:], rotations)]))
    ans = U @ reference_wf
    return np.real(expectation(hamiltonian, ans))

def callback_count(params):
    global iter_count
    iter_count +=1

def mf_rotation_list(n_so):
    '''
    Provides list of k_pq operators
    input: 
    n_so Number of qubits
    '''
    exp = []
    count = 0
    for i in range(n_so):
        for j in range(i+1, n_so):
            exp += [(FermionOperator(iEj(i, j)) - FermionOperator(iEj(j, i)))/2]
            count +=1
    return exp

def mf_rotation_list_complex(n_so):
    '''
    Provides list of K_pq' operators
    input:
    n_so Number of qubits
    '''
    exp = []
    count = 0
    for i in range(n_so):
        for j in range(i+1, n_so):
            exp += [(0. + 1.j)*(FermionOperator(iEj(i, j)) + FermionOperator(iEj(j, i)))/2]
            count +=1
    return exp

def mf_rotation_list_all(n_so):
    '''
    
    '''
    rot_real = mf_rotation_list(n_so)
    rot_complex = mf_rotation_list_complex(n_so)
    return rot_real + rot_complex

def mf_rotation_list_doubles(n_so):
    exp = []
    count = 0
    for i in range(n_so):
        for j in range(i+1, n_so):
            for k in range(j+1, n_so):
                for l in range(k+1, n_so):
                    exp += [(FermionOperator(ijEkl(i, j, k, l)) - FermionOperator(ijEkl(l, k, j, i)))/2]
                    count +=1
    return exp

def evaluate_trial_energy(params,
                          hamiltonian,
                          mf_rotation_exponents,
                          poly,
                          tau,
                          reference_wf):
    V = expm(sum([a*b for a , b in zip(params[:], mf_rotation_exponents)]))
    R = (V.conjugate().T) @ poly @ V
    a = (1. + np.cos(tau))/2
    b = - np.sin(tau)/2
    c = (1. - np.cos(tau))/2

    RH = R @ hamiltonian
    h_rot = a * hamiltonian + 1.j * b * ((hamiltonian @ R) - RH) + c * RH @ R

    return np.real(expectation(h_rot, reference_wf))

def evaluate_trial_energy_tau(params,
                          hamiltonian,
                          mf_rotation_exponents,
                          poly,
                          reference_wf):
    V = expm(sum([a*b for a , b in zip(params[1:], mf_rotation_exponents)]))
    R = (V.conjugate().T) @ poly @ V
    tau = params[0]
    a = (1. + np.cos(tau))/2
    b = - np.sin(tau)/2
    c = (1. - np.cos(tau))/2

    RH = R @ hamiltonian
    h_rot = a * hamiltonian + 1.j * b * ((hamiltonian @ R) - RH) + c * RH @ R

    return np.real(expectation(h_rot, reference_wf))

def evaluate_trial_energy_tau_complex(params,
                          hamiltonian,
                          mf_rotation_exponents,
                          poly,
                          reference_wf):
    V = expm(sum([a*b for a , b in zip(params[1:], mf_rotation_exponents)]))
    R = (V.conjugate().T) @ poly @ V
    tau = params[0]
    a = (1. + np.cos(tau))/2
    b = - np.sin(tau)/2
    c = (1. - np.cos(tau))/2

    RH = R @ hamiltonian
    h_rot = a * hamiltonian + 1.j * b * ((hamiltonian @ R) - RH) + c * RH @ R

    return np.real(expectation(h_rot, reference_wf))

def evaluate_trial_energy_complex(params,
                                  hamiltonian,
                                  mf_rotation_exponents,
                                  poly,
                                  tau,
                                  reference_wf):
    V = expm(sum([a*b for a , b in zip(params[:], mf_rotation_exponents)]))
    R = (V.conjugate().T) @ poly @ V

    a = (1. + np.cos(tau))/2
    b = - np.sin(tau)/2
    c = (1. - np.cos(tau))/2

    RH = R @ hamiltonian
    h_rot = a * hamiltonian + 1.j * b * ((hamiltonian @ R) - RH) + c * RH @ R

    return np.real(expectation(h_rot, reference_wf))

def evaluate_gradient_complex(params,
                              hamiltonian,
                              mf_rotation_exponents,
                              poly,
                              reference_wf):
    V = expm(sum([a*b for a , b in zip(params[:], mf_rotation_exponents)]))
    R = (V.conjugate().T) @ poly @ V
    RH = hamiltonian @ R - R @ hamiltonian
    eval = (reference_wf.conjugate().T @ RH @ reference_wf)[0][0]
    return (-np.abs(eval))

#################

# obtains kappa matrix for MFR only (second degree number preserving terms sum(a_p^ a_q) )
# excludes constant terms
def get_matrix(op, n_so):
    op_terms = list(op.terms.items())
    matrix = np.zeros(shape = (n_so, n_so), dtype=complex)
    for term in op_terms:
        if len(term[0]) == 2:
            matrix[term[0][0][0], term[0][1][0]] = term[1]
    return matrix

#reverse of get_matrix
def get_operator(mat):
    op = 0
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            op += FermionOperator(str(i)+'^ '+str(j), mat[i][j])
    return op

# evaluates C^ B C where C is restricted to MFR only and C = e^A (e^-A B e^A)
def BCH_symb(A, B, n_so):
    B_list = list(B.terms.items())
    C = get_matrix(-A, n_so)
    V_mat = expm(C)
    V_conj = V_mat.conjugate()

    expression = 0
    for term in B_list:
        prod = FermionOperator.identity()
        for op in term[0]:
            if op[1] == 0:
                trans_op = sum([FermionOperator(str(i), V_conj[i, op[0]]) for i in range(n_so)])
            else:
                trans_op = sum([FermionOperator(str(i)+'^', V_mat[i, op[0]]) for i in range(n_so)])
            prod *= trans_op
        expression += term[1]*prod
    return normal_ordered(expression)

################

#to load data
def load_json(name, loc):
    with open(loc + name + '.json') as f:
        data = json.load(f)
    return data

def dress_H_expectation(tau, hamiltonian, R, reference_wf):
    h_rot = dress_H(tau, hamiltonian, R)
    val = reference_wf.conjugate().T @ h_rot @ reference_wf
    return np.real(val[0, 0])

def dress_H(tau, hamiltonian, R):
    a = float((1. + np.cos(tau))/2)
    b = - float(np.sin(tau)/2)
    c = float((1. - np.cos(tau))/2)

    RH = R @ hamiltonian
    h_rot = a * hamiltonian + 1.j * b * ((hamiltonian @ R) - RH) + c * RH @ R
    return h_rot

def dress_H_simult(taus, hamiltonian, R_list, reference_wf):
    for i in range(len(taus))[:-1]:
        hamiltonian = dress_H(taus[i], hamiltonian, R_list[i])
    return dress_H_expectation(taus[-1], hamiltonian, R_list[-1], reference_wf)

#saving data to file
def save_to_file(directory, filename, x_label, x_vals, data, comments = []):
    text = ''
    if comments != []:
        for comment in comments:
            text += '\n#{}'.format(comment)
        text +='\n'
    text = text[1:]
    text += '${} '.format(x_label)
    labels = data.keys()
    for label in labels:
        text += '{} '.format(label)
    for x in x_vals:
        text += '\n{} '.format(x)
        for label in labels:
            if data[label][x] != None:
                text += '{} '.format(data[label][x])
            else:
                text += 'None '

    location = directory + filename
    os.makedirs(os.path.dirname(location), exist_ok=True)
    with open(location, 'a') as file:
        file.write(text)
    return

#plotting data from file
def plot_from_file(directory, filename, title = '', grid = False, ylabel = '', savefig = False, figname = ''):
    file = open(directory + filename, 'r')
    comments = []
    data = {}
    x = []
    for line in file.readlines()[:]:
        data_line = line.strip()
        if data_line[0] == '#':
            comments += [data_line]
            continue
        if data_line[0] == '$':
            labels = data_line[1:].split()
            data = {x : {} for x in labels}
            continue
        data_list = data_line.split()
        x += [float(data_list[0])]
        for i in range(1, len(data_list[:])):
            if data_list[i] != 'None':
                data[labels[i]][x[-1]] = float(data_list[i])
    plt.clf()
    for i in range(1, len(labels)):
        x_vals = data[labels[i]].keys()
        y_vals = data[labels[i]].values()
        plt.plot(x_vals, y_vals, label = labels[i])

    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(labels[0])

    plt.grid(grid)
    if title == '':
        plt.title(comments[0])
    else:
        plt.title(title)

    if savefig:
        if figname != '':
            plt.savefig(figname + '.png')
        else:
            plt.savefig(filename + '.png')

    plt.show()
    return x, data, comments

##################

def get_V(params, mf_rotation_exponents):
    return expm(sum([a*b for a , b in zip(params, mf_rotation_exponents)]))

def get_rotation(params, poly, mf_rotation_exponents):
    V = get_V(params, mf_rotation_exponents)
    return (V.conjugate().T) @ poly @ V

def get_V_exp(params, k_pq):
    return sum(params[i]*k_pq[i] for i in range(len(params)))

def get_rotation_symb(params, poly, k_pq, n_so):
    return BCH_symb(get_V_exp(params, k_pq), poly, n_so)

def evaluate_gradient_complex(params,
                              hamiltonian,
                              mf_rotation_exponents,
                              poly,
                              V_list,
                              lam,
                              reference_wf):
    V = get_V(params, mf_rotation_exponents)
    R = get_rotation(params, poly, mf_rotation_exponents)
    RH = hamiltonian @ R - R @ hamiltonian
    val = (reference_wf.conjugate().T @ RH @ reference_wf)[0][0]
    val = (-np.abs(val)).ravel()[0]
    if len(V_list) > 0:
        penality = sum((lam * np.abs(reference_wf.conjugate().T @ V2.conjugate().T @ V @ reference_wf))[0, 0] for V2 in V_list)
        val += np.real(penality)
    return val

def get_generators(n_so):
    return mf_rotation_list(n_so) + mf_rotation_list_complex(n_so)

def get_generators_sparse(n_so):
    k_pq =  get_generators(n_so)
    k_pq_sparse = [get_sparse_operator(element, n_so).toarray() for element in k_pq]
    return k_pq_sparse

#returns labels of top
def get_top_list(grad_list, n_gens):
    #get minimum
    values = list(grad_list.values())
    selected = sorted(values)[:n_gens]
    keys = list(grad_list.keys())
    sorted_list = [(keys[values.index(item)]) for item in selected]
    return sorted_list

def round_to(x, a):
    return round(x, a - 1 - int(floor(log10(abs(x)))))

#get minimum distinct classes, upto 2 significant digits
def get_top_list_2(grad_list, n_gens):
    keys = list(grad_list.keys())
    rounded_values = [round_to(x, 2) for x in list(grad_list.values())]
    sorted_values = sorted(rounded_values)
    selected = []
    for i in range(len(rounded_values)):
        if sorted_values[i] not in selected:
            selected += [sorted_values[i]]

    if n_gens < len(selected):
        selected =  selected[:n_gens]

    sorted_list = [(keys[rounded_values.index(item)]) for item in selected]
    return sorted_list

def maximize_grad(arg):
    x0, hamilt_sparse, k_pq_sparse, poly, V_list, lam, hf_statevector, method, tol = arg
    result = minimize(evaluate_gradient_complex,
                     x0=x0,
                     args=(hamilt_sparse, k_pq_sparse, poly, V_list, lam, hf_statevector),
                     method=method,
                     tol=tol)
    return result

def minimize_energy(arg):
    x0, hamilt_sparse, R, hf_statevector, method, tol = arg
    result = minimize(dress_H_simult,
                     x0=x0,
                     args=(hamilt_sparse, R, hf_statevector),
                     method=method,
                     tol=tol)
    return result

#an iterative version of FR
def FR_ranked(HAMILTONIAN, HF_OCCUPATIONS, n_gens, layers, cpu_pool, trials=5, lam = 0.01, coeff_tol = 0.001, select_type = 1):

    #generators
    n_so = count_qubits(HAMILTONIAN)
    k_pq = get_generators(n_so)
    k_pq_sparse = get_generators_sparse(n_so)

    hamilt_sparse = get_sparse_operator(HAMILTONIAN, n_so)
    hf_statevector = construct_configuration_wf(HF_OCCUPATIONS)

    #checks
    if (type(layers) is not int) or layers <= 0:
        print("\n Error, enter a positive integer value for number of layers. Current input {}".format(layers))
        return

    if (type(n_gens) is not int) or n_gens <= 0:
        print("\n Error, enter a positive integer value for number of generators. Current input {}".format(n_gens))
        return

    if (type(trials) is not int) or trials <= 0:
        print("\n Error, enter a postive integer value for number of trials. Current input {}".format(trials))
        return

    reps = 1
    if n_gens > len(poly_list):
        reps = ceil(n_gens/len(poly_list))
        print("\n Input number of generators in each layers exceed number of polynomial classes. Generating {} rotations for each class using orthogonality condition.".format(reps))

    if n_so != len(HF_OCCUPATIONS):
        print("\n Mismatch in hf state length ({}) and hamiltonian qubits ({})".format(len(HF_OCCUPATIONS, n_so)))

    print("\nChecks  complete. Starting optimization...")
    final_energies = []
    hamiltonian_length = {0: len(list(HAMILTONIAN.terms.items()))}
    grad_list = {}
    param_list = {}
    for layer in range(layers):
        V_list = {p : [] for p in poly_list}
        for rep in range(reps):
            args = []
            for p in poly_list:
                poly = get_sparse_operator(P[p], n_so).toarray()
                method = 'L-BFGS-B'
                tol = 1e-6
                x0_params = [0.0]*len(k_pq_sparse)
                random_guesses = [
                    np.random.uniform(low=0.0, high=2 * np.pi, size=(1, len(k_pq_sparse)))
                    for _ in range(trials)
                ]

                args += [(x0_params, hamilt_sparse, k_pq_sparse, poly, V_list[p], lam, hf_statevector, method, tol)]
                args += [
                    (
                        random_guesses[idx], hamilt_sparse, k_pq_sparse, poly, V_list[p], lam, hf_statevector, method, tol
                    )
                    for idx in range(trials)
                ]
            opt_runs = cpu_pool.map(maximize_grad, args)
            opt_runs_results = np.reshape(opt_runs, (len(poly_list), trials+1))

            #selecting best run for each class
            for i in range(len(poly_list)):
                A = opt_runs_results[i]
                grads = [result.fun for result in A]
                params = [result.x for result in A]
                grad_min = min(grads)
                grad_min_idx = grads.index(grad_min)
                label = poly_list[i] + '_' + str(rep)
                grad_list[label] = grad_min
                param_list[label] = params[grad_min_idx]
                V_list[poly_list[i]] += [get_V(params[grad_min_idx], k_pq_sparse)]
        #print(grad_list)
        #print(param_list)

        #choose top gradients
        if select_type == 1:
            top_poly = get_top_list(grad_list, n_gens)
        elif select_type == 2:
            top_poly = get_top_list_2(grad_list, n_gens)

        print("\nChosen classes: {}".format(top_poly))

        R_list = {}
        R_list_symb = {}
        for p in top_poly:
            poly = P[p[:3]]
            poly_sparse = get_sparse_operator(poly, n_so).toarray()
            R_list_symb[p] = get_rotation_symb(param_list[p], poly, k_pq, n_so)
            R_list[p] = get_rotation(param_list[p], poly_sparse, k_pq_sparse)

        #optimize hamiltonian rotation energy
        x0_params = [0.0]*n_gens
        args = []
        random_guesses = [
            np.random.uniform(low=0.0, high=2 * np.pi, size=(1, n_gens))
            for _ in range(trials)
        ]
        args += [(x0_params, hamilt_sparse, list(R_list.values()), hf_statevector, method, tol)]
        args += [
            (
                random_guesses[idx], hamilt_sparse, list(R_list.values()), hf_statevector, method, tol
            )
            for idx in range(trials)
        ]

        opt_runs = cpu_pool.map(minimize_energy, args)

        energies = [result.fun for result in opt_runs]
        params = [result.x for result in opt_runs]
        energy_min = min(energies)
        energy_min_idx = energies.index(energy_min)
        params_min = params[energy_min_idx]

        print("\n Energy after {} layers: {}".format(layer + 1, energy_min))
        #dress hamiltonian and continue
        #print('\n # of terms in dressed hamiltonian = {}'.format(len(list(HAMILTONIAN.terms.items()))))
        for i in range(n_gens):
            hamilt_sparse = dress_H(params_min[i], hamilt_sparse, list(R_list.values())[i])
            #print(HAMILTONIAN)
            #print(list(R_list_symb.values())[i])
        #    print('yes')
        #    HAMILTONIAN = dress_H_symb(params_min[i], HAMILTONIAN, list(R_list_symb.values())[i])
        #    print('\n # of terms in dressed hamiltonian = {}'.format(len(list(HAMILTONIAN.terms.items()))))
        hamilt_sparse = csc_matrix(hamilt_sparse)
        final_energies += [energy_min]
        #l = len(list(HAMILTONIAN.terms.items()))
        #print('\n Number of terms in dressed Hamiltonian after layer {}: {} '.format(layer+1, l))
        #hamiltonian_length[layer+1] = [l]
    return final_energies, hamiltonian_length

#####################################################

def construct_configuration_op(hf_occupations):
    op = 1.0
    for i in range(len(hf_occupations)):
        if hf_occupations[i] == 1:
            op *= FermionOperator(str(i), 1.0)
    op = hermitian_conjugated(op)*op
    return op

def evaluate_gradient_complex_symb(params,
                              hamiltonian,
                              mf_rotation_exponents,
                              poly,
                              V_2,
                              lam,
                              hf_occupations):
    n_so = count_qubits(hamiltonian)
    exp_V = sum(params[i]*mf_rotation_exponents[i] for i in range(len(params)))
    R = BCH_symb(exp_V, poly, n_so)
    H = get_sparse_operator(hamiltonian, n_so)
    Rs = get_sparse_operator(R, n_so)
    RH_ = 1.0*(H*Rs - Rs*H)
    reference_wf = construct_configuration_wf(hf_occupations)
    val = (reference_wf.conjugate().T @ RH_ @ reference_wf)[0][0]
    val = (-np.abs(val)).ravel()[0]
    if V_2 != 0:
        penality_op = get_sparse_operator(BCH_symb(exp_V, V_2, n_so))
        penality = (lam * np.abs(reference_wf.conjugate().T @ penality_op @ reference_wf))[0, 0]
        val += np.real(penality)
    return val

def maximize_grad_symb(arg):
    x0, hamiltonian, k_pq, poly, V_list, lam, hf_occupations, method, tol = arg
    result = minimize(evaluate_gradient_complex_symb,
                     x0=x0,
                     args=(hamiltonian, k_pq, poly, V_list, lam, hf_occupations),
                     method=method,
                     tol=tol)
    return result

def dress_H_symb(tau, hamiltonian, R):
    a = float((1. + np.cos(tau))/2)
    b = - float(np.sin(tau)/2)
    c = float((1. - np.cos(tau))/2)

    RH = R * hamiltonian
    h_rot = a * hamiltonian + 1.j * b * ((hamiltonian * R) - RH) + c * RH * R
    h_rot.compress(abs_tol = 1e-6)
    return h_rot

def get_V_exp(params, k_pq):
    return sum(params[i]*k_pq[i] for i in range(len(params)))

def get_v_exp(params, k_pq):
    return sum(params[i]*k_pq[i] for i in range(len(params)))

#an iterative version of FR
def FR_ranked_symb(HAMILTONIAN, HF_OCCUPATIONS, n_gens, layers, cpu_pool, trials=5, lam = 0.01, coeff_tol = 0.001, select_type = 1):

    #generators
    n_so = count_qubits(HAMILTONIAN)
    rot_real = mf_rotation_list(n_so)
    rot_complex = mf_rotation_list_complex(n_so)
    k_pq =  rot_real + rot_complex
    k_pq_sparse = get_generators_sparse(n_so)

    hamilt_sparse = get_sparse_operator(HAMILTONIAN, n_so)
    hf_statevector = construct_configuration_wf(HF_OCCUPATIONS)
    hf_op = construct_configuration_op(HF_OCCUPATIONS)

    #checks
    if (type(layers) is not int) or layers <= 0:
        print("\n Error, enter a positive integer value for number of layers. Current input {}".format(layers))
        return

    if (type(n_gens) is not int) or n_gens <= 0:
        print("\n Error, enter a positive integer value for number of generators. Current input {}".format(n_gens))
        return

    if (type(trials) is not int) or trials <= 0:
        print("\n Error, enter a postive integer value for number of trials. Current input {}".format(trials))
        return

    reps = 1
    if n_gens > len(poly_list):
        reps = ceil(n_gens/len(poly_list))
        print("\n Input number of generators in each layers exceed number of polynomial classes. Generating {} rotations for each class using orthogonality condition.".format(reps))

    if n_so != len(HF_OCCUPATIONS):
        print("\n Mismatch in hf state length ({}) and hamiltonian qubits ({})".format(len(HF_OCCUPATIONS, n_so)))
        return

    print("\nChecks  complete. Starting optimization...")
    final_energies = []
    grad_list = {}
    param_list = {}
    for layer in range(layers):
        V_list = {p : 0 for p in poly_list}
        for rep in range(reps):
            args = []
            for p in poly_list:
                poly = P[p]
                method = 'L-BFGS-B'
                tol = 1e-6
                x0_params = [0.0]*len(k_pq)
                random_guesses = [
                    np.random.uniform(low=0.0, high=2 * np.pi, size=(1, len(k_pq)))
                    for _ in range(trials)
                ]

                args += [(x0_params, HAMILTONIAN, k_pq, poly, V_list[p], lam, HF_OCCUPATIONS, method, tol)]
                args += [
                    (
                        random_guesses[idx], HAMILTONIAN, k_pq, poly, V_list[p], lam, HF_OCCUPATIONS, method, tol
                    )
                    for idx in range(trials)
                ]
            opt_runs = cpu_pool.map(maximize_grad_symb, args)
            opt_runs_results = np.reshape(opt_runs, (len(poly_list), trials+1))
            #opt_runs_results = [maximize_grad_symb(arg) for arg in args]

            #selecting best run for each class
            for i in range(len(poly_list)):
                A = opt_runs_results[i]
                grads = [result.fun for result in A]
                params = [result.x for result in A]
                grad_min = min(grads)
                grad_min_idx = grads.index(grad_min)
                label = poly_list[i] + '_' + str(rep)
                grad_list[label] = grad_min
                param_list[label] = params[grad_min_idx]
                V_list[poly_list[i]] += BCH_symb(-get_V_exp(params[grad_min_idx], k_pq), hf_op, n_so)
        #print(grad_list)
        #print(param_list)

        #choose top gradients
        if select_type == 1:
            top_poly = get_top_list(grad_list, n_gens)
        elif select_type == 2:
            top_poly = get_top_list_2(grad_list, n_gens)

        print("\nChosen classes: {}".format(top_poly))

        R_list = {}
        R_list_sparse = {}
        for p in top_poly:
            poly = P[p[:3]]
            R_list[p] = BCH_symb(get_V_exp(params[grad_min_idx], k_pq), poly, n_so)
            R_list_sparse[p] = get_sparse_operator(R_list[p], n_so)

        #optimize hamiltonian rotation energy
        x0_params = [0.0]*n_gens
        args = []
        random_guesses = [
            np.random.uniform(low=0.0, high=2 * np.pi, size=(1, n_gens))
            for _ in range(trials)
        ]
        args += [(x0_params, hamilt_sparse, list(R_list_sparse.values()), hf_statevector, method, tol)]
        args += [
            (
                random_guesses[idx], hamilt_sparse, list(R_list_sparse.values()), hf_statevector, method, tol
            )
            for idx in range(trials)
        ]

        opt_runs = cpu_pool.map(minimize_energy, args)

        energies = [result.fun for result in opt_runs]
        params = [result.x for result in opt_runs]
        energy_min = min(energies)
        energy_min_idx = energies.index(energy_min)
        params_min = params[energy_min_idx]

        print("\n Energy after {} layers: {}".format(layer + 1, energy_min))
        #dress hamiltonian and continue
        #for i in range(n_gens):
            #hamilt_sparse = dress_H(params_min[i], hamilt_sparse, list(R_list_sparse.values())[i])
            #HAMILTONIAN = dress_H_symb(params_min[i], HAMILTONIAN, list(R_list.values())[i])
        hamilt_sparse = csc_matrix(hamilt_sparse)
        final_energies += [energy_min]
    return final_energies

############################
#get UCC, qcc, MFR gradients for LiH
from qcc_pilot import *
import tequila as tq
from tequila import PauliString
import copy

get_mag = lambda a: np.real(a) + np.imag(a)

def PauliString_from_Operator(op):
    """
    :param op : QubitOperator, converted if FermionOperator
    :return: list of PauliString object instances

    note: currently returns with real coefficients
    """
    if isinstance(op, FermionOperator):
        op = jordan_wigner(op)

    if not isinstance(op, QubitOperator):
        raise TypeError("Incorrect input type")

    return [PauliString.from_openfermion(a, coeff = b) for a,b in zip(list(op.terms.keys()), list(op.terms.values()))]
def filter_empty_strings(pauli_list):
    filtered_list = []
    for a in pauli_list:
        if a._data != {}:
            filtered_list.append(a)
    return filtered_list

def mfr_gradient(params, h, ops, poly, ref):
    V = get_V(params, ops)
    R = get_rotation(params, poly, ops)
    RH = h @ R - R @ h
    val = (ref.conjugate().T @ RH @ ref)[0][0]
    val = 0.5*(-np.abs(val)).ravel()[0]
    return val
def maximize_mfr_grad(arg):
    x0, h, ops, poly, ref, method, tol = arg
    result = minimize(mfr_gradient,
                     x0=x0,
                     args=(h, ops, poly, ref),
                     method=method,
                     tol=tol)
    return result


class get_grads:
    def __init__(self, hamiltonian, ref_occ, symb = 0):
        self.hamiltonian = hamiltonian
        self.n_so = count_qubits(hamiltonian)
        self.symb = symb

        self.occ = ref_occ
        self.ref_state = construct_configuration_wf(ref_occ)

        singles = mf_rotation_list(self.n_so)
        doubles = mf_rotation_list_doubles(self.n_so)
        self.uccsd_ops = singles + doubles
        self.mfr_rot = singles + mf_rotation_list_complex(self.n_so)

        self.uccsd_grads = []
        self.mfr_grads = []
        self.qcc_grads = []

        if symb == 0:
            self.hamilt_sparse = get_sparse_operator(hamiltonian, self.n_so).toarray()
            self.uccsd_ops_sparse = [get_sparse_operator(op, self.n_so) for op in self.uccsd_ops]
            self.mfr_rot_sparse = [get_sparse_operator(op, self.n_so).toarray() for op in self.mfr_rot]

    def grad(self, op):
        if self.symb == 0: #sparse
            return np.abs(np.array(self.ref_state).T.conjugate() @ (self.hamilt_sparse @ op - op @ self.hamilt_sparse) @ self.ref_state).ravel()[0]

    def UCCSD_grads(self, threshold = 1e-6):
        for i in range(len(self.uccsd_ops)):
            self.uccsd_grads.append((self.uccsd_ops[i], self.grad(self.uccsd_ops_sparse[i])))
        self.uccsd_grads.sort(key=lambda x: x[1], reverse=True)
        print('\n UCCSD gradients calculated.')
        return
    def MFR_grads(self, cpu_pool = None, trials = 10, tol = 1e-6, threshold = 1e-6):
        #consider all ten operators and optimize for gradients
        args = []
        for p in poly_list:
            #print('\nBeginning calculation of gradients for polynomial {}'.format(p))
            poly = P[p]
            if self.symb == 0:
                poly = get_sparse_operator(P[p], self.n_so).toarray()
            method = 'L-BFGS-B'
            x0_params = [0.0]*len(self.mfr_rot)
            random_guesses = [
                np.random.uniform(low=0.0, high=2 * np.pi, size=(1, len(self.mfr_rot)))
                for _ in range(trials)
            ]

            args += [(copy.deepcopy(x0_params), copy.deepcopy(self.hamilt_sparse), copy.deepcopy(self.mfr_rot_sparse), copy.deepcopy(poly), copy.deepcopy(self.ref_state), copy.deepcopy(method), copy.deepcopy(tol))]

            args += [
                (
                    random_guesses[idx], self.hamilt_sparse, self.mfr_rot_sparse, poly, self.ref_state, method, tol
                )
                for idx in range(trials)
            ]
        opt_runs = cpu_pool.map(maximize_mfr_grad, args)#[maximize_mfr_grad(arg) for arg in args]#
        opt_runs_results = np.reshape(opt_runs, (len(poly_list), trials+1))

        #selecting best run for each class
        for i in range(len(poly_list)):
            A = opt_runs_results[i]
            grads = [result.fun for result in A]
            params = [result.x for result in A]
            grad_min = min(grads)
            grad_min_idx = grads.index(grad_min)
            label = poly_list[i]
            self.mfr_grads.append((label, grad_min, params[grad_min_idx]))
        self.mfr_grads.sort(key=lambda x: x[1], reverse=False)
        print("\nGradients maximized successfully.")
        return
    def QCC_grads(self, threshold = 1e-6):
        if not isinstance(self.hamiltonian, QubitOperator):
            hamiltonian_openfermion = jordan_wigner(self.hamiltonian)
        else:
            hamiltonian_openfermion = self.hamiltonian
        n_qubits = len(self.occ)
        hamiltonian_x_strings = get_operator_x_strings(hamiltonian_openfermion, n_qubits)
        dis = []
        for x_string in hamiltonian_x_strings:
            representative_entangler = get_entangler_from_x_string(
                x_string, type="minimal", format="openfermion"
            )
            gradient = abs(
                np.imag(
                    compute_matrix_element(
                        identity_op,
                        representative_entangler,
                        hamiltonian_openfermion,
                        self.occ,
                    )
                )
            )
            if gradient > threshold:
                dis.append((x_string, round(gradient, int(floor(log10(1 / threshold))))))
        dis.sort(key=lambda x: x[1], reverse=True)
        self.qcc_grads = dis
        print('\nQCC Gradients evalluated successfully')
        return
    def disp_top_grads(self, n = 5):
        print('\n Top UCCSD gradients:')
        for i in range(min(n, len(self.uccsd_grads))):
            print('\n{}'.format(self.uccsd_grads[i][1]))
        print('\n Top MFR gradients:')
        for i in range(min(n, len(self.mfr_grads))):
            print('\n{}'.format(self.mfr_grads[i][1]))
        print('\n Top QCC gradients:')
        for i in range(min(n, len(self.qcc_grads))):
            print('\n{}'.format(self.qcc_grads[i][1]))
    def mfr_rotation_symb(self, p, target):
        '''
        Effects the optimized rotation of poly class on the target fermionic operator
        '''
        #check if optimized
        for i in range(len(self.mfr_grads)):
            if self.mfr_grads[i][0] == p:
                params = self.mfr_grads[i][2]
        return get_rotation_symb(params, target, self.mfr_rot, self.n_so)
