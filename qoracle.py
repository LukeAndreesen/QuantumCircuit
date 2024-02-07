from single_qubit import SingleQubit
from nqubit import NQubit
import numpy as np

def to_binary_string(n, n_qubits):
    remainder = n
    s = ''
    while remainder > 0:
        if remainder % 2 == 0:
            s = '0' + s  
        else:
            s = '1' + s 
        remainder = remainder // 2
    
    while len(s) < n_qubits:
        s = '0' + s
    return s

def swap_adjacent(i, num_qubits):
        # swap qubits at i and i + 1
        swap_gate =  np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        result_gate = 1
        identity = np.array([[1, 0], [0, 1]])
        for x in range(0, i):
            result_gate = np.kron(result_gate, identity)
        result_gate = np.kron(result_gate, swap_gate)
        for x in range(i + 1, num_qubits-1):
            result_gate = np.kron(result_gate, identity)
        return result_gate

def apply_swap_gate(i, j, num_qubits):
    if i > j:
        temp = i
        i = j
        j = temp
    
    result_gate = np.identity(2 ** num_qubits)
    # Move qubit at index i to index j through adjacent swaps
    for x in range(i, j):
        result_gate = np.matmul(swap_adjacent(x, num_qubits), result_gate)

    # Move qubit at index j-1 back to index i through adjacent swaps
    for y in range(j-1, i, -1):
        result_gate = np.matmul(swap_adjacent(x-1, num_qubits), result_gate)
    
    return result_gate

def apply_cnot_gate(i, j, num_qubits):

    i_original = i
    j_original = j
    result_gate = 1
    if i > j:
        result_gate = apply_swap_gate(i, j, num_qubits)
        temp = i
        i = j
        j = temp
    if abs((i_original - j_original)) > 1:
        result_gate = np.matmul(apply_swap_gate(i+1, j, num_qubits), result_gate) if result_gate != 1 else apply_swap_gate(i+1, j, num_qubits)
    cnot_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    identity = np.array([[1, 0], [0, 1]])
    result_new = 1
    for x in range(0, i):
        result_new = np.kron(result_new, identity)
    result_new = np.kron(result_new, cnot_gate)
    for x in range(i + 1, num_qubits-1):
        result_new = np.kron(result_new, identity)
    result = np.matmul(result_new, result_gate) if type(result_gate) != int else result_new
    reswap = 1
    if abs((i_original - j_original)) > 1:
        reswap = apply_swap_gate(i+1, j, num_qubits)
    if i_original > j_original:
        reswap = np.matmul(apply_swap_gate(i, j, num_qubits), reswap) if reswap != 1 else apply_swap_gate(i, j, num_qubits)

    return np.matmul(reswap, result) if type(reswap) != int else result

class QOracle():
    def __init__(self):
        self.bernvaz = None
        self.archimedes = None

    def set_bernvaz(self, code):
        idendity = np.array([[1, 0], [0, 1]])
        str_code = to_binary_string(code, 3)
        last_digit = int(str_code[2])
        middle_digit = int(str_code[1])
        first_digit = int(str_code[0])
        result_gate = np.identity(2 ** 4)
        if last_digit == 1:
            result_gate = apply_cnot_gate(2, 3, 4)
        if middle_digit == 1:
            result_gate = np.matmul(result_gate, apply_cnot_gate(1, 3, 4))
        if first_digit == 1:
            result_gate = np.matmul(result_gate, apply_cnot_gate(0, 3, 4))
        self.bernvaz = result_gate
        return result_gate
    
    def probe_bernvaz(self, nq):
        nq.state = np.matmul(self.bernvaz, nq.state.reshape(-1, 1)).reshape(1, -1).squeeze()

    
    def set_archimedes(self, codes):
        result = np.identity(2 ** 4)
        for code in codes:
            str_code = str(code)
            if len(str_code) != 3:
                for i in range(3 - len(str_code)):
                    str_code = '0' + str_code
            last_digit = int(str_code[2])
            middle_digit = int(str_code[1])
            first_digit = int(str_code[0])
            row = (last_digit * 2) + (middle_digit * 4) + (first_digit * 8)
            result[row][row] = 0
            result[row][row + 1] = 1
            result[row + 1][row + 1] = 0
            result[row + 1][row] = 1
        self.archimedes = result
    
    def probe_archimedes(self, nq):
        nq.state = np.matmul(self.archimedes, nq.state.reshape(-1, 1)).reshape(1, -1).squeeze()





