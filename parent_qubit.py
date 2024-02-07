from abc import ABC, abstractmethod
import math
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

    

class ParentQubit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2 ** num_qubits, dtype=np.complex128) # Amplitudes
        self.state[0] = 1
        self.phase = np.zeros(2 ** num_qubits, dtype=np.complex128)

    def set_value(self, v, i):
        self.state[i] = v
  
        

    def set_values(self, v):
        for index, _ in enumerate(self.state):
            self.state[index] = v[index]
                          
    def get_value(self, i):
        return self.state[i]
    
    def get_values(self):
        values = []
        for s in self.state:
            values.append(s)
        return values
    
    def set_phase(self, p, i):
        self.phase[i] = p
    
    def set_phases(self, p):
        for index, _ in enumerate(self.phase):
            self.phase[index] = p[index]
    
    def get_phase(self, i):
        if self.phase[i].real < 0:
            return -1
        return 1
    
    def get_num_qubits(self):
        return self.num_qubits
    
    def merge_qubits(self, pq):
        a_reshape = self.state.reshape(-1, 1)
        b_reshape = pq.state.reshape(-1, 1)
        new_state = np.kron(a_reshape, b_reshape).reshape(1,-1).squeeze()
        phase_a_reshape = self.phase.reshape(-1, 1)
        phase_b_reshape = pq.phase.reshape(-1, 1)
        new_phase = np.tensordot(phase_a_reshape, phase_b_reshape, axes=0).reshape(1,-1).squeeze()
        new_qubit = ParentQubit((self.num_qubits + pq.num_qubits))
        new_qubit.state = new_state
        new_qubit.phase = new_phase
        return new_qubit

        return result
        
    
    def to_bra_ket(self):
        result = ''
        phase_strs = []
        for phase in self.phase:
            phase_imag_string = (str(int(phase.imag)) + 'i') if phase.imag > 0 else ''
            phase_str = str(int(phase.real)) + '+' + phase_imag_string if len(phase_imag_string) > 1 else str(int(phase.real))
            if phase_str == '-1': phase_str = '-'
            if phase_str == '0': phase_str = ''
            phase_strs.append(phase_str)
        
        result = f'{phase_strs[0]}{self.state[0].real:.2f}|{to_binary_string(0, self.num_qubits)}>'
        if '-' in phase_strs[1]: 
            result += ' - '
        else: 
            result += ' + '

        for index, amp in enumerate(self.state[1 : -1], start = 1):
            binary = to_binary_string(index, self.num_qubits)
            phase_str = phase_strs[index]
            result += f'{phase_str}{amp.real:.2f}|{binary}>'
            if '-' in phase_strs[index + 1]: 
                result += ' - '
            else: 
                result += ' + '
        result += f'{phase_strs[-1]}{self.state[-1].real:.2f}|{to_binary_string((2 ** self.num_qubits) - 1, self.num_qubits)}>'
        
        return result
    
    def apply_not_gate(self, i = None):
        not_gate = np.array([[0, 1], [1, 0]])
        identity = np.array([[1, 0], [0, 1]])
        reshape = self.state.reshape(-1, 1)
        if i is None:
            result_gate = not_gate
            for q in range(self.num_qubits-1):
                result_gate = np.kron(result_gate, not_gate)
            new_state = np.matmul(result_gate, reshape)
            self.state = new_state.reshape(1, -1).squeeze()
        else:
            result_gate = 1
            for q in range((self.num_qubits)):
                if q == i:
                    result_gate = np.kron(result_gate, not_gate)
                else:
                    result_gate = np.kron(result_gate, identity)
            new_state = np.matmul(result_gate, reshape)
            self.state = new_state.reshape(1, -1).squeeze()
    
    def apply_hadamard_gate(self, i = None):
        coefficent = 1.0/math.sqrt(2)
        h_gate = np.array([[coefficent, coefficent], [coefficent, -coefficent]])
        identity = np.array([[1, 0], [0, 1]])
        reshape = self.state.reshape(-1, 1)
        if i is None:
            result_gate = h_gate
            for q in range(self.num_qubits-1):
                result_gate = np.kron(result_gate, h_gate)
            new_state = np.matmul(result_gate, reshape)
            self.state = new_state.reshape(1, -1).squeeze()
        else:
            result_gate = 1
            for q in range((self.num_qubits)):
                if q == i:
                    result_gate = np.kron(result_gate, h_gate)
                else:
                    result_gate = np.kron(result_gate, identity)
            new_state = np.matmul(result_gate, reshape)
            self.state = new_state.reshape(1, -1).squeeze()
    
    def apply_z_gate(self, i = None):
        z_gate = np.array([[1, 0], [0, -1]])
        identity = np.array([[1, 0], [0, 1]])
        reshape = self.state.reshape(-1, 1)
        if i is None:
            result_gate = z_gate
            for q in range(self.num_qubits-1):
                result_gate = np.kron(result_gate, z_gate)
            new_state = np.matmul(result_gate, reshape)
            self.state = new_state.reshape(1, -1).squeeze()
        else:
            result_gate = None
            if i == 0:
                result_gate = z_gate
                for q in range(self.num_qubits-1):
                    result_gate = np.kron(result_gate, identity)
            else:
                result_gate = identity
                for q in range(self.num_qubits-1):
                    index = q + 1 + 1
                    if index == i:
                        result_gate = np.kron(result_gate, z_gate)
                    else:
                        result_gate = np.kron(result_gate, identity)
            new_state = np.matmul(result_gate, reshape)
            self.state = new_state.reshape(1, -1).squeeze()
    
    def swap_adjacent(self, i):
        # swap qubits at i and i + 1
        swap_gate =  np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        result_gate = 1
        identity = np.array([[1, 0], [0, 1]])
        for x in range(0, i):
            result_gate = np.kron(result_gate, identity)
        result_gate = np.kron(result_gate, swap_gate)
        for x in range(i + 1, self.num_qubits-1):
            result_gate = np.kron(result_gate, identity)
        result = np.matmul(result_gate, self.state.reshape(-1, 1))
        self.state = result.reshape(1, -1).squeeze()

    def apply_swap_gate(self, i, j):
        if i > j:
            temp = i
            i = j
            j = temp
        
        # Move qubit at index i to index j through adjacent swaps
        for x in range(i, j):
            self.swap_adjacent(x)

        # Move qubit at index j-1 back to index i through adjacent swaps
        for y in range(j-1, i, -1):
            self.swap_adjacent(y-1)


    def apply_cnot_gate(self, i, j):
        i_original = i
        j_original = j
        if i > j:
            self.apply_swap_gate(i, j)
            temp = i
            i = j
            j = temp
        if abs((i_original - j_original)) > 1:
            self.apply_swap_gate(i+1, j)
        result_gate = 1
        cnot_gate = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        identity = np.array([[1, 0], [0, 1]])
        for x in range(0, i):
            result_gate = np.kron(result_gate, identity)
        result_gate = np.kron(result_gate, cnot_gate)
        for x in range(i + 1, self.num_qubits-1):
            result_gate = np.kron(result_gate, identity)
      #  print(result_gate.shape)
        result = np.matmul(result_gate, self.state.reshape(-1, 1))
        self.state = result.reshape(1, -1).squeeze()
        if abs((i_original - j_original)) > 1:
            self.apply_swap_gate(i+1, j)
        if i_original > j_original:
            self.apply_swap_gate(i, j)
        


    
    def measure(self):
        # Numpy function that 
        amplitudes = [(state.real * state.real) for state in self.state]
        measurement_index = np.random.choice(len(self.state), p=amplitudes)
        self.set_values([0] * len(self.state))
        self.set_value(1, measurement_index)
        return int(to_binary_string(measurement_index, self.num_qubits))
    