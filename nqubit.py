from parent_qubit import ParentQubit
import numpy as np

class NQubit(ParentQubit):
    def __init__(self, n):
        super().__init__(n)
        self.num_qubits = n
        self.state = np.zeros(2 ** n, dtype=np.complex128) # Amplitudes
        self.state[0] = 1
        self.phase = np.zeros(2 ** n, dtype=np.complex128)
       