from parent_qubit import ParentQubit

class SingleQubit(ParentQubit):
    def __init__(self):
        super().__init__(1)
        self.num_qubits = 1

