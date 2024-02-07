from single_qubit import SingleQubit
from nqubit import NQubit

class QCircuit:
    def same_entangle(qa, i, j):
        qa.apply_hadamard_gate(i)
        qa.apply_cnot_gate(i, j)

    def bernvaz(qa, qo):
        qa.apply_hadamard_gate()
        qo.probe_bernvaz(qa)
        qa.apply_hadamard_gate()
        
    def archimedes(qa, qo):
        qa.apply_hadamard_gate()
        qo.probe_archimedes(qa)
        qa.apply_hadamard_gate()

        
        