from src.stats import getProbs, getState, stitch
from src.coupler import Coupler
from qudit import Circuit
from time import time
import numpy as np

D1, D2 = 2, 3
coupler = Coupler(D1, D2, threshold=1e-12)

def original():
    qc = Circuit(4, dim=[D1, D2, D1, D2])
    G2 = qc.gates[D1]

    qc.gate(G2.H, [0])
    qc.gate(G2.H, [2])

    qc.gate(coupler.CX, [0, 1])
    qc.gate(coupler.CX, [2, 3])
    return qc


def cut():
    terms = coupler.terms

    n_full = D1**2 * D2**2
    amp = np.zeros(n_full, dtype=np.complex64)

    for i, (op_sub1, op_sub2, coeff) in enumerate(terms):
        """CIRCUIT A"""
        qc1 = Circuit(3, dim=[D1, D2, D1])
        G2 = qc1.gates[D1]

        qc1.gate(G2.H, [0])
        qc1.gate(coupler.CX, [0, 1])
        qc1.gate(G2.H, [2])

        if op_sub1 != "I":
            gate = coupler.B1[op_sub1]
            qc1.gate(gate, [2])

        """CIRCUIT B"""
        qc2 = Circuit(1, dim=[D2])

        if op_sub2 != "I":
            gate = coupler.B2[op_sub2]
            qc2.gate(gate, [0])

        psi1 = getState(qc1, flipped=False)
        psi2 = getState(qc2, flipped=False)
        c = coeff.item() if hasattr(coeff, "item") else coeff
        amp += c * np.kron(psi1, psi2)

    bases = [D1, D2, D1, D2]
    perm = [0, 1, 2, 3]
    return stitch(amp, bases, perm)

t0 = time()
orig_probs = getProbs(original(), 4)
t0 = time() - t0

t1 = time()
cut_probs = cut()
t1 = time() - t1

total_error = 0
keys = sorted(orig_probs.keys())

for k in keys:
    p_orig = orig_probs[k]
    p_cut = cut_probs.get(k, 0.0)
    diff = abs(p_orig - p_cut)

    total_error += diff

tvd = total_error / 2
slow = t1 / t0
terms = len(coupler.terms)
print(f"TVD={tvd:.5f} | Slow: {slow:.3f}x | Terms: {terms}")