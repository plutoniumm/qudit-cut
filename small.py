from src.stats import getProbs, getState, stitch
from src.coupler import Coupler
from qudit import Circuit
from time import time
import numpy as np
import argparse

# Ex: python small.py --d 3 4 --t 1e-6
P = argparse.ArgumentParser()
P.add_argument("--d", nargs=2, type=int, required=True, help="Dimensions D1 D2")
P.add_argument("--t", type=float, required=True, help="Coupler threshold")
args = P.parse_args()

D1, D2 = args.d
coupler = Coupler(D1, D2, method="optimal", threshold=args.t)

def original():
    qc = Circuit(4, dim=[D1, D1, D2, D2])
    G1, G2 = qc.gates[D1], qc.gates[D2]

    for i in range(2):
        qc.gate(G1.H, [i])
    for i in range(2, 4):
        qc.gate(G2.H, [i])

    qc.gate(coupler.CX, [1, 2])

    qc.gate(G1.RY, 0, angle=np.pi / 3)
    qc.gate(G1.RZ, 0, angle=np.pi / 4)

    qc.gate(G1.RY, 1, angle=np.pi / 5)
    qc.gate(G1.RZ, 1, angle=np.pi / 6)

    qc.gate(G2.RY, 2, angle=np.pi / 7)
    qc.gate(G2.RZ, 2, angle=np.pi / 8)

    qc.gate(G2.RY, 3, angle=np.pi / 9)
    qc.gate(G2.RZ, 3, angle=np.pi / 10)
    return qc


def cut():
    terms = coupler.terms

    n_full = D1**2 * D2**2
    amp = np.zeros(n_full, dtype=np.complex64)

    for i, (op_sub1, op_sub2, coeff) in enumerate(terms):
        """CIRCUIT A"""
        qc1 = Circuit(2, dim=D1)
        G1 = qc1.gates[D1]

        for j in range(2):
            qc1.gate(G1.H, [j])

        if isinstance(op_sub1, str):
            if op_sub1 != "I":
                qc1.gate(coupler.B1[op_sub1], [1])
        else:
            qc1.gate(op_sub1, [1])

        qc1.gate(G1.RY, 0, angle=np.pi / 3)
        qc1.gate(G1.RZ, 0, angle=np.pi / 4)

        qc1.gate(G1.RY, 1, angle=np.pi / 5)
        qc1.gate(G1.RZ, 1, angle=np.pi / 6)

        """CIRCUIT B"""
        qc2 = Circuit(2, dim=D2)
        G2 = qc2.gates[D2]

        for j in range(2):
            qc2.gate(G2.H, [j])

        if isinstance(op_sub2, str):
            if op_sub2 != "I":
                qc2.gate(coupler.B2[op_sub2], [0])
        else:
            qc2.gate(op_sub2, [0])

        qc2.gate(G2.RY, 0, angle=np.pi / 7)
        qc2.gate(G2.RZ, 0, angle=np.pi / 8)

        qc2.gate(G2.RY, 1, angle=np.pi / 9)
        qc2.gate(G2.RZ, 1, angle=np.pi / 10)

        psi1 = getState(qc1, flipped=False)
        psi2 = getState(qc2, flipped=False)
        c = coeff.item() if hasattr(coeff, "item") else coeff
        amp += c * np.kron(psi2, psi1)

    bases = [D2, D2, D1, D1]
    perm = [2, 3, 0, 1]
    return stitch(amp, bases, perm)

orig_probs = getProbs(original(), 4)
cut_probs = cut()

total_error = 0
keys = sorted(orig_probs.keys())

for k in keys:
    p_orig = orig_probs[k]
    p_cut = cut_probs.get(k, 0.0)
    diff = abs(p_orig - p_cut)

    total_error += diff

tvd = total_error / 2
terms = len(coupler.terms)
print(f"{D1},{D2},{args.t},{terms},{tvd:.5f}")