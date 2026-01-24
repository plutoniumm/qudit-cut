from src.stats import getProbs, getState, stitch
from src.coupler import Coupler
from src.cache import Cache
from qudit import Circuit
from tqdm import tqdm
from time import time
import numpy as np
import gc

D1, D2 = 3, 3

# coupler = Coupler(D1, D2, method="optimal")
coupler = Coupler(D1, D2, method="basis")
C = Cache("./.cache/large.npz")
SZ, HF = 4, 2

def count(start: int, count: int):
    return [start + 2 * i for i in range(count)]

pset1 = count(3, HF)
pset2 = count(3 + 2 * HF, SZ - HF)

def original():
    qc = Circuit(SZ, dim=[D1] * HF + [D2] * HF)
    G1, G2 = qc.gates[D1], qc.gates[D2]

    for i in range(HF):
        qc.gate(G1.H, [i])
    for i in range(HF, SZ):
        qc.gate(G2.H, [i])

    qc.gate(coupler.CX, [HF - 1, HF])

    for i, ang in enumerate(pset1):
        qc.gate(G1.RY, i, angle=np.pi / ang)
        qc.gate(G1.RZ, i, angle=np.pi / (ang + 1))

    for j, ang in enumerate(pset2):
        qi = HF + j
        qc.gate(G2.RY, qi, angle=np.pi / ang)
        qc.gate(G2.RZ, qi, angle=np.pi / (ang + 1))

    return qc


def cut():
    terms = coupler.terms

    for t, (sub1, sub2, coeff) in tqdm(enumerate(terms), total=len(terms)):
        if C.has(f"term:{t}:psi1") and C.has(f"term:{t}:psi2") and C.has(f"term:{t}:c"):
            continue

        qc1 = Circuit(HF, dim=D1)
        G1 = qc1.gates[D1]

        for q in range(HF):
            qc1.gate(G1.H, [q])

        if isinstance(sub1, str):
            if sub1 != "I":
                qc1.gate(coupler.B1[sub1], [HF - 1])
        else:
            qc1.gate(sub1, [HF - 1])

        for i, ang in enumerate(pset1):
            qc1.gate(G1.RY, i, angle=np.pi / ang)
            qc1.gate(G1.RZ, i, angle=np.pi / (ang + 1))

        psi1 = getState(qc1, flipped=False)
        C.set(f"term:{t}:psi1", psi1)
        del qc1, psi1
        gc.collect()

        qc2 = Circuit(HF, dim=D2)
        G2 = qc2.gates[D2]

        for q in range(HF):
            qc2.gate(G2.H, [q])

        if isinstance(sub2, str):
            if sub2 != "I":
                qc2.gate(coupler.B2[sub2], [0])
        else:
            qc2.gate(sub2, [0])

        for j, ang in enumerate(pset2):
            qc2.gate(G2.RY, j, angle=np.pi / ang)
            qc2.gate(G2.RZ, j, angle=np.pi / (ang + 1))

        psi2 = getState(qc2, flipped=False)
        c = coeff.item() if hasattr(coeff, "item") else coeff

        C.set(f"term:{t}:psi2", psi2)
        C.set(f"term:{t}:c", np.array(c))

        del qc2, psi2
        gc.collect()

    n_full = (D1**HF) * (D2**HF)
    amp = np.zeros(n_full, dtype=np.complex64)

    for t, _ in enumerate(terms):
        psi1 = C.fetch(f"term:{t}:psi1")
        psi2 = C.fetch(f"term:{t}:psi2")
        c_arr = C.fetch(f"term:{t}:c")
        c = c_arr.item() if hasattr(c_arr, "item") else c_arr

        amp += c * np.kron(psi2, psi1)

        del psi1, psi2
        gc.collect()

    bases = [D2] * HF + [D1] * HF
    perm = list(range(HF, SZ)) + list(range(0, HF))
    return stitch(amp, bases, perm)


if __name__ == "__main__":
    print("\n[1] Running Original...")
    t0 = time()
    orig_probs = getProbs(original(), SZ)
    print(f"Original: {time() - t0:.3f}s")

    print("\n[2] Running Cut...")
    t1 = time()
    cut_probs = cut()
    print(f"Stitched: {time() - t1:.3f}s")

    print("             RESULTS COMPARISON           ")
    print(f"{'State':<12} | {'Original':<12} | {'Stitched':<12} | {'Diff':<10}")

    total_error = 0.0
    keys = sorted(orig_probs.keys())

    for k in keys:
        p_orig = orig_probs[k]
        p_cut = cut_probs.get(k, 0.0)
        diff = abs(p_orig - p_cut)
        total_error += diff
        if diff > 1e-3:
            print(f"|{k}>    | {p_orig:.5f}      | {p_cut:.5f}      | {diff:.5f}")

    print(f"TVD={total_error/2:.5f}")