"""LO cut and LOCC cut of CSUM_{d1,d2} on random circuits: overhead and largest TVD per pair of dimensions."""
import itertools
import numpy as np
import torch
from qudit import Circuit
from qudit.random import random_unitary

np.random.seed(2)


def haar(d):
    return random_unitary(d).numpy()


def run(dims, ket, *gates):
    qc = Circuit(len(dims), dim=dims)
    for g, idx in gates:
        qc.gate(torch.as_tensor(g), idx)
    return np.abs(qc(ket).detach().numpy().ravel()) ** 2


def shift(d, r):
    X = np.zeros((d, d), dtype=complex)
    for i in range(d):
        X[i, (i + r) % d] = 1
    return X


def csum(d1, d2):
    U = np.zeros((d1 * d2, d1 * d2), dtype=complex)
    for r in range(d1):
        P = np.zeros((d1, d1))
        P[r, r] = 1
        U += np.kron(P, shift(d2, r))
    return U


def probs(prep, kraus, post):
    d = len(prep)
    ket = np.eye(d)[0]
    return np.array([run([d], ket, (prep, [0]), (K, [0]), (post, [0])) for K in kraus])


def signed(prep, Kp, Km, post):
    p = probs(prep, [Kp, Km], post)
    return p[0] - p[1]


def lo_terms(d2):
    x = np.arange(d2)
    q = np.exp(2j * np.pi * x * x * ((d2 + 1) // 2) / d2) if d2 % 2 else np.exp(1j * np.pi * x * x / d2)
    u = np.array([np.mean(q.conj() * np.exp(-2j * np.pi * t * x / d2)) for t in range(d2)])
    F = np.exp(2j * np.pi * np.outer(x, x) / d2) / np.sqrt(d2)
    return q, u, F


def lo_cut(d1, d2, A0, A1, B0, B1):
    q, u, F = lo_terms(d2)
    x, r = np.arange(d2), np.arange(d1)
    A = [u[t] / abs(u[t]) * np.diag(q[r % d2] * np.exp(2j * np.pi * t * r / d2)) for t in range(d2)]
    B = [F @ np.diag(q * np.exp(-2j * np.pi * t * x / d2)) @ F.conj().T for t in range(d2)]
    c = 1 / np.sqrt(d2)
    p = np.zeros(d1 * d2)
    for j in range(d2):
        p += c * c * np.kron(probs(A0, [A[j]], A1)[0], probs(B0, [B[j]], B1)[0])
        for k in range(j + 1, d2):
            aR = signed(A0, (A[j] + A[k]) / 2, (A[j] - A[k]) / 2, A1)
            aI = signed(A0, (A[j] + 1j * A[k]) / 2, (A[j] - 1j * A[k]) / 2, A1)
            bR = signed(B0, (B[j] + B[k]) / 2, (B[j] - B[k]) / 2, B1)
            bI = signed(B0, (B[j] + 1j * B[k]) / 2, (B[j] - 1j * B[k]) / 2, B1)
            p += 2 * c * c * (np.kron(aR, bR) - np.kron(aI, bI))
    return p, 2 * d2 - 1


def locc_cut(d1, d2, A0, A1, B0, B1):
    m = min(d1, d2)
    cls = np.arange(d1) % m
    w = np.exp(2j * np.pi / m)
    X = [shift(d2, v) for v in range(m)]
    Zb = [np.diag(w ** (b * cls)) for b in range(m)]
    tp, n = np.zeros(d1 * d2), 0
    for ph in itertools.product(range(3), repeat=m - 1):
        th = np.exp(2j * np.pi * np.array((0,) + ph) / 3)
        E = [sum(th[v].conj() * w ** (-b * v) * X[v] for v in range(m)) / m for b in range(m)]
        PB = probs(B0, E, B1)
        for b in range(m):
            tp += np.kron(probs(A0, [Zb[b] @ np.diag(th[cls])], A1)[0], PB[b])
        n += 1
    tp /= n
    PA = probs(A0, [np.diag((cls == r).astype(complex)) for r in range(m)], A1)
    PBv = [probs(B0, [X[v]], B1)[0] for v in range(m)]
    tm = sum(np.kron(PA[r], sum(PBv[v] for v in range(m) if v != r) / (m - 1)) for r in range(m))
    return m * tp - (m - 1) * tm, 2 * m - 1


def main():
    print("d1,d2 | bound | LO: kappa, max TVD | LOCC: kappa, max TVD")
    for d1, d2 in [(2, 2), (2, 3), (3, 2), (2, 5), (3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (4, 4),
                   (4, 6), (6, 4), (5, 7), (7, 5), (6, 6)]:
        U = csum(d1, d2)
        e0 = np.zeros(d1 * d2)
        e0[0] = 1
        t_lo, t_locc = [], []
        for _ in range(10):
            A0, A1, B0, B1 = haar(d1), haar(d1), haar(d2), haar(d2)
            ref = run([d1, d2], e0, (A0, [0]), (B0, [1]), (U, [0, 1]), (A1, [0]), (B1, [1]))
            p, k_lo = lo_cut(d1, d2, A0, A1, B0, B1)
            q, k_locc = locc_cut(d1, d2, A0, A1, B0, B1)
            t_lo.append(abs(p - ref).sum() / 2)
            t_locc.append(abs(q - ref).sum() / 2)
        assert max(t_lo) < 1e-5 and max(t_locc) < 1e-5
        print(f"{d1},{d2} | {2 * min(d1, d2) - 1} | {k_lo:5.1f} {max(t_lo):.1e} | {k_locc:3} {max(t_locc):.1e}")


if __name__ == "__main__":
    main()
