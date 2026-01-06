import torch as pt
import numpy as np


def GellMann(j, k, d):
    j0, k0 = j - 1, k - 1
    M = np.zeros((d, d), dtype=np.complex64)

    if j < k:
        M[j0, k0] = 1
        M[k0, j0] = 1

        return M
    # endif

    if j > k:
        M[j0, k0] = -1j
        M[k0, j0] = 1j

        return M
    # endif

    if j == k and j < d:
        n = j
        norm = np.sqrt(2 / (n * (n + 1)))
        M[:n, :n] = np.eye(n) * norm
        M[n, n] = -n * norm

        return M
    # endif
    raise ValueError("Invalid indices for Gell-Mann matrix.")


torchify = lambda x: pt.from_numpy(x).to(pt.complex64)


class G:
    @staticmethod
    def to(a, b, d=None):
        if d == 2:
            if (a, b) == (1, 0):
                return "X"
            elif (a, b) == (0, 1):
                return "Y"
            elif (a, b) == (1, 1):
                return "Z"

        return f"G({a},{b})"

    @staticmethod
    def frm(string: str):
        assert string.startswith("G(") and string.endswith(")")

        a_b = string[2:-1]
        a, b = a_b.split(",")

        return int(a), int(b)

    @staticmethod
    def basis(d):
        B = {"I": pt.eye(d, dtype=pt.complex64)}
        for a in range(1, d + 1):  # 1-based
            for b in range(1, d + 1):
                if a == d and b == d:
                    continue

                M = GellMann(a, b, d).T
                B[G.to(a, b, d)] = torchify(M)

        return B

    @staticmethod
    def Project(d):
        diags = [GellMann(j, j, d) for j in range(1, d)]
        diags = [torchify(M) for M in diags]

        I = pt.eye(d, dtype=pt.complex64)
        P = {}
        for r in range(d):
            Pr = I / d
            for L in diags:
                Pr = Pr + 0.5 * L[r, r] * L
            P[r] = Pr

        return P


def shift(d, r):
    X = pt.zeros((d, d), dtype=pt.complex64)
    for i in range(d):
        X[i, (i + r) % d] = 1
    return X


def decompose(A, basis, threshold):
    out = []
    for label, B in basis.items():
        num = pt.trace(A @ B.conj().T)
        den = pt.trace(B @ B.conj().T)
        c = num / den
        if abs(c) > threshold:
            out.append((label, c))

    return out


class Coupler:
    def __init__(self, dim1, dim2, threshold):
        self.d1 = dim1
        self.d2 = dim2

        self.B1 = G.basis(dim1)
        self.B2 = G.basis(dim2)

        self.P1 = G.Project(dim1)
        self.P2 = G.Project(dim2)

        self.terms = self._build_terms(threshold)

    def X(self, r, which=2):
        if which == 1:
            return shift(self.d1, r)
        else:
            return shift(self.d2, r)

    @property
    def CX(self):
        w = self.d1 * self.d2
        CX = pt.zeros((w, w), dtype=pt.complex64)
        for r in range(self.d1):
            A = self.P1[r]  # |r><r|
            B = shift(self.d2, r)  # X^r on target

            CX += pt.kron(A, B)

        return CX

    def _build_terms(self, threshold):
        terms = []
        for r in range(self.d1):
            A = self.P1[r]  # |r><r|
            B = shift(self.d2, r)  # X^r on target

            left = decompose(A, self.B1, threshold)
            right = decompose(B, self.B2, threshold)

            for a_label, a_c in left:
                for b_label, b_c in right:
                    terms.append((a_label, b_label, a_c * b_c))

        return terms
