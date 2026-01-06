from collections import defaultdict
from .bases import flip, to_mixed
import numpy as np

def getProbs(qc, num_qudits):
    ket0 = np.zeros(qc.width)
    ket0[0] = 1.0

    state = qc(ket0).cpu().detach().numpy().flatten()
    state = flip(state, base=qc.dim)

    probs = defaultdict(float)

    base = qc.dim
    if isinstance(base, (list, tuple, np.ndarray)):
        bases = list(base)[::-1]
    else:
        bases = [int(base)] * int(num_qudits)

    total_len = int(np.prod(bases))

    for i in range(total_len):
        digits = to_mixed(i, bases)
        bstr = "".join(str(d) for d in digits[::-1])
        probs[bstr] = float(np.abs(state[i]) ** 2)

    return probs


def getState(qc, flipped: bool = False):
    ket0 = np.zeros(qc.width)
    ket0[0] = 1.0
    state = qc(ket0).cpu().detach().numpy().flatten()

    if flipped:
        return flip(state, base=qc.dim)

    return state


def stitch(amp, bases, perm=None):
    probs = defaultdict(float)
    total_len = len(amp)

    for i in range(total_len):
        digits = to_mixed(i, bases)

        if perm:
            digits = [digits[p] for p in perm]

        bstr = "".join(str(d) for d in digits)
        probs[bstr] = float(np.abs(amp[i]) ** 2)

    return probs