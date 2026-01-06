import numpy as np
from scripts.replication import QuantumCircuit
from qiskit_aer import AerSimulator
from collections import defaultdict

SIMULATOR = AerSimulator()
SHOTS = 100000


def run_and_get_probs(qc, num_qubits):
    """
    Runs the circuit on the simulator and returns a dictionary
    of probabilities for all possible bitstrings.
    """
    result = SIMULATOR.run(qc, shots=SHOTS).result()
    counts = result.get_counts()

    probs = defaultdict(float)
    total_shots = sum(counts.values())

    for i in range(2**num_qubits):
        bin_str = format(i, f"0{num_qubits}b")
        probs[bin_str] = counts.get(bin_str, 0) / total_shots

    return probs


def print_distribution(probs, title="Distribution"):
    print(f"\n--- {title} ---")
    sorted_keys = sorted(probs.keys())
    for k in sorted_keys:
        if probs[k] > 0.001:
            print(f"State |{k}> : {probs[k]:.4f}")


def make_original_circuit():
    """
    Creates a 4-qubit circuit with entanglement between Q1 and Q2.
    Topology: 0-1-2-3
    """
    qc = QuantumCircuit(4)

    qc.h(range(4))

    qc.cx(1, 2)

    qc.ry(np.pi / 3, 0)
    qc.rz(np.pi / 4, 1)
    qc.ry(np.pi / 5, 2)
    qc.rz(np.pi / 6, 3)

    qc.measure_all()
    return qc


def run_cut_simulation():
    """
    Runs the cut version of the circuit.

    Decomposition of CNOT(c, t):
    CX = 0.5 * (I_c I_t + Z_c I_t + I_c X_t - Z_c X_t)

    We need to run 4 pairs of sub-circuits:
    Pair 1: Sub1(Identity)  + Sub2(Identity)  [Weight +0.5]
    Pair 2: Sub1(Z Gate)    + Sub2(Identity)  [Weight +0.5]
    Pair 3: Sub1(Identity)  + Sub2(X Gate)    [Weight +0.5]
    Pair 4: Sub1(Z Gate)    + Sub2(X Gate)    [Weight -0.5]
    """

    print("Running CutQC Simulation...")

    terms = [("I", "I", 0.5), ("Z", "I", 0.5), ("I", "X", 0.5), ("Z", "X", -0.5)]

    stitched_probs = defaultdict(float)
    for i in range(16):
        stitched_probs[format(i, "04b")] = 0.0

    for i, (op_sub1, op_sub2, coeff) in enumerate(terms):
        print(f"  Executing Term {i+1}/4: Q1={op_sub1}, Q2={op_sub2}, Coeff={coeff}")

        qc1 = QuantumCircuit(2)
        qc1.h(0)
        qc1.h(1)

        if op_sub1 == "Z":
            qc1.z(1)

        qc1.ry(np.pi / 3, 0)
        qc1.rz(np.pi / 4, 1)
        qc1.measure_all()

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.h(1)

        if op_sub2 == "X":
            qc2.x(0)

        qc2.ry(np.pi / 5, 0)
        qc2.rz(np.pi / 6, 1)
        qc2.measure_all()

        probs1 = run_and_get_probs(qc1, 2)
        probs2 = run_and_get_probs(qc2, 2)

        for s1, p1 in probs1.items():
            for s2, p2 in probs2.items():

                full_state = s2 + s1

                stitched_probs[full_state] += coeff * p1 * p2

    return stitched_probs


if __name__ == "__main__":
    print("==========================================")
    print("      CutQC Replication Experiment")
    print("==========================================")

    qc_orig = make_original_circuit()
    print("\n[1] Running Original Uncut Circuit (4 Qubits)...")
    orig_probs = run_and_get_probs(qc_orig, 4)

    print("\n[2] Running Cut Circuits & Stitching Results...")
    cut_probs = run_cut_simulation()

    print("\n==========================================")
    print("             RESULTS COMPARISON           ")
    print("==========================================")
    print(f"{'State':<10} | {'Original':<12} | {'Stitched (CutQC)':<15} | {'Diff':<10}")
    print("-" * 55)

    total_error = 0
    keys = sorted(orig_probs.keys())

    for k in keys:
        p_orig = orig_probs[k]
        p_cut = cut_probs[k]
        diff = abs(p_orig - p_cut)
        total_error += diff

        print(f"|{k}>    | {p_orig:.5f}      | {p_cut:.5f}           | {diff:.5f}")

    print("-" * 55)
    print(f"Total Variation Distance (Sum|diff|/2): {total_error/2:.5f}")

    if total_error / 2 < 0.01:
        print(
            "\nSUCCESS: Stitched distribution matches original within statistical noise."
        )
    else:
        print("\nWARNING: Discrepancy detected (may be due to shot noise).")
