"""Calibration data from Qiskit's snapshot backends, and two ways to simulate a Ramsey scan."""
import warnings
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_aer.noise import (NoiseModel, coherent_unitary_error, depolarizing_error,
                              thermal_relaxation_error)
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

FAMILIES = {27: "falcon", 127: "eagle", 133: "heron", 156: "heron"}
_SIM = AerSimulator(method="density_matrix")


@dataclass
class Qubit:
    backend: str
    index: int
    t1: float
    t2: float
    readout: float
    sx_error: float


def _backends():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [b for b in FakeProviderForBackendV2().backends()
                if b.num_qubits in FAMILIES and b.name != "fake_nighthawk"]


def sample_qubits(rng, per_backend=6):
    """Pick usable qubits. T2 is capped at 2*T1, the physical limit, as Aer does."""
    out = []
    for b in _backends():
        t = b.target
        for i in rng.permutation(b.num_qubits)[:per_backend]:
            i = int(i)
            q = t.qubit_properties[i]
            ro, sx = t["measure"][(i,)].error, t["sx"][(i,)].error
            if not (q.t1 and q.t2) or ro is None or sx is None:
                continue
            t2 = min(q.t2, 2 * q.t1)
            if ro < 0.15 and sx < 0.01 and t2 > 15e-6:
                out.append(Qubit(b.name, i, q.t1, t2, ro, sx))
    return out


def _hadamard(c):
    c.rz(np.pi / 2, 0)
    c.sx(0)
    c.rz(np.pi / 2, 0)


def fast_scan(q, taus, delta, f0):
    """Exact excited-state fractions from a one-qubit density-matrix simulation."""
    nm = NoiseModel(basis_gates=["sx", "rz", "unitary"])
    nm.add_quantum_error(depolarizing_error(q.sx_error, 1), "sx", [0])
    circuits = []
    for k, tau in enumerate(taus):
        a = 2 * np.pi * delta * tau
        rot = coherent_unitary_error(np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)]))
        nm.add_quantum_error(thermal_relaxation_error(q.t1, q.t2, tau).compose(rot), f"idle{k}", [0])
        c = QuantumCircuit(1)
        _hadamard(c)
        c.append(UnitaryGate(np.eye(2), label=f"idle{k}"), [0])
        c.rz(2 * np.pi * f0 * tau, 0)
        _hadamard(c)
        c.save_probabilities()
        circuits.append(c)
    res = _SIM.run(circuits, noise_model=nm, basis_gates=nm.basis_gates).result()
    p1 = np.array([res.data(i)["probabilities"][1] for i in range(len(circuits))])
    return p1 * (1 - q.readout) + (1 - p1) * q.readout


def full_stack_circuits(backend, physical_qubit, taus, delta, f0):
    """Ramsey circuits transpiled to native gates and scheduled for the backend.

    Snapshot noise models have no frequency error, so delta is injected as a virtual Z,
    which is how a detuning appears in the rotating frame.
    """
    circuits = []
    for tau in taus:
        c = QuantumCircuit(1, 1)
        c.h(0)
        c.delay(float(tau), 0, unit="s")
        c.rz(2 * np.pi * (f0 + delta) * float(tau), 0)
        c.h(0)
        c.measure(0, 0)
        circuits.append(c)
    return transpile(circuits, backend, initial_layout=[physical_qubit], optimization_level=1,
                     scheduling_method="alap")


def full_stack_noise(backend):
    """The noise model Aer derives from the backend's snapshot. Slow on large backends, so reuse it."""
    return AerSimulator.from_backend(backend, method="density_matrix").options.noise_model


def full_stack_sampler(noise, seed):
    return SamplerV2(seed=seed, options={"backend_options": {"method": "density_matrix", "noise_model": noise}})


def full_stack_scan(backend, sampler, physical_qubit, taus, delta, f0, shots):
    """Excited-state fractions and the transpiled circuits for one scan."""
    circuits = full_stack_circuits(backend, physical_qubit, taus, delta, f0)
    res = sampler.run(circuits, shots=shots).result()
    return np.array([r.data.c.get_int_counts().get(1, 0) / shots for r in res]), circuits
