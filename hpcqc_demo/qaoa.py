from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import numpy as np

from hpcqc_demo.models import GraphSpec


def bits_from_index(index: int, num_nodes: int) -> tuple[int, ...]:
    return tuple((index >> node) & 1 for node in range(num_nodes))


def bitstring_from_index(index: int, num_nodes: int) -> str:
    return "".join(str(bit) for bit in bits_from_index(index, num_nodes))


def index_from_bitstring(bitstring: str) -> int:
    total = 0
    for node, bit in enumerate(bitstring):
        if bit == "1":
            total |= 1 << node
    return total


def maxcut_value(bits: Sequence[int], graph: GraphSpec) -> float:
    return sum(edge.weight for edge in graph.edges if bits[edge.u] != bits[edge.v])


def exact_maxcut(graph: GraphSpec) -> dict[str, float | str]:
    best_value = -1.0
    best_bitstring = ""
    for index in range(2**graph.num_nodes):
        bitstring = bitstring_from_index(index, graph.num_nodes)
        value = maxcut_value(bits_from_index(index, graph.num_nodes), graph)
        if value > best_value:
            best_value = value
            best_bitstring = bitstring
    return {"bitstring": best_bitstring, "value": best_value}


def _basis_costs(graph: GraphSpec) -> np.ndarray:
    return np.array(
        [maxcut_value(bits_from_index(index, graph.num_nodes), graph) for index in range(2**graph.num_nodes)],
        dtype=float,
    )


def _apply_mixer(state: np.ndarray, beta: float, num_nodes: int) -> np.ndarray:
    mixed = state.copy()
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    for node in range(num_nodes):
        next_state = mixed.copy()
        mask = 1 << node
        for index in range(2**num_nodes):
            if index & mask:
                continue
            paired = index | mask
            amp_zero = mixed[index]
            amp_one = mixed[paired]
            next_state[index] = cos_beta * amp_zero - 1j * sin_beta * amp_one
            next_state[paired] = -1j * sin_beta * amp_zero + cos_beta * amp_one
        mixed = next_state
    return mixed


def qaoa_probabilities(graph: GraphSpec, gammas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have the same length")
    num_states = 2**graph.num_nodes
    state = np.full(num_states, 1 / np.sqrt(num_states), dtype=complex)
    costs = _basis_costs(graph)

    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * gamma * costs)
        state = _apply_mixer(state, beta, graph.num_nodes)

    probabilities = np.abs(state) ** 2
    probabilities = probabilities / probabilities.sum()
    return probabilities


def expectation(probabilities: np.ndarray, graph: GraphSpec) -> float:
    costs = _basis_costs(graph)
    return float(np.dot(probabilities, costs))


def sample_counts(probabilities: np.ndarray, shots: int, rng: np.random.Generator, num_nodes: int) -> dict[str, int]:
    samples = rng.multinomial(shots, probabilities)
    return {
        bitstring_from_index(index, num_nodes): int(count)
        for index, count in enumerate(samples)
        if count > 0
    }


@dataclass(frozen=True)
class OptimizationResult:
    gammas: list[float]
    betas: list[float]
    expected_cut: float
    trace: list[dict[str, float | int | list[float]]]


def optimize_qaoa(graph: GraphSpec, p: int, steps: int, seed: int | None = None) -> OptimizationResult:
    rng = np.random.default_rng(seed)
    candidates: list[tuple[np.ndarray, np.ndarray]] = [
        (np.full(p, pi / 4), np.full(p, pi / 8)),
        (np.full(p, pi / 2), np.full(p, pi / 6)),
    ]
    for _ in range(max(0, steps - len(candidates))):
        candidates.append((rng.uniform(0, pi, size=p), rng.uniform(0, pi / 2, size=p)))

    best_gammas: np.ndarray | None = None
    best_betas: np.ndarray | None = None
    best_expectation = -1.0
    trace: list[dict[str, float | int | list[float]]] = []
    for step, (gammas, betas) in enumerate(candidates, start=1):
        probabilities = qaoa_probabilities(graph, gammas, betas)
        expected_cut = expectation(probabilities, graph)
        trace.append(
            {
                "step": step,
                "expected_cut": expected_cut,
                "gammas": [float(value) for value in gammas],
                "betas": [float(value) for value in betas],
            }
        )
        if expected_cut > best_expectation:
            best_expectation = expected_cut
            best_gammas = gammas.copy()
            best_betas = betas.copy()

    assert best_gammas is not None
    assert best_betas is not None
    return OptimizationResult(
        gammas=[float(value) for value in best_gammas],
        betas=[float(value) for value in best_betas],
        expected_cut=float(best_expectation),
        trace=trace,
    )

