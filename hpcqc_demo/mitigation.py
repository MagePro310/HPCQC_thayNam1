from __future__ import annotations

from itertools import product

import numpy as np

from hpcqc_demo.qaoa import bitstring_from_index, index_from_bitstring


def readout_transition_matrix(num_nodes: int, noise: float) -> np.ndarray:
    size = 2**num_nodes
    matrix = np.zeros((size, size), dtype=float)
    for true_index in range(size):
        true_bits = bitstring_from_index(true_index, num_nodes)
        for measured_bits in product("01", repeat=num_nodes):
            measured = "".join(measured_bits)
            distance = sum(a != b for a, b in zip(true_bits, measured))
            probability = (noise**distance) * ((1 - noise) ** (num_nodes - distance))
            matrix[index_from_bitstring(measured), true_index] = probability
    return matrix


def apply_readout_noise(probabilities: np.ndarray, noise: float, num_nodes: int) -> np.ndarray:
    noisy = readout_transition_matrix(num_nodes, noise) @ probabilities
    return noisy / noisy.sum()


def counts_to_probabilities(counts: dict[str, int], num_nodes: int) -> np.ndarray:
    total = sum(counts.values())
    probabilities = np.zeros(2**num_nodes, dtype=float)
    if total == 0:
        return probabilities
    for bitstring, count in counts.items():
        probabilities[index_from_bitstring(bitstring)] = count / total
    return probabilities


def probabilities_to_dict(probabilities: np.ndarray, num_nodes: int, min_probability: float = 0.0) -> dict[str, float]:
    return {
        bitstring_from_index(index, num_nodes): float(probability)
        for index, probability in enumerate(probabilities)
        if probability >= min_probability
    }


def mitigate_counts(counts: dict[str, int], num_nodes: int, noise: float) -> np.ndarray:
    measured = counts_to_probabilities(counts, num_nodes)
    if measured.sum() == 0:
        return measured
    if noise == 0:
        return measured
    matrix = readout_transition_matrix(num_nodes, noise)
    mitigated = np.linalg.pinv(matrix) @ measured
    mitigated = np.clip(mitigated, 0.0, None)
    if mitigated.sum() == 0:
        return measured
    return mitigated / mitigated.sum()


def top_outcomes(probabilities: np.ndarray, num_nodes: int, limit: int = 8) -> list[dict[str, float]]:
    ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
    return [
        {"bitstring": bitstring_from_index(index, num_nodes), "probability": float(probability)}
        for index, probability in ranked[:limit]
        if probability > 0
    ]

