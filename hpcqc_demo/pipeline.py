from __future__ import annotations

from typing import Callable

import numpy as np

from hpcqc_demo.mitigation import (
    apply_readout_noise,
    counts_to_probabilities,
    mitigate_counts,
    probabilities_to_dict,
    readout_transition_matrix,
    top_outcomes,
)
from hpcqc_demo.models import JobRequest
from hpcqc_demo.qaoa import (
    bitstring_from_index,
    exact_maxcut,
    maxcut_value,
    optimize_qaoa,
    qaoa_probabilities,
    sample_counts,
)


def run_hpcqc_job(request: JobRequest, job_id: str, add_log: Callable[[str], None]) -> dict:
    graph = request.graph
    add_log("File System: loaded job input graph and run parameters.")
    add_log("Job Queue: scheduler assigned the job to a classical worker.")

    add_log("Classical processing: optimizing QAOA angles for MaxCut.")
    optimization = optimize_qaoa(graph, request.p, request.optimizer_steps, request.seed)
    add_log(f"Classical processing: best expected cut is {optimization.expected_cut:.4f}.")

    add_log("Quantum queue: transpiled circuit submitted to simulated QPU.")
    ideal_probabilities = qaoa_probabilities(graph, optimization.gammas, optimization.betas)
    noisy_probabilities = apply_readout_noise(ideal_probabilities, request.noise, graph.num_nodes)
    rng = np.random.default_rng(request.seed)
    raw_counts = sample_counts(noisy_probabilities, request.shots, rng, graph.num_nodes)
    raw_probabilities = counts_to_probabilities(raw_counts, graph.num_nodes)
    add_log(f"QPU: completed {request.shots} noisy measurement shots.")

    add_log("Calibration & Error Mitigation: applying readout calibration matrix.")
    mitigated_probabilities = mitigate_counts(raw_counts, graph.num_nodes, request.noise)
    matrix = readout_transition_matrix(graph.num_nodes, request.noise)

    exact = exact_maxcut(graph)
    best_index = int(np.argmax(mitigated_probabilities))
    best_bitstring = bitstring_from_index(best_index, graph.num_nodes)
    best_bits = tuple(int(bit) for bit in best_bitstring)
    best_cut = maxcut_value(best_bits, graph)
    approximation_ratio = best_cut / exact["value"] if exact["value"] else 0.0
    add_log("Result: final mitigated distribution and best bitstring are ready.")

    return {
        "job_id": job_id,
        "algorithm": "QAOA MaxCut",
        "graph": graph.model_dump(mode="json"),
        "parameters": {
            "p": request.p,
            "shots": request.shots,
            "noise": request.noise,
            "optimizer_steps": request.optimizer_steps,
            "seed": request.seed,
        },
        "optimizer": {
            "best_gammas": optimization.gammas,
            "best_betas": optimization.betas,
            "best_expected_cut": optimization.expected_cut,
            "trace": optimization.trace,
        },
        "qpu": {
            "raw_counts": raw_counts,
            "raw_probabilities": probabilities_to_dict(raw_probabilities, graph.num_nodes, min_probability=0.0),
            "top_raw": top_outcomes(raw_probabilities, graph.num_nodes),
        },
        "mitigation": {
            "model": "independent readout bit-flip",
            "noise": request.noise,
            "calibration_matrix_shape": list(matrix.shape),
            "mitigated_probabilities": probabilities_to_dict(
                mitigated_probabilities,
                graph.num_nodes,
                min_probability=0.0,
            ),
            "top_mitigated": top_outcomes(mitigated_probabilities, graph.num_nodes),
        },
        "final": {
            "best_bitstring": best_bitstring,
            "best_cut_value": best_cut,
            "best_probability": float(mitigated_probabilities[best_index]),
            "exact_best_bitstring": exact["bitstring"],
            "exact_best_cut_value": exact["value"],
            "approximation_ratio": approximation_ratio,
        },
    }

