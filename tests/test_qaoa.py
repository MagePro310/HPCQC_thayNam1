from __future__ import annotations

import numpy as np

from hpcqc_demo.mitigation import apply_readout_noise, mitigate_counts
from hpcqc_demo.models import GraphSpec
from hpcqc_demo.qaoa import maxcut_value, optimize_qaoa, qaoa_probabilities, sample_counts


def test_maxcut_value_scores_cut_edges() -> None:
    graph = GraphSpec.model_validate({"num_nodes": 2, "edges": [{"u": 0, "v": 1, "weight": 2.0}]})

    assert maxcut_value((0, 1), graph) == 2.0
    assert maxcut_value((1, 1), graph) == 0.0


def test_qaoa_probabilities_are_normalized() -> None:
    graph = GraphSpec.model_validate({"num_nodes": 3, "edges": [{"u": 0, "v": 1}, {"u": 1, "v": 2}]})

    probabilities = qaoa_probabilities(graph, [0.5], [0.25])

    assert probabilities.shape == (8,)
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities >= 0)


def test_optimizer_returns_trace_and_angles() -> None:
    graph = GraphSpec.model_validate({"num_nodes": 3, "edges": [{"u": 0, "v": 1}, {"u": 1, "v": 2}]})

    result = optimize_qaoa(graph, p=1, steps=4, seed=11)

    assert len(result.gammas) == 1
    assert len(result.betas) == 1
    assert len(result.trace) == 4
    assert result.expected_cut >= 0


def test_mitigation_returns_normalized_distribution() -> None:
    graph = GraphSpec.model_validate({"num_nodes": 2, "edges": [{"u": 0, "v": 1}]})
    probabilities = qaoa_probabilities(graph, [0.5], [0.25])
    noisy = apply_readout_noise(probabilities, noise=0.1, num_nodes=2)
    counts = sample_counts(noisy, shots=512, rng=np.random.default_rng(3), num_nodes=2)

    mitigated = mitigate_counts(counts, num_nodes=2, noise=0.1)

    assert np.isclose(mitigated.sum(), 1.0)
    assert np.all(mitigated >= 0)

