from __future__ import annotations

from hpcqc_demo.server import create_app


def test_submit_job_runs_to_completion(tmp_path) -> None:
    app = create_app(data_dir=tmp_path, start_worker=True)
    client = app.test_client()

    response = client.post(
        "/rpc/jobs",
        json={
            "title": "integration",
            "graph": {"num_nodes": 3, "edges": [{"u": 0, "v": 1}, {"u": 1, "v": 2}]},
            "p": 1,
            "shots": 128,
            "noise": 0.05,
            "priority": 5,
            "optimizer_steps": 4,
            "seed": 2,
        },
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    manager = app.config["JOB_MANAGER"]
    record = manager.wait_until_terminal(job_id, timeout=5.0)

    assert record.state == "completed"
    assert record.result is not None
    assert record.result["algorithm"] == "QAOA MaxCut"

    fetch = client.get(f"/rpc/jobs/{job_id}")
    assert fetch.status_code == 200
    assert fetch.get_json()["state"] == "completed"

