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


def test_batch_submission_preserves_fifo_sequence(tmp_path) -> None:
    app = create_app(data_dir=tmp_path, start_worker=True, machine_count=1)
    client = app.test_client()

    response = client.post(
        "/rpc/jobs/batch",
        json={
            "jobs": [
                {
                    "title": "first",
                    "graph": {"num_nodes": 3, "edges": [{"u": 0, "v": 1}, {"u": 1, "v": 2}]},
                    "shots": 128,
                    "optimizer_steps": 4,
                    "seed": 1,
                    "simulated_runtime_seconds": 0,
                },
                {
                    "title": "second",
                    "graph": {"num_nodes": 3, "edges": [{"u": 0, "v": 1}, {"u": 0, "v": 2}]},
                    "shots": 128,
                    "optimizer_steps": 4,
                    "seed": 2,
                    "simulated_runtime_seconds": 0,
                },
            ]
        },
    )

    assert response.status_code == 202
    body = response.get_json()
    assert body["batch_id"].startswith("batch-")
    sequences = [job["sequence"] for job in body["jobs"]]
    assert sequences == sorted(sequences)

    manager = app.config["JOB_MANAGER"]
    records = [manager.wait_until_terminal(job["job_id"], timeout=5.0) for job in body["jobs"]]

    assert [record.request.title for record in records] == ["first", "second"]
    assert records[0].started_at <= records[1].started_at
    assert records[0].machine_id == "Machine 1"
    assert records[1].machine_id == "Machine 1"


def test_multiple_machines_record_timeline(tmp_path) -> None:
    app = create_app(data_dir=tmp_path, start_worker=True, machine_count=2)
    client = app.test_client()

    response = client.post(
        "/rpc/jobs/batch",
        json={
            "jobs": [
                {
                    "title": f"job-{index}",
                    "graph": {"num_nodes": 3, "edges": [{"u": 0, "v": 1}, {"u": 1, "v": 2}]},
                    "shots": 128,
                    "optimizer_steps": 4,
                    "seed": index,
                    "simulated_runtime_seconds": 0,
                }
                for index in range(4)
            ]
        },
    )

    assert response.status_code == 202
    jobs = response.get_json()["jobs"]
    manager = app.config["JOB_MANAGER"]
    records = [manager.wait_until_terminal(job["job_id"], timeout=5.0) for job in jobs]

    assert all(record.machine_id in {"Machine 1", "Machine 2"} for record in records)
    assert all(record.started_at and record.completed_at for record in records)
    assert all(record.queue_wait_seconds is not None for record in records)
    assert all(record.runtime_seconds is not None for record in records)

    timeline = client.get("/rpc/timeline")
    assert timeline.status_code == 200
    rows = timeline.get_json()["timeline"]
    assert len(rows) == 4
    assert {"job_id", "machine_id", "chart_start", "chart_end", "sequence"}.issubset(rows[0])


def test_machine_api_updates_worker_count(tmp_path) -> None:
    app = create_app(data_dir=tmp_path, start_worker=True, machine_count=1)
    client = app.test_client()

    response = client.post("/rpc/machines", json={"machine_count": 3})

    assert response.status_code == 200
    assert response.get_json()["machine_count"] == 3

    get_response = client.get("/rpc/machines")
    assert get_response.status_code == 200
    assert get_response.get_json()["machine_count"] == 3
