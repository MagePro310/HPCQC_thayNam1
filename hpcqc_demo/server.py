from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request
from pydantic import ValidationError

from hpcqc_demo.config import DEFAULT_DATA_DIR
from hpcqc_demo.jobs import JobManager
from hpcqc_demo.models import JobBatchRequest, JobRequest


def create_app(data_dir: Path | None = None, start_worker: bool = True, machine_count: int = 3) -> Flask:
    app = Flask(__name__)
    manager = JobManager(data_dir or DEFAULT_DATA_DIR, start_worker=start_worker, machine_count=machine_count)
    app.config["JOB_MANAGER"] = manager

    @app.get("/rpc/health")
    def health():
        return jsonify(manager.health())

    @app.post("/rpc/jobs")
    def submit_job():
        payload = request.get_json(silent=True) or {}
        try:
            job_request = JobRequest.model_validate(payload)
        except ValidationError as exc:
            return jsonify({"error": "invalid job request", "details": exc.errors()}), 400
        record = manager.submit(job_request)
        return jsonify(record.model_dump(mode="json")), 202

    @app.post("/rpc/jobs/batch")
    def submit_batch():
        payload = request.get_json(silent=True) or {}
        try:
            batch_request = JobBatchRequest.model_validate(payload)
        except ValidationError as exc:
            return jsonify({"error": "invalid batch request", "details": exc.errors()}), 400
        batch_id, records = manager.submit_batch(batch_request.jobs)
        return jsonify(
            {
                "batch_id": batch_id,
                "jobs": [record.model_dump(mode="json") for record in records],
            }
        ), 202

    @app.get("/rpc/jobs")
    def list_jobs():
        limit = request.args.get("limit", default=50, type=int)
        jobs = [record.model_dump(mode="json") for record in manager.list(limit=limit)]
        return jsonify({"jobs": jobs})

    @app.get("/rpc/jobs/<job_id>")
    def get_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(record.model_dump(mode="json"))

    @app.get("/rpc/timeline")
    def timeline():
        return jsonify({"timeline": manager.timeline()})

    @app.get("/rpc/machines")
    def get_machines():
        return jsonify(manager.machine_summary())

    @app.post("/rpc/machines")
    def set_machines():
        payload = request.get_json(silent=True) or {}
        machine_count = payload.get("machine_count")
        if not isinstance(machine_count, int):
            return jsonify({"error": "machine_count must be an integer"}), 400
        try:
            summary = manager.set_machine_count(machine_count)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(summary)

    return app


def main() -> None:
    host = os.environ.get("HPCQC_HOST", "127.0.0.1")
    port = int(os.environ.get("HPCQC_PORT", "5050"))
    app = create_app()
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
