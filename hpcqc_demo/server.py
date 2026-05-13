from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request
from pydantic import ValidationError

from hpcqc_demo.config import DEFAULT_DATA_DIR
from hpcqc_demo.jobs import JobManager
from hpcqc_demo.models import JobRequest


def create_app(data_dir: Path | None = None, start_worker: bool = True) -> Flask:
    app = Flask(__name__)
    manager = JobManager(data_dir or DEFAULT_DATA_DIR, start_worker=start_worker)
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

    return app


def main() -> None:
    host = os.environ.get("HPCQC_HOST", "127.0.0.1")
    port = int(os.environ.get("HPCQC_PORT", "5050"))
    app = create_app()
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

