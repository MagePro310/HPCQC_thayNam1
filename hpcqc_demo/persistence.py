from __future__ import annotations

import json
from pathlib import Path

from hpcqc_demo.models import JobRecord


class JobStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.jobs_dir = self.data_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def save(self, record: JobRecord) -> None:
        path = self.job_path(record.job_id)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def load(self, job_id: str) -> JobRecord | None:
        path = self.job_path(job_id)
        if not path.exists():
            return None
        return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self, limit: int | None = None) -> list[JobRecord]:
        records: list[JobRecord] = []
        for path in self.jobs_dir.glob("*.json"):
            records.append(JobRecord.model_validate_json(path.read_text(encoding="utf-8")))
        records.sort(key=lambda record: record.created_at, reverse=True)
        if limit is not None:
            return records[:limit]
        return records

