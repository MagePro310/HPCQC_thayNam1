from __future__ import annotations

import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from hpcqc_demo.models import JobRecord, JobRequest, JobState
from hpcqc_demo.persistence import JobStore
from hpcqc_demo.pipeline import run_hpcqc_job


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self, data_dir: Path, start_worker: bool = True) -> None:
        self.store = JobStore(data_dir)
        self.queue: queue.PriorityQueue[tuple[int, int, str]] = queue.PriorityQueue()
        self.lock = threading.RLock()
        self.sequence = int(time.time() * 1000)
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        if start_worker:
            self.start()

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.worker = threading.Thread(target=self._worker_loop, name="hpcqc-job-worker", daemon=True)
        self.worker.start()

    def submit(self, request: JobRequest) -> JobRecord:
        with self.lock:
            self.sequence += 1
            job_id = f"job-{uuid.uuid4().hex[:12]}"
            now = utc_now()
            record = JobRecord(
                job_id=job_id,
                state=JobState.QUEUED,
                request=request,
                created_at=now,
                updated_at=now,
                sequence=self.sequence,
                logs=["Job Queue: accepted job into priority/FIFO queue."],
                artifacts={"record": str(self.store.job_path(job_id))},
            )
            self.store.save(record)
            self.queue.put((request.priority, record.sequence, job_id))
            return record

    def get(self, job_id: str) -> JobRecord | None:
        return self.store.load(job_id)

    def list(self, limit: int | None = 50) -> list[JobRecord]:
        return self.store.list(limit=limit)

    def health(self) -> dict[str, int | bool | str]:
        return {
            "status": "ok",
            "queued": self.queue.qsize(),
            "worker_alive": bool(self.worker and self.worker.is_alive()),
            "data_dir": str(self.store.data_dir),
        }

    def wait_until_terminal(self, job_id: str, timeout: float = 10.0) -> JobRecord:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(job_id)
            if record and record.state in {JobState.COMPLETED, JobState.FAILED}:
                return record
            time.sleep(0.05)
        raise TimeoutError(f"job {job_id} did not finish within {timeout} seconds")

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                _, _, job_id = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._run(job_id)
            finally:
                self.queue.task_done()

    def _append_log(self, record: JobRecord, message: str) -> JobRecord:
        stamped = f"{utc_now()} | {message}"
        updated = record.model_copy(update={"logs": [*record.logs, stamped], "updated_at": utc_now()})
        self.store.save(updated)
        return updated

    def _run(self, job_id: str) -> None:
        with self.lock:
            record = self.store.load(job_id)
            if record is None:
                return
            now = utc_now()
            record = record.model_copy(update={"state": JobState.RUNNING, "started_at": now, "updated_at": now})
            self.store.save(record)

        def add_log(message: str) -> None:
            nonlocal record
            with self.lock:
                latest = self.store.load(job_id)
                if latest is not None:
                    record = self._append_log(latest, message)

        try:
            result = run_hpcqc_job(record.request, record.job_id, add_log)
            with self.lock:
                latest = self.store.load(job_id) or record
                now = utc_now()
                completed = latest.model_copy(
                    update={
                        "state": JobState.COMPLETED,
                        "result": result,
                        "completed_at": now,
                        "updated_at": now,
                    }
                )
                self.store.save(completed)
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            with self.lock:
                latest = self.store.load(job_id) or record
                now = utc_now()
                failed = latest.model_copy(
                    update={
                        "state": JobState.FAILED,
                        "error": str(exc),
                        "completed_at": now,
                        "updated_at": now,
                    }
                )
                self.store.save(failed)

