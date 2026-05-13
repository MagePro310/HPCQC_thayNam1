from __future__ import annotations

from collections import deque
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


def seconds_between(start: str | None, end: str | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())


class JobManager:
    def __init__(self, data_dir: Path, start_worker: bool = True, machine_count: int = 3) -> None:
        self.store = JobStore(data_dir)
        self.pending: deque[str] = deque()
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.sequence = int(time.time() * 1000)
        self.stop_event = threading.Event()
        self.machine_count = self._validated_machine_count(machine_count)
        self.workers: dict[int, threading.Thread] = {}
        if start_worker:
            self.start()

    def start(self) -> None:
        self.set_machine_count(self.machine_count)

    def set_machine_count(self, machine_count: int) -> dict[str, int | list[str]]:
        count = self._validated_machine_count(machine_count)
        to_start: list[int] = []
        with self.condition:
            self.machine_count = count
            for machine_number in range(1, count + 1):
                worker = self.workers.get(machine_number)
                if worker is None or not worker.is_alive():
                    to_start.append(machine_number)

        for machine_number in to_start:
            worker = threading.Thread(
                target=self._worker_loop,
                args=(machine_number,),
                name=f"hpcqc-machine-{machine_number}",
                daemon=True,
            )
            with self.condition:
                self.workers[machine_number] = worker
            worker.start()

        with self.condition:
            self.condition.notify_all()
            return self.machine_summary()

    def machine_summary(self) -> dict[str, int | list[str]]:
        with self.lock:
            active = [
                f"Machine {machine_number}"
                for machine_number, worker in sorted(self.workers.items())
                if machine_number <= self.machine_count and worker.is_alive()
            ]
            return {
                "machine_count": self.machine_count,
                "active_machines": active,
            }

    def submit(self, request: JobRequest) -> JobRecord:
        return self._submit_many([request], batch_id=None)[0]

    def submit_batch(self, requests: list[JobRequest]) -> tuple[str, list[JobRecord]]:
        batch_id = f"batch-{uuid.uuid4().hex[:12]}"
        records = self._submit_many(requests, batch_id=batch_id)
        return batch_id, records

    def _submit_many(self, requests: list[JobRequest], batch_id: str | None) -> list[JobRecord]:
        with self.condition:
            records: list[JobRecord] = []
            now = utc_now()
            accepted_log = "Job Queue: accepted job into shared FIFO queue."
            if batch_id is not None:
                accepted_log = f"Job Queue: accepted batch {batch_id} into shared FIFO queue."
            for request in requests:
                self.sequence += 1
                job_id = f"job-{uuid.uuid4().hex[:12]}"
                record = JobRecord(
                    job_id=job_id,
                    state=JobState.QUEUED,
                    request=request,
                    created_at=now,
                    updated_at=now,
                    sequence=self.sequence,
                    batch_id=batch_id,
                    queued_at=now,
                    logs=[accepted_log],
                    artifacts={"record": str(self.store.job_path(job_id))},
                )
                self.store.save(record)
                self.pending.append(job_id)
                records.append(record)
            self.condition.notify_all()
            return records

    @staticmethod
    def _validated_machine_count(machine_count: int) -> int:
        if not 1 <= machine_count <= 4:
            raise ValueError("machine_count must be between 1 and 4")
        return machine_count

    def queue_size(self) -> int:
        with self.lock:
            return len(self.pending)

    def health(self) -> dict[str, int | bool | str | list[str]]:
        machines = self.machine_summary()
        return {
            "status": "ok",
            "queued": self.queue_size(),
            "worker_alive": bool(machines["active_machines"]),
            "data_dir": str(self.store.data_dir),
            **machines,
        }

    def get(self, job_id: str) -> JobRecord | None:
        return self.store.load(job_id)

    def list(self, limit: int | None = 50) -> list[JobRecord]:
        return self.store.list(limit=limit)

    def timeline(self) -> list[dict[str, str | int | float | None]]:
        rows: list[dict[str, str | int | float | None]] = []
        now = utc_now()
        for record in self.store.list(limit=None):
            if record.machine_id is None or record.started_at is None:
                continue
            end = record.completed_at or now
            final = record.result.get("final", {}) if record.result else {}
            runtime = record.runtime_seconds
            if runtime is None:
                runtime = seconds_between(record.started_at, end)
            rows.append(
                {
                    "job_id": record.job_id,
                    "title": record.request.title,
                    "state": record.state.value,
                    "batch_id": record.batch_id,
                    "machine_id": record.machine_id,
                    "sequence": record.sequence,
                    "queued_at": record.queued_at,
                    "started_at": record.started_at,
                    "completed_at": record.completed_at,
                    "chart_start": record.started_at,
                    "chart_end": end,
                    "queue_wait_seconds": record.queue_wait_seconds,
                    "runtime_seconds": runtime,
                    "simulated_runtime_seconds": record.request.simulated_runtime_seconds,
                    "best_cut_value": final.get("best_cut_value"),
                }
            )
        return sorted(rows, key=lambda row: (str(row["machine_id"]), int(row["sequence"])))

    def wait_until_terminal(self, job_id: str, timeout: float = 10.0) -> JobRecord:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(job_id)
            if record and record.state in {JobState.COMPLETED, JobState.FAILED}:
                return record
            time.sleep(0.05)
        raise TimeoutError(f"job {job_id} did not finish within {timeout} seconds")

    def _worker_loop(self, machine_number: int) -> None:
        while not self.stop_event.is_set():
            with self.condition:
                while not self.pending and not self.stop_event.is_set() and machine_number <= self.machine_count:
                    self.condition.wait(timeout=0.2)
                if self.stop_event.is_set() or machine_number > self.machine_count:
                    return
                job_id = self.pending.popleft()
            self._run(machine_number, job_id)

    def _append_log(self, record: JobRecord, message: str) -> JobRecord:
        stamped = f"{utc_now()} | {message}"
        updated = record.model_copy(update={"logs": [*record.logs, stamped], "updated_at": utc_now()})
        self.store.save(updated)
        return updated

    def _run(self, machine_number: int, job_id: str) -> None:
        machine_id = f"Machine {machine_number}"
        with self.lock:
            record = self.store.load(job_id)
            if record is None:
                return
            now = utc_now()
            logs = [
                *record.logs,
                f"{now} | Scheduler: {machine_id} started FIFO sequence {record.sequence}.",
            ]
            record = record.model_copy(
                update={
                    "state": JobState.RUNNING,
                    "started_at": now,
                    "machine_id": machine_id,
                    "queue_wait_seconds": seconds_between(record.queued_at or record.created_at, now),
                    "updated_at": now,
                    "logs": logs,
                }
            )
            self.store.save(record)

        def add_log(message: str) -> None:
            nonlocal record
            with self.lock:
                latest = self.store.load(job_id)
                if latest is not None:
                    record = self._append_log(latest, message)

        started_monotonic = time.monotonic()
        try:
            result = run_hpcqc_job(record.request, record.job_id, add_log)
            elapsed = time.monotonic() - started_monotonic
            sleep_for = max(0.0, record.request.simulated_runtime_seconds - elapsed)
            if sleep_for > 0:
                add_log(f"{machine_id}: holding simulated runtime for {sleep_for:.2f}s.")
                time.sleep(sleep_for)

            with self.lock:
                latest = self.store.load(job_id) or record
                now = utc_now()
                completed = latest.model_copy(
                    update={
                        "state": JobState.COMPLETED,
                        "result": result,
                        "completed_at": now,
                        "updated_at": now,
                        "runtime_seconds": seconds_between(latest.started_at, now),
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
                        "runtime_seconds": seconds_between(latest.started_at, now),
                    }
                )
                self.store.save(failed)
