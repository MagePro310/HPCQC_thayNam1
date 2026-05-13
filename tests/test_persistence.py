from __future__ import annotations

from hpcqc_demo.jobs import utc_now
from hpcqc_demo.models import JobRecord, JobRequest, JobState
from hpcqc_demo.persistence import JobStore


def test_job_store_round_trip(tmp_path) -> None:
    store = JobStore(tmp_path)
    now = utc_now()
    record = JobRecord(
        job_id="job-test",
        state=JobState.QUEUED,
        request=JobRequest(),
        created_at=now,
        updated_at=now,
        sequence=1,
    )

    store.save(record)
    loaded = store.load("job-test")

    assert loaded is not None
    assert loaded.job_id == "job-test"
    assert loaded.request.graph.num_nodes == 4
    assert store.list()[0].job_id == "job-test"

