# HPCQC RPC MVP Demo

Python MVP for the HPCQC architecture diagram. It demonstrates a Streamlit user system submitting hybrid QAOA MaxCut jobs to a Flask HTTP-RPC server, where jobs pass through a shared FIFO queue, configurable simulated machines, classical optimizer, simulated QPU, calibration/error-mitigation step, Gantt timeline, and local JSON persistence.

## Run

```bash
conda activate HPQCthayNam
python -m hpcqc_demo.server
```

In a second terminal:

```bash
conda activate HPQCthayNam
streamlit run hpcqc_demo/ui.py
```

The backend defaults to `http://127.0.0.1:5050`. Job artifacts are stored under `hpcqc_data/jobs/`.

The Streamlit dashboard supports:

- Single QAOA job submission.
- Random batch generation and submission.
- Configurable worker machine count, 1-4 machines.
- FIFO queue table sorted by sequence.
- Machine Gantt chart from backend timeline data.

## Test

```bash
conda activate HPQCthayNam
pytest
```

## API

- `GET /rpc/health`
- `POST /rpc/jobs`
- `POST /rpc/jobs/batch`
- `GET /rpc/jobs`
- `GET /rpc/jobs/{job_id}`
- `GET /rpc/timeline`
- `GET /rpc/machines`
- `POST /rpc/machines`
