# HPCQC RPC MVP Demo

Python MVP for the HPCQC architecture diagram. It demonstrates a Streamlit user system submitting hybrid QAOA MaxCut jobs to a Flask HTTP-RPC server, where jobs pass through a queue, classical optimizer, simulated QPU, calibration/error-mitigation step, and local JSON persistence.

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

## Test

```bash
conda activate HPQCthayNam
pytest
```

## API

- `GET /rpc/health`
- `POST /rpc/jobs`
- `GET /rpc/jobs`
- `GET /rpc/jobs/{job_id}`

