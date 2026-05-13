from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(os.environ.get("HPCQC_DATA_DIR", PROJECT_ROOT / "hpcqc_data"))
DEFAULT_BACKEND_URL = os.environ.get("HPCQC_BACKEND_URL", "http://127.0.0.1:5050")

DEFAULT_GRAPH = {
    "num_nodes": 4,
    "edges": [
        {"u": 0, "v": 1, "weight": 1.0},
        {"u": 1, "v": 2, "weight": 1.0},
        {"u": 2, "v": 3, "weight": 1.0},
        {"u": 3, "v": 0, "weight": 1.0},
        {"u": 0, "v": 2, "weight": 0.5},
    ],
}

