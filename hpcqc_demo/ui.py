from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
import streamlit as st

from hpcqc_demo.config import DEFAULT_BACKEND_URL, DEFAULT_GRAPH


def parse_edges(text: str) -> list[dict[str, float | int]]:
    edges: list[dict[str, float | int]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) not in {2, 3}:
            raise ValueError(f"Invalid edge line: {line}")
        u = int(parts[0])
        v = int(parts[1])
        weight = float(parts[2]) if len(parts) == 3 else 1.0
        edges.append({"u": u, "v": v, "weight": weight})
    return edges


def request_json(method: str, backend_url: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
    try:
        response = requests.request(method, f"{backend_url.rstrip('/')}{path}", timeout=10, **kwargs)
    except requests.RequestException as exc:
        return 0, {"error": str(exc)}
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": response.text}
    return response.status_code, payload


def render_result(result: dict[str, Any]) -> None:
    final = result["final"]
    cols = st.columns(4)
    cols[0].metric("Best bitstring", final["best_bitstring"])
    cols[1].metric("Best cut", f"{final['best_cut_value']:.3f}")
    cols[2].metric("Best probability", f"{final['best_probability']:.3f}")
    cols[3].metric("Approx. ratio", f"{final['approximation_ratio']:.3f}")

    st.subheader("Raw vs mitigated outcomes")
    raw = pd.DataFrame(result["qpu"]["top_raw"])
    mitigated = pd.DataFrame(result["mitigation"]["top_mitigated"])
    left, right = st.columns(2)
    with left:
        st.caption("Raw noisy QPU samples")
        st.dataframe(raw, hide_index=True, use_container_width=True)
        if not raw.empty:
            st.bar_chart(raw.set_index("bitstring"))
    with right:
        st.caption("Mitigated probabilities")
        st.dataframe(mitigated, hide_index=True, use_container_width=True)
        if not mitigated.empty:
            st.bar_chart(mitigated.set_index("bitstring"))

    st.subheader("Classical optimizer trace")
    trace = pd.DataFrame(result["optimizer"]["trace"])
    st.line_chart(trace.set_index("step")["expected_cut"])

    with st.expander("Full result JSON"):
        st.json(result)


def main() -> None:
    st.set_page_config(page_title="HPCQC RPC MVP", page_icon="HPCQC", layout="wide")
    st.title("HPCQC RPC MVP")

    with st.sidebar:
        st.header("Backend")
        backend_url = st.text_input("HTTP-RPC URL", value=DEFAULT_BACKEND_URL)
        status, health = request_json("GET", backend_url, "/rpc/health")
        if status == 200:
            st.success(f"Connected: {health['queued']} queued")
            st.caption(f"Data: {health['data_dir']}")
        else:
            st.error("Backend unavailable")
            st.caption(str(health.get("error", health)))

    st.subheader("Submit QAOA MaxCut job")
    default_edges = "\n".join(
        f"{edge['u']} {edge['v']} {edge['weight']}" for edge in DEFAULT_GRAPH["edges"]
    )
    with st.form("job_form"):
        left, right = st.columns([2, 1])
        with left:
            title = st.text_input("Job title", value="QAOA MaxCut demo")
            edge_text = st.text_area("Graph edges: u v weight", value=default_edges, height=150)
        with right:
            num_nodes = st.slider("Nodes", 2, 8, int(DEFAULT_GRAPH["num_nodes"]))
            p = st.select_slider("QAOA depth p", options=[1, 2], value=1)
            shots = st.slider("Shots", 128, 10_000, 1024, step=128)
            noise = st.slider("Readout noise", 0.0, 0.25, 0.08, step=0.01)
            priority = st.slider("Priority, lower runs first", 0, 10, 5)
            optimizer_steps = st.slider("Optimizer trials", 4, 64, 16)
            seed = st.number_input("Seed, -1 for random", min_value=-1, value=7)
        submitted = st.form_submit_button("Submit job", type="primary")

    if submitted:
        try:
            payload = {
                "title": title,
                "graph": {"num_nodes": num_nodes, "edges": parse_edges(edge_text)},
                "p": p,
                "shots": shots,
                "noise": noise,
                "priority": priority,
                "optimizer_steps": optimizer_steps,
                "seed": None if seed < 0 else int(seed),
            }
            status, body = request_json("POST", backend_url, "/rpc/jobs", json=payload)
            if status == 202:
                st.success(f"Submitted {body['job_id']}")
                time.sleep(0.5)
            else:
                st.error("Job submission failed")
                st.json(body)
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Job queue")
    status, body = request_json("GET", backend_url, "/rpc/jobs?limit=20")
    if status != 200:
        st.error("Could not load jobs")
        st.json(body)
        return

    jobs = body.get("jobs", [])
    if not jobs:
        st.info("No jobs submitted yet.")
        return

    rows = [
        {
            "job_id": job["job_id"],
            "state": job["state"],
            "title": job["request"]["title"],
            "priority": job["request"]["priority"],
            "created_at": job["created_at"],
        }
        for job in jobs
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    selected = st.selectbox("Inspect job", [job["job_id"] for job in jobs])
    status, job = request_json("GET", backend_url, f"/rpc/jobs/{selected}")
    if status != 200:
        st.error("Could not load selected job")
        st.json(job)
        return

    st.write(f"State: `{job['state']}`")
    with st.expander("Pipeline logs", expanded=True):
        for entry in job["logs"]:
            st.text(entry)

    if job.get("error"):
        st.error(job["error"])
    if job.get("result"):
        render_result(job["result"])
    else:
        st.info("Job is not complete yet. Refresh the page to update status.")


if __name__ == "__main__":
    main()
