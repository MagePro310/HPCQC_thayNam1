from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import altair as alt
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


def random_graph(num_nodes: int, rng: random.Random) -> dict[str, Any]:
    pairs = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)]
    min_edges = max(1, num_nodes - 1)
    max_edges = min(len(pairs), num_nodes + 2)
    edge_count = rng.randint(min_edges, max_edges)
    edges = [
        {"u": u, "v": v, "weight": round(rng.uniform(0.5, 2.0), 2)}
        for u, v in rng.sample(pairs, edge_count)
    ]
    return {"num_nodes": num_nodes, "edges": edges}


def build_random_jobs(
    batch_size: int,
    seed: int,
    node_range: tuple[int, int],
    shot_range: tuple[int, int],
    noise_range: tuple[float, float],
    runtime_range: tuple[float, float],
    optimizer_steps: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    jobs: list[dict[str, Any]] = []
    for index in range(1, batch_size + 1):
        num_nodes = rng.randint(node_range[0], node_range[1])
        shots = rng.randrange(shot_range[0], shot_range[1] + 1, 128)
        runtime = rng.uniform(runtime_range[0], runtime_range[1])
        jobs.append(
            {
                "title": f"Random QAOA job {index}",
                "graph": random_graph(num_nodes, rng),
                "p": rng.choice([1, 2]),
                "shots": shots,
                "noise": round(rng.uniform(noise_range[0], noise_range[1]), 3),
                "priority": 5,
                "optimizer_steps": optimizer_steps,
                "seed": rng.randint(0, 1_000_000),
                "simulated_runtime_seconds": round(runtime, 2),
            }
        )
    return jobs


def render_timeline(timeline: list[dict[str, Any]], machine_count: int) -> None:
    st.subheader("Machine Gantt chart")
    if not timeline:
        st.info("No running or completed jobs to show yet.")
        return

    df = pd.DataFrame(timeline)
    df["chart_start"] = pd.to_datetime(df["chart_start"])
    df["chart_end"] = pd.to_datetime(df["chart_end"])
    df.loc[df["chart_end"] <= df["chart_start"], "chart_end"] = (
        df.loc[df["chart_end"] <= df["chart_start"], "chart_start"] + pd.Timedelta(milliseconds=200)
    )
    machine_order = [f"Machine {index}" for index in range(1, machine_count + 1)]
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("chart_start:T", title="Start"),
            x2=alt.X2("chart_end:T"),
            y=alt.Y("machine_id:N", sort=machine_order, title="Machine"),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(
                    domain=["running", "completed", "failed"],
                    range=["#2f80ed", "#27ae60", "#c0392b"],
                ),
                title="State",
            ),
            tooltip=[
                "job_id:N",
                "title:N",
                "state:N",
                "batch_id:N",
                "sequence:Q",
                "queue_wait_seconds:Q",
                "runtime_seconds:Q",
                "simulated_runtime_seconds:Q",
                "best_cut_value:Q",
            ],
        )
        .properties(height=max(220, 54 * machine_count))
    )
    st.altair_chart(chart, use_container_width=True)


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
        auto_refresh = st.checkbox("Auto-refresh active jobs", value=True)

        st.header("Machines")
        machine_status, machine_body = request_json("GET", backend_url, "/rpc/machines")
        current_machine_count = int(machine_body.get("machine_count", 3)) if machine_status == 200 else 3
        machine_count = st.slider("Worker machines", 1, 4, current_machine_count)
        if st.button("Apply machine count"):
            status, body = request_json("POST", backend_url, "/rpc/machines", json={"machine_count": machine_count})
            if status == 200:
                st.success(f"Using {body['machine_count']} machines")
            else:
                st.error(body.get("error", "Could not update machine count"))
        if machine_status == 200:
            st.caption(", ".join(machine_body.get("active_machines", [])) or "No active machines")

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
            priority = st.slider("Priority, stored only; FIFO ignores it", 0, 10, 5)
            optimizer_steps = st.slider("Optimizer trials", 4, 64, 16)
            simulated_runtime = st.slider("Simulated runtime seconds", 0.0, 30.0, 3.0, step=0.5)
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
                "simulated_runtime_seconds": simulated_runtime,
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

    st.subheader("Submit random batch")
    with st.form("batch_form"):
        left, middle, right = st.columns(3)
        with left:
            batch_size = st.slider("Batch size", 2, 30, 8)
            batch_seed = st.number_input("Batch seed", min_value=0, value=42)
            node_range = st.slider("Node range", 2, 8, (3, 6))
        with middle:
            shot_range = st.slider("Shot range", 128, 10_000, (512, 2048), step=128)
            noise_range = st.slider("Noise range", 0.0, 0.25, (0.02, 0.12), step=0.01)
            batch_optimizer_steps = st.slider("Batch optimizer trials", 4, 64, 8)
        with right:
            runtime_range = st.slider("Runtime range seconds", 0.0, 30.0, (2.0, 8.0), step=0.5)
            st.caption("Jobs are generated in form order and submitted to one shared FIFO queue.")
        batch_submitted = st.form_submit_button("Generate and submit batch", type="primary")

    if batch_submitted:
        jobs = build_random_jobs(
            batch_size=batch_size,
            seed=int(batch_seed),
            node_range=node_range,
            shot_range=shot_range,
            noise_range=noise_range,
            runtime_range=runtime_range,
            optimizer_steps=batch_optimizer_steps,
        )
        status, body = request_json("POST", backend_url, "/rpc/jobs/batch", json={"jobs": jobs})
        if status == 202:
            st.success(f"Submitted {len(body['jobs'])} jobs in {body['batch_id']}")
            time.sleep(0.5)
        else:
            st.error("Batch submission failed")
            st.json(body)

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
            "sequence": job["sequence"],
            "job_id": job["job_id"],
            "state": job["state"],
            "title": job["request"]["title"],
            "batch_id": job.get("batch_id"),
            "machine_id": job.get("machine_id"),
            "created_at": job["created_at"],
            "queued_at": job.get("queued_at"),
            "started_at": job.get("started_at"),
            "sim_runtime_s": job["request"].get("simulated_runtime_seconds", 0.0),
        }
        for job in jobs
    ]
    jobs_df = pd.DataFrame(rows).sort_values("sequence")
    st.dataframe(jobs_df, hide_index=True, use_container_width=True)

    timeline_status, timeline_body = request_json("GET", backend_url, "/rpc/timeline")
    if timeline_status == 200:
        render_timeline(timeline_body.get("timeline", []), machine_count)
    else:
        st.error("Could not load timeline")
        st.json(timeline_body)

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

    active_jobs = any(job["state"] in {"queued", "running"} for job in jobs)
    if auto_refresh and active_jobs:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
