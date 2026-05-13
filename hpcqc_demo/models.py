from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hpcqc_demo.config import DEFAULT_GRAPH


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EdgeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    u: int = Field(..., ge=0)
    v: int = Field(..., ge=0)
    weight: float = Field(1.0, gt=0.0, le=100.0)


class GraphSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_nodes: int = Field(4, ge=2, le=8)
    edges: list[EdgeSpec]

    @model_validator(mode="after")
    def validate_edges(self) -> "GraphSpec":
        if not self.edges:
            raise ValueError("graph must contain at least one edge")
        for edge in self.edges:
            if edge.u == edge.v:
                raise ValueError("self-loop edges are not supported")
            if edge.u >= self.num_nodes or edge.v >= self.num_nodes:
                raise ValueError("edge endpoint is outside graph node range")
        return self


def default_graph_spec() -> GraphSpec:
    return GraphSpec.model_validate(DEFAULT_GRAPH)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field("QAOA MaxCut demo", min_length=1, max_length=80)
    graph: GraphSpec = Field(default_factory=default_graph_spec)
    p: int = Field(1, ge=1, le=2)
    shots: int = Field(1024, ge=128, le=10_000)
    noise: float = Field(0.08, ge=0.0, le=0.25)
    priority: int = Field(5, ge=0, le=10)
    optimizer_steps: int = Field(16, ge=4, le=64)
    seed: int | None = Field(None, ge=0)
    simulated_runtime_seconds: float = Field(0.0, ge=0.0, le=120.0)
    submitted_by: str = Field("streamlit-ui", min_length=1, max_length=80)


class JobBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jobs: list[JobRequest] = Field(..., min_length=1, max_length=100)


class JobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    state: JobState
    request: JobRequest
    created_at: str
    updated_at: str
    sequence: int
    batch_id: str | None = None
    queued_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    machine_id: str | None = None
    queue_wait_seconds: float | None = None
    runtime_seconds: float | None = None
    logs: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
