from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class LoadStatus(BaseModel):
    status: str = Field(description="one of: idle, inspecting, resolving, success, error, unsupported")
    message: str = ""
    detail: Optional[str] = None


class SourceFile(BaseModel):
    file_id: str
    path: str
    display_name: str
    content: str
    package_version: Optional[str] = None


class SourceReference(BaseModel):
    id: str
    source_file_id: str
    symbol: str = Field(description="class or function identity")
    method: Optional[str] = None
    start_line: int
    end_line: int
    start_col: Optional[int] = None
    end_col: Optional[int] = None
    granularity: str = Field(description="class | method | expression")
    architecture_node_ids: list[str] = []
    origin: str = Field(description="inspected-symbol | adapter-mapping")


class ArchitectureNode(BaseModel):
    id: str
    label: str
    component_type: str = Field(description="model | embedding | block | attention | mlp | layernorm | output | group | other")
    module_path: Optional[str] = None
    parent_id: Optional[str] = None
    repetition: Optional[int] = None
    repetition_index: Optional[int] = None
    source_reference_ids: list[str] = []
    config: dict[str, Any] = {}
    parameter_count: Optional[int] = None
    direct_parameter_count: Optional[int] = None


class ArchitectureEdge(BaseModel):
    source: str
    target: str
    kind: str = Field(description="tensor-flow | residual | contains | group")
    label: Optional[str] = None


class InspectionResult(BaseModel):
    model_id: str
    revision: Optional[str] = None
    config_source: Optional[str] = None
    implementation_package: str
    implementation_version: str
    implementation_file: Optional[str] = None
    family: Optional[str] = None
    loading_mode: str = Field(description="config-only | weights-loaded")
    model_identity: dict[str, Any] = {}
    graph: list[ArchitectureNode] = []
    edges: list[ArchitectureEdge] = []
    source_files: list[SourceFile] = []
    source_references: list[SourceReference] = []
    unsupported: list[str] = []
    parameter_count: Optional[int] = None
    warnings: list[str] = []


class ResolveRequest(BaseModel):
    model_id: str
    revision: Optional[str] = None


class LoadWeightsRequest(BaseModel):
    model_id: str
    revision: Optional[str] = None
