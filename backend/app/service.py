from __future__ import annotations

import ast
import inspect
from typing import Optional

import torch
from transformers import AutoConfig, AutoModel

import adapters
from .inspector import collect_source_file, collect_symbol_reference, method_line_range
from .models import (
    ArchitectureNode,
    InspectionResult,
    SourceFile,
    SourceReference,
)

# GPT-2 module-level classes to inspect and expose as source files. We inspect
# the base GPT2Model implementation so repeated blocks, attention and MLP share
# a single source reference that maps to multiple nodes.
_INSPECT_SYMBOLS = ["GPT2Model", "GPT2Block", "GPT2Attention", "GPT2MLP", "LayerNorm"]


def _package_version(pkg: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None


def inspect_model(model_id: str, revision: Optional[str] = None) -> InspectionResult:
    """Resolve config, build an inspectable model on meta device, and produce
    a serializable inspection result with graph, source, and mappings.

    Does not download or allocate full weight tensors.
    """
    warnings: list[str] = []
    unsupported: list[str] = []

    # --- resolve config (downloads config only, not weights) ---
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    family = config.model_type

    # --- build model on meta device (no weight allocation) ---
    with torch.device("meta"):
        model = AutoModel.from_config(config)

    adapter = adapters.get_adapter(family)
    if adapter is None:
        raise ValueError(
            f"Model family '{family}' is not supported. Supported families: {', '.join(adapters.supported_families())}"
        )
    if not adapter.supports(config, model):
        raise ValueError(f"Model '{model_id}' resolved to family '{family}' but is not a supported implementation.")

    # --- inspect source files ---
    source_files: dict[str, SourceFile] = {}
    source_refs: list[SourceReference] = {}
    module = None

    impl_file = inspect.getsourcefile(type(model))
    impl_cls = type(model).__name__

    # Add the top-level model class file
    root_file = collect_source_file(type(model), _package_version("transformers"))
    if root_file:
        source_files[root_file.file_id] = root_file
        ref = collect_symbol_reference(type(model), root_file)
        if ref:
            ref.source_file_id = root_file.file_id
            source_refs["class:" + impl_cls] = ref

    # Parse the implementation file once for AST method-range lookups
    if root_file:
        try:
            module = ast.parse(root_file.content)
        except SyntaxError:
            module = None
        warnings.append(f"Implementation source from {root_file.path}")

    # Collect class-level references for known shared classes
    for symbol_name in _INSPECT_SYMBOLS:
        sym = _resolve_class(model, symbol_name)
        if sym is None:
            continue
        file = source_files.get("__class__:" + symbol_name) or collect_source_file(sym, _package_version("transformers"))
        if file is None:
            continue
        if file.file_id not in source_files:
            source_files[file.file_id] = file
        ref = collect_symbol_reference(sym, file)
        if ref:
            ref.source_file_id = file.file_id
            source_refs["class:" + symbol_name] = ref

    # Collect method-level references (forward) for the shared classes
    for symbol_name in _INSPECT_SYMBOLS:
        sym = _resolve_class(model, symbol_name)
        if sym is None:
            continue
        cls_ref = source_refs.get("class:" + symbol_name)
        if cls_ref is None:
            continue
        forward_ref = _method_ref(sym, cls_ref, "forward", module)
        if forward_ref:
            key = f"method:{symbol_name}:forward"
            forward_ref.source_file_id = cls_ref.source_file_id
            source_refs[key] = forward_ref

    # --- build graph via adapter ---
    nodes, edges, adapter_unsupported = adapter.build(config, model, list(source_refs.values()))
    unsupported.extend(adapter_unsupported)

    # --- link source references to architecture nodes ---
    # Map each node's component_type to the relevant source references.
    node_by_id = {n.id: n for n in nodes}
    # Assign source ref ids to nodes based on component type and module path.
    _assign_source_links(nodes, source_refs)

    # Also back-fill: each source reference's related node ids
    node_links: dict[str, list[str]] = {}
    for n in nodes:
        for rid in n.source_reference_ids:
            node_links.setdefault(rid, []).append(n.id)
    for ref in source_refs.values():
        ref.architecture_node_ids = node_links.get(ref.id if hasattr(ref, "id") else "", [])

    total_params = sum(p.numel() for p in model.parameters())

    return InspectionResult(
        model_id=model_id,
        revision=revision,
        family=family,
        loading_mode="config-only",
        implementation_package="transformers",
        implementation_version=_package_version("transformers") or "unknown",
        implementation_file=impl_file,
        model_identity={
            "class": impl_cls,
            "config_type": type(config).__name__,
            "n_layer": getattr(config, "n_layer", None),
            "n_head": getattr(config, "n_head", None),
            "n_embd": getattr(config, "n_embd", None),
        },
        graph=nodes,
        edges=edges,
        source_files=list(source_files.values()),
        source_references=list(source_refs.values()),
        unsupported=unsupported,
        parameter_count=total_params,
        warnings=warnings,
    )


def _resolve_class(model, name: str):
    """Resolve a class object by name from the model's module."""
    module = type(model).__module__
    try:
        import importlib
        mod = importlib.import_module(module)
        cls = getattr(mod, name, None)
        if cls is not None and isinstance(cls, type):
            return cls
    except Exception:
        pass
    # Fall back to scanning instance attributes
    for attr in dir(model):
        if attr == name:
            obj = getattr(model, name)
            if isinstance(obj, type):
                return obj
    return None


def _method_ref(cls, cls_ref: SourceReference, method_name: str, module: Optional[ast.Module]) -> Optional[SourceReference]:
    method = getattr(cls, method_name, None)
    if method is None:
        return None
    try:
        lines, start = inspect.getsourcelines(method)
    except (TypeError, OSError, IOError):
        return None
    end = start + len(lines) - 1
    if module is not None:
        ranged = method_line_range(module, cls.__name__, method_name)
        if ranged:
            start, end = ranged
    return SourceReference(
        id="ref:method:" + cls.__name__ + ":" + method_name,
        source_file_id=cls_ref.source_file_id,
        symbol=cls.__name__,
        method=method_name,
        start_line=start,
        end_line=end,
        granularity="method",
        architecture_node_ids=[],
        origin="inspected-symbol",
    )


def _assign_source_links(nodes: list[ArchitectureNode], source_refs: dict[str, SourceReference]):
    """Link nodes to source reference ids based on component type + symbol."""
    for node in nodes:
        refs: list[str] = []
        # Map by component_type to symbol name
        type_to_symbol = {
            "model": "GPT2Model",
            "block": "GPT2Block",
            "attention": "GPT2Attention",
            "mlp": "GPT2MLP",
            "layernorm": "LayerNorm",
        }
        sym = type_to_symbol.get(node.component_type)
        if sym:
            cls_key = "class:" + sym
            method_key = "method:" + sym + ":forward"
            if method_key in source_refs:
                refs.append(source_refs[method_key].id)
            elif cls_key in source_refs:
                refs.append(source_refs[cls_key].id)
        node.source_reference_ids = refs
