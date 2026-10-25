from __future__ import annotations

import ast
import inspect
from typing import Optional

from app.models import ArchitectureEdge, ArchitectureNode, SourceReference

FAMILY = "gpt2"


class Gpt2Adapter:
    """Semantic architecture adapter for the GPT-2 family.

    Recognizes GPT2Model / GPT2LMHeadModel / GPT2Config, enumerates real module
    instances, and builds verified semantic nodes and edges (including residual
    branches). Returns explicit unsupported results when assumptions do not hold.
    """

    def __init__(self) -> None:
        self._node_ids: dict[str, str] = {}

    def _node_id(self, module_path: str) -> str:
        if module_path == "":
            return "model"
        return "m:" + module_path

    def supports(self, config, model) -> bool:
        return config.model_type == "gpt2" and type(model).__name__ in (
            "GPT2Model",
            "GPT2LMHeadModel",
            "GPT2DoubleHeadsModel",
        )

    def build(
        self,
        config,
        model,
        source_refs: list[SourceReference],
    ) -> tuple[list[ArchitectureNode], list[ArchitectureEdge], list[str]]:
        unsupported: list[str] = []
        nodes: list[ArchitectureNode] = []
        edges: list[ArchitectureEdge] = []

        # Reference maps: build a lookup from module_path -> source_reference_id
        # for the class symbols we care about, so shared implementations can be
        # linked to multiple nodes.
        ref_by_class: dict[str, list[str]] = {}
        for ref in source_refs:
            key = ref.symbol
            if key not in ref_by_class:
                ref_by_class[key] = []
            ref_by_class[key].append(ref.id if hasattr(ref, "id") else "")

        def add(node: ArchitectureNode):
            nodes.append(node)

        def add_edge(source: str, target: str, kind: str, label: Optional[str] = None):
            edges.append(ArchitectureEdge(source=source, target=target, kind=kind, label=label))

        # --- model root ---
        model_cls = type(model).__name__
        model_refs = ref_by_class.get(model_cls, [])
        model_count = sum(1 for p in model.parameters())
        add(
            ArchitectureNode(
                id=self._node_id(""),
                label=model_cls,
                component_type="model",
                module_path="",
                source_reference_ids=model_refs,
                config={"model_type": config.model_type, "architectures": list(getattr(config, "architectures", []) or [])},
                parameter_count=model_count,
                direct_parameter_count=None,
            )
        )

        # --- embeddings (wte, wpe) ---
        for emb_path, emb_label in (("wte", "Token Embedding"), ("wpe", "Position Embedding")):
            emb_mod = self._get_module(model, emb_path)
            if emb_mod is None:
                unsupported.append(f"missing {emb_path}")
                continue
            emb_refs = ref_by_class.get("Embedding", [])
            add(
                ArchitectureNode(
                    id=self._node_id(emb_path),
                    label=emb_label,
                    component_type="embedding",
                    module_path=emb_path,
                    parent_id=self._node_id(""),
                    source_reference_ids=emb_refs,
                    config={
                        "embedding_dim": getattr(config, "n_embd", None),
                        "vocab_size": getattr(config, "vocab_size", None) if emb_path == "wte" else getattr(config, "n_positions", None),
                    },
                    parameter_count=sum(p.numel() for p in emb_mod.parameters()),
                )
            )
            add_edge(self._node_id(""), self._node_id(emb_path), "contains")

        # --- repeated transformer blocks group ---
        n_layer = getattr(config, "n_layer", None)
        block_mod = None
        if hasattr(model, "h") and len(model.h) > 0:
            block_mod = model.h[0]
        block_refs = ref_by_class.get(type(block_mod).__name__ if block_mod else "GPT2Block", [])

        block_group_id = "group:h"
        add(
            ArchitectureNode(
                id=block_group_id,
                label="Transformer Blocks",
                component_type="group",
                parent_id=self._node_id(""),
                repetition=n_layer,
                source_reference_ids=block_refs,
                config={"n_layer": n_layer},
            )
        )
        add_edge(self._node_id(""), block_group_id, "contains")
        add_edge(self._node_id("wte"), block_group_id, "tensor-flow")
        add_edge(self._node_id("wpe"), block_group_id, "tensor-flow", "add")

        # Expand individual blocks so users can inspect internals. We render the
        # first block in full detail; the remaining blocks are summarized by the
        # group repetition count (kept compact per spec).
        first_block_idx = 0
        n_blocks = len(model.h) if hasattr(model, "h") else 0
        for idx in range(min(n_blocks, 1)):
            self._build_block(model, idx, block_group_id, block_refs, edges, add, add_edge, ref_by_class)

        # --- final layernorm ---
        if hasattr(model, "ln_f"):
            ln_f = model.ln_f
            ln_refs = ref_by_class.get("LayerNorm", [])
            add(
                ArchitectureNode(
                    id=self._node_id("ln_f"),
                    label="Final LayerNorm",
                    component_type="layernorm",
                    module_path="ln_f",
                    parent_id=self._node_id(""),
                    source_reference_ids=ln_refs,
                    config={"hidden_size": getattr(config, "n_embd", None)},
                    parameter_count=sum(p.numel() for p in ln_f.parameters()),
                )
            )
            add_edge(block_group_id, self._node_id("ln_f"), "tensor-flow")

        # --- lm head for LM head models ---
        if hasattr(model, "lm_head") or hasattr(model, "transformer") and hasattr(getattr(model, "transformer", None), "wte"):
            add(
                ArchitectureNode(
                    id=self._node_id("lm_head"),
                    label="LM Head",
                    component_type="output",
                    module_path="lm_head",
                    parent_id=self._node_id(""),
                    source_reference_ids=[],
                    config={"vocab_size": getattr(config, "vocab_size", None)},
                )
            )
            add_edge(self._node_id("ln_f"), self._node_id("lm_head"), "tensor-flow")

        return nodes, edges, unsupported

    def _build_block(
        self,
        model,
        idx: int,
        parent_group_id: str,
        block_refs: list[str],
        edges: list[ArchitectureEdge],
        add,
        add_edge,
        ref_by_class: dict[str, list[str]],
    ):
        block = model.h[idx]
        path = f"h.{idx}"
        block_id = self._node_id(path)
        add(
            ArchitectureNode(
                id=block_id,
                label=f"Block {idx}",
                component_type="block",
                module_path=path,
                parent_id=parent_group_id,
                repetition_index=idx,
                source_reference_ids=block_refs,
                parameter_count=sum(p.numel() for p in block.parameters()),
            )
        )
        add_edge(parent_group_id, block_id, "contains")

        # ln_1 -> attn
        ln1_id = self._node_id(f"{path}.ln_1")
        add(
            ArchitectureNode(
                id=ln1_id,
                label="LayerNorm 1",
                component_type="layernorm",
                module_path=f"{path}.ln_1",
                parent_id=block_id,
                source_reference_ids=ref_by_class.get("LayerNorm", []),
            )
        )
        add_edge(block_id, ln1_id, "contains")
        add_edge(block_id, ln1_id, "tensor-flow")

        attn_id = self._node_id(f"{path}.attn")
        attn_mod = getattr(block, "attn", None)
        attn_type = type(attn_mod).__name__ if attn_mod else "GPT2Attention"
        add(
            ArchitectureNode(
                id=attn_id,
                label="Self-Attention",
                component_type="attention",
                module_path=f"{path}.attn",
                parent_id=block_id,
                source_reference_ids=ref_by_class.get(attn_type, []),
                config={
                    "num_heads": getattr(model.config, "n_head", None),
                    "head_dim": getattr(model.config, "n_embd", None) // getattr(model.config, "n_head", 1)
                    if getattr(model.config, "n_head", 1)
                    else None,
                },
                parameter_count=sum(p.numel() for p in attn_mod.parameters()) if attn_mod else None,
            )
        )
        add_edge(block_id, attn_id, "contains")
        add_edge(ln1_id, attn_id, "tensor-flow")
        # residual stream enters the first add junction; attention output adds in
        add_id_1 = self._node_id(f"{path}.add_1")
        add(
            ArchitectureNode(
                id=add_id_1,
                label="Residual Add",
                component_type="add",
                module_path=None,
                parent_id=block_id,
                source_reference_ids=[],
                config={},
            )
        )
        add_edge(block_id, add_id_1, "residual", "add")
        add_edge(attn_id, add_id_1, "tensor-flow")

        # ln_2 -> mlp
        ln2_id = self._node_id(f"{path}.ln_2")
        add(
            ArchitectureNode(
                id=ln2_id,
                label="LayerNorm 2",
                component_type="layernorm",
                module_path=f"{path}.ln_2",
                parent_id=block_id,
                source_reference_ids=ref_by_class.get("LayerNorm", []),
            )
        )
        add_edge(block_id, ln2_id, "contains")
        add_edge(add_id_1, ln2_id, "tensor-flow")

        mlp_id = self._node_id(f"{path}.mlp")
        mlp_mod = getattr(block, "mlp", None)
        mlp_type = type(mlp_mod).__name__ if mlp_mod else "GPT2MLP"
        add(
            ArchitectureNode(
                id=mlp_id,
                label="MLP",
                component_type="mlp",
                module_path=f"{path}.mlp",
                parent_id=block_id,
                source_reference_ids=ref_by_class.get(mlp_type, []),
                parameter_count=sum(p.numel() for p in mlp_mod.parameters()) if mlp_mod else None,
            )
        )
        add_edge(block_id, mlp_id, "contains")
        add_edge(ln2_id, mlp_id, "tensor-flow")
        # residual stream continues through the second add junction; mlp adds in
        add_id_2 = self._node_id(f"{path}.add_2")
        add(
            ArchitectureNode(
                id=add_id_2,
                label="Residual Add",
                component_type="add",
                module_path=None,
                parent_id=block_id,
                source_reference_ids=[],
                config={},
            )
        )
        add_edge(add_id_1, add_id_2, "residual", "add")
        add_edge(mlp_id, add_id_2, "tensor-flow")
        # The output stream exits through the block boundary.
        add_edge(add_id_2, block_id, "residual", "out")

    def _get_module(self, model, path: str):
        cur = model
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur
