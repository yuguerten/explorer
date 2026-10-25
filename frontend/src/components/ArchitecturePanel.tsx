import { useEffect, useMemo, useRef, type RefObject } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  BaseEdge,
  getSmoothStepPath,
  useReactFlow,
  type EdgeProps,
  type Node,
  type Edge,
  type NodeProps,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { ArchitectureNode, InspectionResult } from "../types";
import { layoutDecoder } from "../lib/layout";

const EYEBROW: Record<string, string> = {
  model: "MODEL",
  embedding: "EMBEDDING",
  group: "GROUP",
  block: "BLOCK",
  attention: "ATTENTION",
  mlp: "FEED-FORWARD",
  layernorm: "NORMALIZE",
  output: "OUTPUT",
  add: "+",
};

/** Hidden anchors keep React Flow connections available for routed edges. */
function FlowHandles() {
  return (
    <>
      <Handle type="source" id="source-top" position={Position.Top} />
      <Handle type="source" id="source-bottom" position={Position.Bottom} />
      <Handle type="source" id="source-left" position={Position.Left} />
      <Handle type="target" id="target-top" position={Position.Top} />
      <Handle type="target" id="target-bottom" position={Position.Bottom} />
      <Handle type="target" id="target-left" position={Position.Left} />
    </>
  );
}

function ComponentNode({ data, selected }: NodeProps) {
  const node = data.node as ArchitectureNode;
  const eyebrow = EYEBROW[node.component_type] ?? node.component_type;
  const meta = nodeMeta(node);
  return (
    <div className={`rf-node node-${node.component_type}${selected ? " selected" : ""}`}>
      <FlowHandles />
      <span className="node-type">{eyebrow}</span>
      <span className="node-label">{node.label}</span>
      {meta && <span className="node-meta">{meta}</span>}
    </div>
  );
}

function nodeMeta(node: ArchitectureNode): string | null {
  const value = (key: string) => node.config[key];
  const number = (key: string) => typeof value(key) === "number" ? value(key) as number : null;
  if (node.component_type === "embedding") return node.module_path === "wte" ? "learned token vectors" : "learned positions";
  if (node.component_type === "attention") {
    const heads = number("num_heads");
    return heads ? `${heads} heads · causal mask` : "multi-head · causal mask";
  }
  if (node.component_type === "mlp") {
    const hidden = number("hidden_size");
    const inner = number("intermediate_size");
    const factor = hidden && inner ? Math.round(inner / hidden) : 4;
    return `${factor}× hidden · GELU`;
  }
  if (node.component_type === "layernorm") {
    if (node.module_path?.endsWith("ln_1")) return "pre-attention";
    if (node.module_path?.endsWith("ln_2")) return "pre-MLP";
    return "output normalization";
  }
  if (node.component_type === "output") {
    const vocab = number("vocab_size");
    return vocab ? String(vocab.toLocaleString()) + "-way logits" : "vocabulary logits";
  }
  return null;
}

function AddNode({ selected }: NodeProps) {
  return (
    <div className={`rf-node add-node${selected ? " selected" : ""}`}>
      <FlowHandles />
      <span className="add-glyph">+</span>
    </div>
  );
}

function ContainerNode({ data, selected }: NodeProps) {
  const node = data.node as ArchitectureNode;
  const isGroup = node.component_type === "group";
  const isBlock = node.component_type === "block";
  return (
    <div
      className={`container-panel${selected ? " selected" : ""}${isGroup ? " group" : ""}${isBlock ? " block" : ""}`}
      style={{ width: "100%", height: "100%" }}
    >
      <div className="container-panel-head">
        <span className="container-title">
          {node.label}
          {isGroup ? <span className="group-subtitle">N = 12, 24, 36, 48</span> : node.repetition != null && <span className="node-count">×{node.repetition}</span>}
        </span>
        {isGroup && node.repetition != null && <span className="group-current">current N = {node.repetition}</span>}
      </div>
      <FlowHandles />
    </div>
  );
}

function RoutedEdge({ id, data, style, markerEnd }: EdgeProps) {
  const junction = data?.junction as { x: number; y: number; label?: string | null; title?: string } | undefined;
  const label = data?.label as { x: number; y: number } | undefined;
  return (
    <>
      <BaseEdge id={id} path={data?.path as string} style={style} markerEnd={markerEnd} />
      {junction && (
        <g className="flow-input-junction" pointerEvents="none">
          <title>{junction.title ?? "Residual stream branch"}</title>
          <circle cx={junction.x} cy={junction.y} r={3.2} fill="var(--ink)" />
          {junction.label && <text x={junction.x + 9} y={junction.y - 5} fill="var(--ink)" fontSize={11} fontFamily="var(--mono)">{junction.label}</text>}
        </g>
      )}
      {label && <text className="residual-label" x={label.x} y={label.y} pointerEvents="none">residual</text>}
    </>
  );
}

function FitDiagram({ containerRef }: { containerRef: RefObject<HTMLDivElement> }) {
  const { fitBounds, getNodes, getNodesBounds, viewportInitialized } = useReactFlow();
  useEffect(() => {
    if (!viewportInitialized || !containerRef.current) return;
    let frame = 0;
    const resize = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        frame = requestAnimationFrame(() => {
          void fitBounds(getNodesBounds(getNodes()), { padding: 0.15, duration: 0 });
        });
      });
    };
    const observer = new ResizeObserver(resize);
    observer.observe(containerRef.current);
    resize();
    return () => { observer.disconnect(); cancelAnimationFrame(frame); };
  }, [containerRef, fitBounds, getNodes, getNodesBounds, viewportInitialized]);
  return null;
}

const edgeTypes = { architecture: RoutedEdge };

const nodeTypes: NodeTypes = {
  component: ComponentNode,
  add: AddNode,
  container: ContainerNode,
};

export function ArchitecturePanel({
  result,
  selectedNodeId,
  onJumpToNode,
}: {
  result: InspectionResult;
  selectedNodeId: string | null;
  onJumpToNode: (id: string) => void;
}) {
  const graphRef = useRef<HTMLDivElement>(null);
  const { nodes, edges } = useMemo(
    () => layoutDecoder(result.graph, result.edges, new Set()),
    [result.graph, result.edges]
  );

  const rfNodes: Node[] = useMemo(
    () =>
      nodes.map((ln) => {
        const t = ln.node.component_type;
        const kind = isContainerType(t) ? "container" : t === "add" ? "add" : "component";
        return {
          id: ln.node.id,
          position: { x: ln.x, y: ln.y },
          data: { node: ln.node },
          type: kind,
          zIndex: ln.z,
          style: { width: ln.w, height: ln.h },
          selected: selectedNodeId === ln.node.id,
        } as Node;
      }),
    [nodes, selectedNodeId]
  );

  const rfEdges: Edge[] = useMemo(() => {
    const byId = new Map(nodes.map((n) => [n.node.id, n]));
    // Expanded frames need continuous entry/exit wires even when an inspection
    // response omits its boundary edges. Keep the branch attached during HMR
    // and when reopening results produced by the older adapter.
    const connectedEdges = [...edges];
    for (const block of nodes.filter((n) => n.node.component_type === "block")) {
      const children = nodes.filter((n) => n.node.parent_id === block.node.id && !isContainerType(n.node.component_type))
        .sort((a, b) => a.y - b.y);
      const first = children[0];
      const last = children[children.length - 1];
      if (first && !connectedEdges.some((e) => e.source === block.node.id && e.target === first.node.id && e.kind === "tensor-flow")) {
        connectedEdges.push({ source: block.node.id, target: first.node.id, kind: "tensor-flow" });
      }
      if (last && !connectedEdges.some((e) => e.source === last.node.id && e.target === block.node.id && e.label === "out")) {
        connectedEdges.push({ source: last.node.id, target: block.node.id, kind: "residual", label: "out" });
      }
    }
    return connectedEdges.flatMap((e, i) => {
      const source = byId.get(e.source);
      const target = byId.get(e.target);
      if (!source || !target) return [];
      const sx = source.x + source.w / 2;
      const tx = target.x + target.w / 2;
      const output = e.kind === "residual" && e.label === "out";
      const residual = e.kind === "residual" && !output;
      let path: string;
      // Frames continue the same stream; only computational targets need an arrowhead.
      let marker = !isContainerType(target.node.component_type);
      let junction: { x: number; y: number; label?: string | null; title?: string } | undefined;
      let edgeLabel: { x: number; y: number } | undefined;

      if (output) {
        // Leaving a containing frame is a continuation, not a return arrow.
        path = "M " + sx + " " + (source.y + source.h) + " V " + (target.y + target.h);
        marker = false;
      } else if (residual && isContainerType(source.node.component_type)) {
        const first = nodes.filter((n) => n.node.parent_id === source.node.id && !isContainerType(n.node.component_type))
          .sort((a, b) => a.y - b.y)[0];
        const entryY = (first?.y ?? source.y + 34) - 13;
        const siblings = nodes.filter((n) => n.node.parent_id === source.node.id && !isContainerType(n.node.component_type));
        const railX = Math.max(...siblings.map((n) => n.x + n.w)) + 72;
        const targetY = target.y + target.h / 2;
        junction = { x: tx, y: entryY, label: "x", title: "Block input x, before LayerNorm 1" };
        edgeLabel = { x: railX + 8, y: (entryY + targetY) / 2 };
        path = "M " + tx + " " + entryY + " H " + railX + " V " + targetY + " H " + (target.x + target.w);
      } else if (residual) {
        // The second bypass travels beside LayerNorm/MLP, never through them.
        const siblings = nodes.filter((n) => n.node.parent_id === source.node.parent_id && !isContainerType(n.node.component_type));
        const railX = Math.max(...siblings.map((n) => n.x + n.w)) + 38;
        const startX = source.x + source.w / 2;
        const startY = source.y + source.h + 4;
        const targetY = target.y + target.h / 2;
        junction = { x: startX, y: startY, label: null, title: "Residual stream after the attention add; bypasses LayerNorm 2 and the MLP" };
        edgeLabel = { x: railX + 8, y: (startY + targetY) / 2 };
        path = "M " + startX + " " + startY + " H " + railX + " V " + targetY + " H " + (target.x + target.w);
      } else if (target.node.parent_id === source.node.id) {
        // Draw the input down from the group boundary through the clear center
        // of both headers, rather than routing from the container's midpoint.
        const parent = source.node.parent_id ? byId.get(source.node.parent_id) : undefined;
        path = "M " + tx + " " + (parent?.y ?? source.y) + " V " + target.y;
      } else {
        [path] = getSmoothStepPath({
          sourceX: sx,
          sourceY: source.y + source.h,
          sourcePosition: Position.Bottom,
          targetX: tx,
          targetY: target.y,
          targetPosition: Position.Top,
          borderRadius: 4,
          offset: 10,
        });
      }
      const color = residual ? "var(--residual)" : "var(--wire)";
      return [{
        id: "e-" + i,
        source: e.source,
        target: e.target,
        sourceHandle: "source-bottom",
        targetHandle: "target-top",
        type: "architecture",
        data: { path, junction, label: edgeLabel },
        zIndex: 1,
        style: { stroke: color, strokeWidth: residual ? 1.7 : 1.5 },
        markerEnd: marker ? { type: "arrowclosed", width: 12, height: 12, color } : undefined,
      } as Edge];
    });
  }, [edges, nodes]);

  return (
    <div className="architecture-view">
      <div className="graph-wrap" ref={graphRef}>
        <ReactFlow
          nodes={rfNodes}
          edges={rfEdges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          fitViewOptions={{ padding: 0.15, maxZoom: 1.2 }}
          minZoom={0.05}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          proOptions={{ hideAttribution: true }}
          onNodeClick={(_, node) => onJumpToNode(node.id)}
          style={{ width: "100%", height: "100%" }}
        >
          <FitDiagram containerRef={graphRef} />
          <Background gap={18} size={1} color="var(--hairline)" />
          <Controls position="bottom-right" showInteractive={false} />
        </ReactFlow>
      </div>
    </div>
  );
}

function isContainerType(t: string) {
  return t === "model" || t === "block" || t === "group";
}
