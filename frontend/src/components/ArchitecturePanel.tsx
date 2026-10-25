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
  return (
    <div className={`rf-node${selected ? " selected" : ""}`}>
      <FlowHandles />
      <span className="node-type">{eyebrow}</span>
      <span className="node-label">{node.label}</span>
    </div>
  );
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
  return (
    <div
      className={`container-panel${selected ? " selected" : ""}${isGroup ? " group" : ""}`}
      style={{ width: "100%", height: "100%" }}
    >
      <div className="container-panel-head">
        <span className="container-title">
          {node.label}
          {node.repetition != null && <span className="node-count">×{node.repetition}</span>}
        </span>
      </div>
      <FlowHandles />
    </div>
  );
}

function RoutedEdge({ id, data, style, markerEnd }: EdgeProps) {
  const junction = data?.junction as { x: number; y: number } | undefined;
  return (
    <>
      <BaseEdge id={id} path={data?.path as string} style={style} markerEnd={markerEnd} />
      {junction && (
        <g className="flow-input-junction" pointerEvents="none">
          <title>Block input x: the residual stream before LayerNorm 1</title>
          <circle cx={junction.x} cy={junction.y} r={3.2} fill="#1c1a17" />
          <text x={junction.x + 9} y={junction.y - 5} fill="#1c1a17" fontSize={11} fontFamily="var(--mono)">x</text>
        </g>
      )}
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
      let junction: { x: number; y: number } | undefined;

      if (output) {
        // Leaving a containing frame is a continuation, not a return arrow.
        path = "M " + sx + " " + (source.y + source.h) + " V " + (target.y + target.h);
        marker = false;
      } else if (residual && isContainerType(source.node.component_type)) {
        const first = nodes.filter((n) => n.node.parent_id === source.node.id && !isContainerType(n.node.component_type))
          .sort((a, b) => a.y - b.y)[0];
        const entryY = (first?.y ?? source.y + 34) - 13;
        const railX = (first?.x ?? target.x) - 55;
        const targetY = target.y + target.h / 2;
        junction = { x: tx, y: entryY };
        path = "M " + tx + " " + entryY + " H " + railX + " V " + targetY + " H " + target.x;
      } else if (residual) {
        // The second bypass travels beside LayerNorm/MLP, never through them.
        const siblings = nodes.filter((n) => n.node.parent_id === source.node.parent_id && !isContainerType(n.node.component_type));
        const railX = Math.max(...siblings.map((n) => n.x + n.w)) + 55;
        path = "M " + (source.x + source.w) + " " + (source.y + source.h / 2) +
          " H " + railX + " V " + (target.y + target.h / 2) + " H " + (target.x + target.w);
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
      const color = residual ? "#a63d2f" : "#1c1a17";
      return [{
        id: "e-" + i,
        source: e.source,
        target: e.target,
        sourceHandle: "source-bottom",
        targetHandle: "target-top",
        type: "architecture",
        data: { path, junction },
        zIndex: 1,
        style: { stroke: color, strokeWidth: residual ? 1.8 : 1.5, strokeDasharray: residual ? "6 4" : undefined },
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
