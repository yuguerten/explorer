import { useCallback, useRef, useState } from "react";
import { inspectModel } from "./api";
import type { InspectionResult, SourceReference, ArchitectureNode } from "./types";
import { LoadBar } from "./components/LoadBar";
import { SourcePanel } from "./components/SourcePanel";
import { ArchitecturePanel } from "./components/ArchitecturePanel";
import { DetailsPanel } from "./components/DetailsPanel";

type StatusState = { phase: "idle" | "loading" | "success" | "error"; message: string };

export default function App() {
  const [result, setResult] = useState<InspectionResult | null>(null);
  const [status, setStatus] = useState<StatusState>({ phase: "idle", message: "" });
  const [error, setError] = useState<string | null>(null);

  // Selection state (bidirectional)
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedRefId, setSelectedRefId] = useState<string | null>(null);

  // Guard against feedback loops between code-selection and diagram-selection.
  const programmatic = useRef(false);

  const onInspect = useCallback(async (modelId: string, revision?: string) => {
    setStatus({ phase: "loading", message: "Inspecting…" });
    setError(null);
    setResult(null);
    setSelectedNodeId(null);
    setSelectedRefId(null);
    try {
      const r = await inspectModel(modelId, revision);
      setResult(r);
      setStatus({ phase: "success", message: `Resolved ${r.family} · ${r.implementation_version}` });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      setStatus({ phase: "error", message: "Inspection failed" });
    }
  }, []);

  // Navigation from one side triggers the other; suppress the loop.
  const jumpToNode = useCallback(
    (nodeId: string) => {
      programmatic.current = true;
      setSelectedNodeId(nodeId);
      if (result) {
        const node = result.graph.find((n) => n.id === nodeId);
        setSelectedRefId(node?.source_reference_ids[0] ?? null);
      }
      requestAnimationFrame(() => {
        programmatic.current = false;
      });
    },
    [result]
  );

  const jumpToRef = useCallback(
    (refId: string) => {
      programmatic.current = true;
      setSelectedRefId(refId);
      if (result) {
        const ref = result.source_references.find((s) => s.id === refId);
        if (ref && ref.architecture_node_ids.length > 0) {
          const linkedNodeId = selectedNodeId && ref.architecture_node_ids.includes(selectedNodeId)
            ? selectedNodeId
            : ref.architecture_node_ids[0];
          setSelectedNodeId(linkedNodeId);
        }
      }
      requestAnimationFrame(() => {
        programmatic.current = false;
      });
    },
    [result, selectedNodeId]
  );

  const selectedNode: ArchitectureNode | null = result
    ? result.graph.find((n) => n.id === selectedNodeId) ?? null
    : null;
  const selectedRef: SourceReference | null = result
    ? result.source_references.find((s) => s.id === selectedRefId) ?? null
    : null;

  return (
    <div className="app">
      <LoadBar
        status={status}
        onInspect={onInspect}
        modelId={result?.model_id ?? ""}
      />
      {error && <div className="error-box">{error}</div>}
      {!result && !error && (
        <div className="placeholder">
          <div>Enter a Hugging Face model ID to begin.</div>
          <div className="sub">
            Try <code>gpt2</code>. The service resolves the configuration and
            inspects the installed Transformers implementation without loading
            the full weights.
          </div>
        </div>
      )}
      {result && (
        <Workspace
          result={result}
          selectedNodeId={selectedNodeId}
          selectedRefId={selectedRefId}
          onJumpToNode={jumpToNode}
          onJumpToRef={jumpToRef}
          selectedNode={selectedNode}
          selectedRef={selectedRef}
          onClearSelection={() => { setSelectedNodeId(null); setSelectedRefId(null); }}
        />
      )}
    </div>
  );
}

function Workspace({
  result,
  selectedNodeId,
  selectedRefId,
  onJumpToNode,
  onJumpToRef,
  selectedNode,
  selectedRef,
  onClearSelection,
}: {
  result: InspectionResult;
  selectedNodeId: string | null;
  selectedRefId: string | null;
  onJumpToNode: (id: string) => void;
  onJumpToRef: (id: string) => void;
  selectedNode: ArchitectureNode | null;
  selectedRef: SourceReference | null;
  onClearSelection: () => void;
}) {
  const [leftPct, setLeftPct] = useState(50);
  const dragging = useRef(false);

  const onDividerDown = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    const startX = e.clientX;
    const startPct = leftPct;
    const move = (ev: MouseEvent) => {
      const w = window.innerWidth;
      const delta = ((ev.clientX - startX) / w) * 100;
      setLeftPct(Math.min(75, Math.max(25, startPct + delta)));
    };
    const up = () => {
      dragging.current = false;
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }, [leftPct]);

  return (
    <div className="workspace">
      <div className="panel" style={{ width: `${leftPct}%` }}>
        <div className="panel-header">
          <span>Source</span>
          <span className="count">{result.source_files.length} file</span>
        </div>
        <SourcePanel
          result={result}
          selectedRefId={selectedRefId}
          onJumpToRef={onJumpToRef}
        />
      </div>
      <div
        className={`divider${dragging.current ? " dragging" : ""}`}
        onMouseDown={onDividerDown}
        title="Drag to resize"
      />
      <div className="panel" style={{ width: `${100 - leftPct}%` }}>
        <div className="panel-header">
          <span>Architecture</span>
          <span className="count">{result.graph.length} nodes · {result.edges.length} edges</span>
        </div>
        <ArchitecturePanel
          result={result}
          selectedNodeId={selectedNodeId}
          onJumpToNode={onJumpToNode}
        />
        {selectedNode && (
          <aside className="details" aria-label="Component details">
            <DetailsPanel node={selectedNode} refMeta={selectedRef} result={result} onClose={onClearSelection} onJumpToRef={onJumpToRef} />
          </aside>
        )}
      </div>
    </div>
  );
}
