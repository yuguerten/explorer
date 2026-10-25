import type { ArchitectureNode, InspectionResult, SourceReference } from "../types";

export function DetailsPanel({ node, refMeta, result, onClose, onJumpToRef }: {
  node: ArchitectureNode | null;
  refMeta: SourceReference | null;
  result: InspectionResult;
  onClose: () => void;
  onJumpToRef: (refId: string) => void;
}) {
  if (!node) return null;

  const file = refMeta
    ? result.source_files.find((f) => f.file_id === refMeta.source_file_id)
    : undefined;
  const rows: Array<[string, string]> = [];
  if (node.module_path != null) rows.push(["Module", node.module_path || "Model root"]);
  if (node.parameter_count != null) rows.push(["Parameters", node.parameter_count.toLocaleString("en-US")]);
  if (node.repetition != null) rows.push(["Layers", String(node.repetition)]);
  if (node.component_type === "embedding") {
    const dimension = typeof node.config.embedding_dim === "number" ? node.config.embedding_dim : "hidden-size";
    rows.push(["How it works", node.module_path === "wte"
      ? "Token IDs index a learned table and return a " + dimension + "-value vector."
      : "Position IDs index a learned table; GPT-2 adds these vectors to token vectors before Block 0."]);
  }
  for (const [key, value] of Object.entries(node.config ?? {})) {
    if (value != null) rows.push([key.replace(/_/g, " "), typeof value === "object" ? JSON.stringify(value) : String(value)]);
  }
  if (refMeta && file) {
    rows.push(["Source", file.display_name + ":" + refMeta.start_line + "–" + refMeta.end_line]);
  }
  const codeRefs = node.component_type === "embedding"
    ? node.source_reference_ids.map((id) => result.source_references.find((ref) => ref.id === id)).filter((ref): ref is SourceReference => ref != null)
    : [];

  return (
    <>
      <div className="details-header">
        <div className="details-title">
          <h4>{node.label}</h4>
          <span>{node.component_type}</span>
        </div>
        <button className="details-close" onClick={onClose} aria-label="Close component details">×</button>
      </div>
      {rows.length > 0 && (
        <dl className="details-grid">
          {rows.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      )}
      {codeRefs.length > 0 && (
        <div className="details-code-links" aria-label="Related source code">
          <span>Code</span>
          {codeRefs.map((ref) => {
            const implementation = ref.symbol === "Embedding" && ref.method === "forward";
            const label = implementation
              ? "PyTorch nn.Embedding.forward"
              : node.module_path === "wte" ? "GPT-2 token lookup" : "GPT-2 positions + sum";
            return <button key={ref.id} onClick={() => onJumpToRef(ref.id)}>{label}</button>;
          })}
        </div>
      )}
    </>
  );
}
