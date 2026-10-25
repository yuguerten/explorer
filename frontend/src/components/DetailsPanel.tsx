import type { ArchitectureNode, InspectionResult, SourceReference } from "../types";

export function DetailsPanel({ node, refMeta, result, onClose }: {
  node: ArchitectureNode | null;
  refMeta: SourceReference | null;
  result: InspectionResult;
  onClose: () => void;
}) {
  if (!node) return null;

  const file = refMeta
    ? result.source_files.find((f) => f.file_id === refMeta.source_file_id)
    : undefined;
  const rows: Array<[string, string]> = [];
  if (node.module_path != null) rows.push(["Module", node.module_path || "Model root"]);
  if (node.parameter_count != null) rows.push(["Parameters", node.parameter_count.toLocaleString("en-US")]);
  if (node.repetition != null) rows.push(["Layers", String(node.repetition)]);
  for (const [key, value] of Object.entries(node.config ?? {})) {
    if (value != null) rows.push([key.replace(/_/g, " "), typeof value === "object" ? JSON.stringify(value) : String(value)]);
  }
  if (refMeta && file) {
    rows.push(["Source", file.display_name + ":" + refMeta.start_line + "–" + refMeta.end_line]);
  }

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
    </>
  );
}
