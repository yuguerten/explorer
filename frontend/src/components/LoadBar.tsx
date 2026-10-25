import { useState } from "react";

export function LoadBar({
  status,
  onInspect,
  modelId,
}: {
  status: { phase: "idle" | "loading" | "success" | "error"; message: string };
  onInspect: (modelId: string, revision?: string) => void;
  modelId: string;
}) {
  const [id, setId] = useState(modelId || "gpt2");
  const [revision, setRevision] = useState("");

  const submit = () => {
    if (id.trim()) onInspect(id.trim(), revision.trim() || undefined);
  };

  return (
    <div className="loadbar">
      <span className="title">Transformer Explorer</span>
      <input
        type="text"
        value={id}
        placeholder="Hugging Face model ID, e.g. gpt2"
        onChange={(e) => setId(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
        aria-label="Model ID"
      />
      <input
        type="text"
        value={revision}
        placeholder="revision (optional)"
        onChange={(e) => setRevision(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && submit()}
        aria-label="Revision"
      />
      <button onClick={submit}>Inspect</button>
      <span className={`status ${status.phase}`}>
        <span className="dot" />
        {status.phase === "idle" && "ready"}
        {status.phase === "loading" && "inspecting…"}
        {status.phase === "success" && status.message}
        {status.phase === "error" && status.message}
      </span>
    </div>
  );
}
