import type { InspectionResult } from "./types";

export async function inspectModel(modelId: string, revision?: string): Promise<InspectionResult> {
  const res = await fetch("/api/inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId, revision: revision || null }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = (body as { detail?: string }).detail ?? `Request failed (${res.status})`;
    throw new Error(detail);
  }
  return res.json();
}
