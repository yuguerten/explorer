import type { ArchitectureEdge, ArchitectureNode } from "../types";

/**
 * Generic, compact layout for any architecture graph emitted by the backend.
 *
 * The backend produces a generic graph (model root -> embeddings / group /
 * tail, group -> block, block -> internal components, plus tensor-flow and
 * residual edges). This layouter arranges it into a compact vertical flow:
 *
 *   [ Model ]
 *     ├─ [Token Embedding] [Position Embedding]
 *     ▼
 *   ┌─ Transformer Blocks ×N ─────────────────┐
 *   │   x ──────────────► [ LN1 ]             │
 *   │   │                 [ Attn]             │
 *   │   x ─residual add─► [ ⊕  ]              │
 *   │   │                 [ LN2 ]             │
 *   │   │                 [ MLP ]             │
 *   │   x ─residual add─► [ ⊕  ]──►           │
 *   └─────────────────────────────────────────┘
 *     ▼
 *   [ Final LayerNorm ]
 *
 * Containers (model / block / group) frame their children recursively; leaf
 * components render on top. All coordinates are absolute (global) so React
 * Flow renders each node at its final position with container panels behind.
 *
 * It is model-agnostic: it keys off component_type / parent_id, never off
 * model-specific names, so any adapter output can be laid out.
 */

const NODE_W = 190;
const NODE_H = 60;
const ROW_GAP = 16;
const ADD_SIZE = 36;
const EMB_GAP = 32;
const MODEL_H = 40;
const HEADER_H = 34;
const GROUP_HEADER_H = 46;
const CONTAINER_PAD = 10;
const SECTION_GAP = 22;

export interface LayoutNode {
  node: ArchitectureNode;
  x: number;
  y: number;
  w: number;
  h: number;
  z: number;
}

export interface LayoutEdge {
  source: string;
  target: string;
  kind: string;
  label?: string | null;
}

export interface LayoutResult {
  nodes: LayoutNode[];
  edges: LayoutEdge[];
  width: number;
  height: number;
}

interface Pos {
  x: number;
  y: number;
  w: number;
  h: number;
}

const isContainer = (t: string) => t === "model" || t === "block" || t === "group";
const isAdd = (t: string) => t === "add";

const ORDER: Record<string, number> = {
  embedding: 0,
  layernorm: 1,
  attention: 2,
  add: 3,
  mlp: 4,
};

export function layoutDecoder(
  visibleNodes: ArchitectureNode[],
  _visibleEdges: ArchitectureEdge[],
  _expanded: Set<string>
): LayoutResult {
  const childrenOf = new Map<string, ArchitectureNode[]>();
  for (const n of visibleNodes) {
    const key = n.parent_id ?? "__root__";
    if (!childrenOf.has(key)) childrenOf.set(key, []);
    childrenOf.get(key)!.push(n);
  }
  const flowOrder = (n: ArchitectureNode): number => {
    if (n.component_type === "embedding") return 0;
    if (n.module_path?.endsWith(".ln_1")) return 1;
    if (n.component_type === "attention") return 2;
    if (n.id.endsWith(".add_1")) return 3;
    if (n.module_path?.endsWith(".ln_2")) return 4;
    if (n.component_type === "mlp") return 5;
    if (n.id.endsWith(".add_2")) return 6;
    return (ORDER[n.component_type] ?? 99) + 10;
  };
  for (const list of childrenOf.values()) {
    list.sort((a, b) => flowOrder(a) - flowOrder(b));
  }

  const pos = new Map<string, Pos>();
  const placed = new Set<string>();
  const W = 620;
  let cursorY = 16;
  const maxX = { v: 0 };

  const placeContentColumn = (list: ArchitectureNode[], x: number): number => {
    // Place each component in semantic flow order. Add junctions occupy their
    // own row so they cannot overlap a neighboring node.
    let y = cursorY;
    const content = list.filter((n) => !isContainer(n.component_type));
    for (const c of content) {
      if (isAdd(c.component_type)) {
        pos.set(c.id, {
          x: x + (NODE_W - ADD_SIZE) / 2,
          y: y + (NODE_H - ADD_SIZE) / 2,
          w: ADD_SIZE,
          h: ADD_SIZE,
        });
      } else {
        pos.set(c.id, { x, y, w: NODE_W, h: NODE_H });
      }
      placed.add(c.id);
      y += NODE_H + ROW_GAP;
    }
    return content.length ? y - ROW_GAP : cursorY;
  };

  // ---- model root panel ----
  const model = visibleNodes.find((n) => n.component_type === "model");
  if (model) {
    pos.set(model.id, { x: 8, y: cursorY, w: W - 16, h: MODEL_H });
    placed.add(model.id);
    cursorY += MODEL_H + SECTION_GAP;
  }

  const modelKids = childrenOf.get(model?.id ?? "") ?? [];
  const emb = modelKids.filter((c) => c.component_type === "embedding");
  const others = modelKids.filter((c) => c.component_type !== "embedding");

  // ---- embeddings row ----
  if (emb.length) {
    const totalW = emb.length * NODE_W + (emb.length - 1) * EMB_GAP;
    let ex = (W - totalW) / 2;
    for (const c of emb) {
      pos.set(c.id, { x: ex, y: cursorY, w: NODE_W, h: NODE_H });
      placed.add(c.id);
      ex += NODE_W + EMB_GAP;
    }
    cursorY += NODE_H + SECTION_GAP;
  }

  // ---- transformer block group ----
  const group = modelKids.find((c) => c.component_type === "group");
  if (group) {
    const gx = 20;
    const gw = W - 40;
    const gy = cursorY;
    const block = (childrenOf.get(group.id) ?? []).find((c) => c.component_type === "block");

    let innerBottom = gy + GROUP_HEADER_H;
    if (block) {
      const blockKids = childrenOf.get(block.id) ?? [];
      cursorY = gy + GROUP_HEADER_H + HEADER_H + ROW_GAP;
      const mainX = (W - NODE_W) / 2;
      const bottom = placeContentColumn(blockKids, mainX);
      // block panel wraps its children
      const bp: Pos = {
        x: gx + CONTAINER_PAD,
        y: gy + GROUP_HEADER_H,
        w: gw - CONTAINER_PAD * 2,
        h: bottom - (gy + GROUP_HEADER_H) + ROW_GAP,
      };
      pos.set(block.id, bp);
      placed.add(block.id);
      innerBottom = bp.y + bp.h;
      cursorY = innerBottom;
    }
    pos.set(group.id, { x: gx, y: gy, w: gw, h: innerBottom - gy });
    placed.add(group.id);
    cursorY = innerBottom + SECTION_GAP;
  }

  // ---- tail (final LN + output + lm head), non-embedding direct children ----
  const mainX = (W - NODE_W) / 2;
  for (const c of others) {
    if (c.component_type === "group" || placed.has(c.id)) continue;
    pos.set(c.id, { x: mainX, y: cursorY, w: NODE_W, h: NODE_H });
    placed.add(c.id);
    cursorY += NODE_H + SECTION_GAP;
  }

  // ---- fallback for any unplaced nodes ----
  const fallbackX = W + 60;
  let fy = 16;
  for (const n of visibleNodes) {
    if (!placed.has(n.id)) {
      pos.set(n.id, { x: fallbackX, y: fy, w: NODE_W, h: NODE_H });
      placed.add(n.id);
      fy += NODE_H + 18;
      maxX.v = Math.max(maxX.v, fallbackX + NODE_W);
    }
  }

  const nodes: LayoutNode[] = visibleNodes.map((n) => {
    const p = pos.get(n.id) ?? { x: 0, y: 0, w: NODE_W, h: NODE_H };
    const z = isContainer(n.component_type) ? 0 : 3;
    return { node: n, x: p.x, y: p.y, w: p.w, h: p.h, z };
  });

  const edges: LayoutEdge[] = _visibleEdges
    .filter((e) => e.kind !== "contains")
    .map((e) => ({ source: e.source, target: e.target, kind: e.kind, label: e.label }));

  const width = Math.max(W + 40, maxX.v + 40);
  const height = Math.max(cursorY, fy + 40);

  return { nodes, edges, width, height };
}
