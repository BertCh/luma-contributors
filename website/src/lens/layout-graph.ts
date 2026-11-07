// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Layered (Sugiyama-style) left-to-right layout for a LensViewModel. Pure and deterministic so the
// prerendered SVG and the client render agree.

import type {LensViewModel} from './types.ts';

export type LaidOutNode = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  layer: number;
};
export type LaidOutEdge = {
  from: string;
  to: string;
  /** Cubic Bezier endpoints and controls, in layout units. */
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  controlX0: number;
  controlY0: number;
  controlX1: number;
  controlY1: number;
  /** Layers skipped over; long edges are drawn faint because they cross other nodes. */
  span: number;
};
export type GraphLayout = {
  width: number;
  height: number;
  nodes: LaidOutNode[];
  edges: LaidOutEdge[];
};

export type LayoutOptions = {
  nodeWidth?: number;
  nodeHeight?: number;
  layerGap?: number;
  rowGap?: number;
  padding?: number;
  /** Layers wrap into rows so the picture stays near this width. */
  targetWidth?: number;
};

export function layoutLensGraph(model: LensViewModel, options: LayoutOptions = {}): GraphLayout {
  const {
    nodeWidth = 168,
    nodeHeight = 38,
    layerGap = 74,
    rowGap = 14,
    padding = 20,
    targetWidth = 1040
  } = options;
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  for (const node of model.nodes) {
    incoming.set(node.id, []);
    outgoing.set(node.id, []);
  }
  for (const edge of model.edges) {
    incoming.get(edge.to)?.push(edge.from);
    outgoing.get(edge.from)?.push(edge.to);
  }

  // Longest path layering; nodes are in schedule order so every edge points forward.
  const layerOf = new Map<string, number>();
  for (const node of model.nodes) {
    let layer = 0;
    for (const parent of incoming.get(node.id) ?? [])
      layer = Math.max(layer, (layerOf.get(parent) ?? 0) + 1);
    layerOf.set(node.id, layer);
  }
  const layerCount = Math.max(0, ...layerOf.values()) + 1;
  const layers: string[][] = Array.from({length: layerCount}, () => []);
  for (const node of model.nodes) layers[layerOf.get(node.id) ?? 0].push(node.id);

  // Barycenter sweeps reduce crossings.
  const rank = new Map<string, number>();
  const assignRanks = () =>
    layers.forEach(layer => layer.forEach((id, index) => rank.set(id, index)));
  assignRanks();
  const barycenter = (id: string, neighbors: Map<string, string[]>): number => {
    const list = neighbors.get(id) ?? [];
    if (list.length === 0) return rank.get(id) ?? 0;
    return list.reduce((total, other) => total + (rank.get(other) ?? 0), 0) / list.length;
  };
  for (let sweep = 0; sweep < 4; sweep++) {
    for (let layer = 1; layer < layerCount; layer++) {
      layers[layer].sort((a, b) => barycenter(a, incoming) - barycenter(b, incoming));
      assignRanks();
    }
    for (let layer = layerCount - 2; layer >= 0; layer--) {
      layers[layer].sort((a, b) => barycenter(a, outgoing) - barycenter(b, outgoing));
      assignRanks();
    }
  }

  const layersPerRow = Math.max(
    1,
    Math.floor((targetWidth - 2 * padding + layerGap) / (nodeWidth + layerGap))
  );
  const bandCount = Math.ceil(layerCount / layersPerRow);
  const bandHeights = Array.from({length: bandCount}, (_, band) =>
    Math.max(
      1,
      ...layers.slice(band * layersPerRow, (band + 1) * layersPerRow).map(layer => layer.length)
    )
  );
  const bandGap = 40;
  const bandTops: number[] = [];
  let cursor = padding;
  for (const rows of bandHeights) {
    bandTops.push(cursor);
    cursor += rows * nodeHeight + (rows - 1) * rowGap + bandGap;
  }
  const nodes: LaidOutNode[] = [];
  const byId = new Map<string, LaidOutNode>();
  layers.forEach((layer, layerIndex) => {
    const band = Math.floor(layerIndex / layersPerRow);
    const band0 = layerIndex % layersPerRow;
    // Odd rows run right to left, so the end of one row sits above the start of the next.
    const column = band % 2 === 0 ? band0 : layersPerRow - 1 - band0;
    const bandHeight = bandHeights[band] * nodeHeight + (bandHeights[band] - 1) * rowGap;
    const height = layer.length * nodeHeight + (layer.length - 1) * rowGap;
    const offset = (bandHeight - height) / 2;
    layer.forEach((id, index) => {
      const placed: LaidOutNode = {
        id,
        x: padding + column * (nodeWidth + layerGap),
        y: bandTops[band] + offset + index * (nodeHeight + rowGap),
        width: nodeWidth,
        height: nodeHeight,
        layer: layerIndex
      };
      nodes.push(placed);
      byId.set(id, placed);
    });
  });

  const edges: LaidOutEdge[] = [];
  for (const edge of model.edges) {
    const from = byId.get(edge.from);
    const to = byId.get(edge.to);
    if (!from || !to) continue;
    const span = to.layer - from.layer;
    const sameBand = Math.floor(from.layer / layersPerRow) === Math.floor(to.layer / layersPerRow);
    let x0: number;
    let y0: number;
    let x1: number;
    let y1: number;
    let controlX0: number;
    let controlY0: number;
    let controlX1: number;
    let controlY1: number;
    if (sameBand && to.x > from.x) {
      x0 = from.x + from.width;
      y0 = from.y + from.height / 2;
      x1 = to.x;
      y1 = to.y + to.height / 2;
      const reach = Math.max(24, (x1 - x0) * 0.5);
      [controlX0, controlY0, controlX1, controlY1] = [x0 + reach, y0, x1 - reach, y1];
    } else if (sameBand) {
      x0 = from.x;
      y0 = from.y + from.height / 2;
      x1 = to.x + to.width;
      y1 = to.y + to.height / 2;
      const reach = Math.max(24, (x0 - x1) * 0.5);
      [controlX0, controlY0, controlX1, controlY1] = [x0 - reach, y0, x1 + reach, y1];
    } else {
      x0 = from.x + from.width / 2;
      y0 = from.y + from.height;
      x1 = to.x + to.width / 2;
      y1 = to.y;
      const reach = Math.max(20, (y1 - y0) * 0.5);
      [controlX0, controlY0, controlX1, controlY1] = [x0, y0 + reach, x1, y1 - reach];
    }
    edges.push({
      from: edge.from,
      to: edge.to,
      x0,
      y0,
      x1,
      y1,
      controlX0,
      controlY0,
      controlX1,
      controlY1,
      span
    });
  }
  const columns = Math.min(layerCount, layersPerRow);
  return {
    width: padding * 2 + columns * nodeWidth + (columns - 1) * layerGap,
    height: cursor - bandGap + padding,
    nodes,
    edges
  };
}

/** SVG path of an edge: horizontal tangents at both ends. */
export function getEdgePath(edge: LaidOutEdge): string {
  return `M${edge.x0},${edge.y0} C${edge.controlX0},${edge.controlY0} ${edge.controlX1},${edge.controlY1} ${edge.x1},${edge.y1}`;
}
