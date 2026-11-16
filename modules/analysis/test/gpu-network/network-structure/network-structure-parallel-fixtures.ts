// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CSR preserving distinct reverse pairs for parallel edges and self-pairing each loop. */
export function createUndirectedStructureCSR(
  nodeCount: number,
  edges: readonly (readonly [number, number])[],
  padding = 0
) {
  const offsets = new Uint32Array(nodeCount + 1);
  for (const [source, target] of edges) {
    offsets[source + 1]++;
    if (source !== target) offsets[target + 1]++;
  }
  for (let vertex = 0; vertex < nodeCount; vertex++) offsets[vertex + 1] += offsets[vertex];
  const neighbors = new Uint32Array(offsets[nodeCount] + padding).fill(0xffffffff);
  const reverseEdgeSlots = new Uint32Array(neighbors.length).fill(0xffffffff);
  const cursor = offsets.slice();
  for (const [source, target] of edges) {
    const forward = cursor[source]++;
    neighbors[forward] = target;
    if (source === target) {
      reverseEdgeSlots[forward] = forward;
    } else {
      const backward = cursor[target]++;
      neighbors[backward] = source;
      reverseEdgeSlots[forward] = backward;
      reverseEdgeSlots[backward] = forward;
    }
  }
  return {offsets, neighbors, reverseEdgeSlots};
}

/** Two rectangular grids connected by exactly one edge, with 100,000 vertices overall. */
export function createSplitGridStructureCSR() {
  const width = 400;
  const height = 250;
  const boundary = width / 2;
  const edges: [number, number][] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const vertex = row * width + column;
      if (row + 1 < height) edges.push([vertex, vertex + width]);
      if (column + 1 < width && (column + 1 !== boundary || row === 0)) {
        edges.push([vertex, vertex + 1]);
      }
    }
  }
  return {...createUndirectedStructureCSR(width * height, edges), width, boundary};
}
