// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CSR with optional live-looking tail slots that no vertex owns. */
export function createComponentCSR(
  nodeCount: number,
  edges: readonly (readonly [number, number])[],
  padding = 0
): {offsets: Uint32Array; neighbors: Uint32Array} {
  const offsets = new Uint32Array(nodeCount + 1);
  for (const [source] of edges) offsets[source + 1]++;
  for (let vertex = 0; vertex < nodeCount; vertex++) offsets[vertex + 1] += offsets[vertex];
  const cursor = offsets.slice(0, nodeCount);
  const neighbors = new Uint32Array(edges.length + padding);
  for (const [source, target] of edges) neighbors[cursor[source]++] = target;
  return {offsets, neighbors};
}

/** Use high bits so power-of-two vertex counts do not inherit an LCG's low-bit edge pattern. */
export function createRandomComponentEdges(
  nodeCount: number,
  edgeCount: number,
  seed: number
): [number, number][] {
  let state = seed >>> 0;
  const nextVertex = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) % nodeCount;
  };
  return Array.from({length: edgeCount}, () => [nextVertex(), nextVertex()]);
}
