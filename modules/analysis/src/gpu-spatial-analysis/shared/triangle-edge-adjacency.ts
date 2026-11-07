// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Adjacency value of a triangle edge that has no neighbor among the live triangles. */
export const TRIANGLE_ADJACENCY_NONE = 0xffffffff;

/** Properties for {@link createTriangleEdgeAdjacencyNodes}. @internal */
export type TriangleEdgeAdjacencyProps = {
  id: string;
  operation: string;
  /** Number of vertices that triangle corners may reference. */
  pointCount: number;
  triangles: GraphDataView<'uint32x3'>;
  /** Only the first `triangleCount[0]` triangles are live. */
  triangleCount: GraphDataView<'uint32'>;
  /** Optional per-triangle live mask applied on top of `triangleCount`. */
  validity?: GraphDataView<'uint32'>;
  /**
   * Written with one neighbor triangle per directed triangle edge `(corner, corner + 1)`, or
   * {@link TRIANGLE_ADJACENCY_NONE} when no other live triangle shares the undirected edge.
   */
  adjacency: GraphDataView<'uint32'>;
};

/**
 * Builds triangle edge adjacency from one stable sort of undirected edge keys, so the cost is a
 * sort plus a binary search per half-edge instead of a scan of every triangle per edge.
 * @internal
 */
export function createTriangleEdgeAdjacencyNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: TriangleEdgeAdjacencyProps
): GPUCommandNode<Parameters>[] {
  const {id, operation, pointCount, triangles} = props;
  const triangleCapacity = triangles.length;
  const halfEdgeCapacity = 3 * triangleCapacity;
  if (halfEdgeCapacity === 0) {
    return [];
  }
  if (props.adjacency.length !== halfEdgeCapacity) {
    // Adjacency must provide one row per directed triangle edge.
    throw new Error(`${id} adjacency length must equal three times the triangle capacity`);
  }
  const candidateIds = createTransientView(
    graph,
    `${id}-candidate-ids`,
    'uint32',
    halfEdgeCapacity
  );
  const minimumKeys = createTransientView(graph, `${id}-minimum-keys`, 'uint32', halfEdgeCapacity);
  const maximumKeys = createTransientView(graph, `${id}-maximum-keys`, 'uint32', halfEdgeCapacity);
  const keyBits = clamp(32 - Math.clz32(pointCount), 1, 32);
  const edgeSort = createStableLexicographicIndexSortNodes<Parameters>(graph, {
    id: `${id}-sort`,
    operation,
    indices: candidateIds,
    keys: [
      {view: minimumKeys, keyBits},
      {view: maximumKeys, keyBits}
    ]
  });
  const validityBinding = props.validity
    ? [{name: 'validity', view: props.validity, type: 'u32' as const, access: 'read' as const}]
    : [];
  return [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-keys`,
      operation,
      variant: 'triangle-adjacency-keys',
      bindings: [
        {name: 'triangles', view: triangles, type: 'u32', access: 'read'},
        {name: 'triangleCount', view: props.triangleCount, type: 'u32', access: 'read'},
        ...validityBinding,
        {name: 'candidateIds', view: candidateIds, type: 'u32', access: 'read_write'},
        {name: 'minimumKeys', view: minimumKeys, type: 'u32', access: 'read_write'},
        {name: 'maximumKeys', view: maximumKeys, type: 'u32', access: 'read_write'}
      ],
      invocationCount: halfEdgeCapacity,
      declarations: `const POINT_COUNT: u32 = ${pointCount}u;
const TRIANGLE_CAPACITY: u32 = ${triangleCapacity}u;`,
      body: `let triangle = index / 3u;
let edge = index % 3u;
candidateIds[candidateIdsOffset + index] = index;
var minimum = POINT_COUNT; var maximum = POINT_COUNT;
var live = triangle < min(triangleCount[triangleCountOffset], TRIANGLE_CAPACITY);
${props.validity ? 'live = live && validity[validityOffset + triangle] != 0u;' : ''}
if (live) {
  let a = triangles[trianglesOffset + triangle * 3u + edge];
  let b = triangles[trianglesOffset + triangle * 3u + (edge + 1u) % 3u];
  if (a < POINT_COUNT && b < POINT_COUNT && a != b) { minimum = min(a, b); maximum = max(a, b); }
}
minimumKeys[minimumKeysOffset + index] = minimum;
maximumKeys[maximumKeysOffset + index] = maximum;`
    }),
    ...edgeSort.nodes,
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-resolve`,
      operation,
      variant: 'triangle-adjacency-resolve',
      bindings: [
        {name: 'sortedCandidates', view: edgeSort.sortedIndices, type: 'u32', access: 'read'},
        {name: 'sortedMinimumKeys', view: edgeSort.sortedPrimaryKeys, type: 'u32', access: 'read'},
        {name: 'maximumKeys', view: maximumKeys, type: 'u32', access: 'read'},
        {name: 'adjacency', view: props.adjacency, type: 'u32', access: 'read_write'}
      ],
      invocationCount: halfEdgeCapacity,
      declarations: `const POINT_COUNT: u32 = ${pointCount}u;
const HALF_EDGE_COUNT: u32 = ${halfEdgeCapacity}u;
fn sortedMaximumKey(row: u32) -> u32 { return maximumKeys[maximumKeysOffset + sortedCandidates[sortedCandidatesOffset + row]]; }
fn sortedKeyEquals(row: u32, minimum: u32, maximum: u32) -> bool {
  return sortedMinimumKeys[sortedMinimumKeysOffset + row] == minimum && sortedMaximumKey(row) == maximum;
}`,
      // Equal undirected keys are contiguous and ordered by candidate id (stable sort).
      body: `let candidate = sortedCandidates[sortedCandidatesOffset + index];
let triangle = candidate / 3u;
let minimum = sortedMinimumKeys[sortedMinimumKeysOffset + index];
var neighbor = 0xffffffffu;
if (minimum < POINT_COUNT) {
  let maximum = sortedMaximumKey(index);
  var low = 0u; var high = index;
  while (low < high) {
    let middle = (low + high) >> 1u;
    let middleMinimum = sortedMinimumKeys[sortedMinimumKeysOffset + middle];
    if (middleMinimum < minimum || (middleMinimum == minimum && sortedMaximumKey(middle) < maximum)) { low = middle + 1u; } else { high = middle; }
  }
  var row = low;
  while (row < HALF_EDGE_COUNT && sortedKeyEquals(row, minimum, maximum)) {
    let otherTriangle = sortedCandidates[sortedCandidatesOffset + row] / 3u;
    if (otherTriangle != triangle) { neighbor = otherTriangle; break; }
    row++;
  }
}
adjacency[adjacencyOffset + candidate] = neighbor;`
    })
  ];
}
