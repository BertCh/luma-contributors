// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createTriangleEdgeAdjacencyNodes} from '../shared/triangle-edge-adjacency';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGPUStorageBindingByteLength} from '../../analysis-core/kernels/gpu-contributor-utils';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {ADAPTIVE_INCIRCLE_WGSL} from '../delaunay-tessellation/gpu-delaunay-tessellation';
import type {GPUConstrainedDelaunayTriangulationProps} from './gpu-constrained-delaunay-triangulation';

/** One clear, candidate, apply, adjacency-repair and gate pass per order-preserving flip round. */
export const CONSTRAINED_DELAUNAY_NODES_PER_ROUND = 5;
/** Initial triangulation, adjacency sorts, initialization and fixed passes; measured by node specs. */
export const CONSTRAINED_DELAUNAY_FIXED_NODE_ALLOWANCE = 1024;
/** Small meshes keep the single-pass reference; large meshes parallelize edge discovery. */
export const CONSTRAINED_DELAUNAY_PARALLEL_MINIMUM_TRIANGLE_COUNT = 256;

/** Preflight before either initial triangulation or parallel scratch is declared. @internal */
export function validateConstrainedDelaunayParallelScratch<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  triangleCount: number,
  polygonCount: number
): void {
  for (const [name, byteLength] of [
    ['flip candidates and adjacency', Math.max(3 * triangleCount, 1) * 4],
    ['flip records', Math.max(8 * polygonCount, 1) * 4],
    [
      'adjacency sort',
      (3 * triangleCount <= 256
        ? 2 ** Math.ceil(Math.log2(Math.max(3 * triangleCount, 1)))
        : 3 * triangleCount) * 4
    ],
    ['dispatch gate', 24]
  ] as const)
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce the triangle capacity or polygon count'
    );
}

/**
 * Parallel discovery of the same first illegal triangle pair chosen by the serial reference.
 * Each triangle tests only its three indexed neighbors, picks its lowest second triangle / side,
 * and atomicMin selects the lowest first triangle in each polygon. Applying one flip per polygon
 * preserves the tested pair/side ordering exactly. A separate pass repairs the two changed faces
 * and their neighbors from a saved six-word record; no pass reads triangles while they change.
 * Round indices appear only in node IDs. Choices, counters and indirect gates live on the GPU.
 * @internal
 */
export function getConstrainedDelaunayParallelNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUConstrainedDelaunayTriangulationProps,
  offsets: GraphDataView<'uint32'>
): GPUCommandNode<Parameters>[] {
  const id = `${props.id ?? 'constrained-delaunay'}-parallel`;
  const operation = 'GPUConstrainedDelaunayTriangulation';
  const polygonCount = props.polygonOffsets.length - 1;
  const triangleCount = Math.floor(props.output.indices.length / 3);
  validateConstrainedDelaunayParallelScratch(graph, id, triangleCount, polygonCount);
  const view = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
  const owners = view('owners', triangleCount);
  const live = view('live', triangleCount);
  const total = view('total', 1);
  const adjacency = view('adjacency', 3 * triangleCount);
  const choices = view('choices', polygonCount);
  const iterations = view('iterations', polygonCount);
  const candidates = view('candidates', 3 * triangleCount);
  const records = view('records', 8 * polygonCount);
  const remaining = view('remaining', 1);
  const triangles = graph.createDataView(props.output.indices.buffer, {
    format: 'uint32x3',
    byteOffset: props.output.indices.byteOffset,
    length: triangleCount
  });
  const binding = (
    name: string,
    bound: GraphDataView,
    access: 'read' | 'read_write' = 'read',
    type: 'u32' | 'f32' | 'atomic<u32>' = 'u32'
  ): WGSLKernelBinding => ({name, view: bound, access, type});
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-gate`,
    operation,
    shapes: [
      [polygonCount, 256],
      [triangleCount, 32]
    ]
  });
  const triangleDeclarations = `fn triangle(row: u32) -> vec3u { let base = indicesOffset + 3u * row; return vec3u(indices[base], indices[base + 1u], indices[base + 2u]); }
fn point(row: u32) -> vec2f { let base = positionsOffset + 2u * row; return vec2f(positions[base], positions[base + 1u]); }`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-initialize-polygons`,
      operation,
      invocationCount: polygonCount,
      workgroupSize: 256,
      bindings: [
        binding('iterations', iterations, 'read_write'),
        binding('converged', props.output.status.converged, 'read_write')
      ],
      body: 'iterations[iterationsOffset + index] = 0u; converged[convergedOffset + index] = 0u;'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-initialize-gate`,
      operation,
      invocationCount: 1,
      bindings: [
        gate.binding,
        binding('remaining', remaining, 'read_write'),
        binding('total', total, 'read_write')
      ],
      body: `${gate.armWGSL}\nremaining[remainingOffset] = 0u; total[totalOffset] = ${triangleCount}u;`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-owners`,
      operation,
      invocationCount: triangleCount,
      workgroupSize: 256,
      bindings: [
        binding('offsets', offsets),
        binding('valid', props.output.valid),
        binding('indices', props.output.indices),
        binding('owners', owners, 'read_write'),
        binding('live', live, 'read_write')
      ],
      declarations: `fn triangle(row: u32) -> vec3u { let base = indicesOffset + 3u * row; return vec3u(indices[base], indices[base + 1u], indices[base + 2u]); }`,
      body: `let polygonCount = arrayLength(&offsets) - offsetsOffset - 1u;
var low = 0u; var high = polygonCount;
while (low < high) { let middle = low + (high - low + 1u) / 2u; if (offsets[offsetsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; } }
owners[ownersOffset + index] = low;
var usable = false;
if (low < polygonCount) { let corners = triangle(index); usable = valid[validOffset + low] != 0u && corners.x != corners.y && corners.x != corners.z && corners.y != corners.z; }
live[liveOffset + index] = u32(usable);`
    }),
    ...createTriangleEdgeAdjacencyNodes(graph, {
      id: `${id}-adjacency`,
      operation,
      pointCount: props.positions.length,
      triangles,
      triangleCount: total,
      validity: live,
      adjacency
    })
  ];
  const maximumFlips = props.maximumFlips ?? 4096;
  for (let round = 0; round <= maximumFlips; round++) {
    const prefix = `${id}-round-${round}`;
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${prefix}-clear`,
        operation,
        variant: 'parallel-flip-clear',
        invocationCount: polygonCount,
        workgroupSize: 256,
        ...gate.getGate(0),
        bindings: [
          binding('choices', choices, 'read_write'),
          binding('records', records, 'read_write')
        ],
        body: 'choices[choicesOffset + index] = 0xffffffffu; records[recordsOffset + 8u * index] = 0xffffffffu;'
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-evaluate`,
        operation,
        variant: 'parallel-flip-evaluate',
        invocationCount: triangleCount,
        workgroupSize: 32,
        ...gate.getGate(1),
        bindings: [
          binding('positions', props.positions, 'read', 'f32'),
          binding('indices', props.output.indices),
          binding('owners', owners),
          binding('live', live),
          binding('adjacency', adjacency),
          binding('converged', props.output.status.converged),
          binding('candidates', candidates, 'read_write'),
          binding('choices', choices, 'read_write', 'atomic<u32>')
        ],
        declarations: `${EXACT_ORIENTATION_WGSL}\n${ADAPTIVE_INCIRCLE_WGSL}\n${triangleDeclarations}`,
        body: `if (live[liveOffset + index] == 0u) { return; }
let polygon = owners[ownersOffset + index]; if (converged[convergedOffset + polygon] != 0u) { return; }
let first = triangle(index); var best = 0xffffffffu; var bestSide = 0u; var bestOtherSide = 0u;
for (var side = 0u; side < 3u; side++) {
  let secondRow = adjacency[adjacencyOffset + 3u * index + side];
  if (secondRow <= index || secondRow >= arrayLength(&owners) - ownersOffset || owners[ownersOffset + secondRow] != polygon) { continue; }
  let second = triangle(secondRow); let firstVertex = first[side]; let secondVertex = first[(side + 1u) % 3u]; let firstOpposite = first[(side + 2u) % 3u];
  for (var otherSide = 0u; otherSide < 3u; otherSide++) {
    if (second[otherSide] != secondVertex || second[(otherSide + 1u) % 3u] != firstVertex) { continue; }
    let secondOpposite = second[(otherSide + 2u) % 3u];
    if (orientSign(point(firstOpposite), point(secondOpposite), point(secondVertex)) != 1 || orientSign(point(secondOpposite), point(firstOpposite), point(firstVertex)) != 1 || incircleSign(point(firstVertex), point(secondVertex), point(firstOpposite), point(secondOpposite)) <= 0) { continue; }
    if (secondRow < best || (secondRow == best && side < bestSide)) { best = secondRow; bestSide = side; bestOtherSide = otherSide; }
  }
}
if (best != 0xffffffffu) { let base = candidatesOffset + 3u * index; candidates[base] = best; candidates[base + 1u] = bestSide; candidates[base + 2u] = bestOtherSide; atomicMin(&choices[choicesOffset + polygon], index); }`
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-apply`,
        operation,
        variant: 'parallel-flip-apply',
        invocationCount: polygonCount,
        workgroupSize: 256,
        ...gate.getGate(0),
        bindings: [
          binding('choices', choices),
          binding('candidates', candidates),
          binding('adjacency', adjacency),
          binding('indices', props.output.indices, 'read_write'),
          binding('iterations', iterations, 'read_write'),
          binding('records', records, 'read_write'),
          binding('converged', props.output.status.converged, 'read_write'),
          binding('remaining', remaining, 'read_write', 'atomic<u32>')
        ],
        declarations: triangleDeclarations.split('fn point')[0],
        body: `let firstRow = choices[choicesOffset + index];
if (firstRow == 0xffffffffu) { converged[convergedOffset + index] = 1u; return; }
if (iterations[iterationsOffset + index] >= ${maximumFlips}u) { return; }
let candidateBase = candidatesOffset + 3u * firstRow; let secondRow = candidates[candidateBase]; let side = candidates[candidateBase + 1u]; let otherSide = candidates[candidateBase + 2u];
let first = triangle(firstRow); let second = triangle(secondRow); let firstOpposite = first[(side + 2u) % 3u]; let secondOpposite = second[(otherSide + 2u) % 3u];
let recordBase = recordsOffset + 8u * index; records[recordBase] = firstRow; records[recordBase + 1u] = secondRow;
records[recordBase + 2u] = adjacency[adjacencyOffset + 3u * firstRow + (side + 1u) % 3u];
records[recordBase + 3u] = adjacency[adjacencyOffset + 3u * firstRow + (side + 2u) % 3u];
records[recordBase + 4u] = adjacency[adjacencyOffset + 3u * secondRow + (otherSide + 1u) % 3u];
records[recordBase + 5u] = adjacency[adjacencyOffset + 3u * secondRow + (otherSide + 2u) % 3u];
let firstBase = indicesOffset + 3u * firstRow; let secondBase = indicesOffset + 3u * secondRow;
indices[firstBase] = firstOpposite; indices[firstBase + 1u] = secondOpposite; indices[firstBase + 2u] = first[(side + 1u) % 3u];
indices[secondBase] = secondOpposite; indices[secondBase + 1u] = firstOpposite; indices[secondBase + 2u] = first[side];
iterations[iterationsOffset + index] += 1u; atomicStore(&remaining[remainingOffset], 1u);`
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-repair`,
        operation,
        variant: 'parallel-flip-repair',
        invocationCount: triangleCount,
        workgroupSize: 32,
        ...gate.getGate(1),
        bindings: [
          binding('owners', owners),
          binding('live', live),
          binding('records', records),
          binding('indices', props.output.indices),
          binding('adjacency', adjacency, 'read_write')
        ],
        declarations: triangleDeclarations.split('fn point')[0],
        body: `if (live[liveOffset + index] == 0u) { return; }
let recordBase = recordsOffset + 8u * owners[ownersOffset + index]; let firstRow = records[recordBase]; if (firstRow == 0xffffffffu) { return; } let secondRow = records[recordBase + 1u]; let base = adjacencyOffset + 3u * index;
if (index == firstRow) { adjacency[base] = secondRow; adjacency[base + 1u] = records[recordBase + 5u]; adjacency[base + 2u] = records[recordBase + 2u]; return; }
if (index == secondRow) { adjacency[base] = firstRow; adjacency[base + 1u] = records[recordBase + 3u]; adjacency[base + 2u] = records[recordBase + 4u]; return; }
let corners = triangle(index);
for (var side = 0u; side < 3u; side++) {
  let neighbor = adjacency[base + side]; if (neighbor != firstRow && neighbor != secondRow) { continue; }
  var replacement = 0xffffffffu;
  for (var changed = 0u; changed < 2u; changed++) { let row = select(firstRow, secondRow, changed == 1u); let other = triangle(row); for (var otherSide = 0u; otherSide < 3u; otherSide++) { if (other[otherSide] == corners[(side + 1u) % 3u] && other[(otherSide + 1u) % 3u] == corners[side]) { replacement = row; } } }
  adjacency[base + side] = replacement;
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${prefix}-gate`,
        operation,
        variant: 'parallel-flip-gate',
        invocationCount: 1,
        bindings: [binding('remaining', remaining, 'read_write', 'atomic<u32>'), gate.binding],
        body: `if (atomicLoad(&remaining[remainingOffset]) != 0u) { ${gate.armWGSL} } else { ${gate.disarmWGSL} } atomicStore(&remaining[remainingOffset], 0u);`
      })
    );
  }
  // Invalid polygons never enter evaluation, and must retain converged=0 rather than the empty case.
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-valid-status`,
      operation,
      invocationCount: polygonCount,
      bindings: [
        binding('valid', props.output.valid),
        binding('converged', props.output.status.converged, 'read_write')
      ],
      body: 'converged[convergedOffset + index] &= u32(valid[validOffset + index] != 0u);'
    })
  );
  return nodes;
}
