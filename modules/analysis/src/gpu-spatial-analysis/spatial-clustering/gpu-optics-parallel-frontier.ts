// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUOPTICSProps} from './gpu-optics';

/**
 * Exact indexed frontier distributed over workgroups. Nonnegative f32 distance bits sort as
 * uint32, including infinity (roots). Visited/invalid rows use UINT_MAX, beyond infinity.
 * A second published reduction chooses the lowest input row among equal minimum keys.
 * No float mantissa bits are discarded and predecessor updates remain strictly improving.
 * Dispatch boundaries serialize visits; no cross-workgroup spin barriers are used.
 * @internal
 */
export function createOPTICSParallelFrontierNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUOPTICSProps,
  views: {
    distanceSource: string;
    points: GraphDataView<'float32'>;
    offsets: GraphDataView<'uint32'>;
    neighbors: GraphDataView<'uint32'>;
    overflow: GraphDataView<'uint32'>;
    processed: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const id = `${props.id ?? 'optics'}-parallel`;
  const rows = props.positions.length;
  const {output} = props;
  const {distanceSource, points, offsets, neighbors, overflow, processed} = views;
  const keys = createTransientView(graph, `${id}-keys`, 'uint32', rows);
  const ties = createTransientView(graph, `${id}-ties`, 'uint32', rows);
  const minimum = createTransientView(graph, `${id}-minimum`, 'uint32', 1);
  const focus = createTransientView(graph, `${id}-focus`, 'uint32', 1);
  // Reuse reduction scratch across every visit; the graph's storage hazards order each reuse.
  const reduceKeys = new GPUReduction({
    id: `${id}-keys`,
    input: keys,
    output: minimum,
    operation: 'min'
  }).getCommandNodes(graph);
  const reduceTies = new GPUReduction({
    id: `${id}-ties`,
    input: ties,
    output: focus,
    operation: 'min'
  }).getCommandNodes(graph);
  // Public GPUReduction contract: one 256-lane level per ceil(length / 256),
  // followed by a one-workgroup finalize. Integer reductions need no validity pass.
  const reductionWorkgroups: number[] = [];
  let remaining = rows;
  do {
    remaining = Math.ceil(remaining / 256);
    reductionWorkgroups.push(remaining);
  } while (remaining > 1);
  reductionWorkgroups.push(1);
  const workgroupCounts = [...new Set([1, Math.ceil(rows / 256), ...reductionWorkgroups])];
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-dispatch`,
    operation: 'GPUOPTICS',
    shapes: workgroupCounts.map(count => [count * 256, 256] as const)
  });
  const getGate = (workgroups: number) => gate.getGate(workgroupCounts.indexOf(workgroups));
  const rowGate = getGate(Math.ceil(rows / 256));
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-initialize`,
      operation: 'GPUOPTICS',
      variant: 'parallel-initialize',
      invocationCount: rows,
      bindings: [
        {name: 'processed', view: processed, type: 'u32', access: 'read'},
        {name: 'overflow', view: overflow, type: 'u32', access: 'read'},
        {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
        gate.binding
      ],
      body: `keys[keysOffset + index] = select(0x7f800000u, 0xffffffffu, processed[processedOffset + index] != 0u);
  if (index == 0u) {
    status[statusOffset + 1u] = u32(status[statusOffset + 2u] == ${rows}u && overflow[overflowOffset] == 0u);
    ${gate.armWGSL}
    if (overflow[overflowOffset] != 0u || status[statusOffset + 1u] != 0u) { ${gate.disarmWGSL} }
  }`
    })
  ];
  const markTies = createWGSLKernelNode(graph, {
    id: `${id}-mark-ties`,
    operation: 'GPUOPTICS',
    variant: 'parallel-ties',
    invocationCount: rows,
    workgroupSize: 256,
    ...rowGate,
    bindings: [
      {name: 'keys', view: keys, type: 'u32', access: 'read'},
      {name: 'minimum', view: minimum, type: 'u32', access: 'read'},
      {name: 'ties', view: ties, type: 'u32', access: 'read_write'}
    ],
    body: `let key = keys[keysOffset + index];
  ties[tiesOffset + index] = select(0xffffffffu, index, key != 0xffffffffu && key == minimum[minimumOffset]);`
  });
  const commit = createWGSLKernelNode(graph, {
    id: `${id}-commit`,
    operation: 'GPUOPTICS',
    variant: 'parallel-commit',
    invocationCount: 1,
    // The gate owner must be ungated: writable storage and indirect arguments cannot
    // alias in one WebGPU dispatch. Finished rounds run only this scalar gate check.
    bindings: [
      {name: 'focus', view: focus, type: 'u32', access: 'read'},
      {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
      {name: 'processed', view: processed, type: 'u32', access: 'read_write'},
      {name: 'ordering', view: output.ordering, type: 'u32', access: 'read_write'},
      {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
      gate.binding
    ],
    body: `if (dispatch[dispatchOffset] == 0u) { return; }
  let row = focus[focusOffset];
  if (row == 0xffffffffu) { ${gate.disarmWGSL} return; }
  let visited = status[statusOffset];
  ordering[orderingOffset + visited] = row;
  keys[keysOffset + row] = 0xffffffffu;
  processed[processedOffset + row] = 1u;
  status[statusOffset] = visited + 1u;
  if (visited + 1u + status[statusOffset + 2u] == ${rows}u) {
    status[statusOffset + 1u] = 1u;
    ${gate.disarmWGSL}
  }`
  });
  const relax = createWGSLKernelNode(graph, {
    id: `${id}-relax`,
    operation: 'GPUOPTICS',
    variant: 'parallel-relax',
    invocationCount: rows,
    workgroupSize: 256,
    ...rowGate,
    bindings: [
      {name: 'points', view: points, type: 'f32', access: 'read'},
      {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
      {name: 'focus', view: focus, type: 'u32', access: 'read'},
      {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
      {name: 'processed', view: processed, type: 'u32', access: 'read'},
      {name: 'reachability', view: output.reachability, type: 'f32', access: 'read_write'},
      {name: 'predecessors', view: output.predecessors, type: 'u32', access: 'read_write'}
    ],
    declarations: `${distanceSource}
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(points[pointsOffset + 4u * row], points[pointsOffset + 4u * row + 1u]);
}`,
    body: `let selected = focus[focusOffset];
  let begin = offsets[offsetsOffset + selected];
  let end = offsets[offsetsOffset + selected + 1u];
  if (index >= end - begin) { return; }
  let row = neighbors[neighborsOffset + begin + index];
  if (processed[processedOffset + row] != 0u) { return; }
  let coreDistance = points[pointsOffset + 4u * selected + 3u];
  if (bitcast<u32>(coreDistance) == 0x7f800000u) { return; }
  ${props.times ? `if (abs(points[pointsOffset + 4u * selected + 2u] - points[pointsOffset + 4u * row + 2u]) > ${Math.fround(props.maximumTimeDistance!)}f) { return; }` : ''}
  let pairDistance = getPointDistance(point(selected), point(row));
  if (pairDistance > ${Math.fround(props.maximumDistance)}f) { return; }
  let candidate = max(coreDistance, pairDistance);
  if (candidate < reachability[reachabilityOffset + row]) {
    reachability[reachabilityOffset + row] = candidate;
    predecessors[predecessorsOffset + row] = selected;
    keys[keysOffset + row] = bitcast<u32>(candidate);
  }`
  });
  const gateReduction = (node: GPUCommandNode<Parameters>, level: number, iteration: number) => {
    if (node.type !== 'compute') {
      throw new Error(`${id} GPUReduction nodes must be compute nodes`);
    }
    const reductionGate = getGate(reductionWorkgroups[level]);
    return {
      ...node,
      id: `${node.id}-${iteration}`,
      condition: reductionGate.condition,
      resources: [...(node.resources ?? []), ...reductionGate.extraResources]
    };
  };
  for (let iteration = 0; iteration < (props.maximumIterations ?? rows); iteration++) {
    nodes.push(
      ...reduceKeys.map((node, level) => gateReduction(node, level, iteration)),
      {...markTies, id: `${markTies.id}-${iteration}`},
      ...reduceTies.map((node, level) => gateReduction(node, level, iteration)),
      {...commit, id: `${commit.id}-${iteration}`},
      {...relax, id: `${relax.id}-${iteration}`}
    );
  }
  return nodes;
}
