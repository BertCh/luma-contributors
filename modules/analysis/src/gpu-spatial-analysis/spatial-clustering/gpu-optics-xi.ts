// luma.gl
// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright (c) 2007-2024 The scikit-learn developers
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// WGSL adaptation of scikit-learn 1.7.2 _xi_cluster and predecessor correction.
// Original BSD notice and disclaimer are reproduced in optics-xi-license.txt.

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPU_OPTICS_XI_BATCH_SIZE, type GPUOPTICSProps} from './gpu-optics';

/** Stateful reference xi extraction with parallel initialization and cleanup. @internal */
export function createOPTICSXiNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUOPTICSProps,
  id: string
) {
  const rows = props.positions.length;
  const output = props.output;
  // The last row block stores inverse ordering without another storage binding.
  const areas = createTransientView(graph, `${id}-xi-areas`, 'uint32', rows * 4 + 6);
  const intervals =
    output.clusterIntervals ?? createTransientView(graph, `${id}-xi-intervals`, 'uint32x2', rows);
  const summary =
    output.clusterSummary ?? createTransientView(graph, `${id}-xi-summary`, 'uint32', 2);
  // Retain [areaCount, next, maximumBetween bits, label, visited, batchStart].
  // Reading batchStart from storage lets all batches share shader source and pipelines.
  // Per-area scans remain reference-tier, bounded by GPU_OPTICS_GRID_MAXIMUM_ROWS.
  const gate = createWGSLDispatchGate(graph, {
    id: `${id}-xi-dispatch`,
    operation: 'GPUOPTICS',
    shapes: [[1]]
  });
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-xi-initialize`,
      operation: 'GPUOPTICS',
      variant: 'xi-initialize',
      invocationCount: rows,
      bindings: [
        {name: 'ordering', view: output.ordering, type: 'u32', access: 'read'},
        {name: 'status', view: output.status, type: 'u32', access: 'read'},
        {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
        {name: 'areas', view: areas, type: 'u32', access: 'read_write'},
        {name: 'intervals', view: intervals, type: 'u32', access: 'read_write'},
        {name: 'summary', view: summary, type: 'u32', access: 'read_write'},
        gate.binding
      ],
      body: `labels[labelsOffset + index] = 0xffffffffu;
  if (index < status[statusOffset]) {
    areas[areasOffset + ${rows * 3}u + ordering[orderingOffset + index]] = index;
  }
  intervals[intervalsOffset + 2u * index] = 0xffffffffu;
  intervals[intervalsOffset + 2u * index + 1u] = 0xffffffffu;
  if (index == 0u) {
    for (var word = 0u; word < 6u; word++) { areas[areasOffset + ${rows * 4}u + word] = 0u; }
    areas[areasOffset + ${rows * 4 + 4}u] = status[statusOffset];
    summary[summaryOffset] = 0u; summary[summaryOffset + 1u] = 0u;
    ${gate.armWGSL}
    if (status[statusOffset + 1u] == 0u || status[statusOffset] == 0u) { ${gate.disarmWGSL} }
  }`
    })
  ];
  for (let batch = 0; batch < Math.ceil(rows / GPU_OPTICS_XI_BATCH_SIZE); batch++) {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-xi-${batch}`,
        operation: 'GPUOPTICS',
        variant: 'xi-extraction',
        invocationCount: 1,
        ...gate.getGate(0),
        bindings: [
          {name: 'ordering', view: output.ordering, type: 'u32', access: 'read'},
          {name: 'reachability', view: output.reachability, type: 'f32', access: 'read'},
          {name: 'predecessors', view: output.predecessors, type: 'u32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
          {name: 'areas', view: areas, type: 'u32', access: 'read_write'},
          {name: 'intervals', view: intervals, type: 'u32', access: 'read_write'},
          {name: 'summary', view: summary, type: 'u32', access: 'read_write'}
        ],
        declarations: `const VISITED_OFFSET = ${rows * 4 + 4}u;
const BATCH_START_OFFSET = ${rows * 4 + 5}u;
const INVERSE_ORDER_OFFSET = ${rows * 3}u;
const XI_COMPLEMENT = ${Math.fround(1 - (props.xi ?? 0.05))}f;
fn reach(rank: u32) -> f32 {
  if (rank >= areas[areasOffset + VISITED_OFFSET]) { return bitcast<f32>(0x7f800000u | (rank & 0u)); }
  return reachability[reachabilityOffset + ordering[orderingOffset + rank]];
}
fn direction(rank: u32) -> vec4<bool> {
  let first = reach(rank); let second = reach(rank + 1u);
  // Both infinities and both zeros have no direction, matching undefined reference ratios.
  if (first == second) { return vec4<bool>(false); }
  return vec4<bool>(first <= XI_COMPLEMENT * second, first * XI_COMPLEMENT >= second, (first < second), (first > second));
}
fn extend(start: u32, downward: bool) -> u32 {
  var end = start; var nonSteep = 0u;
  for (var rank = start; rank < areas[areasOffset + VISITED_OFFSET]; rank++) {
    let flags = direction(rank);
    if (select(flags.x, flags.y, downward)) { nonSteep = 0u; end = rank; }
    else if (!select(flags.w, flags.z, downward)) {
      nonSteep++; if (nonSteep > ${props.minimumSamples}u) { break; }
    } else { break; }
  }
  return end;
}`,
        body: `let batchStart = areas[areasOffset + BATCH_START_OFFSET];
var areaCount = areas[areasOffset + ${rows * 4}u];
var next = areas[areasOffset + ${rows * 4 + 1}u];
var maximumBetween = bitcast<f32>(areas[areasOffset + ${rows * 4 + 2}u]);
var clusterCount = summary[summaryOffset];
var label = areas[areasOffset + ${rows * 4 + 3}u];
for (var steep = batchStart; steep < min(batchStart + ${GPU_OPTICS_XI_BATCH_SIZE}u, areas[areasOffset + VISITED_OFFSET]); steep++) {
  if (steep < next) { continue; }
  let flags = direction(steep);
  if (!flags.x && !flags.y) { continue; }
  for (var rank = next; rank <= steep; rank++) { maximumBetween = max(maximumBetween, reach(rank)); }
  var retained = 0u;
  for (var area = 0u; area < areaCount; area++) {
    let offset = areasOffset + 3u * area;
    let start = areas[offset];
    if (bitcast<u32>(maximumBetween) != 0x7f800000u && maximumBetween <= reach(start) * XI_COMPLEMENT) {
      let destination = areasOffset + 3u * retained;
      areas[destination] = start; areas[destination + 1u] = areas[offset + 1u];
      areas[destination + 2u] = bitcast<u32>(max(bitcast<f32>(areas[offset + 2u]), maximumBetween)); retained++;
    }
  }
  areaCount = retained;
  let end = extend(steep, flags.y); next = end + 1u; maximumBetween = reach(next);
  if (flags.y) {
    let destination = areasOffset + 3u * areaCount;
    areas[destination] = steep; areas[destination + 1u] = end; areas[destination + 2u] = 0u; areaCount++;
    continue;
  }
  // Reverse the matching downward areas so inner intervals precede containing intervals.
  for (var reverse = areaCount; reverse > 0u; reverse--) {
    let offset = areasOffset + 3u * (reverse - 1u);
    var start = areas[offset]; var finish = end;
    let downwardEnd = areas[offset + 1u];
    if (reach(finish + 1u) * XI_COMPLEMENT < bitcast<f32>(areas[offset + 2u])) { continue; }
    let downwardMaximum = reach(start);
    if (downwardMaximum * XI_COMPLEMENT >= reach(finish + 1u)) {
      while (start < downwardEnd && reach(start + 1u) > reach(finish + 1u)) { start++; }
    } else if (reach(finish + 1u) * XI_COMPLEMENT >= downwardMaximum) {
      while (finish > steep && reach(finish - 1u) > downwardMaximum) { finish--; }
    }
    ${
      props.predecessorCorrection === false
        ? ''
        : `while (start < finish) {
      if (reach(start) > reach(finish)) { break; }
      let predecessor = predecessors[predecessorsOffset + ordering[orderingOffset + finish]];
      if (predecessor != 0xffffffffu) {
        let predecessorRank = areas[areasOffset + INVERSE_ORDER_OFFSET + predecessor];
        if (start <= predecessorRank && predecessorRank < finish) { break; }
      }
      finish--;
    }
    if (start >= finish) { continue; }`
    }
    if (finish - start + 1u < ${props.minimumClusterSize ?? props.minimumSamples}u || start > downwardEnd || finish < steep) { continue; }
    if (clusterCount >= ${rows}u) { summary[summaryOffset + 1u] = 1u; continue; }
    intervals[intervalsOffset + 2u * clusterCount] = start;
    intervals[intervalsOffset + 2u * clusterCount + 1u] = finish; clusterCount++;
    var occupied = false;
    for (var rank = start; rank <= finish; rank++) { occupied = occupied || labels[labelsOffset + ordering[orderingOffset + rank]] != 0xffffffffu; }
    if (!occupied) {
      for (var rank = start; rank <= finish; rank++) { labels[labelsOffset + ordering[orderingOffset + rank]] = label; }
      label++;
    }
  }
}
summary[summaryOffset] = clusterCount;
areas[areasOffset + ${rows * 4}u] = areaCount;
areas[areasOffset + ${rows * 4 + 1}u] = next;
areas[areasOffset + ${rows * 4 + 2}u] = bitcast<u32>(maximumBetween);
areas[areasOffset + ${rows * 4 + 3}u] = label;
areas[areasOffset + BATCH_START_OFFSET] = batchStart + ${GPU_OPTICS_XI_BATCH_SIZE}u;
`
      }),
      // The batch cannot storage-write its own indirect arguments. Update them in an
      // ungated scalar dispatch, after the batch's persistent state has been published.
      createWGSLKernelNode(graph, {
        id: `${id}-xi-gate-${batch}`,
        operation: 'GPUOPTICS',
        variant: 'xi-gate',
        invocationCount: 1,
        bindings: [{name: 'areas', view: areas, type: 'u32', access: 'read'}, gate.binding],
        body: `if (areas[areasOffset + ${rows * 4 + 1}u] >= areas[areasOffset + ${rows * 4 + 4}u] ||
      areas[areasOffset + ${rows * 4 + 5}u] >= areas[areasOffset + ${rows * 4 + 4}u]) {
    ${gate.disarmWGSL}
  }`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-xi-finalize`,
      operation: 'GPUOPTICS',
      variant: 'xi-finalize',
      invocationCount: rows,
      bindings: [
        {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
        {name: 'summary', view: summary, type: 'u32', access: 'read'}
      ],
      body: 'if (summary[summaryOffset + 1u] != 0u) { labels[labelsOffset + index] = 0xffffffffu; }'
    })
  );
  return nodes;
}
