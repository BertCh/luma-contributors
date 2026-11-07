// luma.gl
// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright (c) 2007-2024 The scikit-learn developers
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// WGSL adaptation of scikit-learn 1.7.2 _xi_cluster and predecessor correction.
// Original BSD notice and disclaimer are reproduced in optics-xi-license.txt.

import {createTransientView, type GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUOPTICSProps} from './gpu-optics';

export function createOPTICSXiNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUOPTICSProps,
  id: string
) {
  const rows = props.positions.length;
  const output = props.output;
  // The last row block stores inverse ordering without another storage binding.
  const areas = createTransientView(graph, `${id}-xi-areas`, 'uint32', rows * 4);
  const intervals =
    output.clusterIntervals ?? createTransientView(graph, `${id}-xi-intervals`, 'uint32x2', rows);
  const summary =
    output.clusterSummary ?? createTransientView(graph, `${id}-xi-summary`, 'uint32', 2);
  return createWGSLKernelNode(graph, {
    id: `${id}-xi`,
    operation: 'GPUOPTICS',
    variant: 'xi-extraction',
    invocationCount: 1,
    bindings: [
      {name: 'ordering', view: output.ordering, type: 'u32', access: 'read'},
      {name: 'reachability', view: output.reachability, type: 'f32', access: 'read'},
      {name: 'predecessors', view: output.predecessors, type: 'u32', access: 'read'},
      {name: 'status', view: output.status, type: 'u32', access: 'read'},
      {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
      {name: 'areas', view: areas, type: 'u32', access: 'read_write'},
      {name: 'intervals', view: intervals, type: 'u32', access: 'read_write'},
      {name: 'summary', view: summary, type: 'u32', access: 'read_write'}
    ],
    declarations: `const INVERSE_ORDER_OFFSET = ${rows * 3}u;
const XI_COMPLEMENT = ${Math.fround(1 - (props.xi ?? 0.05))}f;
fn reach(rank: u32) -> f32 {
  if (rank >= status[statusOffset]) { return bitcast<f32>(0x7f800000u | (rank & 0u)); }
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
  for (var rank = start; rank < status[statusOffset]; rank++) {
    let flags = direction(rank);
    if (select(flags.x, flags.y, downward)) { nonSteep = 0u; end = rank; }
    else if (!select(flags.w, flags.z, downward)) {
      nonSteep++; if (nonSteep > ${props.minimumSamples}u) { break; }
    } else { break; }
  }
  return end;
}`,
    body: `for (var row = 0u; row < ${rows}u; row++) {
  labels[labelsOffset + row] = 0xffffffffu;
  if (row < status[statusOffset]) {
    areas[areasOffset + INVERSE_ORDER_OFFSET + ordering[orderingOffset + row]] = row;
  }
  intervals[intervalsOffset + 2u * row] = 0xffffffffu;
  intervals[intervalsOffset + 2u * row + 1u] = 0xffffffffu;
}
summary[summaryOffset] = 0u; summary[summaryOffset + 1u] = 0u;
if (status[statusOffset + 1u] == 0u) { return; }
var areaCount = 0u; var next = 0u; var maximumBetween = 0.0; var clusterCount = 0u; var label = 0u;
for (var steep = 0u; steep < status[statusOffset]; steep++) {
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
if (summary[summaryOffset + 1u] != 0u) {
  for (var row = 0u; row < ${rows}u; row++) { labels[labelsOffset + row] = 0xffffffffu; }
}`
  });
}
