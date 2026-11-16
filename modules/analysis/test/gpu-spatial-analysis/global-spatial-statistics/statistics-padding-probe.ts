// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createStatisticsKernelNode} from '../../../src/gpu-spatial-analysis/spatial-weights/statistics-kernel-node';

/** The published dispatch helper requires a power of two greater than one. */
const PADDING_PROBE_WORKGROUP_SIZE = 2;

/** Share the real headless probe's construction with the CPU-only regression. */
export function createStatisticsPaddingProbe(
  graph: GPUCommandGraph,
  total: GraphDataView<'uint32'>
) {
  const groupCount = graph.device.limits.maxComputeWorkgroupsPerDimension + 2;
  const node = createStatisticsKernelNode(graph, {
    id: 'cooperative-2d-padding',
    operation: 'test',
    invocationCount: groupCount * PADDING_PROBE_WORKGROUP_SIZE,
    workgroupSize: PADDING_PROBE_WORKGROUP_SIZE,
    guardIndex: false,
    bindings: [{name: 'total', view: total, type: 'atomic<u32>', access: 'read_write'}],
    declarations: 'var<workgroup> value: u32;',
    body: `if (localInvocationIndex == 0u) { value = 1u; }
workgroupBarrier();
if (localInvocationIndex == 0u) { atomicAdd(&total[totalOffset], value); }`
  });
  return {node, groupCount};
}
