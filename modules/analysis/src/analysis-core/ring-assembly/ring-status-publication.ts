// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../kernels/wgsl-kernel-nodes';
import type {GPUSegmentRingAssemblyOutput} from './segment-ring-assembly-types';

/** Scalar publication, independent of tracing and polygon regrouping. @internal */
export function getRingStatusNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    output: GPUSegmentRingAssemblyOutput;
    ringScan: GraphDataView<'uint32'>;
    ringCount: GraphDataView<'uint32'>;
    counters: GraphDataView<'uint32'>;
    segments: number;
  }
): GPUCommandNode<Parameters>[] {
  const {id, output, ringScan, ringCount, counters, segments} = props;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const read = (name: string, view: GraphDataView): WGSLKernelBinding => ({
    name,
    view,
    type: 'u32',
    access: 'read'
  });
  const write = (name: string, view: GraphDataView): WGSLKernelBinding => ({
    name,
    view,
    type: 'u32',
    access: 'read_write'
  });
  const kernel = (
    name: string,
    bindings: WGSLKernelBinding[],
    invocationCount: number,
    body: string
  ) =>
    createWGSLKernelNode(graph, {
      id: `${id}-${name}`,
      operation: 'GPUSegmentRingAssembly',
      variant: name,
      bindings,
      invocationCount,
      body
    });
  // 9. Publish scalars.
  nodes.push(
    kernel(
      'publish',
      [
        read('ringScan', ringScan),
        read('ringCount', ringCount),
        read('counters', counters),
        write('outCount', output.status.count),
        write('outOverflow', output.status.overflow),
        ...(output.status.requiredCount ? [write('outTotal', output.status.requiredCount)] : [])
      ],
      1,
      `let total = ringScan[ringScanOffset + ${segments - 1}u];
  let written = ringCount[ringCountOffset];
  outCount[outCountOffset] = written;
  outOverflow[outOverflowOffset] = select(0u, 1u, written < total || counters[countersOffset + 3u] != 0u);
  ${output.status.requiredCount ? 'outTotal[outTotalOffset] = total;' : ''}`
    )
  );
  if (output.status.openSegmentCount || output.status.touchingSegmentCount) {
    nodes.push(
      kernel(
        'publish-counters',
        [
          read('counters', counters),
          ...(output.status.openSegmentCount
            ? [write('outOpen', output.status.openSegmentCount)]
            : []),
          ...(output.status.touchingSegmentCount
            ? [write('outTouching', output.status.touchingSegmentCount)]
            : [])
        ],
        1,
        `${output.status.openSegmentCount ? 'outOpen[outOpenOffset] = counters[countersOffset] - counters[countersOffset + 2u];' : ''}
  ${output.status.touchingSegmentCount ? 'outTouching[outTouchingOffset] = counters[countersOffset + 1u];' : ''}`
      )
    );
  }
  return nodes;
}
