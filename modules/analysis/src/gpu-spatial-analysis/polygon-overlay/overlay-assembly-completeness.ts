// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT} from '../../analysis-core/ring-assembly/gpu-segment-ring-assembly';

/** Adds ring-assembly loss to the overlay's initialized candidate flag. @internal */
export function createOverlayAssemblyCompletenessNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    segmentFlags: GraphDataView<'uint32'>;
    segmentCount: GraphDataView<'uint32'>;
    openSegmentCount: GraphDataView<'uint32'>;
    candidateOverflow: GraphDataView<'uint32'>;
  }
) {
  return createWGSLKernelNode(graph, {
    id: `${props.id}-assembly-completeness`,
    operation: 'GPUPolygonOverlay',
    variant: 'assembly-completeness',
    invocationCount: Math.max(1, props.segmentFlags.length),
    bindings: [
      {name: 'flags', view: props.segmentFlags, type: 'u32', access: 'read'},
      {name: 'count', view: props.segmentCount, type: 'u32', access: 'read'},
      {name: 'openCount', view: props.openSegmentCount, type: 'u32', access: 'read'},
      {
        name: 'candidateOverflow',
        view: props.candidateOverflow,
        type: 'atomic<u32>',
        access: 'read_write'
      }
    ],
    body: `if (index == 0u && openCount[openCountOffset] != 0u) {
    atomicMax(&candidateOverflow[candidateOverflowOffset], 1u);
  }
  if (index < count[countOffset] && (flags[flagsOffset + index] & ${GPU_SEGMENT_RING_ASSEMBLY_FLAG_CONFLICT}u) != 0u) {
    atomicMax(&candidateOverflow[candidateOverflowOffset], 1u);
  }`
  });
}
