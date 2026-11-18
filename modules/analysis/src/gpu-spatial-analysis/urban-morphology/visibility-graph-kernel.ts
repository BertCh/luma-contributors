// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  createWGSLKernelNode,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/**
 * Co-bound visibility output fields use writable storage consistently for each physical buffer.
 * WebGPU rejects mixing read-only and writable storage even at disjoint offsets. Construction
 * preflight rejects co-bound caller inputs and checks all promoted output ranges before allocation.
 * @internal
 */
export function createVisibilityGraphKernelNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>
) {
  const physicalBuffer = (binding: WGSLKernelProps<Parameters>['bindings'][number]) =>
    binding.view.buffer.defaultBuffer
      ? getCoreDefaultBuffer(binding.view.buffer.defaultBuffer)
      : binding.view.buffer;
  const writableBuffers = new Set(
    props.bindings.filter(binding => binding.access === 'read_write').map(physicalBuffer)
  );
  const bindings = props.bindings.map(binding =>
    writableBuffers.has(physicalBuffer(binding))
      ? {...binding, access: 'read_write' as const}
      : binding
  );
  validateDisjointWritableViews(
    props.operation,
    Object.fromEntries(
      bindings
        .filter(binding => binding.access === 'read_write')
        .map(binding => [binding.name, binding.view])
    ),
    {strictAlignedBindings: true}
  );
  return createWGSLKernelNode(graph, {...props, bindings});
}
