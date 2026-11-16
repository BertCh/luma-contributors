// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';
import {
  createWGSLKernelNode,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Check possible private publication allocations at the contributor boundary, before any nodes. @internal */
export function validateSpatialAnalysisPackedOutputAllocations<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  outputs: unknown
): void {
  if (!outputs || typeof outputs !== 'object') return;
  if ('buffer' in outputs && 'format' in outputs && 'byteOffset' in outputs) {
    const view = outputs as GraphDataView;
    validateCapacity(
      id,
      'packed output allocation byteLength (maxBufferSize)',
      getViewBindingRange(view).size,
      0,
      graph.device.limits.maxBufferSize
    );
  } else
    for (const value of Object.values(outputs))
      validateSpatialAnalysisPackedOutputAllocations(graph, id, value);
}

/**
 * Preserve packed spatial-analysis results when a dispatch reads another field of their buffer.
 * WebGPU forbids mixing read-only and writable storage uses of one physical buffer in a dispatch,
 * even for disjoint aligned ranges. Run conflicting writes in private storage and publish them in
 * separate dispatches. Keep the original binding prefix so the analysis shader is unchanged.
 *
 * Copying the initial contents also preserves kernels that read their read_write output. Copies
 * use storage kernels, so caller buffers do not need COPY_SRC/COPY_DST. The private binding has
 * exactly the caller's aligned size: the contributor's initial caller-binding preflight therefore
 * covers this nested scratch as well. This is domain adaptation around the shared kernel builder.
 * The published graph still requires one active imported handle per writable physical buffer.
 * @internal
 */
export function getSpatialAnalysisPackedKernelNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: WGSLKernelProps<Parameters>
): GPUCommandNode<Parameters>[] {
  const physical = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  const readers = new Set(
    props.bindings
      .filter(binding => binding.access === 'read')
      .map(binding => physical(binding.view))
  );
  const conflicts = props.bindings.filter(
    binding => binding.access === 'read_write' && readers.has(physical(binding.view))
  );
  if (conflicts.length === 0) return [createWGSLKernelNode(graph, props)];
  // Validate every affected range before declaring any of this adapter's resources or nodes.
  validateGraphChunkBindingRanges(
    graph,
    props.id,
    Object.fromEntries(props.bindings.map(binding => [binding.name, binding.view]))
  );
  for (const binding of conflicts)
    validateCapacity(
      props.id,
      `${binding.name} packed publication allocation byteLength (maxBufferSize)`,
      getViewBindingRange(binding.view).size,
      0,
      graph.device.limits.maxBufferSize
    );
  const before: GPUCommandNode<Parameters>[] = [];
  const after: GPUCommandNode<Parameters>[] = [];
  const replacements = new Map<GraphDataView, GraphDataView>();
  for (const binding of conflicts) {
    const original = binding.view;
    const range = getViewBindingRange(original);
    const prefix = original.byteOffset - range.offset;
    const buffer = graph.createTransientBuffer({
      id: `${props.id}-packed-${binding.name}`,
      byteLength: range.size,
      usage: Buffer.STORAGE
    });
    const scratch = graph.createDataView(buffer, {
      format: original.format,
      byteOffset: prefix,
      length: original.length,
      byteStride: original.byteStride,
      rowByteLength: original.rowByteLength
    });
    replacements.set(original, scratch);
    const wordCount =
      original.length === 0
        ? 0
        : ((original.length - 1) * original.byteStride + original.rowByteLength) / 4;
    const copy = (step: string, incoming: GraphDataView, outgoing: GraphDataView) =>
      createWGSLKernelNode(graph, {
        id: `${props.id}-packed-${binding.name}-${step}`,
        operation: props.operation,
        variant: 'packed-storage-publication',
        invocationCount: wordCount,
        bindings: [
          {name: 'incoming', view: incoming, type: 'u32', access: 'read'},
          {name: 'outgoing', view: outgoing, type: 'u32', access: 'read_write'}
        ],
        body: 'outgoing[outgoingOffset+index]=incoming[incomingOffset+index];',
        condition: props.condition,
        extraResources: props.extraResources
      });
    before.push(copy('initialize', original, scratch));
    after.push(copy('publish', scratch, original));
  }
  return [
    ...before,
    createWGSLKernelNode(graph, {
      ...props,
      bindings: props.bindings.map(binding => ({
        ...binding,
        view: replacements.get(binding.view) ?? binding.view
      }))
    }),
    ...after
  ];
}
