// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSpatialAnalysisPackedOutputAllocations} from '../geographic-distribution/spatial-analysis-packed-kernel';
import type {Device} from '@luma.gl/core';
import {
  getViewBindingRange,
  GraphVectorView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/contracts/network-topology';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  type GraphChunkDispatchViews
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGraphViewChunks,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Preflight caller bindings before a recipe declares resources. Packed fields coexist; vector chunks dispatch separately. @internal */
export function validateUrbanRecipeBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  inputs: GraphChunkDispatchViews,
  outputs: unknown,
  coBoundGroups?: readonly (readonly string[])[]
): void {
  const writable: Record<string, GraphDataView | GraphVectorView> = {};
  const collect = (value: unknown, name: string): void => {
    if (!value || typeof value !== 'object') return;
    if (
      value instanceof GraphVectorView ||
      ('buffer' in value && 'format' in value && 'byteOffset' in value)
    ) {
      writable[name] = value as GraphDataView | GraphVectorView;
    } else for (const [field, child] of Object.entries(value)) collect(child, `${name}.${field}`);
  };
  collect(outputs, 'output');
  validateSpatialAnalysisPackedOutputAllocations(graph, id, writable);
  const readable = inputs;
  // Checks ownership, STORAGE usage, actual aligned ranges and buffer extents without allocation.
  validateGraphChunkBindingRanges(graph, id, {...readable, ...writable});
  if (coBoundGroups) {
    const physical = (view: GraphDataView) =>
      view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
    const outputs = Object.entries(writable).flatMap(([name, view]) =>
      getGraphViewChunks(view).map(chunk => ({name, view: chunk}))
    );
    for (const [outputIndex, {name, view}] of outputs.entries()) {
      if (view.length === 0) continue;
      const end = view.byteOffset + (view.length - 1) * view.byteStride + view.rowByteLength;
      const others = [
        ...outputs.slice(outputIndex + 1),
        ...Object.entries(inputs).flatMap(([name, input]) =>
          input ? getGraphViewChunks(input).map(view => ({name, view})) : []
        )
      ];
      for (const other of others) {
        if (other.view.length === 0 || physical(view) !== physical(other.view)) continue;
        const otherEnd =
          other.view.byteOffset +
          (other.view.length - 1) * other.view.byteStride +
          other.view.rowByteLength;
        validateRequirement(
          id,
          view.byteOffset >= otherEnd || other.view.byteOffset >= end,
          `${name} and ${other.name} logical byte ranges must not overlap`
        );
      }
    }
    for (const group of coBoundGroups) {
      const writes = Object.fromEntries(
        group.filter(name => writable[name]).map(name => [name, writable[name]])
      );
      validateDisjointWritableViews(id, writes);
      for (const [name, output] of Object.entries(writes))
        for (const readName of group) {
          if (!inputs[readName]) continue;
          validateDisjointWritableViews(id, {[name]: output, [readName]: inputs[readName]});
        }
    }
    return;
  }
  // Aligned ranges within a dispatch; logical ranges across chunks, including shared core buffers.
  validateDisjointWritableViews(id, writable);
  const physicalBuffer = (view: GraphDataView) =>
    view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
  for (const output of Object.values(writable).flatMap(view => getGraphViewChunks(view))) {
    const outputRange = getViewBindingRange(output);
    for (const other of Object.values(inputs).flatMap(view =>
      view ? getGraphViewChunks(view) : []
    )) {
      if (physicalBuffer(output) !== physicalBuffer(other)) continue;
      const otherRange = getViewBindingRange(other);
      validateRequirement(
        id,
        outputRange.offset >= otherRange.offset + otherRange.size ||
          otherRange.offset >= outputRange.offset + outputRange.size,
        'output aligned binding ranges must not overlap inputs'
      );
    }
  }
}

/** Urban scratch is both one storage binding and one physical allocation. @internal */
export function validateUrbanStorageByteLength(
  device: Device,
  id: string,
  name: string,
  byteLength: number,
  hint: string
): void {
  validateGPUStorageBindingByteLength(device, id, name, byteLength, hint);
  validateCapacity(
    id,
    `${name} allocation byteLength (maxBufferSize)`,
    byteLength,
    0,
    device.limits.maxBufferSize
  );
}
