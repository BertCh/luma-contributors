// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getViewBindingRange,
  type GPUCommandGraph,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {getCoreDefaultBuffer} from '../../analysis-core/network-topology';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGraphViewChunks,
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

type SpatialJoinBindingView = GraphDataView | GraphVectorView | undefined;

/** Join kernels address flat 32-bit words, including literal copy counts, with WGSL uint32. */
const SPATIAL_JOIN_MAXIMUM_STORAGE_WORD_COUNT = 0xffffffff;

/** Physical identity includes separately imported handles of the same core buffer. */
function getSpatialJoinPhysicalBuffer(view: GraphDataView): object {
  return view.buffer.defaultBuffer ? getCoreDefaultBuffer(view.buffer.defaultBuffer) : view.buffer;
}

/** Checks caller bindings and every allocation envelope before any graph/device declaration. */
export function validateSpatialJoinBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  views: Readonly<Record<string, SpatialJoinBindingView>>,
  scratchByteLengths: Readonly<Record<string, number>>
): void {
  validateGraphChunkBindingRanges(graph, id, views);
  const handles = new Map<object, GraphDataView['buffer']>();
  for (const [name, column] of Object.entries(views)) {
    for (const view of column ? getGraphViewChunks(column) : []) {
      validateCapacity(
        id,
        `${name} aligned storage words (SPATIAL_JOIN_MAXIMUM_STORAGE_WORD_COUNT)`,
        getViewBindingRange(view).size / 4,
        1,
        SPATIAL_JOIN_MAXIMUM_STORAGE_WORD_COUNT
      );
      const physical = getSpatialJoinPhysicalBuffer(view);
      const previous = handles.get(physical);
      validateRequirement(
        id,
        !previous || previous === view.buffer,
        `${name} physical buffer must be imported once; create subviews of its existing graph handle`
      );
      handles.set(physical, view.buffer);
      validateCapacity(
        id,
        `${name}.buffer.byteLength (device.limits.maxBufferSize)`,
        view.buffer.byteLength,
        0,
        graph.device.limits.maxBufferSize
      );
    }
  }
  for (const [name, bytes] of Object.entries(scratchByteLengths)) {
    const byteLength = Math.max(bytes, 4);
    validateCapacity(
      id,
      `${name} storage words (SPATIAL_JOIN_MAXIMUM_STORAGE_WORD_COUNT)`,
      byteLength / 4,
      1,
      SPATIAL_JOIN_MAXIMUM_STORAGE_WORD_COUNT
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce row counts or capacities'
    );
    validateCapacity(
      id,
      `${name} byteLength (device.limits.maxBufferSize)`,
      byteLength,
      0,
      graph.device.limits.maxBufferSize
    );
  }
}

/** WebGPU forbids mixed read-only/writable storage bindings of one physical buffer in a pass. */
export function validateSpatialJoinReadWriteBindings(
  id: string,
  reads: readonly SpatialJoinBindingView[],
  writes: readonly SpatialJoinBindingView[]
): void {
  const readBuffers = new Set(
    reads.flatMap(view =>
      view
        ? getGraphViewChunks(view)
            .filter(chunk => chunk.length > 0)
            .map(getSpatialJoinPhysicalBuffer)
        : []
    )
  );
  for (const column of writes) {
    for (const view of column ? getGraphViewChunks(column) : []) {
      if (view.length === 0) continue;
      validateRequirement(
        id,
        !readBuffers.has(getSpatialJoinPhysicalBuffer(view)),
        'one physical buffer must not be bound read-only and writable in the same dispatch'
      );
    }
  }
}
