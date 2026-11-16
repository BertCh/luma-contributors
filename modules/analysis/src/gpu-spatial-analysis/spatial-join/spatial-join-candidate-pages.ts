// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createGraphChunkGroupKernelNode} from '../../analysis-core/kernels/graph-chunk-groups';
import type {
  GPUCommandGraph,
  GPUCommandGraphGPUIndirectCondition,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateCommandNodeBudget
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Shader element addresses and linear invocation indices are uint32. */
export const SPATIAL_JOIN_MAXIMUM_INDEX = 0xffffffff;

/** Allows fixed-size primitive scratch and classifiers per page, including their setup nodes. */
const SPATIAL_JOIN_PAGE_COMMAND_NODE_ALLOWANCE = 64;
/** Leaves headroom for bounds, a BVH, global scans, publication and optional query ordering. */
const SPATIAL_JOIN_FIXED_COMMAND_NODE_ALLOWANCE = 4096;

/** Global counts stay uint32; page-local word addresses never span the entire candidate list. */
export type SpatialJoinCandidatePage = {
  firstSlot: number;
  length: number;
  initialize: boolean;
  counts: GraphDataView<'uint32'>;
  offsets: GraphDataView<'uint32'>;
  metadataRow?: number;
  orders?: Map<number, GraphDataView<'uint32'>>;
};

/** Construction-time page envelope; only the final live dispatch may be shorter. @internal */
export function planSpatialJoinCandidatePages(
  device: Device,
  id: string,
  candidateCapacity: number,
  requestedCapacity: number | undefined,
  bytesPerCandidate: number,
  extraBytes: number = 0,
  chunkCount: number = 1,
  invocationsPerCandidate: number = 1
) {
  const deviceCapacity = Math.min(
    Math.floor(
      (Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) -
        extraBytes) /
        bytesPerCandidate
    ),
    Math.floor((SPATIAL_JOIN_MAXIMUM_INDEX - Math.ceil(extraBytes / 4)) / (bytesPerCandidate / 4)),
    Math.floor(SPATIAL_JOIN_MAXIMUM_INDEX / invocationsPerCandidate)
  );
  if (requestedCapacity !== undefined)
    validateCapacity(
      id,
      'candidateTileCapacity',
      requestedCapacity,
      1,
      Math.min(candidateCapacity, deviceCapacity)
    );
  const capacity = requestedCapacity ?? Math.min(candidateCapacity, deviceCapacity);
  validateCapacity(
    id,
    'candidateTileCapacity (device binding/buffer limit)',
    capacity,
    1,
    candidateCapacity
  );
  const pageCount = Math.ceil(candidateCapacity / capacity);
  const estimatedCommandNodeCount =
    SPATIAL_JOIN_FIXED_COMMAND_NODE_ALLOWANCE +
    pageCount * (SPATIAL_JOIN_PAGE_COMMAND_NODE_ALLOWANCE + chunkCount * 4);
  validateCommandNodeBudget(id, estimatedCommandNodeCount);
  return {capacity, pageCount, estimatedCommandNodeCount};
}

/** Query first row, query count and an indirect [x,y,z] triple share the prefix allocation. */
export const SPATIAL_JOIN_PAGE_QUERY_WORD_COUNT = 5;
/** Squaring the per-axis workgroup bound must fit uint32 in the dispatch packer. */
const SPATIAL_JOIN_PAGE_MAXIMUM_DISPATCH_DIMENSION = 65535;

/** Finds the contiguous source-query tile intersecting this candidate page, wholly on the GPU. */
export function createSpatialJoinCandidateQueryRange<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    offsets: GraphDataView<'uint32'>;
    total: GraphDataView<'uint32'>;
    firstSlot: number;
    length: number;
    candidateCapacity: number;
    queryFirst: number;
    queryLength: number;
    queryCount: number;
    metadataRow: number;
  }
) {
  const updateNode = createGraphChunkGroupKernelNode(graph, {
    id: `${props.id}-query-range`,
    operation: 'SpatialJoinCandidatePages',
    invocationCount: 1,
    parameterValues: [
      props.firstSlot,
      props.length,
      props.queryFirst,
      props.queryLength,
      props.queryCount,
      props.metadataRow,
      Math.min(
        graph.device.limits.maxComputeWorkgroupsPerDimension,
        SPATIAL_JOIN_PAGE_MAXIMUM_DISPATCH_DIMENSION
      ),
      props.candidateCapacity
    ],
    bindings: [
      {name: 'offsets', view: props.offsets, type: 'u32', access: 'read_write'},
      {name: 'total', view: props.total, type: 'u32', access: 'read'}
    ],
    body: `let pageFirst = getChunkParameter(36u);
  let retained = min(total[totalOffset], getChunkParameter(43u));
  let pageEnd = min(retained, pageFirst + getChunkParameter(37u));
  let chunkFirst = getChunkParameter(38u);
  let chunkEnd = chunkFirst + getChunkParameter(39u);
  let rows = getChunkParameter(40u);
  var lower = chunkFirst;
  var upper = chunkEnd;
  loop {
    if (lower >= upper) { break; }
    let middle = lower + (upper - lower) / 2u;
    var end = retained;
    if (middle + 1u < rows) { end = min(retained, offsets[offsetsOffset + middle + 1u]); }
    if (end <= pageFirst) { lower = middle + 1u; } else { upper = middle; }
  }
  let first = lower;
  upper = chunkEnd;
  loop {
    if (lower >= upper) { break; }
    let middle = lower + (upper - lower) / 2u;
    if (offsets[offsetsOffset + middle] < pageEnd) { lower = middle + 1u; } else { upper = middle; }
  }
  let count = select(0u, lower - first, pageFirst < pageEnd);
  let metadata = offsetsOffset + getChunkParameter(41u);
  offsets[metadata] = first;
  offsets[metadata + 1u] = count;
  let groups = count / 256u + select(0u, 1u, count % 256u != 0u);
  let maximum = getChunkParameter(42u);
  var x = min(groups, maximum);
  var y = 1u;
  var z = 1u;
  if (groups > maximum) { y = min((groups + maximum - 1u) / maximum, maximum); }
  if (groups > maximum * maximum) { z = (groups + maximum * maximum - 1u) / (maximum * maximum); }
  offsets[metadata + 2u] = x;
  offsets[metadata + 3u] = y;
  offsets[metadata + 4u] = z;`
  });
  const condition: GPUCommandGraphGPUIndirectCondition = {
    id: `${props.id}-query-range`,
    source: 'gpu',
    mode: 'indirect',
    buffer: props.offsets.buffer,
    byteOffset: props.offsets.byteOffset + (props.metadataRow + 2) * 4
  };
  return {updateNode, condition};
}
