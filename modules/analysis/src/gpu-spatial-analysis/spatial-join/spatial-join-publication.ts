// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUFlagOffsets,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUCompactOutput,
  GPUUint32Rows
} from '../../analysis-core/kernels/gpu-contributor-types';
import type {WGSLKernelBinding} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getGraphViewChunks} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createSpatialJoinChunkNode,
  getChunkNodeId,
  SPATIAL_JOIN_WGSL_HELPERS
} from './spatial-join-passes';

/**
 * Writes per-point feature IDs (and optional distances) and accumulates per-feature counts.
 *
 * @internal
 */
export function getSpatialJoinAssignNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    assignment: GraphDataView<'uint32'>;
    pointFeatureIds?: GPUUint32Rows;
    pointFeatureRows?: GPUUint32Rows;
    featureIds?: GraphDataView<'uint32'>;
    featureCounts?: GraphDataView<'uint32'>;
    polygonAssignmentCounts?: GraphDataView<'uint32'>;
    bestDistanceBits?: GraphDataView<'uint32'>;
    /** Apply sqrt only when publishing a squared-distance reduction. */
    squaredDistances?: boolean;
    /** Decode the robust nearest squared-distance rank when publishing. */
    distanceRankWGSL?: string;
    distances?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const distanceChunks = props.distances ? getGraphViewChunks(props.distances) : [];
  let chunkFirstRow = 0;
  const primaryOutput = props.pointFeatureIds ?? props.pointFeatureRows!;
  const rowChunks = props.pointFeatureRows ? getGraphViewChunks(props.pointFeatureRows) : [];
  for (const [chunkIndex, chunk] of getGraphViewChunks(primaryOutput).entries()) {
    if (chunk.length > 0) {
      const bindings: WGSLKernelBinding[] = [
        {name: 'assignment', view: props.assignment, type: 'u32', access: 'read'},
        {name: 'pointFeatureIds', view: chunk, type: 'u32', access: 'read_write'}
      ];
      const rowChunk = props.pointFeatureIds ? rowChunks[chunkIndex] : undefined;
      if (rowChunk)
        bindings.push({
          name: 'pointFeatureRows',
          view: rowChunk,
          type: 'u32',
          access: 'read_write'
        });
      if (props.featureIds && props.pointFeatureIds) {
        bindings.push({name: 'featureIds', view: props.featureIds, type: 'u32', access: 'read'});
      }
      if (props.featureCounts) {
        bindings.push({
          name: 'featureCounts',
          view: props.featureCounts,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      if (props.polygonAssignmentCounts)
        bindings.push({
          name: 'assignmentCounts',
          view: props.polygonAssignmentCounts,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      const distanceChunk = distanceChunks[chunkIndex];
      if (props.bestDistanceBits && distanceChunk) {
        bindings.push(
          {name: 'bestDistanceBits', view: props.bestDistanceBits, type: 'u32', access: 'read'},
          {name: 'distances', view: distanceChunk, type: 'f32', access: 'read_write'}
        );
      }
      nodes.push(
        createSpatialJoinChunkNode<Parameters>(graph, {
          id: getChunkNodeId(props.id, primaryOutput, chunkIndex),
          operation: props.operation,
          variant: 'assign',
          bindings,
          invocationCount: chunk.length,
          firstRow: chunkFirstRow,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
${props.distanceRankWGSL ?? ''}`,
          body: `let row = CHUNK_FIRST_ROW + index;
  let featureRow = assignment[assignmentOffset + row];
  let matched = featureRow != NO_FEATURE;
  var featureId = featureRow;
  ${props.featureIds && props.pointFeatureIds ? 'if (matched) { featureId = featureIds[featureIdsOffset + featureRow]; }' : ''}
  pointFeatureIds[pointFeatureIdsOffset + index] = featureId;
  ${rowChunk ? 'pointFeatureRows[pointFeatureRowsOffset + index] = featureRow;' : ''}
  ${props.featureCounts ? 'if (matched) { atomicAdd(&featureCounts[featureCountsOffset + featureRow], 1u); }' : ''}
  ${props.polygonAssignmentCounts ? 'if (matched) { atomicAdd(&assignmentCounts[assignmentCountsOffset + featureRow], 1u); }' : ''}
  ${
    props.bestDistanceBits && distanceChunk
      ? `let bestDistance = bitcast<f32>(bestDistanceBits[bestDistanceBitsOffset + row]);
  distances[distancesOffset + index] = select(-1.0, ${props.distanceRankWGSL ? 'decodeNearestDistance(bestDistance)' : props.squaredDistances ? 'sqrt(bestDistance)' : 'bestDistance'}, matched);`
      : ''
  }`
        })
      );
    }
    chunkFirstRow += chunk.length;
  }
  return nodes;
}

/** Stably compacts matched point IDs and writes the total to `state[2]`. @internal */
export function getSpatialJoinCollectNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    points: GraphDataView | GraphVectorView;
    assignment: GraphDataView<'uint32'>;
    state: GraphDataView<'uint32'>;
    sourceIds?: GPUUint32Rows;
    matches: GPUCompactOutput;
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const sourceChunks = props.sourceIds ? getGraphViewChunks(props.sourceIds) : [];
  const capacity = props.matches.ids.length;
  const pointCount = props.points.length;
  if (pointCount === 0) {
    return nodes;
  }
  const flags = createTransientView(graph, `${props.id}-flags`, 'uint32', pointCount);
  const offsets = createTransientView(graph, `${props.id}-offsets`, 'uint32', pointCount);
  let chunkFirstRow = 0;
  for (const [chunkIndex, chunk] of getGraphViewChunks(props.points).entries()) {
    if (chunk.length > 0) {
      nodes.push(
        createSpatialJoinChunkNode<Parameters>(graph, {
          id: getChunkNodeId(props.id, props.points, chunkIndex),
          operation: props.operation,
          variant: 'collect-match-flags',
          bindings: [
            {name: 'assignment', view: props.assignment, type: 'u32', access: 'read'},
            {name: 'flags', view: flags, type: 'u32', access: 'read_write'}
          ],
          invocationCount: chunk.length,
          firstRow: chunkFirstRow,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}
`,
          body: `let row = CHUNK_FIRST_ROW + index;
  flags[flagsOffset + row] = select(0u, 1u, assignment[assignmentOffset + row] != NO_FEATURE);`
        })
      );
    }
    chunkFirstRow += chunk.length;
  }
  // The exclusive scan and the single-word match count (state word 2) come from one primitive.
  nodes.push(
    ...new GPUFlagOffsets({
      id: props.id,
      flags,
      offsets,
      count: graph.createDataView(props.state.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: props.state.byteOffset + 2 * Uint32Array.BYTES_PER_ELEMENT
      })
    }).getCommandNodes(graph)
  );
  if (capacity === 0) {
    return nodes;
  }
  chunkFirstRow = 0;
  for (const [chunkIndex, chunk] of getGraphViewChunks(props.points).entries()) {
    if (chunk.length > 0) {
      const sourceChunk = sourceChunks[chunkIndex];
      const bindings: WGSLKernelBinding[] = [
        {name: 'flags', view: flags, type: 'u32', access: 'read'},
        {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
        {name: 'matchIds', view: props.matches.ids, type: 'u32', access: 'read_write'}
      ];
      if (sourceChunk) {
        bindings.push({name: 'sourceIds', view: sourceChunk, type: 'u32', access: 'read'});
      }
      nodes.push(
        createSpatialJoinChunkNode<Parameters>(graph, {
          id: `${getChunkNodeId(props.id, props.points, chunkIndex)}-write`,
          operation: props.operation,
          variant: 'collect-match-write',
          bindings,
          invocationCount: chunk.length,
          firstRow: chunkFirstRow,
          declarations: `${SPATIAL_JOIN_WGSL_HELPERS}

const MATCH_CAPACITY: u32 = ${capacity}u;`,
          body: `let row = CHUNK_FIRST_ROW + index;
  if (flags[flagsOffset + row] == 0u) { return; }
  let slot = offsets[offsetsOffset + row];
  if (slot < MATCH_CAPACITY) {
    matchIds[matchIdsOffset + slot] = ${sourceChunk ? 'sourceIds[sourceIdsOffset + index]' : 'row'};
  }`
        })
      );
    }
    chunkFirstRow += chunk.length;
  }
  return nodes;
}
