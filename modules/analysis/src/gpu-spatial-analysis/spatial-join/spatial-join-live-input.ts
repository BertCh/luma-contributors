// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createWeightsLossNodes} from '../spatial-weights/spatial-weights-loss';
import type {GPUSpatialJoinGeometry} from './spatial-join-types';
import {
  validateCapacity,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Resident left-operand completeness, captured at construction. */
export type GPUSpatialJoinLeftStatus = {
  overflow?: GraphDataView<'uint32'>;
  candidateOverflow?: GraphDataView<'uint32'>;
  /** One certifies the entire live feature prefix. */
  complete?: GraphDataView<'uint32'>;
};

/** @internal Validate resident cardinality/completeness once before graph declaration. */
export function validateSpatialJoinLeftInput(
  id: string,
  count?: GraphDataView<'uint32'>,
  status?: GPUSpatialJoinLeftStatus
): void {
  for (const [name, view] of Object.entries({
    leftFeatureCount: count,
    ...Object.fromEntries(
      Object.entries(status ?? {}).map(([name, view]) => [`leftStatus.${name}`, view])
    )
  })) {
    if (view) {
      validateSinglePackedView(view, ['uint32'], `${id} ${name}`);
      validateCapacity(id, `${name}.length`, view.length, 1);
    }
  }
}

/** @internal Clear diagnostics for rows that no longer belong to the live left domain. */
export function createSpatialJoinLiveTruncationNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  count: GraphDataView<'uint32'>,
  truncated: GraphDataView<'uint32'>
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode(graph, {
    id: `${id}-live-truncation`,
    operation: 'GPUSpatialPredicateJoin',
    invocationCount: truncated.length,
    bindings: [
      {name: 'liveCount', view: count, type: 'u32', access: 'read'},
      {name: 'truncated', view: truncated, type: 'u32', access: 'read_write'}
    ],
    body: `let count = liveCount[liveCountOffset];
  if (count > ${truncated.length}u || index >= count) { truncated[truncatedOffset + index] = 0u; }`
  });
}

/** @internal Repeats the terminal live ring offset so allocated stale rings are empty to the shared check. */
export function createSpatialJoinLiveRingOffsets<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  geometry: Extract<GPUSpatialJoinGeometry, {kind: 'polygons'}>,
  count: GraphDataView<'uint32'>
) {
  const ringOffsets = createTransientView(
    graph,
    `${id}-live-ring-offset-view`,
    'uint32',
    geometry.ringOffsets.length
  );
  const node = createWGSLKernelNode(graph, {
    id: `${id}-live-ring-offsets`,
    operation: 'GPUSpatialPredicateJoin',
    invocationCount: ringOffsets.length,
    bindings: [
      {name: 'liveCount', view: count, type: 'u32', access: 'read'},
      {name: 'featureOffsets', view: geometry.featureOffsets, type: 'u32', access: 'read'},
      {name: 'polygonOffsets', view: geometry.polygonOffsets, type: 'u32', access: 'read'},
      {name: 'inputRings', view: geometry.ringOffsets, type: 'u32', access: 'read'},
      {name: 'outputRings', view: ringOffsets, type: 'u32', access: 'read_write'}
    ],
    body: `let featureCount = liveCount[liveCountOffset];
  let polygonEnd = min(featureOffsets[featureOffsetsOffset + min(featureCount, ${geometry.featureOffsets.length - 1}u)], ${geometry.polygonOffsets.length - 1}u);
  let ringEnd = select(min(polygonOffsets[polygonOffsetsOffset + polygonEnd], ${geometry.ringOffsets.length - 1}u), 0u, featureCount > ${geometry.featureOffsets.length - 1}u);
  outputRings[outputRingsOffset + index] = inputRings[inputRingsOffset + min(index, ringEnd)];`
  });
  return {node, ringOffsets};
}

/** @internal Masks spare left feature slots and combines upstream loss with BVH loss. */
export function createSpatialJoinLiveInputNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: {
    capacity: number;
    count?: GraphDataView<'uint32'>;
    status?: GPUSpatialJoinLeftStatus;
    minima: GraphDataView<'float32x2'>;
    maxima: GraphDataView<'float32x2'>;
    bvhOverflow: GraphDataView<'uint32'>;
  }
): {nodes: GPUCommandNode<Parameters>[]; loss: GraphDataView<'uint32'>} {
  if (!props.count && !props.status) return {nodes: [], loss: props.bvhOverflow};
  const loss = createTransientView(graph, `${id}-left-input-loss`, 'uint32', 1);
  const nodes = createWeightsLossNodes(
    graph,
    `${id}-left-loss`,
    [
      props.bvhOverflow,
      ...[props.status?.overflow, props.status?.candidateOverflow].filter(
        (view): view is GraphDataView<'uint32'> => Boolean(view)
      )
    ],
    loss
  );
  if (!props.count && !props.status?.complete) return {nodes, loss};
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-live-left-bounds`,
      operation: 'GPUSpatialPredicateJoin',
      invocationCount: props.capacity,
      bindings: [
        {name: 'loss', view: loss, type: 'u32', access: 'read_write'},
        ...(props.count
          ? [{name: 'liveCount', view: props.count, type: 'u32', access: 'read'} as const]
          : []),
        ...(props.status?.complete
          ? [{name: 'complete', view: props.status.complete, type: 'u32', access: 'read'} as const]
          : []),
        {name: 'minima', view: props.minima, type: 'f32', access: 'read_write'},
        {name: 'maxima', view: props.maxima, type: 'f32', access: 'read_write'}
      ],
      body: `let invalidCount = ${props.count ? `liveCount[liveCountOffset] > ${props.capacity}u` : 'false'};
  let incomplete = ${props.status?.complete ? 'complete[completeOffset] == 0u' : 'false'};
  if (index == 0u && (invalidCount || incomplete)) { loss[lossOffset] = 1u; }
  if (${props.count ? 'index >= liveCount[liveCountOffset] || ' : ''}invalidCount || incomplete) {
    minima[minimaOffset + index * 2u] = 3.4028234663852886e38;
    minima[minimaOffset + index * 2u + 1u] = 3.4028234663852886e38;
    maxima[maximaOffset + index * 2u] = -3.4028234663852886e38;
    maxima[maximaOffset + index * 2u + 1u] = -3.4028234663852886e38;
  }`
    })
  );
  return {nodes, loss};
}
