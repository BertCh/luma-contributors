// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_ORDERED_KEY_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';

/**
 * Sorts packed `[first.x, first.y, second.x, second.y, group]` records by exact uint32 group,
 * then minimum segment height. Every group's leaves are contiguous, irrespective of ID gaps,
 * outliers or float32 collisions. Query traversal intersects that exact leaf range before the
 * conservative height test. The published complete-binary BVH keeps this sorted leaf order.
 * @internal
 */
export function getMakeValidSegmentIndexNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; operation: string; records: GraphDataView<'uint32'>; leafCapacity: number}
) {
  const {id, operation, records, leafCapacity} = props;
  const segmentCapacity = records.length / 5;
  const transient = (name: string) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', segmentCapacity);
  const heightKeys = transient('height-keys');
  const rows = transient('rows');
  const sortedHeightKeys = transient('sorted-height-keys');
  const heightRows = transient('height-rows');
  const groupKeys = transient('group-keys');
  const sortedGroups = transient('sorted-groups');
  const sortedRows = transient('sorted-rows');
  const sortedRecords = createTransientView(
    graph,
    `${id}-sorted-records`,
    'uint32',
    records.length
  );
  const minima = createTransientView(graph, `${id}-minima`, 'float32x2', segmentCapacity);
  const maxima = createTransientView(graph, `${id}-maxima`, 'float32x2', segmentCapacity);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-height-keys`,
      operation,
      variant: 'parity-height-keys',
      bindings: [
        {name: 'records', view: records, type: 'u32', access: 'read'},
        {name: 'keys', view: heightKeys, type: 'u32', access: 'read_write'},
        {name: 'rows', view: rows, type: 'u32', access: 'read_write'}
      ],
      invocationCount: segmentCapacity,
      declarations: `${GET_ORDERED_KEY_WGSL}\n${IS_FINITE_VALUE_WGSL}`,
      body: `let base = recordsOffset + 5u * index;
  let first = vec2f(bitcast<f32>(records[base]), bitcast<f32>(records[base + 1u]));
  let second = vec2f(bitcast<f32>(records[base + 2u]), bitcast<f32>(records[base + 3u]));
  let finite = isFiniteValue(first.x) && isFiniteValue(first.y) && isFiniteValue(second.x) && isFiniteValue(second.y);
  let height = select(-3.402823466e+38, min(first.y, second.y), finite);
  keys[keysOffset + index] = getOrderedKey(height);
  rows[rowsOffset + index] = index;`
    }),
    ...new GPUSort({
      id: `${id}-height-sort`,
      keys: heightKeys,
      values: rows,
      outputKeys: sortedHeightKeys,
      outputValues: heightRows,
      algorithm: 'radix'
    }).getCommandNodes<Parameters>(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-group-keys`,
      operation,
      variant: 'parity-group-keys',
      bindings: [
        {name: 'records', view: records, type: 'u32', access: 'read'},
        {name: 'rows', view: heightRows, type: 'u32', access: 'read'},
        {name: 'keys', view: groupKeys, type: 'u32', access: 'read_write'}
      ],
      invocationCount: segmentCapacity,
      body: `keys[keysOffset + index] = records[recordsOffset + 5u * rows[rowsOffset + index] + 4u];`
    }),
    ...new GPUSort({
      id: `${id}-group-sort`,
      keys: groupKeys,
      values: heightRows,
      outputKeys: sortedGroups,
      outputValues: sortedRows,
      algorithm: 'radix'
    }).getCommandNodes<Parameters>(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-gather`,
      operation,
      variant: 'parity-gather',
      bindings: [
        {name: 'records', view: records, type: 'u32', access: 'read'},
        {name: 'rows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'sortedRecords', view: sortedRecords, type: 'u32', access: 'read_write'},
        {name: 'minima', view: minima, type: 'f32', access: 'read_write'},
        {name: 'maxima', view: maxima, type: 'f32', access: 'read_write'}
      ],
      invocationCount: segmentCapacity,
      declarations: IS_FINITE_VALUE_WGSL,
      body: `let source = recordsOffset + 5u * rows[rowsOffset + index];
  for (var component = 0u; component < 5u; component++) {
    sortedRecords[sortedRecordsOffset + 5u * index + component] = records[source + component];
  }
  let first = vec2f(bitcast<f32>(records[source]), bitcast<f32>(records[source + 1u]));
  let second = vec2f(bitcast<f32>(records[source + 2u]), bitcast<f32>(records[source + 3u]));
  let finite = isFiniteValue(first.x) && isFiniteValue(first.y) && isFiniteValue(second.x) && isFiniteValue(second.y);
  let unused = records[source + 4u] == 0xffffffffu;
  // Non-finite edges still reach the exact crossing helper and preserve its uncertainty flag.
  let minimum = select(select(-3.402823466e+38, min(first.y, second.y), finite), 3.402823466e+38, unused);
  let maximum = select(select(3.402823466e+38, max(first.y, second.y), finite), -3.402823466e+38, unused);
  minima[minimaOffset + 2u * index] = 0.0;
  minima[minimaOffset + 2u * index + 1u] = minimum;
  maxima[maximaOffset + 2u * index] = 0.0;
  maxima[maximaOffset + 2u * index + 1u] = maximum;`
    })
  ];
  const index = getSegmentBVHNodes<Parameters>(graph, {
    id,
    operation,
    minima,
    maxima,
    leafCapacity,
    spatialSort: false
  });
  nodes.push(...index.nodes);
  return {nodes, bvh: index.bvh, records: sortedRecords, groups: sortedGroups};
}
