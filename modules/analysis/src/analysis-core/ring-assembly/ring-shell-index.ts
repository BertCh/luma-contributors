// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUBVH,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../kernels/wgsl-kernel-nodes';
import {GET_ORDERED_KEY_WGSL, SPREAD_BITS_WGSL} from '../kernels/wgsl-common';

/** Area, ring kind (0 shell, 1 hole, 2 unused) and group; bounds reuse the ring-stats view. */
export const RING_SHELL_RECORD_WORDS = 3;

/**
 * Builds a shell BVH ordered by the full 64-bit Morton code of the two float32 center keys.
 * Interleaving the order-preserving coordinate bits retains every stored center distinction:
 * an outlier cannot collapse nearby centers into one quantized key. Two stable uint32 sorts
 * (low word, then high word) preserve original ring order only for identical center keys.
 * Leaves retain original ring IDs, independently of their spatial order.
 * @internal
 */
export function getRingShellIndexNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    bounds: GraphDataView<'float32'>;
    areas: GraphDataView<'float32'>;
    isHole: GraphDataView<'uint32'>;
    count: GraphDataView<'uint32'>;
    groups?: GraphDataView<'uint32'>;
  }
): {
  records: GraphDataView<'uint32'>;
  bvh: GPUBVH;
  nodes: GPUCommandNode<Parameters>[];
} {
  const {id, operation} = props;
  const ringCapacity = props.isHole.length;
  const leafCapacity = 2 ** Math.ceil(Math.log2(ringCapacity));
  const nodeCount = 2 * leafCapacity - 1;
  const records = createTransientView(
    graph,
    `${id}-records`,
    'uint32',
    ringCapacity * RING_SHELL_RECORD_WORDS
  );
  const keys = (name: string) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', ringCapacity);
  const lowKeys = keys('low-keys');
  const highKeys = keys('high-keys');
  const rows = keys('rows');
  const sortedLowKeys = keys('sorted-low-keys');
  const lowRows = keys('low-rows');
  const orderedHighKeys = keys('ordered-high-keys');
  const sortedHighKeys = keys('sorted-high-keys');
  const sortedRows = keys('sorted-rows');
  const minima = createTransientView(graph, `${id}-minima`, 'float32x2', ringCapacity);
  const maxima = createTransientView(graph, `${id}-maxima`, 'float32x2', ringCapacity);
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-records`,
      operation,
      variant: 'shell-records',
      bindings: [
        {name: 'areas', view: props.areas, type: 'f32', access: 'read'},
        {name: 'isHole', view: props.isHole, type: 'u32', access: 'read'},
        {name: 'count', view: props.count, type: 'u32', access: 'read'},
        ...(props.groups
          ? [{name: 'groups', view: props.groups, type: 'u32' as const, access: 'read' as const}]
          : []),
        {name: 'records', view: records, type: 'u32', access: 'read_write'}
      ],
      invocationCount: ringCapacity,
      body: `let base = recordsOffset + ${RING_SHELL_RECORD_WORDS}u * index;
  records[base] = bitcast<u32>(areas[areasOffset + index]);
  records[base + 1u] = select(2u, isHole[isHoleOffset + index], index < count[countOffset]);
  records[base + 2u] = ${props.groups ? 'groups[groupsOffset + index]' : '0u'};`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-keys`,
      operation,
      variant: 'shell-keys',
      bindings: [
        {name: 'bounds', view: props.bounds, type: 'f32', access: 'read'},
        {name: 'records', view: records, type: 'u32', access: 'read'},
        {name: 'lowKeys', view: lowKeys, type: 'u32', access: 'read_write'},
        {name: 'highKeys', view: highKeys, type: 'u32', access: 'read_write'},
        {name: 'rows', view: rows, type: 'u32', access: 'read_write'}
      ],
      invocationCount: ringCapacity,
      declarations: `${GET_ORDERED_KEY_WGSL}\n${SPREAD_BITS_WGSL}`,
      body: `let base = recordsOffset + ${RING_SHELL_RECORD_WORDS}u * index;
  var low = 0xffffffffu;
  var high = 0xffffffffu;
  if (records[base + 1u] == 0u) {
    // Halve before adding so finite bounds of either sign cannot overflow at their center.
    let boundsBase = boundsOffset + 4u * index;
    let centerX = 0.5 * bounds[boundsBase] + 0.5 * bounds[boundsBase + 2u];
    let centerY = 0.5 * bounds[boundsBase + 1u] + 0.5 * bounds[boundsBase + 3u];
    let coordinateKeyX = getOrderedKey(centerX);
    let coordinateKeyY = getOrderedKey(centerY);
    low = spreadBits(coordinateKeyX & 0xffffu) | (spreadBits(coordinateKeyY & 0xffffu) << 1u);
    high = spreadBits(coordinateKeyX >> 16u) | (spreadBits(coordinateKeyY >> 16u) << 1u);
  }
  lowKeys[lowKeysOffset + index] = low;
  highKeys[highKeysOffset + index] = high;
  rows[rowsOffset + index] = index;`
    }),
    ...new GPUSort({
      id: `${id}-sort-low`,
      keys: lowKeys,
      values: rows,
      outputKeys: sortedLowKeys,
      outputValues: lowRows,
      algorithm: 'radix'
    }).getCommandNodes<Parameters>(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-order-high-keys`,
      operation,
      variant: 'shell-order-high-keys',
      bindings: [
        {name: 'highKeys', view: highKeys, type: 'u32', access: 'read'},
        {name: 'rows', view: lowRows, type: 'u32', access: 'read'},
        {name: 'orderedKeys', view: orderedHighKeys, type: 'u32', access: 'read_write'}
      ],
      invocationCount: ringCapacity,
      body: 'orderedKeys[orderedKeysOffset + index] = highKeys[highKeysOffset + rows[rowsOffset + index]];'
    }),
    ...new GPUSort({
      id: `${id}-sort-high`,
      keys: orderedHighKeys,
      values: lowRows,
      outputKeys: sortedHighKeys,
      outputValues: sortedRows,
      algorithm: 'radix'
    }).getCommandNodes<Parameters>(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-gather`,
      operation,
      variant: 'shell-gather',
      bindings: [
        {name: 'bounds', view: props.bounds, type: 'f32', access: 'read'},
        {name: 'records', view: records, type: 'u32', access: 'read'},
        {name: 'rows', view: sortedRows, type: 'u32', access: 'read'},
        {name: 'minima', view: minima, type: 'f32', access: 'read_write'},
        {name: 'maxima', view: maxima, type: 'f32', access: 'read_write'}
      ],
      invocationCount: ringCapacity,
      body: `let ring = rows[rowsOffset + index];
  let unused = records[recordsOffset + ${RING_SHELL_RECORD_WORDS}u * ring + 1u] != 0u;
  let base = boundsOffset + 4u * ring;
  minima[minimaOffset + 2u * index] = select(bounds[base], 3.402823466e+38, unused);
  minima[minimaOffset + 2u * index + 1u] = select(bounds[base + 1u], 3.402823466e+38, unused);
  maxima[maximaOffset + 2u * index] = select(bounds[base + 2u], -3.402823466e+38, unused);
  maxima[maximaOffset + 2u * index + 1u] = select(bounds[base + 3u], -3.402823466e+38, unused);`
    })
  ];
  const bvh = new GPUBVH({
    id: `${id}-bvh`,
    minima,
    maxima,
    sourceIds: sortedRows,
    leafCapacity,
    nodeMinima: createTransientView(graph, `${id}-node-minima`, 'float32x2', nodeCount),
    nodeMaxima: createTransientView(graph, `${id}-node-maxima`, 'float32x2', nodeCount),
    nodeChildren: createTransientView(graph, `${id}-node-children`, 'uint32x2', nodeCount),
    leafIds: createTransientView(graph, `${id}-leaf-ids`, 'uint32', leafCapacity),
    count: createTransientView(graph, `${id}-count`, 'uint32', 1),
    overflow: createTransientView(graph, `${id}-overflow`, 'uint32', 1)
  });
  nodes.push(...bvh.getCommandNodes<Parameters>(graph));
  return {records, bvh, nodes};
}
