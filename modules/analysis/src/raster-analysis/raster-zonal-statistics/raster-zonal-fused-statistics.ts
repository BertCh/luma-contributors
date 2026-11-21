// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

const FUSED_WORKGROUP_SIZE = 256;

/**
 * One-pass zone reduction of counts, minimums and maximums with workgroup-private tables.
 *
 * Each workgroup walks a contiguous tile of `cellsPerThread * 256` cells, accumulates into
 * `var<workgroup>` atomic tables (shared-memory atomics, no global traffic), and merges the non-empty
 * entries into the global result once per workgroup. Compared with one `GPUGroupAggregation` per
 * column, zones and values are read once instead of once per column, the zone and value masks are
 * never materialised, and global atomic traffic is at most `zoneCapacity` per tile instead of one
 * per cell. Minimums and maximums use order-preserving u32 keys (the same encoding as the core
 * aggregation), so counts, minimums and maximums stay exact.
 *
 * The table size is `zoneCapacity` words per requested column, so callers must check that it
 * fits in workgroup storage first.
 */
export function getFusedStatisticsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    zones: GraphDataView<'uint32'>;
    values?: GraphDataView<'float32'>;
    validity?: GraphDataView<'uint32'>;
    zoneCapacity: number;
    ignoredZone: number;
    cellCount: number;
    cellCounts?: GraphDataView<'uint32'>;
    valueCounts?: GraphDataView<'uint32'>;
    minimums?: GraphDataView<'float32'>;
    maximums?: GraphDataView<'float32'>;
    overflow?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, zoneCapacity, cellCount} = props;
  const columns = [
    {name: 'cellCounts', view: props.cellCounts, initial: '0u', merge: 'atomicAdd'},
    {name: 'valueCounts', view: props.valueCounts, initial: '0u', merge: 'atomicAdd'},
    {name: 'minimums', view: props.minimums, initial: '0xffffffffu', merge: 'atomicMin'},
    {name: 'maximums', view: props.maximums, initial: '0u', merge: 'atomicMax'}
  ].filter(column => column.view) as {
    name: string;
    view: GraphDataView;
    initial: string;
    merge: string;
  }[];
  const needsValues = Boolean(props.valueCounts || props.minimums || props.maximums);
  const cellsPerThread = clamp(Math.ceil(zoneCapacity / 32), 8, 64);
  const nodes: GPUCommandNode<Parameters>[] = [];

  // Global results start at the merge identity; the tables merge into them atomically.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-zone-reduce-init`,
      operation: 'GPURasterZonalStatistics',
      variant: 'zone-reduce-init',
      bindings: columns.map(column => ({
        name: column.name,
        view: column.view,
        type: 'u32' as const,
        access: 'read_write' as const
      })),
      invocationCount: zoneCapacity,
      body: columns
        .map(column => `${column.name}[${column.name}Offset + index] = ${column.initial};`)
        .join('\n  ')
    })
  );

  const bindings: WGSLKernelBinding[] = [
    {name: 'zones', view: props.zones, type: 'u32', access: 'read'}
  ];
  if (needsValues) {
    bindings.push(
      {name: 'values', view: props.values!, type: 'f32', access: 'read'},
      {name: 'validity', view: props.validity!, type: 'u32', access: 'read'}
    );
  }
  for (const column of columns) {
    bindings.push({
      name: column.name,
      view: column.view,
      type: 'atomic<u32>',
      access: 'read_write'
    });
  }
  if (props.overflow) {
    bindings.push({
      name: 'overflowFlag',
      view: props.overflow,
      type: 'atomic<u32>',
      access: 'read_write'
    });
  }
  const tables = columns
    .map(column => `var<workgroup> local_${column.name}: array<atomic<u32>, ${zoneCapacity}>;`)
    .join('\n');
  const hasMinimum = Boolean(props.minimums);
  const hasMaximum = Boolean(props.maximums);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-zone-reduce`,
      operation: 'GPURasterZonalStatistics',
      variant: 'zone-reduce',
      bindings,
      invocationCount: Math.ceil(cellCount / cellsPerThread),
      guardIndex: false,
      workgroupSize: FUSED_WORKGROUP_SIZE,
      declarations: `const CAPACITY: u32 = ${zoneCapacity}u;
const IGNORED: u32 = ${props.ignoredZone}u;
const CELL_COUNT: u32 = ${cellCount}u;
const CELLS_PER_THREAD: u32 = ${cellsPerThread}u;
${tables}
${IS_FINITE_VALUE_WGSL}
fn encodeOrderedFloat(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}`,
      body: `for (var slot = localInvocationIndex; slot < CAPACITY; slot += ${FUSED_WORKGROUP_SIZE}u) {
    ${columns.map(column => `atomicStore(&local_${column.name}[slot], ${column.initial});`).join('\n    ')}
  }
  workgroupBarrier();
  let tileStart = (index - localInvocationIndex) * CELLS_PER_THREAD;
  for (var step = 0u; step < CELLS_PER_THREAD; step++) {
    let cell = tileStart + step * ${FUSED_WORKGROUP_SIZE}u + localInvocationIndex;
    if (cell >= CELL_COUNT) {
      break;
    }
    let zone = zones[zonesOffset + cell];
    if (zone >= CAPACITY || zone == IGNORED) {
      ${props.overflow ? 'if (zone != IGNORED) {\n        atomicStore(&overflowFlag[overflowFlagOffset], 1u);\n      }' : ''}
      continue;
    }
    ${props.cellCounts ? 'atomicAdd(&local_cellCounts[zone], 1u);' : ''}
    ${
      needsValues
        ? `let value = values[valuesOffset + cell];
    if (validity[validityOffset + cell] != 0u && isFiniteValue(value)) {
      ${props.valueCounts ? 'atomicAdd(&local_valueCounts[zone], 1u);' : ''}
      ${hasMinimum || hasMaximum ? 'let key = encodeOrderedFloat(value);' : ''}
      ${hasMinimum ? 'atomicMin(&local_minimums[zone], key);' : ''}
      ${hasMaximum ? 'atomicMax(&local_maximums[zone], key);' : ''}
    }`
        : ''
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < CAPACITY; slot += ${FUSED_WORKGROUP_SIZE}u) {
    ${columns
      .map(
        column => `let partial_${column.name} = atomicLoad(&local_${column.name}[slot]);
    if (partial_${column.name} != ${column.initial}) {
      ${column.merge}(&${column.name}[${column.name}Offset + slot], partial_${column.name});
    }`
      )
      .join('\n    ')}
  }`
    })
  );

  // Decode the order-preserving keys in place; empty zones become NaN.
  const decoded = columns.filter(
    column => column.name === 'minimums' || column.name === 'maximums'
  );
  if (decoded.length > 0) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-zone-reduce-decode`,
        operation: 'GPURasterZonalStatistics',
        variant: 'zone-reduce-decode',
        bindings: decoded.map(column => ({
          name: column.name,
          view: column.view,
          type: 'u32' as const,
          access: 'read_write' as const
        })),
        invocationCount: zoneCapacity,
        declarations: `fn decodeOrderedKey(key: u32) -> u32 {
  return select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u);
}`,
        body: decoded
          .map(
            column => `let key_${column.name} = ${column.name}[${column.name}Offset + index];
  ${column.name}[${column.name}Offset + index] = select(decodeOrderedKey(key_${column.name}), 0x7fc00000u, key_${column.name} == ${column.initial});`
          )
          .join('\n  ')
      })
    );
  }
  return nodes;
}
