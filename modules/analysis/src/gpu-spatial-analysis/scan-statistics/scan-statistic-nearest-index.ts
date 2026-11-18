// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createScanNearestZonesKernelNode} from './scan-statistic-nearest-zones';

/** Compile-time exact neighbor discovery. Auto indexes lists of at most 65 rows, including lookahead. */
export type GPUScanNearestZoneSearch = 'auto' | 'indexed' | 'full-scan';
/** Bounds two per-query private uint32 lists to 520 bytes, including a spatial scan's lookahead. */
export const GPU_SCAN_MAXIMUM_INDEXED_NEIGHBOR_COUNT = 65;
/** Upper bound for two published radix sorts, their scans and the key/query passes. @internal */
export const SCAN_INDEX_BUILD_COMMAND_NODE_ALLOWANCE = 2048;

/** Resolve discovery once at construction; the estimate counts separation evaluations, excluding sorting. @internal */
export function getScanNearestZoneSearchPlan(
  id: string,
  zones: number,
  listLength: number,
  requested: GPUScanNearestZoneSearch = 'auto'
): {
  nearestZoneSearch: 'indexed' | 'full-scan';
  estimatedFullScanSeparationCount: number;
} {
  validateRequirement(
    id,
    ['auto', 'indexed', 'full-scan'].includes(requested),
    'nearestZoneSearch must be auto, indexed or full-scan'
  );
  const nearestZoneSearch =
    requested === 'auto'
      ? listLength < zones && listLength <= GPU_SCAN_MAXIMUM_INDEXED_NEIGHBOR_COUNT
        ? 'indexed'
        : 'full-scan'
      : requested;
  if (nearestZoneSearch === 'indexed')
    validateMaximumCount(
      id,
      'nearestZoneSearch indexed listLength',
      listLength,
      GPU_SCAN_MAXIMUM_INDEXED_NEIGHBOR_COUNT
    );
  return {
    nearestZoneSearch,
    estimatedFullScanSeparationCount: zones * zones * (listLength < zones ? 5 : 1)
  };
}

/** Shared exact scan discovery: published radix-sort indexes or the original exhaustive route. @internal */
export function createScanNearestZoneNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    positions: GraphDataView<'float32x2'>;
    neighbors: GraphDataView<'uint32'>;
    distances: GraphDataView<'float32'>;
    zones: number;
    listLength: number;
    separationWGSL: string;
    geographic: boolean;
    nearestZoneSearch: 'indexed' | 'full-scan';
  }
): GPUCommandNode<Parameters>[] {
  const {
    id,
    operation,
    positions,
    neighbors,
    distances,
    zones,
    listLength,
    separationWGSL,
    geographic
  } = props;
  const view = (name: string, length = zones) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  // Both callers preflight zones * listLength words. Every coordinate index buffer is
  // zones words, so it fits that checked binding bound; published sorts preflight their scratch.
  if (props.nearestZoneSearch === 'full-scan')
    return [
      createScanNearestZonesKernelNode(graph, {
        ...props,
        sortKeys: view('sort-keys', zones * listLength),
        sortZones: view('sort-zones', zones * listLength)
      })
    ];
  const keysX = view('keys-x');
  const keysY = view('keys-y');
  const rows = view('rows');
  const sortedKeysX = view('sorted-keys-x');
  const sortedKeysY = view('sorted-keys-y');
  const sortedRowsX = view('sorted-rows-x');
  const sortedRowsY = view('sorted-rows-y');
  const positionKeyWGSL = `// Float-flip orders finite coordinates numerically; the two zeros share one key.
fn getPositionKey(value: f32) -> u32 {
  var bits = bitcast<u32>(value);
  if (value == 0.0) { bits = 0u; }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn getPositionValue(key: u32) -> f32 {
  return bitcast<f32>(select(key ^ 0x80000000u, ~key, (key & 0x80000000u) == 0u));
}`;
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-index-keys`,
      operation,
      variant: 'nearest-index-keys',
      invocationCount: zones,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'keysX', view: keysX, type: 'u32', access: 'read_write'},
        {name: 'keysY', view: keysY, type: 'u32', access: 'read_write'},
        {name: 'rows', view: rows, type: 'u32', access: 'read_write'}
      ],
      declarations: positionKeyWGSL,
      body: `keysX[keysXOffset + index] = getPositionKey(positions[positionsOffset + index * 2u]);
  keysY[keysYOffset + index] = getPositionKey(positions[positionsOffset + index * 2u + 1u]);
  rows[rowsOffset + index] = index;`
    }),
    ...new GPUSort({
      id: `${id}-index-x`,
      keys: keysX,
      values: rows,
      outputKeys: sortedKeysX,
      outputValues: sortedRowsX,
      algorithm: 'radix'
    }).getCommandNodes(graph),
    ...new GPUSort({
      id: `${id}-index-y`,
      keys: keysY,
      values: rows,
      outputKeys: sortedKeysY,
      outputValues: sortedRowsY,
      algorithm: 'radix'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id,
      operation,
      variant: 'nearest-indexed',
      invocationCount: zones,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'sortedKeysX', view: sortedKeysX, type: 'u32', access: 'read'},
        {name: 'sortedKeysY', view: sortedKeysY, type: 'u32', access: 'read'},
        {name: 'sortedRowsX', view: sortedRowsX, type: 'u32', access: 'read'},
        {name: 'sortedRowsY', view: sortedRowsY, type: 'u32', access: 'read'},
        {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
        {name: 'distances', view: distances, type: 'u32', access: 'read_write'}
      ],
      declarations: `${separationWGSL}
${positionKeyWGSL}
const ZONES: u32 = ${zones}u;
const LIST: u32 = ${listLength}u;
const GEOGRAPHIC: bool = ${geographic};
fn getSortedKey(rank: u32, axis: u32) -> u32 {
  if (axis == 0u) { return sortedKeysX[sortedKeysXOffset + rank]; }
  return sortedKeysY[sortedKeysYOffset + rank];
}
fn getSortedRow(rank: u32, axis: u32) -> u32 {
  if (axis == 0u) { return sortedRowsX[sortedRowsXOffset + rank]; }
  return sortedRowsY[sortedRowsYOffset + rank];
}
fn getZoneSeparationKey(zone: u32, centerX: f32, centerY: f32) -> u32 {
  let zoneX = positions[positionsOffset + zone * 2u];
  let zoneY = positions[positionsOffset + zone * 2u + 1u];
  return bitcast<u32>(getSeparationKey(vec2f(zoneX, zoneY), vec2f(centerX, centerY)));
}
// A same-longitude haversine drops a nonnegative term, so latitude is a lower bound even
// across the antimeridian and at a pole. Planar projection similarly drops one squared term.
fn getAxisBound(rank: u32, axis: u32, center: vec2f, canPrune: bool) -> u32 {
  if (!canPrune) { return 0u; }
  let coordinate = getPositionValue(getSortedKey(rank, axis));
  var projection = vec2f(coordinate, center.y);
  if (axis == 1u) { projection = vec2f(center.x, coordinate); }
  let bound = getSeparationKey(projection, center);
  // Retain a conservative margin for the GPU's trigonometric rounding near geographic bounds.
  return bitcast<u32>(select(bound, bound * 0.99, GEOGRAPHIC));
}`,
      body: `let row = index * LIST;
  let centerX = positions[positionsOffset + index * 2u];
  let centerY = positions[positionsOffset + index * 2u + 1u];
  let minimumY = getPositionValue(sortedKeysY[sortedKeysYOffset]);
  let maximumY = getPositionValue(sortedKeysY[sortedKeysYOffset + ZONES - 1u]);
  var axis = 1u;
  if (!GEOGRAPHIC) {
    let minimumX = getPositionValue(sortedKeysX[sortedKeysXOffset]);
    let maximumX = getPositionValue(sortedKeysX[sortedKeysXOffset + ZONES - 1u]);
    axis = select(1u, 0u, maximumX - minimumX >= maximumY - minimumY);
  }
  // Preserve the exhaustive result for out-of-range finite latitudes too: its trigonometric
  // expression need not have a nonnegative longitude term there, so disable pruning.
  let canPrune = !GEOGRAPHIC || (minimumY >= -90.0 && maximumY <= 90.0);
  let center = vec2f(centerX, centerY);
  let centerKey = getPositionKey(select(centerX, centerY, axis == 1u));
  var lower = 0u;
  var upper = ZONES;
  while (lower < upper) {
    let middle = lower + (upper - lower) / 2u;
    if (getSortedKey(middle, axis) < centerKey) { lower = middle + 1u; }
    else { upper = middle; }
  }
  var left = lower;
  var right = lower;
  var filled = 0u;
  var selectedKeys: array<u32, LIST>;
  var selectedRows: array<u32, LIST>;
  loop {
    if (left == 0u && right == ZONES) { break; }
    var leftBound = 0xffffffffu;
    var rightBound = 0xffffffffu;
    if (left > 0u) { leftBound = getAxisBound(left - 1u, axis, center, canPrune); }
    if (right < ZONES) { rightBound = getAxisBound(right, axis, center, canPrune); }
    if (filled == LIST) {
      let worst = selectedKeys[LIST - 1u];
      // Strict comparison retains every separation-key tie, including lower row IDs.
      if (leftBound > worst) { left = 0u; }
      if (rightBound > worst) { right = ZONES; }
    }
    if (left == 0u && right == ZONES) { break; }
    var rank = right;
    if (left > 0u && (right == ZONES || leftBound <= rightBound)) {
      left--;
      rank = left;
    } else { right++; }
    let zone = getSortedRow(rank, axis);
    let key = getZoneSeparationKey(zone, centerX, centerY);
    var destination = min(filled, LIST - 1u);
    if (filled == LIST && (key > selectedKeys[destination] ||
        (key == selectedKeys[destination] && zone >= selectedRows[destination]))) { continue; }
    while (destination > 0u) {
      let previous = destination - 1u;
      if (selectedKeys[previous] < key || (selectedKeys[previous] == key && selectedRows[previous] < zone)) { break; }
      selectedKeys[destination] = selectedKeys[previous];
      selectedRows[destination] = selectedRows[previous];
      destination--;
    }
    selectedKeys[destination] = key;
    selectedRows[destination] = zone;
    filled = min(filled + 1u, LIST);
  }
  for (var slot = 0u; slot < LIST; slot++) {
    neighbors[neighborsOffset + row + slot] = selectedRows[slot];
    distances[distancesOffset + row + slot] = selectedKeys[slot];
  }`
    })
  ];
}
