// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_GRID_COORDINATE_WGSL,
  getViewBindingRange,
  getBoundedDispatchLayout
} from '@luma.gl/graph/gpu-core';
import type {
  GPUCommandGraph,
  GPUCommandGraphGPUIndirectCondition,
  GPUCommandNode,
  GPUComputeCommandNode,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {getGraphViewChunks} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getSegmentTreeSumWGSL,
  SEGMENT_PARTIAL_SUMS_WGSL,
  SEGMENT_WORKGROUP_SIZE
} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelProps,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPU_POINT_DENSITY_HEXAGON_WGSL} from '../../gpu-spatial-analysis/point-density/point-density-hexagon';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {RECTANGULAR_GRID_COORDINATE_WGSL} from '../../analysis-core/kernels/rectangular-grid-coordinate';

const OPERATION = 'GPUFlowAggregation';

/** Fixed lane count keeps zone and contribution pipelines independent of chunk length. @internal */
export const FLOW_CHUNK_WORKGROUP_SIZE = 256;
/** Guard the linear workgroup index before multiplying into a uint32 row index. */
const MAXIMUM_FLOW_CHUNK_WORKGROUP_COUNT = Math.floor(0xffffffff / FLOW_CHUNK_WORKGROUP_SIZE) + 1;

/**
 * Adapts the shared kernel's direct dispatch to a flow chunk's own bounded geometry. The shared
 * kernel's overrides are fixed; row count, row base and dispatch strides come from runtime slots.
 * The executable dispatches only this chunk's workgroups, independently of pipeline compilation.
 */
function createFlowChunkNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: Omit<WGSLKernelProps<Parameters>, 'invocationCount' | 'guardIndex' | 'workgroupSize'> & {
    chunkLength: number;
  }
): GPUCommandNode<Parameters> {
  const layout = getBoundedDispatchLayout(
    OPERATION,
    props.chunkLength,
    FLOW_CHUNK_WORKGROUP_SIZE,
    graph.device.limits.maxComputeWorkgroupsPerDimension
  );
  const node = createWGSLKernelNode(graph, {
    ...props,
    invocationCount: 1,
    guardIndex: false,
    workgroupSize: FLOW_CHUNK_WORKGROUP_SIZE,
    body: `let chunkWorkgroupIndex = (workgroupId.z * rowBase[rowBaseOffset + 4u] + workgroupId.y) * rowBase[rowBaseOffset + 3u] + workgroupId.x;
  if (chunkWorkgroupIndex >= ${MAXIMUM_FLOW_CHUNK_WORKGROUP_COUNT}u) { return; }
  let rowIndex = chunkWorkgroupIndex * ${FLOW_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  if (rowIndex >= rowBase[rowBaseOffset + 1u]) { return; }
  ${props.body}`
  }) as GPUComputeCommandNode<Parameters>;
  return {
    ...node,
    workload: {
      ...node.workload,
      maximumInvocationCount: props.chunkLength,
      maximumWorkgroupCount: layout.x * layout.y * layout.z
    },
    compile: (context): ReturnType<GPUComputeCommandNode<Parameters>['compile']> => {
      const executable = node.compile(context);
      return {
        ...executable,
        encode: encodeContext => {
          const pass = encodeContext.computePass;
          const chunkPass = new Proxy(pass, {
            get(currentPass, property) {
              if (property === 'dispatch')
                return () => currentPass.dispatch(layout.x, layout.y, layout.z);
              const value = Reflect.get(currentPass, property, currentPass);
              return typeof value === 'function' ? value.bind(currentPass) : value;
            }
          });
          executable.encode({...encodeContext, computePass: chunkPass});
        }
      };
    }
  };
}

/** Bounds resolved to either a literal or one GPU `float32x4` row. @internal */
export type FlowResolvedBounds =
  | readonly [number, number, number, number]
  | GraphDataView<'float32x4'>;

/**
 * Writes one zone ID per position of a chunk into `zones[rowBase[0] + rowIndex]`, or the no-zone
 * marker for rejected rows. Grid cells mirror `GPUGridBinning`; hexagons mirror point-density.
 *
 * @internal
 */
export function createFlowZoneNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    kind: 'grid' | 'hexagon';
    positions: GraphDataView;
    zones: GraphDataView<'uint32'>;
    rowBase: GraphDataView<'uint32'>;
    gridSize: readonly [number, number];
    bounds: FlowResolvedBounds;
    radius?: number | GraphDataView<'float32'>;
    /** Optional per-frame `[columns, rows]`, clamped to `1..gridSize` (the compile-time capacity). */
    activeGridSize?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const range = getViewBindingRange(props.positions);
  const positions = graph.createDataView(props.positions.buffer, {
    format: 'float32',
    byteOffset: range.offset,
    length: range.size / 4
  });
  const bindings: WGSLKernelBinding[] = [
    {name: 'positions', view: positions, type: 'f32', access: 'read'},
    {name: 'zones', view: props.zones, type: 'u32', access: 'read_write'},
    {name: 'rowBase', view: props.rowBase, type: 'u32', access: 'read'}
  ];
  const boundsIsView = !Array.isArray(props.bounds);
  if (boundsIsView) {
    bindings.push({
      name: 'boundsValues',
      view: props.bounds as GraphDataView,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.kind === 'hexagon' && typeof props.radius !== 'number' && props.radius) {
    bindings.push({
      name: 'radiusValues',
      view: props.radius,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.activeGridSize) {
    bindings.push({
      name: 'activeGridSizeValues',
      view: props.activeGridSize,
      type: 'u32',
      access: 'read'
    });
  }
  const literal = boundsIsView
    ? undefined
    : (props.bounds as readonly number[]).map(getWGSLFloatLiteral);
  const boundsSource = literal
    ? `let minimumX = ${literal[0]}; let minimumY = ${literal[1]};
  let maximumX = ${literal[2]}; let maximumY = ${literal[3]};`
    : `let minimumX = boundsValues[boundsValuesOffset];
  let minimumY = boundsValues[boundsValuesOffset + 1u];
  let maximumX = boundsValues[boundsValuesOffset + 2u];
  let maximumY = boundsValues[boundsValuesOffset + 3u];`;
  const radiusSource =
    typeof props.radius === 'number'
      ? `let radius = ${getWGSLFloatLiteral(props.radius)};`
      : 'let radius = radiusValues[radiusValuesOffset];';
  const zoneSource =
    props.kind === 'hexagon'
      ? `${radiusSource}
  let validRadius = radius > 0.0 && isFiniteValue(radius);
  if (finite && inside && validRadius) {
    let cell = getPointDensityHexagonCell(x, y, minimumX, minimumY, radius);
    if (cell.x >= 0 && cell.y >= 0 && u32(cell.x) < columns && u32(cell.y) < rows) {
      zone = u32(cell.y) * columns + u32(cell.x);
    }
  }`
      : `if (finite && inside) {
    zone = getRectangularGridCoordinate(y, minimumY, maximumY, rows) * columns +
      getRectangularGridCoordinate(x, minimumX, maximumX, columns);
  }`;
  return createFlowChunkNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: `${props.kind}-zones`,
    bindings,
    chunkLength: props.positions.length,
    declarations: `const COLUMNS: u32 = ${props.gridSize[0]}u;
const ROWS: u32 = ${props.gridSize[1]}u;
const NO_ZONE: u32 = 0xffffffffu;
${IS_FINITE_VALUE_WGSL}
${props.kind === 'hexagon' ? GPU_POINT_DENSITY_HEXAGON_WGSL : ''}
${GPU_GRID_COORDINATE_WGSL}
${props.kind === 'grid' ? RECTANGULAR_GRID_COORDINATE_WGSL : ''}`,
    body: `${boundsSource}
  ${
    props.activeGridSize
      ? `let columns = clamp(activeGridSizeValues[activeGridSizeValuesOffset], 1u, COLUMNS);
  let rows = clamp(activeGridSizeValues[activeGridSizeValuesOffset + 1u], 1u, ROWS);`
      : 'let columns = COLUMNS;\n  let rows = ROWS;'
  }
  let x = positions[positionsOffset + rowBase[rowBaseOffset + 2u] + rowIndex * 2u];
  let y = positions[positionsOffset + rowBase[rowBaseOffset + 2u] + rowIndex * 2u + 1u];
  var zone = NO_ZONE;
  let finite = isFiniteValue(x) && isFiniteValue(y);
  let inX = x >= minimumX && x <= maximumX && (maximumX != minimumX || x == minimumX);
  let inY = y >= minimumY && y <= maximumY && (maximumY != minimumY || y == minimumY);
  let inside = maximumX >= minimumX && maximumY >= minimumY && inX && inY;
  ${zoneSource}
  zones[zonesOffset + rowBase[rowBaseOffset] + rowIndex] = zone;`
  });
}

/**
 * Combines zones, mask, and time mask into one pair key per row (`0xffffffff` when rejected) and
 * writes accepted origin and destination zones (no-zone when rejected).
 *
 * @internal
 */
export function createFlowPairKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    originZones: GraphDataView<'uint32'>;
    destinationZones: GraphDataView<'uint32'>;
    mask?: GraphDataView<'uint32'>;
    timeMask?: GraphDataView<'uint32'>;
    pairKeys: GraphDataView<'uint32'>;
    acceptedOriginZones: GraphDataView<'uint32'>;
    acceptedDestinationZones: GraphDataView<'uint32'>;
    zoneCount: number;
    useRowKeys?: boolean;
    excludeSelfFlows: boolean;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'originZones',
      view: props.originZones,
      type: 'u32',
      access: 'read'
    },
    {
      name: 'destinationZones',
      view: props.destinationZones,
      type: 'u32',
      access: 'read'
    }
  ];
  if (props.mask) {
    bindings.push({
      name: 'rowMask',
      view: props.mask,
      type: 'u32',
      access: 'read'
    });
  }
  if (props.timeMask) {
    bindings.push({
      name: 'timeMask',
      view: props.timeMask,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push(
    {
      name: 'pairKeys',
      view: props.pairKeys,
      type: 'u32',
      access: 'read_write'
    },
    {
      name: 'acceptedOriginZones',
      view: props.acceptedOriginZones,
      type: 'u32',
      access: 'read_write'
    },
    {
      name: 'acceptedDestinationZones',
      view: props.acceptedDestinationZones,
      type: 'u32',
      access: 'read_write'
    }
  );
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pair-keys',
    bindings,
    invocationCount: props.pairKeys.length,
    declarations: `const ZONE_COUNT: u32 = ${props.zoneCount}u;
const NO_ZONE: u32 = 0xffffffffu;`,
    body: `let origin = originZones[originZonesOffset + index];
  let destination = destinationZones[destinationZonesOffset + index];
  var accepted = origin < ZONE_COUNT && destination < ZONE_COUNT;
  ${props.excludeSelfFlows ? 'accepted = accepted && origin != destination;' : ''}
  ${props.mask ? 'accepted = accepted && rowMask[rowMaskOffset + index] != 0u;' : ''}
  ${props.timeMask ? 'accepted = accepted && timeMask[timeMaskOffset + index] != 0u;' : ''}
  pairKeys[pairKeysOffset + index] = select(NO_ZONE, ${props.useRowKeys ? 'index' : 'origin * ZONE_COUNT + destination'}, accepted);
  acceptedOriginZones[acceptedOriginZonesOffset + index] = select(NO_ZONE, origin, accepted);
  acceptedDestinationZones[acceptedDestinationZonesOffset + index] =
    select(NO_ZONE, destination, accepted);`
  });
}

/**
 * slotKeys[slot] = tableKeys[slot] and slotIndices[slot] = slot, one row per hash slot. Empty slots
 * get the key `zoneCount * zoneCount`, one above every pair key, so the sort needs only
 * `ceil(log2(zoneCount^2 + 1))` key bits instead of 32 and empty slots still sort last in slot
 * order. @internal
 */
export function createFlowSlotKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    tableKeys: GraphDataView<'uint32'>;
    slotKeys: GraphDataView<'uint32'>;
    slotIndices: GraphDataView<'uint32'>;
    zoneCount: number;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'slot-keys',
    bindings: [
      {name: 'tableKeys', view: props.tableKeys, type: 'u32', access: 'read'},
      {
        name: 'slotKeys',
        view: props.slotKeys,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'slotIndices',
        view: props.slotIndices,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.tableKeys.length,
    declarations: `const EMPTY_SLOT_KEY: u32 = ${props.zoneCount * props.zoneCount}u;`,
    body: `let key = tableKeys[tableKeysOffset + index];
  slotKeys[slotKeysOffset + index] = select(key, EMPTY_SLOT_KEY, key == 0xffffffffu);
  slotIndices[slotIndicesOffset + index] = index;`
  });
}

/**
 * Writes one ascending sort key per pair-key-ordered slot such that ascending order is descending
 * weight (or count). With weights the key is `0xffffffff` only for empty slots: finite weight sums
 * never map to the all-ones ordered pattern. With counts the key is `rowCounts.length - count`
 * and empty slots take `rowCounts.length` (see {@link getFlowWeightKeyBits}).
 *
 * @internal
 */
export function createFlowWeightKeysNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    slotsByPair: GraphDataView<'uint32'>;
    tableKeys: GraphDataView<'uint32'>;
    tableValues: GraphDataView<'uint32'>;
    rowCounts: GraphDataView<'uint32'>;
    rowWeights?: GraphDataView<'float32'>;
    weightKeys: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {
      name: 'slotsByPair',
      view: props.slotsByPair,
      type: 'u32',
      access: 'read'
    },
    {name: 'tableKeys', view: props.tableKeys, type: 'u32', access: 'read'},
    {
      name: 'tableValues',
      view: props.tableValues,
      type: 'u32',
      access: 'read'
    }
  ];
  // Only the binding that the generated WGSL uses may be declared: unused bindings are dropped
  // from the pipeline layout.
  if (props.rowWeights) {
    bindings.push({
      name: 'rowWeights',
      view: props.rowWeights,
      type: 'f32',
      access: 'read'
    });
  } else {
    bindings.push({
      name: 'rowCounts',
      view: props.rowCounts,
      type: 'u32',
      access: 'read'
    });
  }
  bindings.push({
    name: 'weightKeys',
    view: props.weightKeys,
    type: 'u32',
    access: 'read_write'
  });
  const keySource = props.rowWeights
    ? `var weight = rowWeights[rowWeightsOffset + row];
      weight = select(weight, 0.0, weight == 0.0);
      let bits = bitcast<u32>(weight);
      let ordered = bits ^ select(0x80000000u, 0xffffffffu, (bits >> 31u) != 0u);
      key = ~ordered;`
    : 'key = ROW_TOTAL - rowCounts[rowCountsOffset + row];';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'weight-keys',
    bindings,
    invocationCount: props.slotsByPair.length,
    // Without weights the key is `ROW_TOTAL - count`: ascending is descending count, a stored pair
    // has at least one and at most `ROW_TOTAL` rows, so stored keys stay below `ROW_TOTAL`, which
    // empty slots take. The sort then needs only `bits(ROW_TOTAL)` key bits.
    declarations: props.rowWeights
      ? undefined
      : `const ROW_TOTAL: u32 = ${props.rowCounts.length}u;`,
    body: `let slot = slotsByPair[slotsByPairOffset + index];
  var key = ${props.rowWeights ? '0xffffffffu' : 'ROW_TOTAL'};
  if (tableKeys[tableKeysOffset + slot] != 0xffffffffu) {
    let row = tableValues[tableValuesOffset + slot];
    ${keySource}
  }
  weightKeys[weightKeysOffset + index] = key;`
  });
}

/** Writes the top-K pair keys of `output.ids`, sentinel for rows past the count. @internal */
export function createFlowGatherIdsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    statistics: GraphDataView<'uint32'>;
    rankedSlots: GraphDataView<'uint32'>;
    tableKeys: GraphDataView<'uint32'>;
    ids: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'gather-ids',
    bindings: [
      {
        name: 'statistics',
        view: props.statistics,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'rankedSlots',
        view: props.rankedSlots,
        type: 'u32',
        access: 'read'
      },
      {name: 'tableKeys', view: props.tableKeys, type: 'u32', access: 'read'},
      {name: 'ids', view: props.ids, type: 'u32', access: 'read_write'}
    ],
    invocationCount: props.ids.length,
    declarations: `const CAPACITY: u32 = ${props.ids.length}u;
const SLOT_COUNT: u32 = ${props.rankedSlots.length}u;`,
    body: `let count = min(statistics[statisticsOffset], min(CAPACITY, SLOT_COUNT));
  var id = 0xffffffffu;
  if (index < count) {
    id = tableKeys[tableKeysOffset + rankedSlots[rankedSlotsOffset + index]];
  }
  ids[idsOffset + index] = id;`
  });
}

/** Writes the top-K flow counts and weights, zero for rows past the count. @internal */
export function createFlowGatherValuesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    capacity: number;
    statistics: GraphDataView<'uint32'>;
    rankedSlots: GraphDataView<'uint32'>;
    tableValues: GraphDataView<'uint32'>;
    rowCounts: GraphDataView<'uint32'>;
    rowWeights?: GraphDataView<'float32'>;
    flowCounts?: GraphDataView<'uint32'>;
    flowWeights?: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'statistics', view: props.statistics, type: 'u32', access: 'read'},
    {
      name: 'rankedSlots',
      view: props.rankedSlots,
      type: 'u32',
      access: 'read'
    },
    {
      name: 'tableValues',
      view: props.tableValues,
      type: 'u32',
      access: 'read'
    }
  ];
  const usesRowWeights = Boolean(props.rowWeights && props.flowWeights);
  if (props.flowCounts || (props.flowWeights && !usesRowWeights)) {
    bindings.push({
      name: 'rowCounts',
      view: props.rowCounts,
      type: 'u32',
      access: 'read'
    });
  }
  if (props.rowWeights && usesRowWeights) {
    bindings.push({
      name: 'rowWeights',
      view: props.rowWeights,
      type: 'f32',
      access: 'read'
    });
  }
  if (props.flowCounts) {
    bindings.push({
      name: 'flowCounts',
      view: props.flowCounts,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.flowWeights) {
    bindings.push({
      name: 'flowWeights',
      view: props.flowWeights,
      type: 'f32',
      access: 'read_write'
    });
  }
  const weightSource = usesRowWeights
    ? 'rowWeights[rowWeightsOffset + row]'
    : 'f32(rowCounts[rowCountsOffset + row])';
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'gather-values',
    bindings,
    invocationCount: props.capacity,
    declarations: `const CAPACITY: u32 = ${props.capacity}u;
const SLOT_COUNT: u32 = ${props.rankedSlots.length}u;`,
    body: `let count = min(statistics[statisticsOffset], min(CAPACITY, SLOT_COUNT));
  var flowCount = 0u;
  var flowWeight = 0.0;
  if (index < count) {
    let row = tableValues[tableValuesOffset + rankedSlots[rankedSlotsOffset + index]];
    ${props.flowCounts ? 'flowCount = rowCounts[rowCountsOffset + row];' : ''}
    ${props.flowWeights ? `flowWeight = ${weightSource};` : ''}
  }
  ${props.flowCounts ? 'flowCounts[flowCountsOffset + index] = flowCount;' : ''}
  ${props.flowWeights ? 'flowWeights[flowWeightsOffset + index] = flowWeight;' : ''}`
  });
}

/** Splits output pair keys into origin and destination zone columns. @internal */
export function createFlowDecodeZonesNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    zoneCount: number;
    ids: GraphDataView<'uint32'>;
    sourceOrigins?: GraphDataView<'uint32'>;
    sourceDestinations?: GraphDataView<'uint32'>;
    originZoneIds?: GraphDataView<'uint32'>;
    destinationZoneIds?: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  const bindings: WGSLKernelBinding[] = [
    {name: 'ids', view: props.ids, type: 'u32', access: 'read'}
  ];
  if (props.sourceOrigins && props.sourceDestinations) {
    bindings.push(
      {name: 'sourceOrigins', view: props.sourceOrigins, type: 'u32', access: 'read'},
      {name: 'sourceDestinations', view: props.sourceDestinations, type: 'u32', access: 'read'}
    );
  }
  if (props.originZoneIds) {
    bindings.push({
      name: 'originOut',
      view: props.originZoneIds,
      type: 'u32',
      access: 'read_write'
    });
  }
  if (props.destinationZoneIds) {
    bindings.push({
      name: 'destinationOut',
      view: props.destinationZoneIds,
      type: 'u32',
      access: 'read_write'
    });
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'decode-zones',
    bindings,
    invocationCount: props.ids.length,
    declarations: `const ZONE_COUNT: u32 = ${props.zoneCount}u;`,
    body: `let key = ids[idsOffset + index];
  let valid = key != 0xffffffffu;
  let sourceRow = select(0u, key, valid);
  ${props.originZoneIds ? `originOut[originOutOffset + index] = select(0xffffffffu, ${props.sourceOrigins ? 'sourceOrigins[sourceOriginsOffset + sourceRow]' : 'key / ZONE_COUNT'}, valid);` : ''}
  ${props.destinationZoneIds ? `destinationOut[destinationOutOffset + index] = select(0xffffffffu, ${props.sourceDestinations ? 'sourceDestinations[sourceDestinationsOffset + sourceRow]' : 'key % ZONE_COUNT'}, valid);` : ''}`
  });
}

/** pairOverflow[0] = 1 when any accepted row's pair could not be stored. @internal */
export function createFlowPairOverflowNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    statistics: GraphDataView<'uint32'>;
    pairOverflow: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pair-overflow',
    bindings: [
      {
        name: 'statistics',
        view: props.statistics,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'pairOverflow',
        view: props.pairOverflow,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: 1,
    body: 'pairOverflow[pairOverflowOffset] = select(0u, 1u, statistics[statisticsOffset + 2u] != 0u);'
  });
}

/**
 * Writes one finite-or-zero weight per source row into `contributions` (one node per weight chunk),
 * so non-finite weights add 0 to every requested sum, matching the atomic path's skip policy.
 *
 * @internal
 */
export function createFlowContributionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    weights: GraphDataView<'float32'> | GraphVectorView<'float32'>;
    contributions: GraphDataView<'float32'>;
    rowBases: readonly GraphDataView<'uint32'>[];
  }
): GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const chunks = getGraphViewChunks(props.weights);
  for (const [chunkIndex, chunk] of chunks.entries()) {
    if (chunk.length > 0) {
      const range = getViewBindingRange(chunk);
      const weights = graph.createDataView(chunk.buffer, {
        format: 'float32',
        byteOffset: range.offset,
        length: range.size / 4
      });
      nodes.push(
        createFlowChunkNode<Parameters>(graph, {
          id: `${props.id}-contributions${chunks.length > 1 ? `-chunk-${chunkIndex}` : ''}`,
          operation: OPERATION,
          variant: 'contributions',
          bindings: [
            {name: 'weights', view: weights, type: 'f32', access: 'read'},
            {
              name: 'contributions',
              view: props.contributions,
              type: 'f32',
              access: 'read_write'
            },
            {name: 'rowBase', view: props.rowBases[chunkIndex], type: 'u32', access: 'read'}
          ],
          chunkLength: chunk.length,
          declarations: IS_FINITE_VALUE_WGSL,
          body: `let weight = weights[weightsOffset + rowBase[rowBaseOffset + 2u] + rowIndex];
  let isFinite = isFiniteValue(weight);
  contributions[contributionsOffset + rowBase[rowBaseOffset] + rowIndex] = select(0.0, weight, isFinite);`
        })
      );
    }
  }
  return nodes;
}

/** Sort keys for the per-pair sum: representative row (clamped to `rowCount`) and row index. @internal */
export function createFlowPairSortPrepareNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    representativeRows: GraphDataView<'uint32'>;
    sortKeys: GraphDataView<'uint32'>;
    sortIndices: GraphDataView<'uint32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pair-sum-sort-prepare',
    bindings: [
      {
        name: 'representativeRows',
        view: props.representativeRows,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'sortKeys',
        view: props.sortKeys,
        type: 'u32',
        access: 'read_write'
      },
      {
        name: 'sortIndices',
        view: props.sortIndices,
        type: 'u32',
        access: 'read_write'
      }
    ],
    invocationCount: props.representativeRows.length,
    declarations: `const ROW_COUNT: u32 = ${props.representativeRows.length}u;`,
    // Rows whose pair was not stored have representative row 0xffffffff and sort last.
    body: `sortKeys[sortKeysOffset + index] = min(representativeRows[representativeRowsOffset + index], ROW_COUNT);
  sortIndices[sortIndicesOffset + index] = index;`
  });
}

/**
 * Sums each stored pair's contributions with one 256-thread workgroup per stored pair: a strided
 * per-thread partial sum, then a fixed binary tree (the same order as the zonal-statistics segment
 * sum). The pair's rows are the segment `[rowOffsets[r], rowOffsets[r] + rowCounts[r])` of the
 * sorted contributions, where `r` is the pair's representative row; the sum is written to
 * `rowWeights[r]`. The order depends only on those bounds, so the result is bitwise reproducible.
 *
 * Workgroup `w` sums the stored slot `slotsByPair[w]` (stored slots sort first). The node is
 * dispatched indirectly by the stored-pair count (`statistics[0]` of the hash build) through
 * `condition`, so empty table slots launch no workgroups (review B-18).
 *
 * @internal
 */
export function createFlowPairSumNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    tableKeys: GraphDataView<'uint32'>;
    tableValues: GraphDataView<'uint32'>;
    rowCounts: GraphDataView<'uint32'>;
    rowOffsets: GraphDataView<'uint32'>;
    sortedContributions: GraphDataView<'float32'>;
    rowWeights: GraphDataView<'float32'>;
    /** Table slots ordered by pair key, stored slots first. */
    slotsByPair: GraphDataView<'uint32'>;
    /** Hash build statistics; word 0 is the stored-pair count. */
    statistics: GraphDataView<'uint32'>;
    /** Indirect dispatch sized by the stored-pair count. */
    condition: GPUCommandGraphGPUIndirectCondition;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'pair-sum',
    condition: props.condition,
    bindings: [
      {name: 'slotsByPair', view: props.slotsByPair, type: 'u32', access: 'read'},
      {name: 'statistics', view: props.statistics, type: 'u32', access: 'read'},
      {name: 'tableKeys', view: props.tableKeys, type: 'u32', access: 'read'},
      {
        name: 'tableValues',
        view: props.tableValues,
        type: 'u32',
        access: 'read'
      },
      {name: 'rowCounts', view: props.rowCounts, type: 'u32', access: 'read'},
      {
        name: 'rowOffsets',
        view: props.rowOffsets,
        type: 'u32',
        access: 'read'
      },
      {
        name: 'sortedContributions',
        view: props.sortedContributions,
        type: 'f32',
        access: 'read'
      },
      {
        name: 'rowWeights',
        view: props.rowWeights,
        type: 'f32',
        access: 'read_write'
      }
    ],
    workgroupSize: SEGMENT_WORKGROUP_SIZE,
    invocationCount: props.tableKeys.length * SEGMENT_WORKGROUP_SIZE,
    guardIndex: false,
    declarations: `const INPUT_COUNT: u32 = ${props.sortedContributions.length}u;
${SEGMENT_PARTIAL_SUMS_WGSL}`,
    // No early return: every invocation of a workgroup must reach the barriers.
    body: `let pairIndex = index / ${SEGMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT && pairIndex < statistics[statisticsOffset];
  var partialSum = 0.0;
  var row = 0u;
  var isStored = false;
  let slot = select(0u, slotsByPair[slotsByPairOffset + pairIndex], isInRange);
  if (isInRange && tableKeys[tableKeysOffset + slot] != 0xffffffffu) {
    isStored = true;
    row = tableValues[tableValuesOffset + slot];
    let begin = rowOffsets[rowOffsetsOffset + row];
    let end = min(begin + rowCounts[rowCountsOffset + row], INPUT_COUNT);
    for (var position = begin + localInvocationIndex; position < end; position += ${SEGMENT_WORKGROUP_SIZE}u) {
      partialSum += sortedContributions[sortedContributionsOffset + position];
    }
  }
  ${getSegmentTreeSumWGSL()}
  if (isStored && localInvocationIndex == 0u) {
    rowWeights[rowWeightsOffset + row] = partialSums[0];
  }`
  });
}

/**
 * Writes a `float32` `[minimum, maximum]` pair from a `uint32` `[minimum, maximum]` pair, so the
 * extent of integer zone counts can feed float color ranges directly.
 *
 * @internal
 */
export function createFlowExtentToFloatNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    extent: GraphDataView<'uint32'>;
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    id: props.id,
    operation: OPERATION,
    variant: 'extent-to-float',
    bindings: [
      {name: 'extentIn', view: props.extent, type: 'u32', access: 'read'},
      {name: 'extentOut', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: 2,
    body: `extentOut[extentOutOffset + index] = f32(extentIn[extentInOffset + index]);`
  });
}

/** Radix key bits that {@link createFlowWeightKeysNode} needs: 32 with weights, else count bits. @internal */
export function getFlowWeightKeyBits(hasWeights: boolean, rowTotal: number): number {
  return hasWeights ? 32 : Math.max(1, Math.floor(rowTotal).toString(2).length);
}
