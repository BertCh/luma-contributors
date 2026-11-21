// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createSegmentSumNode} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {COMPENSATED_SUM_WGSL} from '../../analysis-core/kernels/compensated-sum-wgsl';
import {IS_NAN_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SHAPE_MOMENT_RANGE_WGSL} from './shape-moment-range-wgsl';
import {
  getRingBindings,
  getRingDeclarations,
  type FeatureRingInputs
} from './shape-descriptors-kernels';

/** Each invocation walks at most this many ordered edges before the fixed-order reductions. */
export const SHAPE_MOMENT_TILE_ROW_COUNT = 256;
/** Small features keep the low-overhead compensated reference; large row sets use fixed tiles. */
export const SHAPE_MOMENT_TILED_MINIMUM_ROW_COUNT = 4096;
/** Tile IDs and all scan offsets are uint32. */
const SHAPE_MOMENT_MAXIMUM_TILE_COUNT = 0xffffffff;

export function getShapeMomentTileCapacity(rowCount: number, ringCount: number): number {
  // sum ceil(ringRows / tileRows) <= floor(totalRows / tileRows) + ringCount.
  return Math.floor(rowCount / SHAPE_MOMENT_TILE_ROW_COUNT) + ringCount;
}

/** Auto retains the serial route if an unusually large ring table cannot fit the new planes. */
export function canUseShapeMomentTiles(
  device: Device,
  rowCount: number,
  ringCount: number
): boolean {
  const capacity = getShapeMomentTileCapacity(rowCount, ringCount);
  return (
    capacity <= SHAPE_MOMENT_MAXIMUM_TILE_COUNT &&
    Math.max(capacity * 4, ringCount * 8, (ringCount + 1) * 4) <=
      device.limits.maxStorageBufferBindingSize
  );
}

/** Must run before any enclosing contributor allocation. @internal */
export function validateShapeMomentTileScratch(
  device: Device,
  id: string,
  rowCount: number,
  ringCount: number
): void {
  const capacity = getShapeMomentTileCapacity(rowCount, ringCount);
  validateCapacity(id, 'moment tile capacity', capacity, 0, SHAPE_MOMENT_MAXIMUM_TILE_COUNT);
  for (const [name, byteLength] of [
    ['moment tiles', Math.max(capacity, 1) * 4],
    ['moment ring centroids', Math.max(ringCount, 1) * 8],
    ['moment ring offsets', (ringCount + 1) * 4]
  ] as const)
    validateGPUStorageBindingByteLength(
      device,
      id,
      name,
      byteLength,
      "reduce ring count or use momentMethod: 'serial'"
    );
}

/**
 * Fixed, source-ordered 256-edge tiles, then deterministic tree reductions over tiles/rings.
 * Ring roles and the first-ring signed area match the serial reference. Local tile sums carry
 * integer-derived addition errors; reductions use the shared fixed-order segment tree. No atomics
 * or scheduling order enter the floating-point result. Scratch is four scalar planes per tile,
 * ring, and feature, plus ring centroids/roles/offsets. @internal
 */
export function getShapeMomentTileNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: FeatureRingInputs & {
    centroids: GraphDataView<'float32x2'>;
    moments: GraphDataView<'float32x4'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, featureCount} = props;
  const ringCount = props.ringOffsets.length - 1;
  const capacity = getShapeMomentTileCapacity(props.positions.length, ringCount);
  validateShapeMomentTileScratch(graph.device, id, props.positions.length, ringCount);
  const unsigned = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
  const scalar = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
  const featureRings = unsigned('feature-rings', featureCount + 1);
  const tileCounts = unsigned('tile-counts', ringCount + 1);
  const tileOffsets = unsigned('tile-offsets', ringCount + 1);
  const ringRoles = unsigned('ring-roles', ringCount);
  const ringCentroids = createTransientView(
    graph,
    `${id}-ring-centroids`,
    'float32x2',
    Math.max(ringCount, 1)
  );
  const tiles = ['area', 'xx', 'yy', 'xy'].map(name => scalar(`tile-${name}`, capacity));
  const rings = ['area', 'xx', 'yy', 'xy'].map(name => scalar(`ring-${name}`, ringCount));
  const weightedArea = scalar('weighted-area', ringCount);
  const features = ['area', 'xx', 'yy', 'xy'].map(name => scalar(`feature-${name}`, featureCount));
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-feature-rings`,
      operation,
      variant: 'moment-feature-rings',
      invocationCount: featureCount + 1,
      bindings: [
        ...getRingBindings(props),
        {name: 'featureRings', view: featureRings, type: 'u32', access: 'read_write'}
      ],
      declarations: getRingDeclarations(props),
      body: 'featureRings[featureRingsOffset + index] = getRingStart(index);'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-ring-plan`,
      operation,
      variant: 'moment-ring-plan',
      invocationCount: ringCount + 1,
      bindings: [
        ...getRingBindings(props),
        {name: 'centroids', view: props.centroids, type: 'f32', access: 'read'},
        {name: 'ringCentroids', view: ringCentroids, type: 'f32', access: 'read_write'},
        {name: 'ringRoles', view: ringRoles, type: 'u32', access: 'read_write'},
        {name: 'tileCounts', view: tileCounts, type: 'u32', access: 'read_write'}
      ],
      declarations: `${getRingDeclarations(props)}\n${IS_NAN_BITS_WGSL}`,
      body: `tileCounts[tileCountsOffset + index] = 0u;
if (index == RING_COUNT) { return; }
let rowStart = getRingRowStart(index); let rowEnd = max(getRingRowEnd(index), rowStart);
let rows = select(0u, rowEnd - rowStart, rowEnd - rowStart >= 3u);
tileCounts[tileCountsOffset + index] = rows / ${SHAPE_MOMENT_TILE_ROW_COUNT}u + u32(rows % ${SHAPE_MOMENT_TILE_ROW_COUNT}u != 0u);
var low = 0u; var high = ${featureCount}u;
while (low < high) { let middle = low + (high - low + 1u) / 2u; if (getRingStart(middle) <= index) { low = middle; } else { high = middle - 1u; } }
var centroid = vec2<f32>(getNan(index)); var role = 0u;
if (low < ${featureCount}u && index >= getRingStart(low)) { centroid = vec2<f32>(centroids[centroidsOffset + 2u * low], centroids[centroidsOffset + 2u * low + 1u]); role = u32(isShellRing(index, index == getRingStart(low))); }
ringCentroids[ringCentroidsOffset + 2u * index] = centroid.x; ringCentroids[ringCentroidsOffset + 2u * index + 1u] = centroid.y; ringRoles[ringRolesOffset + index] = role;`
    }),
    ...new GPUScan({
      id: `${id}-tile-scan`,
      input: tileCounts,
      output: tileOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-tiles`,
      operation,
      variant: 'moment-tiles',
      invocationCount: capacity,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
        {name: 'tileOffsets', view: tileOffsets, type: 'u32', access: 'read'},
        {name: 'ringCentroids', view: ringCentroids, type: 'f32', access: 'read'},
        ...tiles.map((view, component) => ({
          name: `tileComponent${component}`,
          view,
          type: 'f32' as const,
          access: 'read_write' as const
        }))
      ],
      declarations: `${COMPENSATED_SUM_WGSL}\n${IS_NAN_BITS_WGSL}\n${SHAPE_MOMENT_RANGE_WGSL}
fn addMoment(accumulator: vec2<f32>, value: f32) -> vec2<f32> { return addCompensatedValue(accumulator, value); }
fn position(row: u32) -> vec2<f32> { return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]); }`,
      body: `var area = vec2<f32>(0.0); var xx = vec2<f32>(0.0); var yy = vec2<f32>(0.0); var xy = vec2<f32>(0.0);
let ringCount = arrayLength(&tileOffsets) - tileOffsetsOffset - 1u;
var low = 0u; var high = ringCount;
while (low < high) { let middle = low + (high - low + 1u) / 2u; if (tileOffsets[tileOffsetsOffset + middle] <= index) { low = middle; } else { high = middle - 1u; } }
if (low < ringCount) {
  let rowCount = (arrayLength(&positions) - positionsOffset) / 2u;
  let ringStart = min(ringOffsets[ringOffsetsOffset + low], rowCount); let ringEnd = max(min(ringOffsets[ringOffsetsOffset + low + 1u], rowCount), ringStart);
  let start = ringStart + (index - tileOffsets[tileOffsetsOffset + low]) * ${SHAPE_MOMENT_TILE_ROW_COUNT}u;
  let end = min(start + ${SHAPE_MOMENT_TILE_ROW_COUNT}u, ringEnd);
  let centroid = vec2<f32>(ringCentroids[ringCentroidsOffset + 2u * low], ringCentroids[ringCentroidsOffset + 2u * low + 1u]);
  if (start < end && !isNanBits(centroid.x) && !isNanBits(centroid.y)) {
    var previous = position(select(start - 1u, ringEnd - 1u, start == ringStart)) - centroid;
    for (var row = start; row < end; row++) {
      let current = position(row) - centroid; let cross = previous.x * current.y - current.x * previous.y;
      let terms = getShapeMomentTerms(previous, current);
      area = addMoment(area, cross);
      xx = addMoment(xx, terms.x);
      yy = addMoment(yy, terms.y);
      xy = addMoment(xy, terms.z);
      previous = current;
    }
  }
}
tileComponent0[tileComponent0Offset + index] = area.x + area.y; tileComponent1[tileComponent1Offset + index] = xx.x + xx.y; tileComponent2[tileComponent2Offset + index] = yy.x + yy.y; tileComponent3[tileComponent3Offset + index] = xy.x + xy.y;`
    })
  ];
  for (let component = 0; component < 4; component++)
    nodes.push(
      createSegmentSumNode(graph, {
        id: `${id}-ring-reduce-${component}`,
        operation,
        segmentCount: ringCount,
        input: tiles[component],
        segmentOffsets: tileOffsets,
        output: rings[component]
      })
    );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-ring-roles`,
      operation,
      variant: 'moment-ring-roles',
      invocationCount: ringCount,
      bindings: [
        {name: 'areas', view: rings[0], type: 'f32', access: 'read'},
        {name: 'roles', view: ringRoles, type: 'u32', access: 'read'},
        {name: 'weightedArea', view: weightedArea, type: 'f32', access: 'read_write'},
        ...rings.slice(1).map((view, component) => ({
          name: `component${component}`,
          view,
          type: 'f32' as const,
          access: 'read_write' as const
        }))
      ],
      body: `let area = areas[areasOffset + index]; let factor = select(-1.0, 1.0, roles[rolesOffset + index] != 0u) * sign(area);
weightedArea[weightedAreaOffset + index] = factor * area;
component0[component0Offset + index] *= factor; component1[component1Offset + index] *= factor; component2[component2Offset + index] *= factor;`
    })
  );
  for (let component = 0; component < 4; component++)
    nodes.push(
      createSegmentSumNode(graph, {
        id: `${id}-feature-reduce-${component}`,
        operation,
        segmentCount: featureCount,
        input: component === 0 ? weightedArea : rings[component],
        segmentOffsets: featureRings,
        output: features[component]
      })
    );
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-publish`,
      operation,
      variant: 'moment-publish',
      invocationCount: featureCount,
      bindings: [
        ...features.map((view, component) => ({
          name: `featureComponent${component}`,
          view,
          type: 'f32' as const,
          access: 'read' as const
        })),
        {name: 'featureRings', view: featureRings, type: 'u32', access: 'read'},
        {name: 'ringAreas', view: rings[0], type: 'f32', access: 'read'},
        {name: 'moments', view: props.moments, type: 'f32', access: 'read_write'}
      ],
      body: `let factor = select(1.0, -1.0, featureComponent0[featureComponent0Offset + index] < 0.0); let base = momentsOffset + 4u * index;
moments[base] = factor * featureComponent1[featureComponent1Offset + index]; moments[base + 1u] = factor * featureComponent2[featureComponent2Offset + index]; moments[base + 2u] = factor * featureComponent3[featureComponent3Offset + index];
let first = featureRings[featureRingsOffset + index]; var firstArea = 0.0; if (first < featureRings[featureRingsOffset + index + 1u]) { firstArea = 0.5 * ringAreas[ringAreasOffset + first]; } moments[base + 3u] = firstArea;`
    })
  );
  return nodes;
}
