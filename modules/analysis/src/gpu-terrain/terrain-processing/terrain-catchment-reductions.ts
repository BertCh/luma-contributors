// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  FIXED_POINT_WGSL,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  validateGPUStorageBindingByteLength,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {terrainBinding, TERRAIN_FINITE_WGSL} from './terrain-processing-utils';

/** A thread reduces at most this many consecutive sorted cells, regardless of basin size. */
export const TERRAIN_CATCHMENT_TILE_SIZE = 256;
/** Scratch and tile/lane products are indexed with uint32 in WGSL. */
const MAXIMUM_CATCHMENT_WORD_COUNT = 0xffffffff;

/** Checks shader index products at construction, before any scratch is allocated. @internal */
export function validateTerrainCatchmentReductionSizes(
  id: string,
  cellCount: number,
  basinCount: number,
  histogramBins: number
): void {
  for (const [name, length] of [
    ['basinCount x 9 sum words', basinCount * 9],
    ['basinCount x histogramBins x 3 histogram words', basinCount * histogramBins * 3],
    [
      'cell tiles x (histogramBins + 1)',
      Math.ceil(cellCount / TERRAIN_CATCHMENT_TILE_SIZE) * (histogramBins + 1)
    ]
  ] as const) {
    validateMaximumCount(id, name, length, MAXIMUM_CATCHMENT_WORD_COUNT);
  }
}

/** Checks every new row/basin-proportional binding before graph nodes are created. @internal */
export function validateTerrainCatchmentReductionBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  cellCount: number,
  basinCount: number,
  histogramBins: number
): void {
  for (const [name, length] of [
    ['basin metadata', basinCount * 5],
    ['basin sum words', basinCount * 9],
    ['histogram sum words', basinCount * histogramBins * 3],
    ['profile state', cellCount * 4]
  ] as const) {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      length * 4,
      'reduce grid size, basinCount or histogramBins'
    );
  }
}

/**
 * Sorted, bounded tile reductions. A tile visits at most 256 cells and emits one partial per
 * basin run. Fixed-point atomic carries join partials exactly, including signed elevation sums;
 * only a completed basin sum is rounded to float32. Histogram lanes use the same area unit as
 * the basin statistics. Empty basins and unused bins are initialized on every encode.
 * @internal
 */
export function createTerrainCatchmentReductionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    cellCount: number;
    basinCount: number;
    histogramBins: number;
    isUniform: boolean;
    area: string;
    declarations: string;
    inputs: readonly WGSLKernelBinding[];
    order: GraphDataView<'uint32'>;
    ranges: GraphDataView<'uint32'>;
    statistics: GraphDataView<'float32'>;
    hypsometry: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, cellCount, basinCount, histogramBins, isUniform, area, declarations, inputs} = props;
  const operation = 'GPUTerrainCatchmentSummary';
  const tileCount = Math.ceil(cellCount / TERRAIN_CATCHMENT_TILE_SIZE);
  const metadata = createTransientView(graph, `${id}-basin-metadata`, 'uint32', basinCount * 5);
  const sums = createTransientView(graph, `${id}-basin-sums`, 'uint32', basinCount * 9);
  const histogram = createTransientView(
    graph,
    `${id}-histogram-sums`,
    'uint32',
    basinCount * histogramBins * 3
  );
  const headroomBits = getFixedPointHeadroomBits(cellCount);
  const numericDeclarations = `${FIXED_POINT_WGSL}
fn getElevationKey(value: f32) -> u32 {
 let bits = bitcast<u32>(select(value, 0.0, value == 0.0));
 return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}
fn getElevationFromKey(key: u32) -> f32 {
 return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}`;
  const atomicBinding = (name: string, view: GraphDataView<'uint32'>): WGSLKernelBinding => ({
    name,
    view,
    type: 'atomic<u32>',
    access: 'read_write'
  });
  return [
    createWGSLKernelNode(graph, {
      id: `${id}-reduction-clear`,
      operation,
      invocationCount: Math.max(metadata.length, sums.length, histogram.length),
      bindings: [
        terrainBinding('metadata', metadata, 'read_write'),
        terrainBinding('sums', sums, 'read_write'),
        terrainBinding('histogram', histogram, 'read_write')
      ],
      body: `if (index < ${metadata.length}u) { metadata[metadataOffset + index] = select(0u, 0xffffffffu, index % 5u == 0u); }
if (index < ${sums.length}u) { sums[sumsOffset + index] = 0u; }
if (index < ${histogram.length}u) { histogram[histogramOffset + index] = 0u; }`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-metadata-tiles`,
      operation,
      invocationCount: tileCount,
      bindings: [
        ...inputs,
        terrainBinding('order', props.order),
        atomicBinding('metadata', metadata)
      ],
      declarations: `${declarations}\n${numericDeclarations}
fn publishMetadata(basin: u32, lower: u32, upper: u32, exponents: vec3<u32>) {
 if (basin >= ${basinCount}u) { return; }
 let offset = metadataOffset + basin * 5u;
 atomicMin(&metadata[offset], lower); atomicMax(&metadata[offset + 1u], upper);
 atomicMax(&metadata[offset + 2u], exponents.x); atomicMax(&metadata[offset + 3u], exponents.y); atomicMax(&metadata[offset + 4u], exponents.z);
}`,
      body: `var basin = ${basinCount}u; var lower = 0xffffffffu; var upper = 0u; var exponents = vec3<u32>(0u);
for (var local = 0u; local < ${TERRAIN_CATCHMENT_TILE_SIZE}u; local++) {
 let entry = index * ${TERRAIN_CATCHMENT_TILE_SIZE}u + local; if (entry >= ${cellCount}u) { break; }
 let cell = order[orderOffset + entry]; let nextBasin = basins[basinsOffset + cell]; let value = elevation[elevationOffset + cell];
 if (nextBasin >= ${basinCount}u || !isFiniteValue(value)) { break; }
 if (nextBasin != basin) { publishMetadata(basin, lower, upper, exponents); basin = nextBasin; lower = 0xffffffffu; upper = 0u; exponents = vec3<u32>(0u); }
 let key = getElevationKey(value); lower = min(lower, key); upper = max(upper, key);
 exponents = max(exponents, vec3<u32>(getFixedPointExponentBits(value), ${isUniform ? '0u' : 'getFixedPointExponentBits(getGroundCellArea(cell))'}, getFixedPointExponentBits(getStreamEdgeLength(cell))));
}
publishMetadata(basin, lower, upper, exponents);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-sum-tiles`,
      operation,
      invocationCount: tileCount * (histogramBins + 1),
      bindings: [
        ...inputs,
        terrainBinding('order', props.order),
        terrainBinding('metadata', metadata),
        atomicBinding('sums', sums),
        atomicBinding('histogram', histogram)
      ],
      declarations: `${declarations}\n${numericDeclarations}
${getFixedPointAtomicAddWGSL('sums')}
${getFixedPointAtomicAddWGSL('histogram')}
fn publishSums(basin: u32, lane: u32, elevationSum: vec3<u32>, areaSum: vec3<u32>, lengthSum: vec3<u32>, belowCount: u32) {
 if (basin >= ${basinCount}u) { return; }
 if (lane == 0u) { addFixedPointToSums(basin * 3u, elevationSum); addFixedPointToSums(basin * 3u + 1u, areaSum); addFixedPointToSums(basin * 3u + 2u, lengthSum); }
 else { ${isUniform ? `atomicAdd(&histogram[histogramOffset + (basin * ${histogramBins}u + lane - 1u) * 3u], belowCount);` : `addFixedPointToHistogram(basin * ${histogramBins}u + lane - 1u, areaSum);`} }
}`,
      body: `let tile = index / ${histogramBins + 1}u; let lane = index % ${histogramBins + 1}u;
var basin = ${basinCount}u; var elevationSum = vec3<u32>(0u); var areaSum = vec3<u32>(0u); var lengthSum = vec3<u32>(0u); var belowCount = 0u;
for (var local = 0u; local < ${TERRAIN_CATCHMENT_TILE_SIZE}u; local++) {
 let entry = tile * ${TERRAIN_CATCHMENT_TILE_SIZE}u + local; if (entry >= ${cellCount}u) { break; }
 let cell = order[orderOffset + entry]; let nextBasin = basins[basinsOffset + cell]; let value = elevation[elevationOffset + cell];
 if (nextBasin >= ${basinCount}u || !isFiniteValue(value)) { break; }
 if (nextBasin != basin) { publishSums(basin, lane, elevationSum, areaSum, lengthSum, belowCount); basin = nextBasin; elevationSum = vec3<u32>(0u); areaSum = vec3<u32>(0u); lengthSum = vec3<u32>(0u); belowCount = 0u; }
 let offset = metadataOffset + basin * 5u;
 let areaUnit = getFixedPointUnitExponent(metadata[offset + 3u], ${headroomBits}u);
 if (lane == 0u) {
  elevationSum = addFixedPoint(elevationSum, quantizeFixedPoint(value, getFixedPointUnitExponent(metadata[offset + 2u], ${headroomBits}u)));
  ${isUniform ? '' : 'areaSum = addFixedPoint(areaSum, quantizeFixedPoint(getGroundCellArea(cell), areaUnit));'}
  lengthSum = addFixedPoint(lengthSum, quantizeFixedPoint(getStreamEdgeLength(cell), getFixedPointUnitExponent(metadata[offset + 4u], ${headroomBits}u)));
 } else {
  let lower = getElevationFromKey(metadata[offset]); let upper = getElevationFromKey(metadata[offset + 1u]);
  let threshold = lower + (upper - lower) * f32(lane) / ${histogramBins}.0;
  if (value <= threshold || lane == ${histogramBins}u) { ${isUniform ? 'belowCount++;' : 'areaSum = addFixedPoint(areaSum, quantizeFixedPoint(getGroundCellArea(cell), areaUnit));'} }
 }
}
publishSums(basin, lane, elevationSum, areaSum, lengthSum, belowCount);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-statistics`,
      operation,
      invocationCount: basinCount,
      bindings: [
        terrainBinding('metadata', metadata),
        terrainBinding('sums', sums),
        terrainBinding('ranges', props.ranges),
        terrainBinding('statistics', props.statistics, 'read_write')
      ],
      declarations: `${TERRAIN_FINITE_WGSL}\n${numericDeclarations}
fn loadSum(slot: u32, exponent: u32) -> f32 {
 let offset = sumsOffset + slot * 3u;
 return getFixedPointFloat(vec3<u32>(sums[offset], sums[offset + 1u], sums[offset + 2u]), getFixedPointUnitExponent(exponent, ${headroomBits}u));
}`,
      body: `let sampleCount = ranges[rangesOffset + index + 1u] - ranges[rangesOffset + index];
let offset = metadataOffset + index * 5u;
let sum = loadSum(index * 3u, metadata[offset + 2u]);
let streamLength = loadSum(index * 3u + 2u, metadata[offset + 4u]);
let area = ${isUniform ? `f32(sampleCount) * ${area}` : 'loadSum(index * 3u + 1u, metadata[offset + 3u])'};
let destination = statisticsOffset + index * 6u;
statistics[destination] = area; statistics[destination + 1u] = select(getNaN(), sum / f32(max(sampleCount, 1u)), sampleCount > 0u);
statistics[destination + 2u] = select(getNaN(), getElevationFromKey(metadata[offset]), sampleCount > 0u);
statistics[destination + 3u] = select(getNaN(), getElevationFromKey(metadata[offset + 1u]), sampleCount > 0u);
statistics[destination + 4u] = streamLength; statistics[destination + 5u] = select(getNaN(), streamLength / max(area, 1e-30), area > 0.0);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-hypsometry`,
      operation,
      invocationCount: basinCount * histogramBins,
      bindings: [
        terrainBinding('metadata', metadata),
        terrainBinding('histogram', histogram),
        terrainBinding('hypsometry', props.hypsometry, 'read_write')
      ],
      declarations: FIXED_POINT_WGSL,
      body: `let offset = histogramOffset + index * 3u;
hypsometry[hypsometryOffset + index] = ${isUniform ? `f32(histogram[offset]) * ${area}` : `getFixedPointFloat(vec3<u32>(histogram[offset], histogram[offset + 1u], histogram[offset + 2u]), getFixedPointUnitExponent(metadata[metadataOffset + (index / ${histogramBins}u) * 5u + 3u], ${headroomBits}u))`};`
    })
  ];
}
