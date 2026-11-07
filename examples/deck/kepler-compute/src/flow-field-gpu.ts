// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GPU-resident Flow Field velocity grid. Samples are uploaded once as four float32 columns;
 * every filter revision then runs compute passes only (clear, mask-aware int32 fixed-point
 * binning, cell means, optional downhill gradient, box smoothing, maximum-speed reduction,
 * velocity scale) and copies the result into the `FlowParticleSimulation` rgba32float texture
 * with a buffer-to-texture copy. Nothing is read back: not the mask, not the grid, not the
 * maximum speed. `flow-field-gpu-reference.ts` is the node-tested CPU design of these passes.
 */

import {Buffer, type Device, type Texture} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import type {GPUVector} from '@luma.gl/graph/gpu-data';
import type {KeplerMemoryReservation} from '@luma-contributors/kepler';
import {
  METERS_PER_DEGREE_LATITUDE,
  type FlowFieldColumnMode,
  type FlowFieldSamples
} from './flow-field-grid';
import {
  getFlowFieldFixedPoint,
  type FlowFieldExtent,
  type FlowFieldFixedPoint
} from './flow-field-gpu-reference';
import {MAXIMUM_FLOW_FIELD_LATITUDE} from './flow-field-grid';

const WORKGROUP_SIZE = 128;
const MAXIMUM_WORKGROUPS_PER_DIMENSION = 65535;
const MODE_CODES: Record<FlowFieldColumnMode, number> = {UV: 0, SPEED_DIR: 1, ELEVATION: 2};
/** Uniform words (vec4<u32> x 6); floats are stored as bit patterns. */
const PARAMETER_WORDS = 32;
/** Bit pattern of the largest finite float32 (initial minimum of the extent reduction). */
const MAXIMUM_FLOAT_BITS = 0x7f7fffff;

const HEADER = /* wgsl */ `
struct Parameters { words: array<vec4<u32>, 8> };
@group(0) @binding(0) var<uniform> parameters: Parameters;
fn getWord(index: u32) -> u32 { let vector = parameters.words[index / 4u]; return vector[index % 4u]; }
fn getFloat(index: u32) -> f32 { return bitcast<f32>(getWord(index)); }
fn getInvocation(workgroupId: vec3<u32>, localIndex: u32) -> u32 {
  return (workgroupId.y * ${MAXIMUM_WORKGROUPS_PER_DIMENSION}u + workgroupId.x) * ${WORKGROUP_SIZE}u + localIndex;
}
fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
`;
// Word indices: 0 mode, 1 columns, 2 rows, 3 chunkLength, 4 rowStart, 5 maskWordOffset, 6 useMask,
// 7 smoothRadius, 8 spanLongitude, 9 spanLatitude, 10 west, 11 south, 12 east, 13 north,
// 14 scaleFirst, 15 scaleSecond, 16 centerFirst, 17 centerSecond, 18 velocityNumerator, 19 paddedColumns,
// 20 hasAltitude, 21 altitudeCenter, 22 altitudeScale, 23 longitudeShift, 24 latitudeShift (active bounds vs full).
const ENTRY = (body: string) => /* wgsl */ `
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>, @builtin(local_invocation_index) localIndex: u32) {
  let index = getInvocation(workgroupId, localIndex);
  let columns = getWord(1u);
  let rows = getWord(2u);
  let cellCount = columns * rows;
${body}
}`;

const SHADERS = {
  clear:
    HEADER +
    /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> sums: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> maximumSpeed: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> altitudeSums: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> altitudeCounts: array<atomic<u32>>;` +
    ENTRY(`
  if (index >= cellCount) { return; }
  atomicStore(&sums[index * 2u], 0);
  atomicStore(&sums[index * 2u + 1u], 0);
  atomicStore(&counts[index], 0u);
  if (getWord(20u) == 1u) {
    atomicStore(&altitudeSums[index], 0);
    atomicStore(&altitudeCounts[index], 0u);
  }
  if (index == 0u) {
    atomicStore(&maximumSpeed[0], 0u);
    atomicStore(&maximumSpeed[1], ${MAXIMUM_FLOAT_BITS}u);
  }`),
  extent:
    HEADER +
    /* wgsl */ `
@group(0) @binding(1) var<storage, read> longitudes: array<f32>;
@group(0) @binding(2) var<storage, read> latitudes: array<f32>;
@group(0) @binding(3) var<storage, read> selection: array<u32>;
@group(0) @binding(4) var<storage, read_write> extentWords: array<atomic<u32>>;` +
    ENTRY(`
  if (index >= getWord(3u)) { return; }
  if (getWord(6u) == 1u && selection[getWord(5u) + index] == 0u) { return; }
  let row = getWord(4u) + index;
  let longitude = longitudes[row];
  if (!isFiniteValue(longitude)) { return; }
  // Offsets from the full-extent origin are non-negative: float32 bit patterns order like values.
  let latitude = latitudes[row];
  atomicMin(&extentWords[0], bitcast<u32>(longitude));
  atomicMin(&extentWords[1], bitcast<u32>(latitude));
  atomicMax(&extentWords[2], bitcast<u32>(longitude));
  atomicMax(&extentWords[3], bitcast<u32>(latitude));
  atomicAdd(&extentWords[4], 1u);`),
  bin:
    HEADER +
    /* wgsl */ `
@group(0) @binding(1) var<storage, read> longitudes: array<f32>;
@group(0) @binding(2) var<storage, read> latitudes: array<f32>;
@group(0) @binding(3) var<storage, read> firstValues: array<f32>;
@group(0) @binding(4) var<storage, read> secondValues: array<f32>;
@group(0) @binding(5) var<storage, read> selection: array<u32>;
@group(0) @binding(6) var<storage, read_write> sums: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> counts: array<atomic<u32>>;` +
    ENTRY(`
  if (index >= getWord(3u)) { return; }
  if (getWord(6u) == 1u && selection[getWord(5u) + index] == 0u) { return; }
  let row = getWord(4u) + index;
  let sourceLongitude = longitudes[row];
  // Rows that were not finite or outside the latitude domain were uploaded as NaN.
  if (!isFiniteValue(sourceLongitude)) { return; }
  // Offsets are relative to the full extent; the active bounds follow the filtered rows.
  let longitude = sourceLongitude - getFloat(23u);
  let latitude = latitudes[row] - getFloat(24u);
  let column = min(columns - 1u, u32(floor(longitude / getFloat(8u) * f32(columns))));
  let cellRow = min(rows - 1u, u32(floor(latitude / getFloat(9u) * f32(rows))));
  let cell = cellRow * columns + column;
  var valueFirst = firstValues[row];
  var valueSecond = secondValues[row];
  let mode = getWord(0u);
  if (mode == 1u) {
    let angle = valueSecond * 0.017453292519943295;
    let speed = valueFirst;
    valueFirst = -speed * sin(angle);
    valueSecond = -speed * cos(angle);
  } else if (mode == 2u) {
    valueSecond = 0.0;
  }
  atomicAdd(&sums[cell * 2u], i32(floor((valueFirst - getFloat(16u)) * getFloat(14u) + 0.5)));
  atomicAdd(&sums[cell * 2u + 1u], i32(floor((valueSecond - getFloat(17u)) * getFloat(15u) + 0.5)));
  atomicAdd(&counts[cell], 1u);`),
  // Separate from `bin` to stay within the default eight storage buffers per stage.
  binAltitude:
    HEADER +
    /* wgsl */ `
@group(0) @binding(1) var<storage, read> longitudes: array<f32>;
@group(0) @binding(2) var<storage, read> latitudes: array<f32>;
@group(0) @binding(3) var<storage, read> altitudes: array<f32>;
@group(0) @binding(4) var<storage, read> selection: array<u32>;
@group(0) @binding(5) var<storage, read_write> altitudeSums: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> altitudeCounts: array<atomic<u32>>;` +
    ENTRY(`
  if (index >= getWord(3u)) { return; }
  if (getWord(6u) == 1u && selection[getWord(5u) + index] == 0u) { return; }
  let row = getWord(4u) + index;
  let sourceLongitude = longitudes[row];
  if (!isFiniteValue(sourceLongitude)) { return; }
  let altitude = altitudes[row];
  if (!isFiniteValue(altitude)) { return; }
  let longitude = sourceLongitude - getFloat(23u);
  let latitude = latitudes[row] - getFloat(24u);
  let column = min(columns - 1u, u32(floor(longitude / getFloat(8u) * f32(columns))));
  let cellRow = min(rows - 1u, u32(floor(latitude / getFloat(9u) * f32(rows))));
  let cell = cellRow * columns + column;
  atomicAdd(&altitudeSums[cell], i32(floor((altitude - getFloat(21u)) * getFloat(22u) + 0.5)));
  atomicAdd(&altitudeCounts[cell], 1u);`),
  resolve:
    HEADER +
    /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> sums: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> field: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> altitudeSums: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> altitudeCounts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> altitudeField: array<vec4<f32>>;` +
    ENTRY(`
  if (index >= cellCount) { return; }
  if (getWord(20u) == 1u) {
    let altitudeCount = atomicLoad(&altitudeCounts[index]);
    altitudeField[index] = select(vec4<f32>(0.0),
      vec4<f32>(f32(atomicLoad(&altitudeSums[index])) / getFloat(22u) / f32(max(altitudeCount, 1u)) + getFloat(21u), 0.0, 1.0, 0.0),
      altitudeCount > 0u);
  }
  let count = atomicLoad(&counts[index]);
  if (count == 0u) { field[index] = vec4<f32>(0.0); return; }
  let divisor = f32(count);
  let first = f32(atomicLoad(&sums[index * 2u])) / getFloat(14u) / divisor + getFloat(16u);
  var second = f32(atomicLoad(&sums[index * 2u + 1u])) / getFloat(15u) / divisor + getFloat(17u);
  if (getWord(0u) == 2u) { second = 0.0; }
  field[index] = vec4<f32>(first, second, 1.0, 0.0);`),
  gradient:
    HEADER +
    /* wgsl */ `
@group(0) @binding(1) var<storage, read> source: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> destination: array<vec4<f32>>;
fn isFilled(cell: u32) -> bool { return source[cell].z > 0.5; }` +
    ENTRY(`
  if (index >= cellCount) { return; }
  if (!isFilled(index)) { destination[index] = vec4<f32>(0.0); return; }
  let column = index % columns;
  let row = index / columns;
  let south = getFloat(11u);
  let latitudeStep = select(1.0, (getFloat(13u) - south) / f32(rows - 1u), rows > 1u);
  let longitudeStep = select(1.0, (getFloat(12u) - getFloat(10u)) / f32(columns - 1u), columns > 1u);
  let metersNorth = ${METERS_PER_DEGREE_LATITUDE}.0 * latitudeStep;
  let latitude = south + f32(row) * latitudeStep;
  let metersEast = ${METERS_PER_DEGREE_LATITUDE}.0 * max(0.2, cos(latitude * 0.017453292519943295)) * longitudeStep;
  var hasEast = true;
  var gradientEast = 0.0;
  if (column > 0u && column < columns - 1u && isFilled(index - 1u) && isFilled(index + 1u)) {
    gradientEast = (source[index + 1u].x - source[index - 1u].x) / (2.0 * metersEast);
  } else if (column < columns - 1u && isFilled(index + 1u)) {
    gradientEast = (source[index + 1u].x - source[index].x) / metersEast;
  } else if (column > 0u && isFilled(index - 1u)) {
    gradientEast = (source[index].x - source[index - 1u].x) / metersEast;
  } else { hasEast = false; }
  var hasNorth = true;
  var gradientNorth = 0.0;
  if (row > 0u && row < rows - 1u && isFilled(index - columns) && isFilled(index + columns)) {
    gradientNorth = (source[index + columns].x - source[index - columns].x) / (2.0 * metersNorth);
  } else if (row < rows - 1u && isFilled(index + columns)) {
    gradientNorth = (source[index + columns].x - source[index].x) / metersNorth;
  } else if (row > 0u && isFilled(index - columns)) {
    gradientNorth = (source[index].x - source[index - columns].x) / metersNorth;
  } else { hasNorth = false; }
  if (!hasEast || !hasNorth) { destination[index] = vec4<f32>(0.0); return; }
  destination[index] = vec4<f32>(-gradientEast, -gradientNorth, 1.0, 0.0);`),
  smooth:
    HEADER +
    /* wgsl */ `
@group(0) @binding(1) var<storage, read> source: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> destination: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> altitudeSource: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> altitudeDestination: array<vec4<f32>>;` +
    ENTRY(`
  if (index >= cellCount) { return; }
  let column = i32(index % columns);
  let row = i32(index / columns);
  let reach = i32(getWord(7u));
  if (getWord(20u) == 1u) {
    if (altitudeSource[index].z < 0.5) { altitudeDestination[index] = vec4<f32>(0.0); }
    else {
      var altitudeSum = 0.0;
      var altitudeCount = 0.0;
      for (var rowOffset = -reach; rowOffset <= reach; rowOffset++) {
        let neighborRow = row + rowOffset;
        if (neighborRow < 0 || neighborRow >= i32(rows)) { continue; }
        for (var columnOffset = -reach; columnOffset <= reach; columnOffset++) {
          let neighborColumn = column + columnOffset;
          if (neighborColumn < 0 || neighborColumn >= i32(columns)) { continue; }
          let neighbor = altitudeSource[u32(neighborRow) * columns + u32(neighborColumn)];
          if (neighbor.z < 0.5) { continue; }
          altitudeSum += neighbor.x;
          altitudeCount += 1.0;
        }
      }
      altitudeDestination[index] = vec4<f32>(altitudeSum / altitudeCount, 0.0, 1.0, 0.0);
    }
  }
  if (source[index].z < 0.5) { destination[index] = vec4<f32>(0.0); return; }
  var sum = vec2<f32>(0.0);
  var count = 0.0;
  for (var rowOffset = -reach; rowOffset <= reach; rowOffset++) {
    let neighborRow = row + rowOffset;
    if (neighborRow < 0 || neighborRow >= i32(rows)) { continue; }
    for (var columnOffset = -reach; columnOffset <= reach; columnOffset++) {
      let neighborColumn = column + columnOffset;
      if (neighborColumn < 0 || neighborColumn >= i32(columns)) { continue; }
      let neighbor = source[u32(neighborRow) * columns + u32(neighborColumn)];
      if (neighbor.z < 0.5) { continue; }
      sum += neighbor.xy;
      count += 1.0;
    }
  }
  destination[index] = vec4<f32>(sum / count, 1.0, 0.0);`),
  speed:
    HEADER +
    /* wgsl */ `
@group(0) @binding(1) var<storage, read> field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> maximumSpeed: array<atomic<u32>>;` +
    ENTRY(`
  if (index >= cellCount) { return; }
  let texel = field[index];
  if (texel.z < 0.5) { return; }
  // Non-negative float32 bit patterns order like the values.
  atomicMax(&maximumSpeed[0], bitcast<u32>(length(texel.xy)));
  atomicMin(&maximumSpeed[1], bitcast<u32>(length(texel.xy)));`),
  write:
    HEADER +
    /* wgsl */ `
@group(0) @binding(1) var<storage, read> field: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> maximumSpeed: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> texels: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> altitudeField: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> altitudeTexels: array<vec4<f32>>;` +
    ENTRY(`
  let paddedColumns = getWord(19u);
  if (index >= paddedColumns * rows) { return; }
  let x = index % paddedColumns;
  let y = index / paddedColumns;
  if (x >= columns) {
    texels[index] = vec4<f32>(0.0);
    if (getWord(20u) == 1u) { altitudeTexels[index] = vec4<f32>(0.0); }
    return;
  }
  let texel = field[y * columns + x];
  let maximum = bitcast<f32>(atomicLoad(&maximumSpeed[0]));
  let minimum = min(bitcast<f32>(atomicLoad(&maximumSpeed[1])), maximum);
  let numerator = getFloat(18u);
  let scale = select(1.0, numerator / maximum, numerator > 0.0 && maximum > 0.0);
  // The w channel carries speed normalized over the grid's [minimum, maximum] (color by speed).
  let span = maximum - minimum;
  let speed = length(texel.xy);
  // The fastest cell is exactly 1; the division can round just below it.
  let normalizedSpeed = select(0.0, select(clamp((speed - minimum) / span, 0.0, 1.0), 1.0, speed >= maximum), span > 0.000000000001);
  texels[index] = select(vec4<f32>(0.0), vec4<f32>(texel.xy * scale, 1.0, normalizedSpeed), texel.z > 0.5);
  if (getWord(20u) == 1u) {
    let altitudeTexel = altitudeField[y * columns + x];
    altitudeTexels[index] = select(vec4<f32>(0.0), altitudeTexel, altitudeTexel.z > 0.5 && texel.z > 0.5);
  }`)
};

type StageName = keyof typeof SHADERS;

/** Exposed for the static binding-consistency spec (no device needed). */
export const FLOW_FIELD_GPU_SHADERS: Record<string, string> = SHADERS;

const BINDINGS: Record<StageName, {name: string; type: 'read-only-storage' | 'storage'}[]> = {
  clear: [
    {name: 'sums', type: 'storage'},
    {name: 'counts', type: 'storage'},
    {name: 'maximumSpeed', type: 'storage'},
    {name: 'altitudeSums', type: 'storage'},
    {name: 'altitudeCounts', type: 'storage'}
  ],
  extent: [
    {name: 'longitudes', type: 'read-only-storage'},
    {name: 'latitudes', type: 'read-only-storage'},
    {name: 'selection', type: 'read-only-storage'},
    {name: 'extentWords', type: 'storage'}
  ],
  bin: [
    {name: 'longitudes', type: 'read-only-storage'},
    {name: 'latitudes', type: 'read-only-storage'},
    {name: 'firstValues', type: 'read-only-storage'},
    {name: 'secondValues', type: 'read-only-storage'},
    {name: 'selection', type: 'read-only-storage'},
    {name: 'sums', type: 'storage'},
    {name: 'counts', type: 'storage'}
  ],
  binAltitude: [
    {name: 'longitudes', type: 'read-only-storage'},
    {name: 'latitudes', type: 'read-only-storage'},
    {name: 'altitudes', type: 'read-only-storage'},
    {name: 'selection', type: 'read-only-storage'},
    {name: 'altitudeSums', type: 'storage'},
    {name: 'altitudeCounts', type: 'storage'}
  ],
  resolve: [
    {name: 'sums', type: 'storage'},
    {name: 'counts', type: 'storage'},
    {name: 'field', type: 'storage'},
    {name: 'altitudeSums', type: 'storage'},
    {name: 'altitudeCounts', type: 'storage'},
    {name: 'altitudeField', type: 'storage'}
  ],
  gradient: [
    {name: 'source', type: 'read-only-storage'},
    {name: 'destination', type: 'storage'}
  ],
  smooth: [
    {name: 'source', type: 'read-only-storage'},
    {name: 'destination', type: 'storage'},
    {name: 'altitudeSource', type: 'read-only-storage'},
    {name: 'altitudeDestination', type: 'storage'}
  ],
  speed: [
    {name: 'field', type: 'read-only-storage'},
    {name: 'maximumSpeed', type: 'storage'}
  ],
  write: [
    {name: 'field', type: 'read-only-storage'},
    {name: 'maximumSpeed', type: 'storage'},
    {name: 'texels', type: 'storage'},
    {name: 'altitudeField', type: 'read-only-storage'},
    {name: 'altitudeTexels', type: 'storage'}
  ]
};

/** Exposed for the static binding-consistency spec (no device needed). */
export const FLOW_FIELD_GPU_BINDINGS: Record<string, {name: string; type: string}[]> = BINDINGS;

export type FlowFieldGpuGridOptions = {
  mode: FlowFieldColumnMode;
  columns: number;
  rows: number;
  extent: FlowFieldExtent;
  smoothing: number;
  /** Visual-policy `share * extent / lifetime`; zero keeps unit scale. */
  velocityScaleNumerator: number;
  /** Terrain height source: the first column (ELEVATION), a separate column, or none. */
  altitude?: 'first' | 'column' | 'none';
  memoryReservation?: KeplerMemoryReservation;
};

/** Source columns as float32 offsets from the grid origin; invalid rows are NaN longitudes. */
export function makeFlowFieldGpuColumns(
  samples: Omit<FlowFieldSamples, 'selected'>,
  extent: FlowFieldExtent
): {
  longitudes: Float32Array;
  latitudes: Float32Array;
  first: Float32Array;
  second: Float32Array;
  altitudes?: Float32Array;
} {
  const count = samples.longitudes.length;
  const altitudes = samples.altitude ? new Float32Array(count) : undefined;
  const longitudes = new Float32Array(count);
  const latitudes = new Float32Array(count);
  const first = new Float32Array(count);
  const second = new Float32Array(count);
  const [west, south] = extent.bounds;
  for (let index = 0; index < count; index++) {
    const longitude = samples.longitudes[index];
    const latitude = samples.latitudes[index];
    const firstValue = samples.first[index];
    const secondValue = samples.second ? samples.second[index] : 0;
    if (
      !Number.isFinite(longitude) ||
      !Number.isFinite(latitude) ||
      !Number.isFinite(firstValue) ||
      !Number.isFinite(secondValue) ||
      Math.abs(latitude) > MAXIMUM_FLOW_FIELD_LATITUDE
    ) {
      longitudes[index] = Number.NaN;
      continue;
    }
    longitudes[index] = longitude - west;
    latitudes[index] = latitude - south;
    first[index] = firstValue;
    second[index] = secondValue;
    if (altitudes) altitudes[index] = samples.altitude?.[index] ?? Number.NaN;
  }
  // Rows without a finite altitude stay NaN so the shader skips them.
  if (altitudes)
    for (let index = 0; index < count; index++)
      if (!Number.isFinite(longitudes[index]) && !Number.isFinite(altitudes[index]))
        altitudes[index] = Number.NaN;
  return {longitudes, latitudes, first, second, ...(altitudes ? {altitudes} : {})};
}

export type FlowFieldGpuStats = {
  /** Compute dispatch rounds (one per grid build). */
  buildCount: number;
  /** Always zero: the grid, mask and maximum speed are never read back. */
  readbackCount: number;
  /** Filtered-bounds reductions read back (one 20-byte result per filter revision). */
  extentReadbackCount: number;
  byteLength: number;
};

/** Owns the GPU columns, accumulators and kernels of one resident Flow Field grid. */
export class FlowFieldGpuGrid {
  buildCount = 0;
  readonly readbackCount = 0;
  extentReadbackCount = 0;
  /** Whether a terrain height grid is produced next to the velocity grid. */
  readonly hasAltitude: boolean;
  /** Bounds mapping the texture: follows the filtered rows, defaults to the full extent. */
  activeBounds: [number, number, number, number];
  /** True while the filtered rows cannot form a grid; the next build produces an invalid field. */
  emptyField = false;
  readonly byteLength: number;
  readonly paddedColumns: number;
  readonly rowCount: number;
  private readonly fixedPoint: FlowFieldFixedPoint;
  private readonly kernels = {} as Record<StageName, Kernel>;
  private readonly buffers: Buffer[] = [];
  private readonly longitudes: Buffer;
  private readonly latitudes: Buffer;
  private readonly firstValues: Buffer;
  private readonly secondValues: Buffer;
  private readonly emptySelection: Buffer;
  private readonly sums: Buffer;
  private readonly counts: Buffer;
  private readonly fieldA: Buffer;
  private readonly fieldB: Buffer;
  private readonly maximumSpeed: Buffer;
  private readonly texelBuffer: Buffer;
  private readonly altitudes: Buffer;
  private readonly altitudeSums: Buffer;
  private readonly altitudeCounts: Buffer;
  private readonly altitudeFieldA: Buffer;
  private readonly altitudeFieldB: Buffer;
  private readonly altitudeTexelBuffer: Buffer;
  private readonly extentBuffer: Buffer;
  private destroyed = false;

  constructor(
    readonly device: Device,
    readonly id: string,
    columnsData: ReturnType<typeof makeFlowFieldGpuColumns>,
    readonly options: FlowFieldGpuGridOptions
  ) {
    this.rowCount = columnsData.longitudes.length;
    this.activeBounds = [...options.extent.bounds];
    this.hasAltitude =
      (options.altitude ?? 'none') !== 'none' &&
      (options.altitude === 'first' || Boolean(columnsData.altitudes));
    const cellCount = options.columns * options.rows;
    // Row pitch of buffer-to-texture copies must be a multiple of 256 bytes (16 texels).
    this.paddedColumns = Math.ceil(options.columns / 16) * 16;
    this.fixedPoint = getFlowFieldFixedPoint(options.mode, options.extent);
    const upload = (name: string, data: Float32Array) =>
      this.makeBuffer(name, Buffer.STORAGE | Buffer.COPY_DST, data.byteLength, data);
    this.longitudes = upload('longitudes', columnsData.longitudes);
    this.latitudes = upload('latitudes', columnsData.latitudes);
    this.firstValues = upload('first', columnsData.first);
    this.secondValues =
      options.mode === 'UV' || options.mode === 'SPEED_DIR'
        ? upload('second', columnsData.second)
        : this.firstValues;
    this.emptySelection = this.makeBuffer('empty-selection', Buffer.STORAGE, 16);
    this.extentBuffer = this.makeBuffer(
      'extent',
      Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST,
      32
    );
    // Absent altitude binds distinct 16-byte placeholders (writable bindings must not alias).
    const altitudeCells = this.hasAltitude ? options.columns * options.rows : 1;
    this.altitudes =
      options.altitude === 'column' && columnsData.altitudes
        ? upload('altitudes', columnsData.altitudes)
        : this.firstValues;
    this.altitudeSums = this.makeBuffer('altitude-sums', Buffer.STORAGE, altitudeCells * 4);
    this.altitudeCounts = this.makeBuffer('altitude-counts', Buffer.STORAGE, altitudeCells * 4);
    this.altitudeFieldA = this.makeBuffer('altitude-field-a', Buffer.STORAGE, altitudeCells * 16);
    this.altitudeFieldB = this.makeBuffer('altitude-field-b', Buffer.STORAGE, altitudeCells * 16);
    this.altitudeTexelBuffer = this.makeBuffer(
      'altitude-texels',
      Buffer.STORAGE | Buffer.COPY_SRC,
      this.hasAltitude ? this.paddedColumns * options.rows * 16 : 16
    );
    this.sums = this.makeBuffer('sums', Buffer.STORAGE, cellCount * 8);
    this.counts = this.makeBuffer('counts', Buffer.STORAGE, cellCount * 4);
    this.fieldA = this.makeBuffer('field-a', Buffer.STORAGE, cellCount * 16);
    this.fieldB = this.makeBuffer('field-b', Buffer.STORAGE, cellCount * 16);
    this.maximumSpeed = this.makeBuffer('maximum-speed', Buffer.STORAGE, 16);
    this.texelBuffer = this.makeBuffer(
      'texels',
      Buffer.STORAGE | Buffer.COPY_SRC,
      this.paddedColumns * options.rows * 16
    );
    this.byteLength = this.buffers.reduce((total, buffer) => total + buffer.byteLength, 0);
    options.memoryReservation?.resize(this.byteLength);
    for (const stage of Object.keys(SHADERS) as StageName[])
      this.kernels[stage] = new Kernel(device, {
        id: `${id}-flow-field-${stage}`,
        source: SHADERS[stage],
        shaderLayout: {
          bindings: [
            {name: 'parameters', type: 'uniform', group: 0, location: 0},
            ...BINDINGS[stage].map((binding, index) => ({
              name: binding.name,
              type: binding.type,
              group: 0,
              location: index + 1
            }))
          ]
        }
      });
  }

  get stats(): FlowFieldGpuStats {
    return {
      buildCount: this.buildCount,
      readbackCount: this.readbackCount,
      extentReadbackCount: this.extentReadbackCount,
      byteLength: this.byteLength
    };
  }

  /** Estimated GPU bytes for a grid, for memory reservations before allocation. */
  static estimateByteLength(
    rowCount: number,
    columns: number,
    rows: number,
    altitude: 'first' | 'column' | 'none' = 'none'
  ): number {
    const paddedColumns = Math.ceil(columns / 16) * 16;
    const altitudeBytes =
      altitude === 'none'
        ? 64
        : columns * rows * (4 + 4 + 32) +
          paddedColumns * rows * 16 +
          (altitude === 'column' ? rowCount * 4 : 0);
    return (
      rowCount * 16 + columns * rows * (8 + 4 + 32) + paddedColumns * rows * 16 + 64 + altitudeBytes
    );
  }

  private makeBuffer(name: string, usage: number, byteLength: number, data?: Float32Array): Buffer {
    const buffer = this.device.createBuffer({
      id: `${this.id}-flow-field-${name}`,
      byteLength: Math.max(16, byteLength),
      usage,
      ...(data ? {data} : {})
    });
    this.buffers.push(buffer);
    return buffer;
  }

  private writeParameters(overrides: {
    chunkLength?: number;
    rowStart?: number;
    maskWordOffset?: number;
    useMask?: boolean;
  }): Buffer {
    const {options, fixedPoint} = this;
    const [west, south, east, north] = this.activeBounds;
    const bytes = new ArrayBuffer(PARAMETER_WORDS * 4);
    const integers = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    integers[0] = MODE_CODES[options.mode];
    integers[1] = options.columns;
    integers[2] = options.rows;
    integers[3] = overrides.chunkLength ?? 0;
    integers[4] = overrides.rowStart ?? 0;
    integers[5] = overrides.maskWordOffset ?? 0;
    integers[6] = overrides.useMask ? 1 : 0;
    integers[7] = Math.max(0, Math.floor(options.smoothing));
    floats[8] = east - west;
    floats[9] = north - south;
    floats[10] = west;
    floats[11] = south;
    floats[12] = east;
    floats[13] = north;
    floats[14] = fixedPoint.scaleFirst;
    floats[15] = fixedPoint.scaleSecond;
    floats[16] = fixedPoint.centerFirst;
    floats[17] = fixedPoint.centerSecond;
    floats[18] = options.velocityScaleNumerator;
    integers[19] = this.paddedColumns;
    integers[20] = this.hasAltitude ? 1 : 0;
    floats[21] = fixedPoint.centerAltitude;
    floats[22] = fixedPoint.scaleAltitude;
    floats[23] = Math.fround(west - options.extent.bounds[0]);
    floats[24] = Math.fround(south - options.extent.bounds[1]);
    return this.device.createBuffer({
      id: `${this.id}-flow-field-parameters`,
      usage: Buffer.UNIFORM | Buffer.COPY_DST,
      data: new Uint8Array(bytes)
    });
  }

  /**
   * Re-bins the table under `selection` (all rows when omitted) and writes the scaled velocity
   * texels into `texture`. Pure GPU work: one submit, no readback.
   */
  build(texture: Texture, selection?: GPUVector<'uint32'>, altitudeTexture?: Texture): void {
    if (this.destroyed) return;
    const {device, options} = this;
    const cellCount = options.columns * options.rows;
    const encoder = device.createCommandEncoder({id: `${this.id}-flow-field-build`});
    const pass = encoder.beginComputePass({id: `${this.id}-flow-field-passes`});
    const parameterBuffers: Buffer[] = [];
    const dispatch = (
      stage: StageName,
      count: number,
      parameters: Buffer,
      bindings: Record<string, Buffer>
    ) => {
      const workgroups = Math.max(1, Math.ceil(count / WORKGROUP_SIZE));
      this.kernels[stage].dispatch(pass, {
        bindings: {parameters, ...bindings},
        x: Math.min(workgroups, MAXIMUM_WORKGROUPS_PER_DIMENSION),
        y: Math.ceil(workgroups / MAXIMUM_WORKGROUPS_PER_DIMENSION)
      });
    };
    const common = this.writeParameters({});
    parameterBuffers.push(common);
    dispatch('clear', cellCount, common, {
      sums: this.sums,
      counts: this.counts,
      maximumSpeed: this.maximumSpeed,
      altitudeSums: this.altitudeSums,
      altitudeCounts: this.altitudeCounts
    });
    const binChunk = (
      chunkLength: number,
      rowStart: number,
      maskBuffer: Buffer,
      maskWordOffset: number,
      useMask: boolean
    ) => {
      const parameters = this.writeParameters({chunkLength, rowStart, maskWordOffset, useMask});
      parameterBuffers.push(parameters);
      dispatch('bin', chunkLength, parameters, {
        longitudes: this.longitudes,
        latitudes: this.latitudes,
        firstValues: this.firstValues,
        secondValues: this.secondValues,
        selection: maskBuffer,
        sums: this.sums,
        counts: this.counts
      });
      if (this.hasAltitude)
        dispatch('binAltitude', chunkLength, parameters, {
          longitudes: this.longitudes,
          latitudes: this.latitudes,
          altitudes: this.altitudes,
          selection: maskBuffer,
          altitudeSums: this.altitudeSums,
          altitudeCounts: this.altitudeCounts
        });
    };
    const {rowCount} = this;
    // An empty field (filtered rows too few for a grid) skips binning: every cell stays invalid.
    if (this.emptyField) {
      // no rows binned
    } else if (selection) {
      let rowStart = 0;
      for (const chunk of selection.data) {
        const maskBuffer = chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
        if (chunk.length) binChunk(chunk.length, rowStart, maskBuffer, chunk.byteOffset / 4, true);
        rowStart += chunk.length;
      }
    } else binChunk(rowCount, 0, this.emptySelection, 0, false);
    dispatch('resolve', cellCount, common, {
      sums: this.sums,
      counts: this.counts,
      field: this.fieldA,
      altitudeSums: this.altitudeSums,
      altitudeCounts: this.altitudeCounts,
      altitudeField: this.altitudeFieldA
    });
    let current = this.fieldA;
    let spare = this.fieldB;
    let currentAltitude = this.altitudeFieldA;
    let spareAltitude = this.altitudeFieldB;
    if (options.mode === 'ELEVATION') {
      dispatch('gradient', cellCount, common, {source: current, destination: spare});
      [current, spare] = [spare, current];
    }
    if (Math.floor(options.smoothing) > 0) {
      dispatch('smooth', cellCount, common, {
        source: current,
        destination: spare,
        altitudeSource: currentAltitude,
        altitudeDestination: spareAltitude
      });
      [current, spare] = [spare, current];
      [currentAltitude, spareAltitude] = [spareAltitude, currentAltitude];
    }
    dispatch('speed', cellCount, common, {field: current, maximumSpeed: this.maximumSpeed});
    dispatch('write', this.paddedColumns * options.rows, common, {
      field: current,
      maximumSpeed: this.maximumSpeed,
      texels: this.texelBuffer,
      altitudeField: currentAltitude,
      altitudeTexels: this.altitudeTexelBuffer
    });
    pass.end();
    encoder.copyBufferToTexture({
      sourceBuffer: this.texelBuffer,
      destinationTexture: texture,
      bytesPerRow: this.paddedColumns * 16,
      rowsPerImage: options.rows,
      size: [options.columns, options.rows, 1]
    });
    if (this.hasAltitude && altitudeTexture)
      encoder.copyBufferToTexture({
        sourceBuffer: this.altitudeTexelBuffer,
        destinationTexture: altitudeTexture,
        bytesPerRow: this.paddedColumns * 16,
        rowsPerImage: options.rows,
        size: [options.columns, options.rows, 1]
      });
    device.submit(encoder.finish());
    for (const buffer of parameterBuffers) buffer.destroy();
    this.buildCount++;
  }

  /**
   * Reduces the masked valid rows to their bounds on the GPU (atomic min/max over non-negative
   * float32 offsets) and reads back only that 20-byte result. Null when fewer than four rows are
   * selected or they span no area. The mask and the grid are never read back.
   */
  async measureSelectionBounds(
    selection: GPUVector<'uint32'>
  ): Promise<[number, number, number, number] | null> {
    if (this.destroyed) return null;
    const {device, options} = this;
    this.extentBuffer.write(
      new Uint32Array([MAXIMUM_FLOAT_BITS, MAXIMUM_FLOAT_BITS, 0, 0, 0, 0, 0, 0])
    );
    const encoder = device.createCommandEncoder({id: `${this.id}-flow-field-extent`});
    const pass = encoder.beginComputePass({id: `${this.id}-flow-field-extent-pass`});
    const parameterBuffers: Buffer[] = [];
    // Offsets are relative to the full origin, so the active bounds must not shift them.
    const savedBounds = this.activeBounds;
    this.activeBounds = [...options.extent.bounds];
    let rowStart = 0;
    for (const chunk of selection.data) {
      const maskBuffer = chunk.buffer instanceof Buffer ? chunk.buffer : chunk.buffer.buffer;
      if (chunk.length) {
        const parameters = this.writeParameters({
          chunkLength: chunk.length,
          rowStart,
          maskWordOffset: chunk.byteOffset / 4,
          useMask: true
        });
        parameterBuffers.push(parameters);
        const workgroups = Math.max(1, Math.ceil(chunk.length / WORKGROUP_SIZE));
        this.kernels.extent.dispatch(pass, {
          bindings: {
            parameters,
            longitudes: this.longitudes,
            latitudes: this.latitudes,
            selection: maskBuffer,
            extentWords: this.extentBuffer
          },
          x: Math.min(workgroups, MAXIMUM_WORKGROUPS_PER_DIMENSION),
          y: Math.ceil(workgroups / MAXIMUM_WORKGROUPS_PER_DIMENSION)
        });
      }
      rowStart += chunk.length;
    }
    this.activeBounds = savedBounds;
    pass.end();
    device.submit(encoder.finish());
    const bytes = await this.extentBuffer.readAsync(0, 32);
    for (const buffer of parameterBuffers) buffer.destroy();
    this.extentReadbackCount++;
    const words = new Uint32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + 32));
    const offsets = new Float32Array(words.buffer);
    const [west, south] = options.extent.bounds;
    const count = words[4];
    const bounds: [number, number, number, number] = [
      west + offsets[0],
      south + offsets[1],
      west + offsets[2],
      south + offsets[3]
    ];
    return count >= 4 && bounds[2] > bounds[0] && bounds[3] > bounds[1] ? bounds : null;
  }

  /** Test and debug readback of the altitude grid (columns * rows, NaN where a cell has none). */
  async readAltitudeForDebug(): Promise<Float32Array | undefined> {
    if (!this.hasAltitude) return undefined;
    const {options} = this;
    const bytes = await this.altitudeTexelBuffer.readAsync(
      0,
      this.paddedColumns * options.rows * 16
    );
    const padded = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    );
    const altitude = new Float32Array(options.columns * options.rows).fill(Number.NaN);
    for (let row = 0; row < options.rows; row++)
      for (let column = 0; column < options.columns; column++) {
        const offset = (row * this.paddedColumns + column) * 4;
        if (padded[offset + 2] > 0.5) altitude[row * options.columns + column] = padded[offset];
      }
    return altitude;
  }

  /** Test-only readback of the final texels (counted separately; never used by rendering). */
  async readTexelsForDebug(): Promise<Float32Array> {
    const {options} = this;
    const bytes = await this.texelBuffer.readAsync(0, this.paddedColumns * options.rows * 16);
    const padded = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    );
    const texels = new Float32Array(options.columns * options.rows * 4);
    for (let row = 0; row < options.rows; row++)
      texels.set(
        padded.subarray(
          row * this.paddedColumns * 4,
          row * this.paddedColumns * 4 + options.columns * 4
        ),
        row * options.columns * 4
      );
    return texels;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    for (const kernel of Object.values(this.kernels)) kernel.destroy();
    for (const buffer of new Set(this.buffers)) buffer.destroy();
    this.options.memoryReservation?.release();
  }
}
