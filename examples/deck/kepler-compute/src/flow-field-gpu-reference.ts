// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Node-testable design of the GPU-resident Flow Field grid (`flow-field-gpu.ts`). The GPU path
 * differs from `buildFlowFieldGrid` in three deliberate ways, mirrored here exactly:
 *
 * 1. The grid has the cell count of the extent of every finite sample in the retained table, so a
 *    filter change re-bins into the same texture instead of resizing it. Like native, the bounds
 *    that map that texture follow the filtered rows: a GPU min/max reduction over the masked rows
 *    produces them and only that 20-byte result is read back per filter revision (the mask and
 *    the grid never are). Cells are then slightly non-square, which sampling does not mind.
 * 2. WGSL has no float atomics: per-cell sums are int32 fixed point of `(value - center) * scale`
 *    with a per-column center and a scale that keeps `rowCount * halfRange * scale < 2^30`.
 * 3. The visual-policy velocity scale is `numerator / maximumSpeed`, with the maximum reduced on
 *    the GPU, so the scalar is never read back.
 */

import {
  MAXIMUM_FLOW_FIELD_LATITUDE,
  convertElevationToVector,
  convertSpeedDirectionToVector,
  getFlowFieldBinCounts,
  smoothFlowFieldGrid,
  smoothFlowFieldScalarGrid,
  type FlowFieldColumnMode,
  type FlowFieldGrid,
  type FlowFieldGridOptions,
  type FlowFieldSamples
} from './flow-field-grid';

/** Largest magnitude a cell sum may reach: keeps int32 atomic accumulation overflow free. */
export const FLOW_FIELD_FIXED_POINT_LIMIT = 2 ** 30;
export const MAXIMUM_FLOW_FIELD_FIXED_POINT_SCALE = 65536;

export type FlowFieldExtent = {
  bounds: [number, number, number, number];
  /** Rows with finite coordinates, values and |latitude| <= 85. */
  validCount: number;
  /** Per value column: [minimum, maximum] over valid rows (second is [0, 0] when absent). */
  firstRange: [number, number];
  secondRange: [number, number];
  /** [minimum, maximum] of the optional altitude column over valid rows ([0, 0] when absent). */
  altitudeRange: [number, number];
  /** Largest |speed| (SPEED_DIR) or component magnitude, used to size fixed-point scales. */
  maximumMagnitude: number;
};

/** Scans the unfiltered table once on the CPU (no mask, no GPU readback). */
export function getFlowFieldExtent(
  samples: Omit<FlowFieldSamples, 'selected'>,
  /** Honors a row mask: the extent then follows the selected rows only. */
  selected?: ArrayLike<number>
): FlowFieldExtent | null {
  const count = samples.longitudes.length;
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  let validCount = 0;
  const firstRange: [number, number] = [Infinity, -Infinity];
  const secondRange: [number, number] = [Infinity, -Infinity];
  const altitudeRange: [number, number] = [Infinity, -Infinity];
  for (let index = 0; index < count; index++) {
    if (selected && !selected[index]) continue;
    const longitude = samples.longitudes[index];
    const latitude = samples.latitudes[index];
    const first = samples.first[index];
    const second = samples.second ? samples.second[index] : 0;
    if (
      !Number.isFinite(longitude) ||
      !Number.isFinite(latitude) ||
      !Number.isFinite(first) ||
      !Number.isFinite(second) ||
      Math.abs(latitude) > MAXIMUM_FLOW_FIELD_LATITUDE
    )
      continue;
    validCount++;
    west = Math.min(west, longitude);
    east = Math.max(east, longitude);
    south = Math.min(south, latitude);
    north = Math.max(north, latitude);
    firstRange[0] = Math.min(firstRange[0], first);
    firstRange[1] = Math.max(firstRange[1], first);
    secondRange[0] = Math.min(secondRange[0], second);
    secondRange[1] = Math.max(secondRange[1], second);
    const altitude = samples.altitude ? samples.altitude[index] : Number.NaN;
    if (Number.isFinite(altitude)) {
      altitudeRange[0] = Math.min(altitudeRange[0], altitude);
      altitudeRange[1] = Math.max(altitudeRange[1], altitude);
    }
  }
  if (validCount < 4 || !(east > west) || !(north > south)) return null;
  return {
    bounds: [west, south, east, north],
    validCount,
    firstRange,
    secondRange,
    altitudeRange: altitudeRange[0] <= altitudeRange[1] ? altitudeRange : [0, 0],
    maximumMagnitude: Math.max(
      Math.abs(firstRange[0]),
      Math.abs(firstRange[1]),
      Math.abs(secondRange[0]),
      Math.abs(secondRange[1])
    )
  };
}

export type FlowFieldFixedPoint = {
  centerFirst: number;
  centerSecond: number;
  scaleFirst: number;
  scaleSecond: number;
  /** Altitude accumulation (ELEVATION mode shares the first column's values). */
  centerAltitude: number;
  scaleAltitude: number;
};

function getFixedPointScale(halfRange: number, rowCount: number): number {
  if (!(halfRange > 0)) return MAXIMUM_FLOW_FIELD_FIXED_POINT_SCALE;
  return Math.min(
    MAXIMUM_FLOW_FIELD_FIXED_POINT_SCALE,
    FLOW_FIELD_FIXED_POINT_LIMIT / (Math.max(1, rowCount) * halfRange)
  );
}

/** Per-column centers and scales; centers shrink the accumulated magnitude. Values are float32-exact. */
export function getFlowFieldFixedPoint(
  mode: FlowFieldColumnMode,
  extent: FlowFieldExtent
): FlowFieldFixedPoint {
  const f32 = Math.fround;
  const altitudeRange = mode === 'ELEVATION' ? extent.firstRange : extent.altitudeRange;
  const altitude = {
    centerAltitude: f32((altitudeRange[0] + altitudeRange[1]) / 2),
    scaleAltitude: f32(
      getFixedPointScale((altitudeRange[1] - altitudeRange[0]) / 2, extent.validCount)
    )
  };
  if (mode === 'SPEED_DIR') {
    // Accumulated quantities are u and v, each bounded by the maximum speed.
    const scale = f32(getFixedPointScale(extent.maximumMagnitude, extent.validCount));
    return {centerFirst: 0, centerSecond: 0, scaleFirst: scale, scaleSecond: scale, ...altitude};
  }
  const centerFirst = f32((extent.firstRange[0] + extent.firstRange[1]) / 2);
  const centerSecond = mode === 'UV' ? f32((extent.secondRange[0] + extent.secondRange[1]) / 2) : 0;
  return {
    ...altitude,
    centerFirst,
    centerSecond,
    scaleFirst: f32(
      getFixedPointScale((extent.firstRange[1] - extent.firstRange[0]) / 2, extent.validCount)
    ),
    scaleSecond:
      mode === 'UV'
        ? f32(
            getFixedPointScale(
              (extent.secondRange[1] - extent.secondRange[0]) / 2,
              extent.validCount
            )
          )
        : 1
  };
}

/** WGSL `round` is half-to-even; the shader uses floor(x + 0.5) and so does the reference. */
function quantize(value: number, center: number, scale: number): number {
  return Math.floor((value - center) * scale + 0.5);
}

/** Filtered bounds need at least four selected valid rows spanning a non-degenerate area. */
export function getFlowFieldFilteredBounds(
  samples: FlowFieldSamples
): [number, number, number, number] | null {
  const {selected, ...unfiltered} = samples;
  return getFlowFieldExtent(unfiltered, selected)?.bounds ?? null;
}

export type FlowFieldGridReference = FlowFieldGrid & {
  velocityScale: number;
  scaledTexels: Float32Array;
};

/**
 * CPU reference of the GPU grid: same cell counts, filtered bounds, binning, fixed-point
 * accumulation, gradient, box smoothing, normalized speed and altitude as the compute passes.
 * Returns `velocityScale` applied texels like the runtime texture (`numerator` 0 keeps unit
 * scale). Null when the selected rows cannot form a grid (the GPU path then shows an empty field).
 */
export function buildFlowFieldGridReference(
  samples: FlowFieldSamples,
  options: FlowFieldGridOptions,
  velocityScaleNumerator = 0
): FlowFieldGridReference | null {
  const {selected, ...unfiltered} = samples;
  const fullExtent = getFlowFieldExtent(unfiltered);
  if (!fullExtent) return null;
  const {columns, rows} = getFlowFieldBinCounts(...fullExtent.bounds, options.gridResolution);
  const bounds = selected ? getFlowFieldFilteredBounds(samples) : fullExtent.bounds;
  if (!bounds) return null;
  const [west, south, east, north] = bounds;
  const cellCount = columns * rows;
  const fixedPoint = getFlowFieldFixedPoint(options.mode, fullExtent);
  const sumFirst = new Float64Array(cellCount);
  const sumSecond = new Float64Array(cellCount);
  const sumAltitude = new Float64Array(cellCount);
  const counts = new Uint32Array(cellCount);
  const altitudeCounts = new Uint32Array(cellCount);
  const hasAltitude = options.mode === 'ELEVATION' || Boolean(samples.altitude);
  let sampleCount = 0;
  const count = samples.longitudes.length;
  for (let index = 0; index < count; index++) {
    if (selected && !selected[index]) continue;
    const longitude = samples.longitudes[index];
    const latitude = samples.latitudes[index];
    const first = samples.first[index];
    const second = samples.second ? samples.second[index] : 0;
    if (
      !Number.isFinite(longitude) ||
      !Number.isFinite(latitude) ||
      !Number.isFinite(first) ||
      !Number.isFinite(second) ||
      Math.abs(latitude) > MAXIMUM_FLOW_FIELD_LATITUDE
    )
      continue;
    sampleCount++;
    const column = Math.min(
      columns - 1,
      Math.floor(((longitude - west) / (east - west)) * columns)
    );
    const row = Math.min(rows - 1, Math.floor(((latitude - south) / (north - south)) * rows));
    const cell = row * columns + column;
    let valueFirst = first;
    let valueSecond = second;
    if (options.mode === 'SPEED_DIR')
      [valueFirst, valueSecond] = convertSpeedDirectionToVector(first, second);
    else if (options.mode === 'ELEVATION') valueSecond = 0;
    sumFirst[cell] += quantize(valueFirst, fixedPoint.centerFirst, fixedPoint.scaleFirst);
    sumSecond[cell] += quantize(valueSecond, fixedPoint.centerSecond, fixedPoint.scaleSecond);
    counts[cell]++;
    const altitudeValue = options.mode === 'ELEVATION' ? first : (samples.altitude?.[index] ?? NaN);
    if (hasAltitude && Number.isFinite(altitudeValue)) {
      sumAltitude[cell] += quantize(
        altitudeValue,
        fixedPoint.centerAltitude,
        fixedPoint.scaleAltitude
      );
      altitudeCounts[cell]++;
    }
  }
  const filled = new Uint8Array(cellCount);
  let eastward: Float32Array = new Float32Array(cellCount);
  let northward: Float32Array = new Float32Array(cellCount);
  let altitude: Float32Array | undefined = hasAltitude
    ? new Float32Array(cellCount).fill(Number.NaN)
    : undefined;
  for (let cell = 0; cell < cellCount; cell++) {
    if (!counts[cell]) continue;
    filled[cell] = 1;
    eastward[cell] = sumFirst[cell] / fixedPoint.scaleFirst / counts[cell] + fixedPoint.centerFirst;
    northward[cell] =
      options.mode === 'ELEVATION'
        ? 0
        : sumSecond[cell] / fixedPoint.scaleSecond / counts[cell] + fixedPoint.centerSecond;
    if (altitude && altitudeCounts[cell])
      altitude[cell] =
        sumAltitude[cell] / fixedPoint.scaleAltitude / altitudeCounts[cell] +
        fixedPoint.centerAltitude;
  }
  if (options.mode === 'ELEVATION') {
    const gradient = convertElevationToVector(
      eastward,
      filled,
      columns,
      rows,
      west,
      south,
      east,
      north
    );
    eastward = gradient.east;
    northward = gradient.north;
  }
  smoothFlowFieldGrid(eastward, northward, filled, columns, rows, options.smoothing);
  if (altitude) altitude = smoothFlowFieldScalarGrid(altitude, columns, rows, options.smoothing);
  let maximumSpeed = 0;
  let minimumSpeed = Infinity;
  for (let cell = 0; cell < cellCount; cell++) {
    if (!filled[cell]) continue;
    const speed = Math.hypot(eastward[cell], northward[cell]);
    maximumSpeed = Math.max(maximumSpeed, speed);
    minimumSpeed = Math.min(minimumSpeed, speed);
  }
  const velocityScale =
    velocityScaleNumerator > 0 && maximumSpeed > 0 ? velocityScaleNumerator / maximumSpeed : 1;
  const speedSpan = maximumSpeed - (Number.isFinite(minimumSpeed) ? minimumSpeed : 0);
  const texels = new Float32Array(cellCount * 4);
  const scaledTexels = new Float32Array(cellCount * 4);
  for (let cell = 0; cell < cellCount; cell++) {
    if (!filled[cell]) continue;
    const normalizedSpeed =
      speedSpan > 1e-12
        ? Math.max(
            0,
            Math.min(1, (Math.hypot(eastward[cell], northward[cell]) - minimumSpeed) / speedSpan)
          )
        : 0;
    texels[cell * 4] = eastward[cell];
    texels[cell * 4 + 1] = northward[cell];
    texels[cell * 4 + 2] = 1;
    texels[cell * 4 + 3] = normalizedSpeed;
    scaledTexels[cell * 4] = eastward[cell] * velocityScale;
    scaledTexels[cell * 4 + 1] = northward[cell] * velocityScale;
    scaledTexels[cell * 4 + 2] = 1;
    scaledTexels[cell * 4 + 3] = normalizedSpeed;
  }
  return {
    columns,
    rows,
    bounds,
    texels,
    ...(altitude ? {altitude} : {}),
    minimumSpeed: Number.isFinite(minimumSpeed) ? minimumSpeed : 0,
    maximumSpeed,
    sampleCount,
    skippedCount: count - fullExtent.validCount,
    velocityScale,
    scaledTexels
  };
}
