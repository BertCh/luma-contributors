// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp, toDegrees, toRadians} from '@math.gl/core';
import {MEAN_EARTH_RADIUS_METERS} from './earth-constants';

/**
 * Pure CPU preparation for the native Flow Field layer. It reproduces the pinned Kepler 3.3
 * `flow-field-layer` field semantics (u/v, speed + direction-from, downhill altitude gradient,
 * scatter binning and box smoothing) and emits the rgba32float texel layout of
 * `FlowParticleSimulation` (east m/s, north m/s, validity, unused). Integration, lifecycle and
 * rendering run on the GPU; this file only builds the small (<= 512 x 512) velocity grid.
 */

export type FlowFieldColumnMode = 'UV' | 'SPEED_DIR' | 'ELEVATION';

export const METERS_PER_DEGREE_LATITUDE = 111320;
export const MINIMUM_FLOW_FIELD_GRID_AXIS = 4;
export const MAXIMUM_FLOW_FIELD_GRID_AXIS = 512;
/** `FlowParticleSimulation` restricts lnglat fields to this latitude. */
export const MAXIMUM_FLOW_FIELD_LATITUDE = 85;

export type FlowFieldSamples = {
  longitudes: ArrayLike<number>;
  latitudes: ArrayLike<number>;
  /** UV: east component. SPEED_DIR: speed. ELEVATION: altitude. */
  first: ArrayLike<number>;
  /** UV: north component. SPEED_DIR: direction in degrees the flow comes FROM. Unused for ELEVATION. */
  second?: ArrayLike<number>;
  /** Terrain height per row for streamline Z (ELEVATION mode uses `first`). NaN or absent: none. */
  altitude?: ArrayLike<number>;
  /** Resident filter mask (one word per row); rows with 0 are excluded like native filtered rows. */
  selected?: ArrayLike<number>;
};

export type FlowFieldGrid = {
  columns: number;
  rows: number;
  /** [west, south, east, north] of the first and last samples. */
  bounds: [number, number, number, number];
  /** rgba32float texels, row zero south: east, north, validity, 0. */
  texels: Float32Array;
  minimumSpeed: number;
  maximumSpeed: number;
  sampleCount: number;
  skippedCount: number;
  /** Cell terrain height in meters (NaN where a cell has none), row zero south; absent without altitude. */
  altitude?: Float32Array;
};

/** Validity-aware box blur of a scalar grid where NaN marks cells without a value. */
export function smoothFlowFieldScalarGrid(
  values: Float32Array,
  columns: number,
  rows: number,
  radius: number
): Float32Array {
  const reach = Math.max(0, Math.floor(radius));
  if (!reach) return values;
  const smoothed = values.slice();
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (Number.isNaN(values[index])) continue;
      let sum = 0;
      let count = 0;
      for (let rowOffset = -reach; rowOffset <= reach; rowOffset++) {
        const neighborRow = row + rowOffset;
        if (neighborRow < 0 || neighborRow >= rows) continue;
        for (let columnOffset = -reach; columnOffset <= reach; columnOffset++) {
          const neighborColumn = column + columnOffset;
          if (neighborColumn < 0 || neighborColumn >= columns) continue;
          const neighbor = values[neighborRow * columns + neighborColumn];
          if (Number.isNaN(neighbor)) continue;
          sum += neighbor;
          count++;
        }
      }
      smoothed[index] = sum / count;
    }
  }
  return smoothed;
}

export type FlowFieldGridOptions = {
  mode: FlowFieldColumnMode;
  gridResolution: number;
  smoothing: number;
};

/** Native convention: direction is the bearing the flow comes FROM (meteorological). */
export function convertSpeedDirectionToVector(
  speed: number,
  directionDegrees: number
): [number, number] {
  const radians = toRadians(directionDegrees);
  return [-speed * Math.sin(radians), -speed * Math.cos(radians)];
}

export function getFlowFieldBinCounts(
  west: number,
  south: number,
  east: number,
  north: number,
  gridResolution: number
): {columns: number; rows: number} {
  const clampAxis = (value: number) =>
    clamp(value, MINIMUM_FLOW_FIELD_GRID_AXIS, MAXIMUM_FLOW_FIELD_GRID_AXIS);
  const resolution = clampAxis(Math.round(gridResolution || 64));
  const longitudeSpan = Math.max(1e-9, east - west);
  const latitudeSpan = Math.max(1e-9, north - south);
  return longitudeSpan >= latitudeSpan
    ? {
        columns: resolution,
        rows: clampAxis(Math.round((resolution * latitudeSpan) / longitudeSpan))
      }
    : {
        columns: clampAxis(Math.round((resolution * longitudeSpan) / latitudeSpan)),
        rows: resolution
      };
}

/** Validity-aware box blur; unfilled cells stay unfilled, like native `boxBlur`. */
export function smoothFlowFieldGrid(
  east: Float32Array,
  north: Float32Array,
  filled: Uint8Array,
  columns: number,
  rows: number,
  radius: number
): void {
  const reach = Math.max(0, Math.floor(radius));
  if (!reach) return;
  const sourceEast = east.slice();
  const sourceNorth = north.slice();
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (!filled[index]) continue;
      let sumEast = 0;
      let sumNorth = 0;
      let count = 0;
      for (let rowOffset = -reach; rowOffset <= reach; rowOffset++) {
        const neighborRow = row + rowOffset;
        if (neighborRow < 0 || neighborRow >= rows) continue;
        for (let columnOffset = -reach; columnOffset <= reach; columnOffset++) {
          const neighborColumn = column + columnOffset;
          if (neighborColumn < 0 || neighborColumn >= columns) continue;
          const neighbor = neighborRow * columns + neighborColumn;
          if (!filled[neighbor]) continue;
          sumEast += sourceEast[neighbor];
          sumNorth += sourceNorth[neighbor];
          count++;
        }
      }
      east[index] = sumEast / count;
      north[index] = sumNorth / count;
    }
  }
}

/** Downhill gradient in m/s-equivalent units, matching native `elevationFieldToUV`. */
export function convertElevationToVector(
  altitude: Float32Array,
  filled: Uint8Array,
  columns: number,
  rows: number,
  west: number,
  south: number,
  east: number,
  north: number
): {east: Float32Array; north: Float32Array} {
  const eastward = new Float32Array(columns * rows);
  const northward = new Float32Array(columns * rows);
  const sourceFilled = filled.slice();
  const latitudeStep = rows > 1 ? (north - south) / (rows - 1) : 1;
  const longitudeStep = columns > 1 ? (east - west) / (columns - 1) : 1;
  const metersNorth = METERS_PER_DEGREE_LATITUDE * latitudeStep;
  for (let row = 0; row < rows; row++) {
    const latitude = south + row * latitudeStep;
    const metersEast =
      METERS_PER_DEGREE_LATITUDE * Math.max(0.2, Math.cos(toRadians(latitude))) * longitudeStep;
    for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (!sourceFilled[index]) continue;
      let gradientEast: number | null = null;
      if (column > 0 && column < columns - 1 && sourceFilled[index - 1] && sourceFilled[index + 1])
        gradientEast = (altitude[index + 1] - altitude[index - 1]) / (2 * metersEast);
      else if (column < columns - 1 && sourceFilled[index + 1])
        gradientEast = (altitude[index + 1] - altitude[index]) / metersEast;
      else if (column > 0 && sourceFilled[index - 1])
        gradientEast = (altitude[index] - altitude[index - 1]) / metersEast;
      let gradientNorth: number | null = null;
      if (
        row > 0 &&
        row < rows - 1 &&
        sourceFilled[index - columns] &&
        sourceFilled[index + columns]
      )
        gradientNorth = (altitude[index + columns] - altitude[index - columns]) / (2 * metersNorth);
      else if (row < rows - 1 && sourceFilled[index + columns])
        gradientNorth = (altitude[index + columns] - altitude[index]) / metersNorth;
      else if (row > 0 && sourceFilled[index - columns])
        gradientNorth = (altitude[index] - altitude[index - columns]) / metersNorth;
      if (gradientEast === null || gradientNorth === null) {
        filled[index] = 0;
        continue;
      }
      eastward[index] = -gradientEast;
      northward[index] = -gradientNorth;
    }
  }
  return {east: eastward, north: northward};
}

/** Bins scattered samples into the simulation grid. Non-finite or out-of-range rows are skipped. */
export function buildFlowFieldGrid(
  samples: FlowFieldSamples,
  options: FlowFieldGridOptions
): FlowFieldGrid | null {
  const count = samples.longitudes.length;
  const valid = new Uint8Array(count);
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  let skippedCount = 0;
  let filteredCount = 0;
  for (let index = 0; index < count; index++) {
    if (samples.selected && !samples.selected[index]) {
      filteredCount++;
      continue;
    }
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
    ) {
      skippedCount++;
      continue;
    }
    valid[index] = 1;
    west = Math.min(west, longitude);
    east = Math.max(east, longitude);
    south = Math.min(south, latitude);
    north = Math.max(north, latitude);
  }
  const sampleCount = count - skippedCount - filteredCount;
  // Native requires at least four points; a degenerate extent cannot form a two-sample grid.
  if (sampleCount < 4 || !(east > west) || !(north > south)) return null;
  const {columns, rows} = getFlowFieldBinCounts(west, south, east, north, options.gridResolution);
  const cellCount = columns * rows;
  const sumEast = new Float32Array(cellCount);
  const sumNorth = new Float32Array(cellCount);
  const counts = new Uint32Array(cellCount);
  const hasAltitude = options.mode === 'ELEVATION' || Boolean(samples.altitude);
  const sumAltitude = hasAltitude ? new Float64Array(cellCount) : undefined;
  const altitudeCounts = hasAltitude ? new Uint32Array(cellCount) : undefined;
  for (let index = 0; index < count; index++) {
    if (!valid[index]) continue;
    const column = Math.min(
      columns - 1,
      Math.floor(((samples.longitudes[index] - west) / (east - west)) * columns)
    );
    const row = Math.min(
      rows - 1,
      Math.floor(((samples.latitudes[index] - south) / (north - south)) * rows)
    );
    const cell = row * columns + column;
    if (options.mode === 'UV') {
      sumEast[cell] += samples.first[index];
      sumNorth[cell] += samples.second ? samples.second[index] : 0;
    } else if (options.mode === 'SPEED_DIR') {
      const [u, v] = convertSpeedDirectionToVector(
        samples.first[index],
        samples.second ? samples.second[index] : 0
      );
      sumEast[cell] += u;
      sumNorth[cell] += v;
    } else {
      sumEast[cell] += samples.first[index];
    }
    counts[cell]++;
    if (sumAltitude && altitudeCounts) {
      const altitudeValue =
        options.mode === 'ELEVATION' ? samples.first[index] : (samples.altitude?.[index] ?? NaN);
      if (Number.isFinite(altitudeValue)) {
        sumAltitude[cell] += altitudeValue;
        altitudeCounts[cell]++;
      }
    }
  }
  const filled = new Uint8Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    if (!counts[cell]) continue;
    sumEast[cell] /= counts[cell];
    sumNorth[cell] /= counts[cell];
    filled[cell] = 1;
  }
  let eastward: Float32Array = sumEast;
  let northward: Float32Array = sumNorth;
  if (options.mode === 'ELEVATION') {
    const gradient = convertElevationToVector(
      sumEast,
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
  const texels = new Float32Array(cellCount * 4);
  let minimumSpeed = Infinity;
  let maximumSpeed = 0;
  for (let cell = 0; cell < cellCount; cell++) {
    if (!filled[cell]) continue;
    texels[cell * 4] = eastward[cell];
    texels[cell * 4 + 1] = northward[cell];
    texels[cell * 4 + 2] = 1;
    const speed = Math.hypot(eastward[cell], northward[cell]);
    minimumSpeed = Math.min(minimumSpeed, speed);
    maximumSpeed = Math.max(maximumSpeed, speed);
  }
  // The w channel carries speed normalized over the grid's speed range (color by speed).
  const speedSpan = maximumSpeed - (Number.isFinite(minimumSpeed) ? minimumSpeed : 0);
  for (let cell = 0; cell < cellCount; cell++) {
    if (!filled[cell] || !(speedSpan > 1e-12)) continue;
    texels[cell * 4 + 3] = Math.max(
      0,
      Math.min(1, (Math.hypot(eastward[cell], northward[cell]) - minimumSpeed) / speedSpan)
    );
  }
  let altitude: Float32Array | undefined;
  if (sumAltitude && altitudeCounts) {
    altitude = new Float32Array(cellCount).fill(Number.NaN);
    for (let cell = 0; cell < cellCount; cell++)
      if (altitudeCounts[cell]) altitude[cell] = sumAltitude[cell] / altitudeCounts[cell];
    altitude = smoothFlowFieldScalarGrid(altitude, columns, rows, options.smoothing);
  }
  return {
    ...(altitude ? {altitude} : {}),
    columns,
    rows,
    bounds: [west, south, east, north],
    texels,
    minimumSpeed: Number.isFinite(minimumSpeed) ? minimumSpeed : 0,
    maximumSpeed,
    sampleCount,
    skippedCount
  };
}

/** Bilinear sample requiring all four corners valid, like `FlowParticleSimulation`. */
export function sampleFlowFieldGrid(
  grid: FlowFieldGrid,
  longitude: number,
  latitude: number
): {east: number; north: number} | null {
  const [west, south, east, north] = grid.bounds;
  const x = ((longitude - west) / (east - west)) * (grid.columns - 1);
  const y = ((latitude - south) / (north - south)) * (grid.rows - 1);
  if (!(x >= 0 && y >= 0 && x <= grid.columns - 1 && y <= grid.rows - 1)) return null;
  const column = Math.min(grid.columns - 2, Math.floor(x));
  const row = Math.min(grid.rows - 2, Math.floor(y));
  const fractionX = x - column;
  const fractionY = y - row;
  let sumEast = 0;
  let sumNorth = 0;
  for (const [offsetColumn, offsetRow, weight] of [
    [0, 0, (1 - fractionX) * (1 - fractionY)],
    [1, 0, fractionX * (1 - fractionY)],
    [0, 1, (1 - fractionX) * fractionY],
    [1, 1, fractionX * fractionY]
  ] as const) {
    const texel = ((row + offsetRow) * grid.columns + column + offsetColumn) * 4;
    if (grid.texels[texel + 2] < 0.5) return null;
    sumEast += grid.texels[texel] * weight;
    sumNorth += grid.texels[texel + 1] * weight;
  }
  return {east: sumEast, north: sumNorth};
}

const EARTH_RADIUS_METERS = MEAN_EARTH_RADIUS_METERS;

/**
 * CPU reference of one midpoint (RK2) step in degrees, mirroring the GPU shader's spherical
 * conversion. Returns null when the particle leaves the valid domain.
 */
export function advanceFlowFieldParticle(
  grid: FlowFieldGrid,
  longitude: number,
  latitude: number,
  deltaSeconds: number
): [number, number] | null {
  const start = sampleFlowFieldGrid(grid, longitude, latitude);
  if (!start) return null;
  const degreesPerMeter = toDegrees(1 / EARTH_RADIUS_METERS);
  const move = (
    sample: {east: number; north: number},
    fromLatitude: number,
    seconds: number
  ): [number, number] => [
    (sample.east * seconds * degreesPerMeter) / Math.cos(toRadians(fromLatitude)),
    sample.north * seconds * degreesPerMeter
  ];
  const [halfLongitude, halfLatitude] = move(start, latitude, deltaSeconds * 0.5);
  const middle = sampleFlowFieldGrid(grid, longitude + halfLongitude, latitude + halfLatitude);
  if (!middle) return null;
  const [stepLongitude, stepLatitude] = move(middle, latitude + halfLatitude, deltaSeconds);
  const nextLongitude = longitude + stepLongitude;
  const nextLatitude = latitude + stepLatitude;
  return sampleFlowFieldGrid(grid, nextLongitude, nextLatitude)
    ? [nextLongitude, nextLatitude]
    : null;
}
