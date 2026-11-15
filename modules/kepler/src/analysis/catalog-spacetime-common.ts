// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  getGPUSpaceTimeGuardedRelativeTimes,
  getGPUSpaceTimeRelativeTimes,
  type GPUSpatialWeights
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogFeatureCollection, CatalogCoordinateSystem} from './catalog-geojson';
import {appendCatalogRow, getFeatureIdentity, makeCatalogTable} from './catalog-geojson';
import {makeLocalFrame} from './catalog-local-frame';
import type {CatalogGpuSession} from './catalog-gpu-session';
import {collectAllPositions, type Values} from './catalog-platform3-b-common';

export const SPACETIME_MAXIMUM_FEATURES = 512;
export const SPACETIME_MAXIMUM_LOCATIONS = 64;
export const SPACETIME_MAXIMUM_PERIODS = 128;
export const SPACETIME_MAXIMUM_VOXELS = 16384;

/** Host mapping only: require one finite Point and numeric time per resident row. No numerical analysis. */
export function readSpacetimePoints(
  collection: CatalogFeatureCollection,
  values: Values,
  guardTimeResolution = true
) {
  if (!collection.features.length || collection.features.length > SPACETIME_MAXIMUM_FEATURES)
    throw new Error('Spacetime points requires 1 to 512 features');
  const frame = makeLocalFrame(
    values['coordinateSystem'] as CatalogCoordinateSystem,
    collectAllPositions(collection),
    {shift: 'minimum'}
  );
  const positions: number[] = [];
  const absoluteTimes: number[] = [];
  const timeProperty = String(values['timeProperty']);
  collection.features.forEach((feature, index) => {
    const geometry = feature.geometry;
    const time = feature.properties?.[timeProperty];
    if (geometry?.type !== 'Point' || typeof time !== 'number' || !Number.isFinite(time))
      throw new Error(
        `Spacetime points row ${index} requires a Point and finite numeric ${timeProperty}`
      );
    const position = frame.forward(geometry.coordinates as [number, number]);
    if (!position.every(Number.isFinite))
      throw new Error(`Spacetime points row ${index} cannot be projected`);
    positions.push(...position);
    absoluteTimes.push(time);
  });
  const timeScale = values['timeUnit'] === 'milliseconds' ? 1000 : 1;
  const timeOrigin = Math.min(...absoluteTimes);
  const relativeSeconds = absoluteTimes.map(time => (time - timeOrigin) / timeScale);
  const orderedTimes = [...new Set(relativeSeconds)].sort((first, second) => first - second);
  const minimumInterval =
    orderedTimes.length > 1
      ? Math.min(...orderedTimes.slice(1).map((time, index) => time - orderedTimes[index]))
      : 1;
  const narrowestTimeWidth = Number(
    values['temporalBandwidth'] ??
      values['temporalThreshold'] ??
      (values['maximumTimeLag'] === undefined
        ? minimumInterval
        : Number(values['maximumTimeLag']) / Number(values['classCount']))
  );
  const relative = guardTimeResolution
    ? getGPUSpaceTimeGuardedRelativeTimes(relativeSeconds, narrowestTimeWidth)
    : getGPUSpaceTimeRelativeTimes(relativeSeconds);
  const packedPositions = Float32Array.from(positions);
  if (!packedPositions.every(Number.isFinite))
    throw new Error('Spacetime positions exceed float32 range');
  return {
    collection,
    frame,
    positions: packedPositions,
    times: relative.times,
    absoluteTimes,
    timeOrigin,
    timeScale,
    count: absoluteTimes.length
  };
}
export type SpacetimePoints = ReturnType<typeof readSpacetimePoints>;

/** Dense cell-major series. Locations use an explicit ID property or exact original Point coordinates. */
export function readSpacetimeSeries(
  points: SpacetimePoints,
  values: Values,
  requireSecond = false,
  requireBaseline = false
) {
  const locationProperty = String(values['locationProperty'] ?? '');
  const locationIndices = new Map<string, number>();
  const periods = [...new Set(points.absoluteTimes)].sort((first, second) => first - second);
  const periodIndices = new Map(periods.map((time, index) => [time, index]));
  const sourceCells: number[] = [];
  const sourcePeriods: number[] = [];
  const locationSources: number[] = [];
  points.collection.features.forEach((feature, sourceIndex) => {
    const identity = locationProperty
      ? feature.properties?.[locationProperty]
      : JSON.stringify(feature.geometry);
    if (identity === undefined || identity === null)
      throw new Error(`Spacetime series missing ${locationProperty}`);
    const key = `${typeof identity}:${identity}`;
    let cell = locationIndices.get(key);
    if (cell === undefined) {
      cell = locationIndices.size;
      locationIndices.set(key, cell);
      locationSources.push(sourceIndex);
    } else {
      const original = points.collection.features[locationSources[cell]].geometry;
      if (JSON.stringify(original) !== JSON.stringify(feature.geometry))
        throw new Error('Spacetime series location coordinates must be constant');
    }
    sourceCells.push(cell);
    sourcePeriods.push(periodIndices.get(points.absoluteTimes[sourceIndex])!);
  });
  const cellCount = locationIndices.size;
  if (
    cellCount > SPACETIME_MAXIMUM_LOCATIONS ||
    periods.length < 2 ||
    periods.length > SPACETIME_MAXIMUM_PERIODS
  )
    throw new Error('Spacetime series requires at most 64 locations and 2 to 128 periods');
  // A regular axis is required: these contributors use period offsets, not irregular time differences.
  const interval = periods[1] - periods[0];
  if (
    periods.some(
      (time, index) =>
        Math.abs(time - periods[0] - index * interval) > Math.max(1e-6, interval * 1e-9)
    )
  )
    throw new Error('Spacetime series requires equally spaced periods');
  const rowCount = cellCount * periods.length;
  const cube = new Float32Array(rowCount).fill(Number.NaN);
  const second = new Float32Array(rowCount).fill(Number.NaN);
  const baseline = new Float32Array(rowCount).fill(Number.NaN);
  const sourceRows = new Int32Array(rowCount).fill(-1);
  points.collection.features.forEach((feature, sourceIndex) => {
    const row = sourceCells[sourceIndex] * periods.length + sourcePeriods[sourceIndex];
    if (sourceRows[row] !== -1)
      throw new Error('Spacetime series has duplicate location/time rows');
    sourceRows[row] = sourceIndex;
    for (const [property, target, required] of [
      [values['valueProperty'], cube, true],
      [values['secondValueProperty'], second, requireSecond],
      [values['baselineProperty'], baseline, requireBaseline]
    ] as const) {
      if (!required) continue;
      const value = feature.properties?.[String(property)];
      if (
        typeof value !== 'number' ||
        !Number.isFinite(value) ||
        !Number.isFinite(Math.fround(value))
      )
        throw new Error(`Spacetime series requires finite numeric ${String(property)}`);
      target[row] = value;
    }
  });
  if (sourceRows.some(row => row === -1))
    throw new Error('Spacetime series requires a complete location/time rectangle');
  const positions = Float32Array.from(
    locationSources.flatMap(source => [
      points.positions[source * 2],
      points.positions[source * 2 + 1]
    ])
  );
  return {
    cube,
    second,
    baseline,
    sourceCells,
    sourcePeriods,
    sourceRows,
    locationSources,
    positions,
    periods,
    cellCount,
    periodCount: periods.length,
    intervalSeconds: interval / points.timeScale
  };
}
export type SpacetimeSeries = ReturnType<typeof readSpacetimeSeries>;

/** Deterministic distance/nearest-neighbor CSR host mapping; row ties use the source index. */
export function makeSpacetimeNeighbors(
  positions: Float32Array,
  radius: number,
  neighborCount?: number
) {
  const count = positions.length / 2;
  const offsets = new Uint32Array(count + 1);
  const neighbors: number[] = [];
  const distances: number[] = [];
  for (let row = 0; row < count; row++) {
    const candidates = Array.from({length: count}, (_, index) => ({
      index,
      distance: Math.hypot(
        positions[row * 2] - positions[index * 2],
        positions[row * 2 + 1] - positions[index * 2 + 1]
      )
    }))
      // Match the distance words uploaded below and the contributor's Float32 closed bands.
      .filter(
        candidate =>
          candidate.index !== row && Math.fround(candidate.distance) <= Math.fround(radius)
      )
      .sort((first, second) => first.distance - second.distance || first.index - second.index)
      .slice(0, neighborCount ?? count)
      .sort((first, second) => first.index - second.index);
    for (const candidate of candidates) {
      neighbors.push(candidate.index);
      distances.push(candidate.distance);
    }
    offsets[row + 1] = neighbors.length;
  }
  return {offsets, neighbors: Uint32Array.from(neighbors), distances: Float32Array.from(distances)};
}

export function uploadSpacetimeNeighbors(
  session: CatalogGpuSession,
  neighbors: ReturnType<typeof makeSpacetimeNeighbors>
): GPUSpatialWeights {
  return {
    offsets: session.input('offsets', neighbors.offsets, 'uint32'),
    neighbors: session.input('neighbors', neighbors.neighbors, 'uint32'),
    distances: session.input('distances', neighbors.distances, 'float32'),
    weights: session.input(
      'weights',
      new Float32Array(neighbors.neighbors.length).fill(1),
      'float32'
    )
  };
}

export function makeSpacetimeGrid(
  points: SpacetimePoints,
  values: Values
): {grid: GPURasterMetadata; bounds: [number, number, number, number]} {
  const padding = Number(values['spatialBandwidth']);
  const horizontal = Array.from(points.positions).filter((_, index) => index % 2 === 0);
  const vertical = Array.from(points.positions).filter((_, index) => index % 2 === 1);
  const bounds: [number, number, number, number] = [
    Math.min(...horizontal) - padding,
    Math.min(...vertical) - padding,
    Math.max(...horizontal) + padding,
    Math.max(...vertical) + padding
  ];
  const width = Number(values['gridSize']);
  return {
    bounds,
    grid: makeGPURasterMetadata({
      width,
      height: width,
      cellSize: [(bounds[2] - bounds[0]) / width, (bounds[3] - bounds[1]) / width],
      origin: [bounds[0], bounds[1]],
      rowDirection: 'north'
    })
  };
}

export function makeSpacetimePointTable(
  points: SpacetimePoints,
  columns: Record<string, readonly (number | null)[]>
) {
  const table = makeCatalogTable(
    'points',
    'Spacetime point results',
    {
      source_index: 'int32',
      source_id: 'string',
      geometry: 'string',
      ...Object.fromEntries(Object.keys(columns).map(name => [name, 'float64' as const]))
    },
    'geometry'
  );
  points.collection.features.forEach((feature, sourceIndex) =>
    appendCatalogRow(table, {
      source_index: sourceIndex,
      source_id: getFeatureIdentity(feature),
      geometry: JSON.stringify(feature.geometry),
      ...Object.fromEntries(
        Object.entries(columns).map(([name, entries]) => [name, entries[sourceIndex] ?? null])
      )
    })
  );
  return table;
}
export const finiteSpacetimeValue = (value: number): number | null =>
  Number.isFinite(value) && value !== 0xffffffff ? value : null;

export function makeSpacetimeSummaryTable(summary: Record<string, string | number>) {
  const table = makeCatalogTable('statistics', 'Spacetime statistics', {
    statistic: 'string',
    value: 'float64',
    description: 'string'
  });
  for (const [name, value] of Object.entries(summary))
    appendCatalogRow(table, {
      statistic: name,
      value: typeof value === 'number' ? finiteSpacetimeValue(value) : null,
      description: typeof value === 'string' ? value : ''
    });
  return table;
}

export function getSpacetimeSummary(
  values: Float32Array,
  fields: Readonly<Record<string, number>>,
  offset = 0
): Record<string, number | string> {
  return Object.fromEntries(
    Object.entries(fields).map(([name, index]) => [
      name,
      Number.isFinite(values[offset + index]) ? values[offset + index] : 'undefined'
    ])
  );
}
