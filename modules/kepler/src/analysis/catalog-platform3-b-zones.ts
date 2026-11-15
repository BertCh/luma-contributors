// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import type {CatalogRunOutput} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  appendCatalogRow,
  getFeatureIdentity,
  getPointPositions,
  getPolygonParts,
  makeCatalogGeometryText,
  makeCatalogTable,
  type CatalogCoordinateSystem,
  type CatalogDiagnostic,
  type CatalogFeatureCollection
} from './catalog-geojson';
import {isCatalogPointInPolygon} from './catalog-gpu-pack';
import {
  addDroppedDiagnostic,
  collectAllPositions,
  getFeatureGeometryText,
  readNumericProperty
} from './catalog-platform3-b-common';
import {makeLocalFrame} from './catalog-local-frame';
import {isInsideFramePolygonParts, readFramePolygons} from './catalog-platform3-b-grids';

export type CellStatistic = 'count' | 'sum' | 'mean' | 'min' | 'max';

type Accumulator = {count: number; validCount: number; sum: number; min: number; max: number};

function addToAccumulator(accumulator: Accumulator, value: number | null): void {
  accumulator.count++;
  if (value === null) return;
  accumulator.validCount++;
  accumulator.sum += value;
  accumulator.min = Math.min(accumulator.min, value);
  accumulator.max = Math.max(accumulator.max, value);
}

function makeAccumulator(): Accumulator {
  return {count: 0, validCount: 0, sum: 0, min: Infinity, max: -Infinity};
}

function getStatisticValue(accumulator: Accumulator | undefined, statistic: CellStatistic) {
  if (!accumulator || !accumulator.count)
    return statistic === 'count' || statistic === 'sum' ? 0 : null;
  if (statistic === 'count') return accumulator.count;
  if (statistic === 'sum') return accumulator.sum;
  if (!accumulator.validCount) return null;
  if (statistic === 'mean') return accumulator.sum / accumulator.validCount;
  return statistic === 'min' ? accumulator.min : accumulator.max;
}

function makeLatticeRingText(cellX: number, cellY: number, cellSize: number): string {
  const west = cellX * cellSize;
  const south = cellY * cellSize;
  return makeCatalogGeometryText('Polygon', [
    [
      [west, south],
      [west + cellSize, south],
      [west + cellSize, south + cellSize],
      [west, south + cellSize],
      [west, south]
    ]
  ]);
}

type LatticeSamples = {
  keys: Map<string, {cellX: number; cellY: number; accumulator: Accumulator}>;
  droppedCount: number;
  sampleCount: number;
  missingValueCount: number;
};

function accumulateLattice(
  points: CatalogFeatureCollection,
  valueProperty: string,
  cellSize: number,
  keepMissingValues = false
): LatticeSamples {
  const keys: LatticeSamples['keys'] = new Map();
  let droppedCount = 0;
  let sampleCount = 0;
  let missingValueCount = 0;
  for (const feature of points.features) {
    const positions = getPointPositions(feature.geometry);
    const value = valueProperty ? readNumericProperty(feature, valueProperty) : 1;
    if (!positions.length || (value === null && !keepMissingValues)) {
      droppedCount++;
      continue;
    }
    for (const [x, y] of positions) {
      const cellX = Math.floor(x / cellSize);
      const cellY = Math.floor(y / cellSize);
      const key = `${cellX}:${cellY}`;
      let entry = keys.get(key);
      if (!entry) {
        entry = {cellX, cellY, accumulator: makeAccumulator()};
        keys.set(key, entry);
      }
      addToAccumulator(entry.accumulator, value);
      if (value === null) missingValueCount++;
      sampleCount++;
    }
  }
  return {keys, droppedCount, sampleCount, missingValueCount};
}

/** Square-lattice aggregation: cells are `floor(coordinate / cellSize)` in the input's own units. */
export function runCellAggregation(
  points: CatalogFeatureCollection,
  parameters: {cellSize: number; valueProperty: string}
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const lattice = accumulateLattice(points, parameters.valueProperty, parameters.cellSize, true);
  addDroppedDiagnostic(diagnostics, lattice.droppedCount);
  if (lattice.missingValueCount)
    addCatalogDiagnostic(
      diagnostics,
      'missing-value',
      'warning',
      'cell-aggregation valueProperty: points without a numeric value are included in count but excluded from sum, mean, min and max.',
      lattice.missingValueCount
    );
  const table = makeCatalogTable(
    'cells',
    'Aggregated cells',
    {
      cell_id: 'string',
      cell_x: 'int32',
      cell_y: 'int32',
      count: 'int32',
      sum: 'float64',
      mean: 'float64',
      min: 'float64',
      max: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  const entries = [...lattice.keys.values()].sort((a, b) => a.cellY - b.cellY || a.cellX - b.cellX);
  for (const {cellX, cellY, accumulator} of entries)
    appendCatalogRow(table, {
      cell_id: `${cellX}:${cellY}`,
      cell_x: cellX,
      cell_y: cellY,
      count: accumulator.count,
      sum: accumulator.sum,
      mean: getStatisticValue(accumulator, 'mean'),
      min: getStatisticValue(accumulator, 'min'),
      max: getStatisticValue(accumulator, 'max'),
      geometry: makeLatticeRingText(cellX, cellY, parameters.cellSize)
    });
  return {
    tables: [table],
    diagnostics,
    summary: {
      cellCount: table.rowCount,
      sampleCount: lattice.sampleCount,
      missingValueCount: lattice.missingValueCount
    }
  };
}

/** Multi-level lattice rollup: each level multiplies the cell size by `factor`. */
export function runCellRollup(
  points: CatalogFeatureCollection,
  parameters: {cellSize: number; valueProperty: string; levels: number; factor: number}
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const table = makeCatalogTable(
    'rollup',
    'Rolled-up cells',
    {
      cell_id: 'string',
      parent_cell_id: 'string',
      level: 'int32',
      cell_x: 'int32',
      cell_y: 'int32',
      count: 'int32',
      sum: 'float64',
      mean: 'float64',
      min: 'float64',
      max: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  let sampleCount = 0;
  for (let level = 0; level <= parameters.levels; level++) {
    const cellSize = parameters.cellSize * parameters.factor ** level;
    const lattice = accumulateLattice(points, parameters.valueProperty, cellSize);
    if (level === 0) {
      addDroppedDiagnostic(diagnostics, lattice.droppedCount);
      sampleCount = lattice.sampleCount;
    }
    const entries = [...lattice.keys.values()].sort(
      (a, b) => a.cellY - b.cellY || a.cellX - b.cellX
    );
    for (const {cellX, cellY, accumulator} of entries)
      appendCatalogRow(table, {
        cell_id: `${level}:${cellX}:${cellY}`,
        parent_cell_id:
          level < parameters.levels
            ? `${level + 1}:${Math.floor(cellX / parameters.factor)}:${Math.floor(cellY / parameters.factor)}`
            : null,
        level,
        cell_x: cellX,
        cell_y: cellY,
        count: accumulator.count,
        sum: accumulator.sum,
        mean: accumulator.sum / accumulator.count,
        min: accumulator.min,
        max: accumulator.max,
        geometry: makeLatticeRingText(cellX, cellY, cellSize)
      });
  }
  return {
    tables: [table],
    diagnostics,
    summary: {rowCount: table.rowCount, levels: parameters.levels + 1, sampleCount}
  };
}

/** Per-cell comparison of two point sets on one lattice. */
export function runCellCompare(
  before: CatalogFeatureCollection,
  after: CatalogFeatureCollection,
  parameters: {cellSize: number; valueProperty: string; statistic: 'count' | 'sum' | 'mean'}
): CatalogRunOutput {
  validateRequirement(
    'cell-compare',
    parameters.statistic === 'count' || Boolean(parameters.valueProperty.trim()),
    'valueProperty must name a numeric value column when statistic is sum or mean; choose count to compare without a value column'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  const first = accumulateLattice(before, parameters.valueProperty, parameters.cellSize);
  const second = accumulateLattice(after, parameters.valueProperty, parameters.cellSize);
  addDroppedDiagnostic(diagnostics, first.droppedCount + second.droppedCount);
  const table = makeCatalogTable(
    'comparison',
    'Cell comparison',
    {
      cell_id: 'string',
      cell_x: 'int32',
      cell_y: 'int32',
      before_value: 'float64',
      after_value: 'float64',
      change: 'float64',
      relative_change: 'float64',
      status: 'string',
      geometry: 'string'
    },
    'geometry'
  );
  const merged = new Map<string, {cellX: number; cellY: number}>();
  for (const lattice of [first, second])
    for (const [key, entry] of lattice.keys) merged.set(key, entry);
  const entries = [...merged.entries()].sort(
    (a, b) => a[1].cellY - b[1].cellY || a[1].cellX - b[1].cellX
  );
  for (const [key, {cellX, cellY}] of entries) {
    const beforeValue = getStatisticValue(first.keys.get(key)?.accumulator, parameters.statistic);
    const afterValue = getStatisticValue(second.keys.get(key)?.accumulator, parameters.statistic);
    const beforeNumber = beforeValue ?? 0;
    const afterNumber = afterValue ?? 0;
    appendCatalogRow(table, {
      cell_id: key,
      cell_x: cellX,
      cell_y: cellY,
      before_value: beforeValue,
      after_value: afterValue,
      change: afterNumber - beforeNumber,
      relative_change: beforeNumber ? (afterNumber - beforeNumber) / Math.abs(beforeNumber) : null,
      status: !first.keys.has(key) ? 'appeared' : !second.keys.has(key) ? 'disappeared' : 'both',
      geometry: makeLatticeRingText(cellX, cellY, parameters.cellSize)
    });
  }
  return {
    tables: [table],
    diagnostics,
    summary: {
      cellCount: table.rowCount,
      beforeSampleCount: first.sampleCount,
      afterSampleCount: second.sampleCount
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Zonal statistics of points in polygons
// ---------------------------------------------------------------------------------------------

export function runZonalPointStatistics(
  polygons: CatalogFeatureCollection,
  points: CatalogFeatureCollection,
  parameters: {valueProperty: string}
): CatalogRunOutput {
  const diagnostics: CatalogDiagnostic[] = [];
  const zones = polygons.features.map(feature => {
    const parts = getPolygonParts(feature.geometry);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const polygon of parts)
      for (const [x, y] of polygon[0] ?? []) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    return {parts, minX, minY, maxX, maxY, accumulator: makeAccumulator(), sumSquares: 0};
  });
  let droppedCount = 0;
  let unassignedCount = 0;
  for (const feature of points.features) {
    const positions = getPointPositions(feature.geometry);
    const value = parameters.valueProperty
      ? readNumericProperty(feature, parameters.valueProperty)
      : 1;
    if (!positions.length || value === null) {
      droppedCount++;
      continue;
    }
    for (const position of positions) {
      let assigned = false;
      for (const zone of zones) {
        if (
          position[0] < zone.minX ||
          position[0] > zone.maxX ||
          position[1] < zone.minY ||
          position[1] > zone.maxY
        )
          continue;
        if (!zone.parts.some(rings => rings.length && isCatalogPointInPolygon(position, rings)))
          continue;
        addToAccumulator(zone.accumulator, value);
        zone.sumSquares += value * value;
        assigned = true;
      }
      if (!assigned) unassignedCount++;
    }
  }
  addDroppedDiagnostic(diagnostics, droppedCount);
  if (unassignedCount)
    addCatalogDiagnostic(
      diagnostics,
      'points-outside-zones',
      'info',
      'Points inside no polygon contribute to no zone.',
      unassignedCount
    );
  const table = makeCatalogTable(
    'zones',
    'Zone statistics',
    {
      polygon_index: 'int32',
      polygon_id: 'string',
      count: 'int32',
      sum: 'float64',
      mean: 'float64',
      min: 'float64',
      max: 'float64',
      standard_deviation: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  polygons.features.forEach((feature, polygonIndex) => {
    const {accumulator, sumSquares} = zones[polygonIndex];
    const mean = accumulator.count ? accumulator.sum / accumulator.count : null;
    appendCatalogRow(table, {
      polygon_index: polygonIndex,
      polygon_id: getFeatureIdentity(feature),
      count: accumulator.count,
      sum: accumulator.sum,
      mean,
      min: accumulator.count ? accumulator.min : null,
      max: accumulator.count ? accumulator.max : null,
      standard_deviation:
        accumulator.count && mean !== null
          ? Math.sqrt(Math.max(0, sumSquares / accumulator.count - mean * mean))
          : null,
      geometry: getFeatureGeometryText(feature)
    });
  });
  return {
    tables: [table],
    diagnostics,
    summary: {
      zoneCount: table.rowCount,
      assignedPointCount: zones.reduce((n, z) => n + z.accumulator.count, 0)
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Areal interpolation (change of support) by area-proportional sampling
// ---------------------------------------------------------------------------------------------

export function runArealInterpolation(
  sources: CatalogFeatureCollection,
  targets: CatalogFeatureCollection,
  parameters: {
    valueProperty: string;
    variableType: 'extensive' | 'intensive';
    samplesPerSide: number;
    coordinateSystem: CatalogCoordinateSystem;
  }
): CatalogRunOutput {
  validateRequirement(
    'areal-interpolation',
    parameters.variableType === 'extensive' || parameters.variableType === 'intensive',
    'variableType must be explicitly selected as extensive (counts) or intensive (rates); no variable type is inferred from valueProperty'
  );
  const diagnostics: CatalogDiagnostic[] = [];
  const frame = makeLocalFrame(
    parameters.coordinateSystem,
    [...collectAllPositions(sources), ...collectAllPositions(targets)],
    {kind: 'lambert-equal-area', shift: 'minimum'}
  );
  const sourceRegions = readFramePolygons(sources, frame);
  const targetRegions = readFramePolygons(targets, frame);
  const n = parameters.samplesPerSide;
  // Source-times-sample-times-target work exceeds the CPU reference capacity.
  if (sourceRegions.length * n * n * targetRegions.length > 2e8)
    throw new KeplerAllocationError('capacity', {message: 'Work capacity exceeded'});
  const targetBounds = targetRegions.map(region => {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const rings of region.rings)
      for (const [x, y] of rings[0] ?? []) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    return {minX, minY, maxX, maxY};
  });
  const overlapArea = new Float64Array(targetRegions.length);
  const weightedValue = new Float64Array(targetRegions.length);
  const contributorCount = new Uint32Array(targetRegions.length);
  let droppedCount = 0;
  let emptySourceCount = 0;
  sourceRegions.forEach(source => {
    const value = readNumericProperty(
      sources.features[source.featureIndex],
      parameters.valueProperty
    );
    if (value === null || source.area <= 0) {
      droppedCount++;
      return;
    }
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const rings of source.rings)
      for (const [x, y] of rings[0] ?? []) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    const inside: [number, number][] = [];
    for (let row = 0; row < n; row++)
      for (let column = 0; column < n; column++) {
        const x = minX + ((column + 0.5) / n) * (maxX - minX);
        const y = minY + ((row + 0.5) / n) * (maxY - minY);
        if (isInsideFramePolygonParts(x, y, source.rings)) inside.push([x, y]);
      }
    if (!inside.length) {
      emptySourceCount++;
      return;
    }
    const sampleArea = source.area / inside.length;
    const counts = new Map<number, number>();
    for (const [x, y] of inside)
      for (let target = 0; target < targetRegions.length; target++) {
        const box = targetBounds[target];
        if (x < box.minX || x > box.maxX || y < box.minY || y > box.maxY) continue;
        if (!isInsideFramePolygonParts(x, y, targetRegions[target].rings)) continue;
        counts.set(target, (counts.get(target) ?? 0) + 1);
        break;
      }
    for (const [target, count] of counts) {
      const overlap = count * sampleArea;
      overlapArea[target] += overlap;
      contributorCount[target]++;
      weightedValue[target] +=
        parameters.variableType === 'extensive' ? (value * count) / inside.length : value * overlap;
    }
  });
  addDroppedDiagnostic(
    diagnostics,
    droppedCount,
    'Source polygons without a numeric value or with no area are skipped.'
  );
  if (emptySourceCount)
    addCatalogDiagnostic(
      diagnostics,
      'source-not-sampled',
      'warning',
      'Source polygons smaller than one sample cell contribute nothing; raise samplesPerSide.',
      emptySourceCount
    );
  addCatalogDiagnostic(
    diagnostics,
    'sampled-overlap',
    'info',
    'Overlaps are estimated from a regular sample grid inside each source polygon, not exact intersections.'
  );
  const table = makeCatalogTable(
    'targets',
    'Interpolated target values',
    {
      target_index: 'int32',
      target_id: 'string',
      value: 'float64',
      source_count: 'int32',
      overlap_area: 'float64',
      target_area: 'float64',
      coverage: 'float64',
      geometry: 'string'
    },
    'geometry'
  );
  targetRegions.forEach((region, targetIndex) => {
    const feature = targets.features[region.featureIndex];
    appendCatalogRow(table, {
      target_index: targetIndex,
      target_id: getFeatureIdentity(feature),
      value:
        parameters.variableType === 'extensive'
          ? weightedValue[targetIndex]
          : overlapArea[targetIndex]
            ? weightedValue[targetIndex] / overlapArea[targetIndex]
            : null,
      source_count: contributorCount[targetIndex],
      overlap_area: overlapArea[targetIndex],
      target_area: region.area,
      coverage: region.area > 0 ? Math.min(1, overlapArea[targetIndex] / region.area) : null,
      geometry: getFeatureGeometryText(feature)
    });
  });
  return {
    tables: [table],
    diagnostics,
    summary: {targetCount: table.rowCount, sourceCount: sourceRegions.length}
  };
}
