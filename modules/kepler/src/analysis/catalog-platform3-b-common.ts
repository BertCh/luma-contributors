// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {KeplerAllocationError} from '../runtime/dataset-registry';
import type {LocalFrame} from './catalog-local-frame';
import type {
  CatalogInputSpec,
  CatalogOutputSpec,
  CatalogParameterSpec,
  CatalogParameterValue,
  CatalogRunOutput
} from './catalog-binding-types';
import {COORDINATE_SYSTEM_PARAMETER} from './catalog-binding-types';
import {
  addCatalogDiagnostic,
  getLineParts,
  getPointPositions,
  getPolygonParts,
  makeCatalogGeometryText,
  type CatalogDiagnostic,
  type CatalogFeature,
  type CatalogFeatureCollection,
  type CatalogPosition
} from './catalog-geojson';

export type Values = Readonly<Record<string, CatalogParameterValue>>;
export type Inputs = Readonly<Record<string, CatalogFeatureCollection>>;

export const PLATFORM3_CPU_CLEANUP =
  'CPU only; no retained GPU resources. Published results are removed through native Sources.';

export const PLATFORM3_GPU_OR_CPU_CLEANUP =
  'CPU engine retains nothing. The GPU engine creates storage buffers and compiled graphs per run and destroys them before the result returns; results are plain arrays and publication goes through native Sources.';

export const MAXIMUM_GRID_CELLS = 250000;

export function makeInputSpec(
  name: string,
  label: string,
  accepts: CatalogInputSpec['accepts'],
  maximumFeatures = 20000,
  maximumVertices = 200000
): CatalogInputSpec {
  return {name, label, accepts, maximumFeatures, maximumVertices};
}

export function makeOutputSpec(
  tableId: string,
  label: string,
  geometry: CatalogOutputSpec['geometry'],
  identityColumns: readonly string[],
  maximumRows = 20000
): CatalogOutputSpec {
  return {tableId, label, geometry, identityColumns, maximumRows};
}

export function makeNumberParameter(
  name: string,
  label: string,
  defaultValue: number,
  minimum: number,
  maximum: number,
  unit?: string,
  kind: 'number' | 'integer' = 'number'
): CatalogParameterSpec {
  return {name, label, kind, minimum, maximum, defaultValue, ...(unit ? {unit} : {})};
}

export function makeStringParameter(
  name: string,
  label: string,
  defaultValue = ''
): CatalogParameterSpec {
  return {name, label, kind: 'string', maximumLength: 64, defaultValue};
}

export const SEED_PARAMETER_SPEC: CatalogParameterSpec = {
  name: 'seed',
  label: 'Random seed',
  kind: 'integer',
  minimum: 0,
  maximum: 4294967295,
  defaultValue: 1
};

export const PERMUTATION_PARAMETER_SPEC: CatalogParameterSpec = {
  name: 'permutations',
  label: 'Conditional permutations (0 disables)',
  kind: 'integer',
  minimum: 0,
  maximum: 999,
  defaultValue: 99
};

export {COORDINATE_SYSTEM_PARAMETER};

/** Deterministic 32-bit generator (mulberry32). */
export function makeSeededGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Reads a numeric feature property; numeric strings count, everything else is null. */
export function readNumericProperty(feature: CatalogFeature, name: string): number | null {
  if (!name) return null;
  const value = feature.properties?.[name];
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

/** Every vertex of a collection, for frame construction. */
export function collectAllPositions(collection: CatalogFeatureCollection): CatalogPosition[] {
  const positions: CatalogPosition[] = [];
  for (const feature of collection.features) {
    positions.push(...getPointPositions(feature.geometry));
    for (const part of getLineParts(feature.geometry)) positions.push(...part);
    for (const polygon of getPolygonParts(feature.geometry))
      for (const ring of polygon) positions.push(...ring);
  }
  return positions;
}

export type Sample = {
  /** Frame coordinates in meters (or planar units) from the data minimum. */
  x: Float64Array;
  y: Float64Array;
  value: Float64Array;
  sourceIndex: Int32Array;
  partIndex: Int32Array;
  count: number;
  droppedCount: number;
};

/**
 * Point samples of a collection. `valueProperty === ''` gives every sample the value one. Features
 * without a point position, or with a missing or non-numeric value, are dropped and counted.
 */
export function readPointSamples(
  collection: CatalogFeatureCollection,
  valueProperty: string,
  frame: LocalFrame
): Sample {
  const x: number[] = [];
  const y: number[] = [];
  const value: number[] = [];
  const sourceIndex: number[] = [];
  const partIndex: number[] = [];
  let droppedCount = 0;
  collection.features.forEach((feature, featureIndex) => {
    const positions = getPointPositions(feature.geometry);
    const numeric = valueProperty ? readNumericProperty(feature, valueProperty) : 1;
    if (!positions.length || numeric === null) {
      droppedCount++;
      return;
    }
    positions.forEach((position, index) => {
      const [px, py] = frame.forward(position);
      x.push(px);
      y.push(py);
      value.push(numeric);
      sourceIndex.push(featureIndex);
      partIndex.push(index);
    });
  });
  return {
    x: Float64Array.from(x),
    y: Float64Array.from(y),
    value: Float64Array.from(value),
    sourceIndex: Int32Array.from(sourceIndex),
    partIndex: Int32Array.from(partIndex),
    count: x.length,
    droppedCount
  };
}

export type GridSpec = {
  /** Frame coordinates of the minimum corner of cell (0, 0). */
  minX: number;
  minY: number;
  cellSize: number;
  columns: number;
  rows: number;
};

/** Grid covering [0, extent] of the frame plus padding; rows advance with y. Throws over capacity. */
export function makeGridSpec(
  extentX: number,
  extentY: number,
  cellSize: number,
  padding = 0
): GridSpec {
  const columns = Math.max(1, Math.floor((extentX + 2 * padding) / cellSize) + 1);
  const rows = Math.max(1, Math.floor((extentY + 2 * padding) / cellSize) + 1);
  // Grid cell count exceeds the 250,000 cell capacity; raise the cell size.
  if (columns * rows > MAXIMUM_GRID_CELLS)
    throw new KeplerAllocationError('capacity', {message: 'Grid capacity exceeded'});
  return {minX: -padding, minY: -padding, cellSize, columns, rows};
}

export function getSampleExtent(sample: Sample): {width: number; height: number} {
  let width = 0;
  let height = 0;
  for (let index = 0; index < sample.count; index++) {
    width = Math.max(width, sample.x[index]);
    height = Math.max(height, sample.y[index]);
  }
  return {width, height};
}

export function makeCellPolygonText(
  frame: LocalFrame,
  grid: GridSpec,
  column: number,
  row: number
): string {
  const west = grid.minX + column * grid.cellSize;
  const south = grid.minY + row * grid.cellSize;
  const east = west + grid.cellSize;
  const north = south + grid.cellSize;
  const ring = [
    frame.inverse(west, south),
    frame.inverse(east, south),
    frame.inverse(east, north),
    frame.inverse(west, north),
    frame.inverse(west, south)
  ];
  return makeCatalogGeometryText('Polygon', [ring]);
}

export function makeEmptyOutput(
  tables: CatalogRunOutput['tables'],
  diagnostics: CatalogDiagnostic[],
  summary: CatalogRunOutput['summary'],
  dispatchedContributors?: readonly string[]
): CatalogRunOutput {
  return {
    tables,
    diagnostics,
    summary,
    ...(dispatchedContributors ? {dispatchedContributors} : {})
  };
}

export function addDroppedDiagnostic(
  diagnostics: CatalogDiagnostic[],
  droppedCount: number,
  what = 'Features without a point position or numeric value are skipped.'
): void {
  if (droppedCount)
    addCatalogDiagnostic(diagnostics, 'feature-dropped', 'warning', what, droppedCount);
}

/** Two-sided standard normal tail probability. */
export function getNormalTwoSidedPValue(z: number): number {
  if (!Number.isFinite(z)) return Number.NaN;
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const density = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const tail =
    density *
    t *
    (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return Math.min(1, 2 * tail);
}

export function getFeatureGeometryText(feature: CatalogFeature): string | null {
  return feature.geometry ? JSON.stringify(feature.geometry) : null;
}

/** Representative frame position of any feature: point, mean of line vertices or ring centroid. */
export function getFeatureAnchor(
  feature: CatalogFeature,
  frame: LocalFrame
): [number, number] | null {
  const geometry = feature.geometry;
  if (!geometry) return null;
  const points = getPointPositions(geometry);
  if (points.length) return frame.forward(points[0]);
  const polygons = getPolygonParts(geometry);
  if (polygons.length) {
    let areaSum = 0;
    let cx = 0;
    let cy = 0;
    let fallbackX = 0;
    let fallbackY = 0;
    let fallbackCount = 0;
    for (const polygon of polygons) {
      const ring = polygon[0].map(position => frame.forward(position));
      for (const [px, py] of ring) {
        fallbackX += px;
        fallbackY += py;
        fallbackCount++;
      }
      for (let index = 0; index < ring.length - 1; index++) {
        const cross = ring[index][0] * ring[index + 1][1] - ring[index + 1][0] * ring[index][1];
        areaSum += cross;
        cx += (ring[index][0] + ring[index + 1][0]) * cross;
        cy += (ring[index][1] + ring[index + 1][1]) * cross;
      }
    }
    if (Math.abs(areaSum) > 1e-12) return [cx / (3 * areaSum), cy / (3 * areaSum)];
    return fallbackCount ? [fallbackX / fallbackCount, fallbackY / fallbackCount] : null;
  }
  const lines = getLineParts(geometry).flat();
  if (!lines.length) return null;
  let sx = 0;
  let sy = 0;
  for (const position of lines) {
    const [px, py] = frame.forward(position);
    sx += px;
    sy += py;
  }
  return [sx / lines.length, sy / lines.length];
}
