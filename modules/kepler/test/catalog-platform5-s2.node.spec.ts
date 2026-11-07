// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {runCatalogOperation, type CatalogRunResult} from '../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {getDggsCpuBoundary} from '../src/analysis/catalog-platform4-dggs-cells';
import {PLATFORM5_S2_BINDINGS} from '../src/analysis/catalog-platform5-s2';

function getBinding(id: string) {
  const binding = PLATFORM5_S2_BINDINGS.find(candidate => candidate.id === id);
  if (!binding) throw new Error(`missing ${id}`);
  return binding;
}

function run(
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
): CatalogRunResult {
  return runCatalogOperation(getBinding(id), parameters, inputs);
}

function points(coordinates: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map(position => ({
      geometry: {type: 'Point', coordinates: position},
      properties: {}
    }))
  };
}

function polygon(ring: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: [{geometry: {type: 'Polygon', coordinates: [ring]}, properties: {}}]
  };
}

function column(result: CatalogRunResult, name: string): (string | number | boolean | null)[] {
  return result.tables[0].columns[name];
}

// One point per cube face plus seam and polar probes.
const FACE_POINTS = [
  [0, 0],
  [90, 0],
  [0, 89],
  [180, 0],
  [-90, 0],
  [0, -89],
  [-179.9, 10],
  [45.01, 35.2]
];

test('registers four S2 bindings with GPU contributors and CPU references', () => {
  expect(PLATFORM5_S2_BINDINGS.map(binding => binding.id)).toEqual([
    's2-cell-hierarchy',
    's2-cell-measures',
    's2-cap-covering',
    's2-polygon-covering'
  ]);
  for (const binding of PLATFORM5_S2_BINDINGS) {
    expect(binding.engine).toBe('gpu-or-cpu');
    expect(binding.run).toBeTypeOf('function');
    expect(binding.runGpu).toBeTypeOf('function');
    expect(binding.gpuContributorNames).toEqual(binding.contributorNames);
  }
});

test('hierarchy children, parent and range are consistent integers', () => {
  const children = run(
    's2-cell-hierarchy',
    {level: 9, operation: 'children'},
    {
      points: points(FACE_POINTS)
    }
  );
  expect(children.tables[0].rowCount).toBe(FACE_POINTS.length * 4);
  const source = column(children, 'source_cell_id');
  const child = column(children, 'cell_id');
  const parentOfChild = run(
    's2-cell-hierarchy',
    {level: 10, operation: 'parent', parentLevel: 9},
    {
      points: points(FACE_POINTS)
    }
  );
  // The level-10 parent at level 9 equals the level-9 cell of the same point.
  expect(column(parentOfChild, 'cell_id')).toEqual(source.filter((_, row) => row % 4 === 0));
  expect(new Set(child).size).toBe(child.length);
  for (const level of column(children, 'level')) expect(level).toBe(10);
  const range = run(
    's2-cell-hierarchy',
    {level: 12, operation: 'range'},
    {
      points: points([[10, 10]])
    }
  );
  const [low, high] = column(range, 'cell_id').map(value => BigInt(`0x${value}`));
  const cell = BigInt(`0x${column(range, 'source_cell_id')[0]}`);
  expect(cell - low).toBe(high - cell);
  expect(column(range, 'level')).toEqual([30, 30]);
});

test('hierarchy neighbors are symmetric across cube-face seams', () => {
  for (const level of [0, 1, 5, 14]) {
    const forward = run(
      's2-cell-hierarchy',
      {level, operation: 'neighbors'},
      {
        points: points(FACE_POINTS)
      }
    );
    const sources = column(forward, 'source_cell_id') as string[];
    const neighbors = column(forward, 'cell_id') as string[];
    const relations = column(forward, 'relation') as string[];
    neighbors.forEach((neighbor, row) => {
      if (level === 0 && neighbor === sources[row]) return;
      // The neighbor's own neighbors include the source, through some relation.
      const back = run(
        's2-cell-hierarchy',
        {level, operation: 'neighbors'},
        {
          points: points([cellCenter(neighbor)])
        }
      );
      expect(column(back, 'cell_id'), `${level} ${relations[row]}`).toContain(sources[row]);
    });
  }
});

// Longitude/latitude of a cell center: the spherical mean of its four boundary corners.
function cellCenter(cellId: string): number[] {
  const corners = getDggsCpuBoundary('s2', BigInt(`0x${cellId}`)).slice(0, 4);
  let x = 0;
  let y = 0;
  let z = 0;
  for (const [longitude, latitude] of corners) {
    const lng = (longitude * Math.PI) / 180;
    const lat = (latitude * Math.PI) / 180;
    x += Math.cos(lat) * Math.cos(lng);
    y += Math.cos(lat) * Math.sin(lng);
    z += Math.sin(lat);
  }
  return [(Math.atan2(y, x) * 180) / Math.PI, (Math.atan2(z, Math.hypot(x, y)) * 180) / Math.PI];
}

test('measures: six faces tile the sphere and children sum to the parent', () => {
  const faces = run(
    's2-cell-measures',
    {level: 0, radius: 1},
    {points: points(FACE_POINTS.slice(0, 6))}
  );
  for (const area of column(faces, 'area'))
    expect(area as number).toBeCloseTo((4 * Math.PI) / 6, 10);
  const parent = run('s2-cell-measures', {level: 8, radius: 1}, {points: points([[12, 34]])});
  const children = run(
    's2-cell-hierarchy',
    {level: 8, operation: 'children'},
    {
      points: points([[12, 34]])
    }
  );
  const childAreas = (column(children, 'cell_id') as string[]).map(cellId => {
    const [longitude, latitude] = cellCenter(cellId);
    return column(
      run('s2-cell-measures', {level: 9, radius: 1}, {points: points([[longitude, latitude]])}),
      'area'
    )[0] as number;
  });
  const sum = childAreas.reduce((total, area) => total + area, 0);
  expect(sum / (column(parent, 'area')[0] as number)).toBeCloseTo(1, 8);
  const meters = run('s2-cell-measures', {level: 4}, {points: points([[0, 0]])});
  expect(column(meters, 'perimeter')[0] as number).toBeGreaterThan(1e6);
});

test('cap covering: exterior contains the center cell, interior is a subset, a full sphere takes all', () => {
  const inputs = {points: points([[10, 20]])};
  const exterior = run(
    's2-cap-covering',
    {radiusMeters: 800000, level: 3, mode: 'exterior'},
    inputs
  );
  const interior = run(
    's2-cap-covering',
    {radiusMeters: 800000, level: 3, mode: 'interior'},
    inputs
  );
  const exteriorCells = new Set(column(exterior, 'cell_id') as string[]);
  const interiorCells = column(interior, 'cell_id') as string[];
  expect(exteriorCells.size).toBeGreaterThan(1);
  expect(interiorCells.length).toBeLessThan(exteriorCells.size);
  for (const cell of interiorCells) expect(exteriorCells.has(cell)).toBe(true);
  const centerCell = column(
    run('s2-cell-hierarchy', {level: 3, operation: 'parent', parentLevel: 3}, inputs),
    'cell_id'
  )[0] as string;
  expect(exteriorCells.has(centerCell)).toBe(true);
  const everything = run(
    's2-cap-covering',
    {radiusMeters: 20000000, level: 2, mode: 'exterior', maximumCells: 6144},
    inputs
  );
  expect(everything.tables[0].rowCount).toBe(6 * 16);
  const truncated = run(
    's2-cap-covering',
    {radiusMeters: 20000000, level: 2, maximumCells: 10},
    inputs
  );
  expect(truncated.tables[0].rowCount).toBe(10);
  expect(truncated.diagnostics.map(diagnostic => diagnostic.code)).toContain('s2-cover-truncated');
});

test('polygon covering: exterior covers the polygon, interior is inside it, levels respected', () => {
  const square = polygon([
    [-30, -20],
    [30, -20],
    [30, 25],
    [-30, 25],
    [-30, -20]
  ]);
  const exterior = run(
    's2-polygon-covering',
    {maximumLevel: 4, mode: 'exterior'},
    {polygons: square}
  );
  const interior = run(
    's2-polygon-covering',
    {maximumLevel: 4, mode: 'interior'},
    {polygons: square}
  );
  expect(exterior.tables[0].rowCount).toBeGreaterThan(interior.tables[0].rowCount);
  expect(interior.tables[0].rowCount).toBeGreaterThan(0);
  for (const level of column(exterior, 'level')) expect(level as number).toBeLessThanOrEqual(4);
  // Every probe point inside the square lies in some exterior cell.
  const covered = new Set(column(exterior, 'cell_id') as string[]);
  for (const probe of [
    [0, 0],
    [25, 20],
    [-29, -19],
    [10, 24]
  ]) {
    const hit = [...covered].some(cellId => {
      const parentLevels = Array.from({length: 5}, (_, level) => level);
      return parentLevels.some(level =>
        column(
          run(
            's2-cell-hierarchy',
            {level: 4, operation: 'parent', parentLevel: level},
            {
              points: points([probe])
            }
          ),
          'cell_id'
        ).includes(cellId)
      );
    });
    expect(hit, `probe ${probe}`).toBe(true);
  }
  // A cell strictly outside the polygon is not in the interior cover: the antipodal point's cells.
  const antipode = run(
    's2-cell-hierarchy',
    {level: 4, operation: 'parent', parentLevel: 4},
    {
      points: points([[150, -5]])
    }
  );
  expect(column(exterior, 'cell_id')).not.toContain(column(antipode, 'cell_id')[0]);
  const bounded = run(
    's2-polygon-covering',
    {maximumLevel: 6, maximumCells: 5},
    {polygons: square}
  );
  expect(bounded.tables[0].rowCount).toBe(5);
  expect(bounded.diagnostics.map(diagnostic => diagnostic.code)).toContain('s2-cover-truncated');
});

test('polygon covering skips self-crossing rings and accepts clockwise input', () => {
  const bowtie = polygon([
    [0, 0],
    [10, 10],
    [10, 0],
    [0, 10],
    [0, 0]
  ]);
  const skipped = run('s2-polygon-covering', {maximumLevel: 3}, {polygons: bowtie});
  expect(skipped.tables[0].rowCount).toBe(0);
  expect(skipped.diagnostics.map(diagnostic => diagnostic.code)).toContain(
    'invalid-spherical-polygon'
  );
  const counterClockwise = polygon([
    [0, 0],
    [20, 0],
    [20, 20],
    [0, 20],
    [0, 0]
  ]);
  const clockwise = polygon([
    [0, 0],
    [0, 20],
    [20, 20],
    [20, 0],
    [0, 0]
  ]);
  const first = run('s2-polygon-covering', {maximumLevel: 4}, {polygons: counterClockwise});
  const second = run('s2-polygon-covering', {maximumLevel: 4}, {polygons: clockwise});
  expect(column(second, 'cell_id')).toEqual(column(first, 'cell_id'));
});
