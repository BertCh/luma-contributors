// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {getCatalogBinding, runCatalogOperation} from '../src/analysis/catalog-bindings';
import {
  type CatalogFeatureCollection,
  validateCatalogCoordinateSystem
} from '../src/analysis/catalog-geojson';
import {validateGeometryOutputCapacity} from '../src/analysis/catalog-extra-geometry';
import {makeLocalFrame} from '../src/analysis/catalog-local-frame';
import {computeDbscanLabels} from '../src/analysis/catalog-platform3-b-clusters';
import {readPointSamples} from '../src/analysis/catalog-platform3-b-common';
import {PLATFORM4_GEOMETRY_GPU_RUNNERS} from '../src/analysis/catalog-platform4-geometry-gpu';

function collection(type: string, coordinates: unknown): CatalogFeatureCollection {
  return {type: 'FeatureCollection', features: [{geometry: {type, coordinates}}]};
}
function points(coordinates: number[][]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: coordinates.map(position => ({geometry: {type: 'Point', coordinates: position}}))
  };
}
function run(
  bindingId: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogFeatureCollection>
) {
  return runCatalogOperation(getCatalogBinding(bindingId)!, parameters, inputs);
}
const projectedPoints = points([
  [390000, 5820000],
  [390001, 5820001]
]);
const projectedPolygon = collection('Polygon', [
  [
    [390000, 5820000],
    [390010, 5820000],
    [390010, 5820010],
    [390000, 5820000]
  ]
]);

test.each([
  ['geometry-shape-descriptors', {polygons: projectedPolygon}],
  ['geometry-label-point', {polygons: projectedPolygon}],
  ['voronoi-diagram', {points: projectedPoints}],
  ['nearest-feature-join', {points: projectedPoints, targets: projectedPolygon}]
])('%s rejects projected input with default geographic coordinates', (bindingId, inputs) => {
  expect(() =>
    run(bindingId as string, {}, inputs as Record<string, CatalogFeatureCollection>)
  ).toThrow(`${bindingId}: coordinateSystem`);
});

test('shared coordinate validation accepts planar input and catches projected data for remaining bindings', () => {
  expect(() =>
    validateCatalogCoordinateSystem('geometry-measures', 'EPSG:4326', {features: projectedPolygon})
  ).toThrow('geometry-measures: coordinateSystem');
  expect(() =>
    validateCatalogCoordinateSystem('network-shortest-path', 'EPSG:4326', {lines: projectedPolygon})
  ).toThrow('network-shortest-path: coordinateSystem');
  expect(() =>
    validateCatalogCoordinateSystem('geometry-measures', 'planar', {features: projectedPolygon})
  ).not.toThrow();
});

test('GPU geometry validation rejects invalid coordinates before touching a device', async () => {
  await expect(
    PLATFORM4_GEOMETRY_GPU_RUNNERS['geometry-label-point'].run(
      {device: null!},
      {polygons: projectedPolygon},
      {coordinateSystem: 'EPSG:4326'}
    )
  ).rejects.toThrow('geometry-label-point: coordinateSystem');
});

test('DBSCAN includes points exactly epsilon apart after shifting, matching scikit-learn', () => {
  const input = points([
    [-0.1, 100],
    [0.1, 0],
    [1.1, 0],
    [5.1, 0],
    [6.1, 0]
  ]);
  const sample = readPointSamples(
    input,
    '',
    makeLocalFrame(
      'planar',
      [
        [-0.1, 0],
        [3, 0]
      ],
      {shift: 'minimum'}
    )
  );
  expect(sample.x[2] - sample.x[1]).toBeGreaterThan(1);
  expect(Array.from(computeDbscanLabels(sample, 1, 2).labels)).toEqual([-1, 0, 0, 1, 1]);
  expect(Array.from(computeDbscanLabels(sample, 0.999999, 2).labels)).toEqual([-1, -1, -1, -1, -1]);
});

test('CPU DBSCAN honors the 20,000 point declaration without pair-work or quadratic memory failure', () => {
  const input = points(Array.from({length: 20000}, (_, index) => [index * 10, 0]));
  const sample = readPointSamples(input, '', makeLocalFrame('planar', []));
  expect(computeDbscanLabels(sample, 1, 2).labels.every(label => label === -1)).toBe(true);
  expect(() => computeDbscanLabels({...sample, count: 20001}, 1, 2)).toThrow(
    'density-clustering: points count exceeds the 20,000 point limit'
  );
});

test('triangulation rejects a part over 1,500 vertices with a named limit and remedy', () => {
  const ring = Array.from({length: 1501}, (_, index) => [
    Math.cos((index / 1501) * 2 * Math.PI),
    Math.sin((index / 1501) * 2 * Math.PI)
  ]);
  ring.push(ring[0]);
  expect(() =>
    run(
      'polygon-triangulation',
      {coordinateSystem: 'planar'},
      {polygons: collection('Polygon', [ring])}
    )
  ).toThrow('polygon-triangulation: polygons part vertex count exceeds the 1,500 vertex limit');
  expect(getCatalogBinding('polygon-triangulation')!.limitation).toContain('1,500');
});

test.each([
  [
    'grid-generator',
    {cellWidth: 1, cellHeight: 1},
    {
      extent: points([
        [0, 0],
        [201, 100]
      ])
    },
    '20,000'
  ],
  [
    'random-points-in-polygon',
    {pointsPerPolygon: 1000},
    {polygons: {type: 'FeatureCollection', features: Array(21).fill(projectedPolygon.features[0])}},
    '20,000'
  ],
  [
    'random-points-on-line',
    {pointsPerLine: 1000, coordinateSystem: 'planar'},
    {
      lines: {
        type: 'FeatureCollection',
        features: Array(21).fill(
          collection('LineString', [
            [0, 0],
            [1, 0]
          ]).features[0]
        )
      }
    },
    '20,000'
  ],
  [
    'line-chunk',
    {maximumChunkLength: 1, coordinateSystem: 'planar'},
    {
      lines: collection('LineString', [
        [0, 0],
        [50001, 0]
      ])
    },
    '50,000'
  ]
])('%s rejects projected output overflow before generation', (bindingId, parameters, inputs, limit) => {
  expect(() =>
    run(
      bindingId as string,
      parameters as Record<string, unknown>,
      inputs as Record<string, CatalogFeatureCollection>
    )
  ).toThrow(`${bindingId}:`);
  expect(() =>
    run(
      bindingId as string,
      parameters as Record<string, unknown>,
      inputs as Record<string, CatalogFeatureCollection>
    )
  ).toThrow(limit as string);
  expect(getCatalogBinding(bindingId as string)!.limitation).toContain(limit as string);
});

test('segmentize preflight counts every output vertex, including the first vertex of each part', () => {
  expect(() =>
    validateGeometryOutputCapacity(
      'geometry-line-segmentize',
      {
        lines: collection('LineString', [
          [0, 0],
          [500000, 0]
        ])
      },
      {maximumSegmentLength: 1, coordinateSystem: 'planar'}
    )
  ).toThrow('500,000 limit');
  expect(() =>
    validateGeometryOutputCapacity(
      'geometry-line-segmentize',
      {
        lines: collection('LineString', [
          [0, 0],
          [499999, 0]
        ])
      },
      {maximumSegmentLength: 1, coordinateSystem: 'planar'}
    )
  ).not.toThrow();
});

test('centrality declares the built-node cap separately from raw input capacity and rejects overflow by name', () => {
  const binding = getCatalogBinding('network-centrality')!;
  expect(binding.inputs[0].label).toContain('1,500 built nodes');
  expect(binding.outputs[0].maximumRows).toBe(1500);
  const lines = collection(
    'LineString',
    Array.from({length: 1501}, (_, index) => [index, 0])
  );
  expect(() => run(binding.id, {coordinateSystem: 'planar'}, {lines})).toThrow(
    'network-centrality: lines exceed the 1,500 node limit'
  );
});

test('S2 cap clamps maximumCells to the effective level cap with a diagnostic', () => {
  const result = run('s2-cap-covering', {level: 2, maximumCells: 97}, {points: points([[0, 0]])});
  expect(result.diagnostics).toContainEqual({
    code: 's2-maximum-cells-clamped',
    severity: 'info',
    count: 1,
    message: 'maximumCells 97 was clamped to the 96 cell limit at level 2.'
  });
  expect(() =>
    run('s2-cap-covering', {level: 0, maximumCells: 6}, {points: points([[0, 0]])})
  ).not.toThrow();
  const binding = getCatalogBinding('s2-cap-covering')!;
  expect(
    binding.parameters.find(parameter => parameter.name === 'maximumCells')!.defaultValue
  ).toBe(384);
  expect(binding.declaredBehavior.join(' ')).toContain('default is 384');
});

test('polygonize declares its actual coordinate-unit snap tolerance honestly', () => {
  const binding = getCatalogBinding('geometry-polygonize')!;
  expect(binding.parameters.find(parameter => parameter.name === 'snapTolerance')!.unit).toBe(
    'coordinate units'
  );
  expect(binding.declaredBehavior.join(' ')).toContain('degrees for EPSG:4326');
});

test('hierarchical density results visibly identify the single flat cut', () => {
  const binding = getCatalogBinding('hierarchical-density-clustering')!;
  expect(binding.title).toBe('Mutual-reachability density clustering');
  expect(binding.limitation).toContain('no stability-based selection');
  const result = run(
    binding.id,
    {minimumSamples: 1, minimumClusterSize: 2, cutDistance: 2, coordinateSystem: 'planar'},
    {
      points: points([
        [0, 0],
        [1, 0],
        [10, 0],
        [11, 0]
      ])
    }
  );
  expect(
    result.diagnostics.find(diagnostic => diagnostic.code === 'flat-density-cut')?.message
  ).toContain('without HDBSCAN stability selection');
});

test('GPU triangulation rejects oversized parts before touching a device', async () => {
  const ring = Array.from({length: 1501}, (_, index) => [index, index % 2]);
  await expect(
    PLATFORM4_GEOMETRY_GPU_RUNNERS['polygon-triangulation'].run(
      {device: null!},
      {polygons: collection('Polygon', [ring])},
      {coordinateSystem: 'planar'}
    )
  ).rejects.toThrow(
    'polygon-triangulation: polygons part vertex count exceeds the 1,500 vertex limit'
  );
});
