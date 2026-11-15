// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {tessellateArrowPolygons} from '@luma.gl/engine/arrow';
import {
  makeKeplerPolygonPositionsLow,
  prepareKeplerGeometrySource
} from '../../../../examples/deck/kepler-compute/src/geometry-preparation';

const OUTER = [
  [-122.4194155123456, 37.7749295123457],
  [-122.4184155123457, 37.7749295123457],
  [-122.4184155123457, 37.7759295123457],
  [-122.4194155123456, 37.7759295123457],
  [-122.4194155123456, 37.7749295123457]
];
const HOLE = [
  [-122.4192155123456, 37.7751295123457],
  [-122.4192155123456, 37.7757295123457],
  [-122.4186155123456, 37.7757295123457],
  [-122.4186155123456, 37.7751295123457],
  [-122.4192155123456, 37.7751295123457]
];
const batches = [
  {
    id: 'fp64',
    rows: [
      {type: 'Polygon', coordinates: [OUTER, HOLE]},
      {type: 'LineString', coordinates: OUTER}
    ]
  }
];

it('keeps default preparation free of low streams', () => {
  const source = prepareKeplerGeometrySource(batches);
  expect(source.fp64).toBeUndefined();
  expect(source.batches[0].paths.positionsLow).toBeUndefined();
  expect(source.batches[0].polygonOutlines.positionsLow).toBeUndefined();
});

it('retains path and outline low parts that reconstruct canonical coordinates', () => {
  const source = prepareKeplerGeometrySource(batches, {fp64: true});
  expect(source.fp64).toBe(true);
  const [batch] = source.batches;
  const paths = batch.paths;
  expect(paths.positionsLow?.length).toBe((paths.positions.length / 3) * 2);
  OUTER.forEach(([longitude, latitude], index) => {
    expect(paths.positions[index * 3] + paths.positionsLow![index * 2]).toBeCloseTo(longitude, 11);
    expect(paths.positions[index * 3 + 1] + paths.positionsLow![index * 2 + 1]).toBeCloseTo(
      latitude,
      11
    );
  });
  // The high part alone is off by more than a Float32 low part can hide.
  expect(Math.abs(paths.positions[0] - OUTER[0][0])).toBeGreaterThan(1e-9);
  const outlines = batch.polygonOutlines;
  expect(outlines.positionsLow?.length).toBe((outlines.positions.length / 3) * 2);
});

it('derives tessellated polygon low parts in tessellation vertex order', () => {
  const source = prepareKeplerGeometrySource(batches, {fp64: true});
  const polygons = source.batches[0].polygons;
  const tessellation = tessellateArrowPolygons({polygons});
  // Closing vertices are dropped: 4 outer + 4 hole vertices.
  expect(tessellation.vertexCount).toBe(8);
  const positionsLow = makeKeplerPolygonPositionsLow(polygons, tessellation.vertexCount);
  expect(positionsLow.length).toBe(tessellation.vertexCount * 2);
  const expected = [...OUTER.slice(0, 4), ...HOLE.slice(0, 4)];
  expected.forEach(([longitude, latitude], index) => {
    expect(tessellation.positions[index * 4] + positionsLow[index * 2]).toBeCloseTo(longitude, 11);
    expect(tessellation.positions[index * 4 + 1] + positionsLow[index * 2 + 1]).toBeCloseTo(
      latitude,
      11
    );
  });
  // A vertex count that disagrees with the walk falls back to zeros instead of misaligned residuals.
  expect([...makeKeplerPolygonPositionsLow(polygons, 7)].every(value => value === 0)).toBe(true);
});

it('drops low parts when every coordinate is already Float32-representable', () => {
  const source = prepareKeplerGeometrySource(
    [
      {
        id: 'exact',
        rows: [
          {
            type: 'Polygon',
            coordinates: [
              [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 0]
              ]
            ]
          }
        ]
      }
    ],
    {fp64: true}
  );
  expect(source.fp64).toBeUndefined();
  expect(source.batches[0].polygonOutlines.positionsLow).toBeUndefined();
});
