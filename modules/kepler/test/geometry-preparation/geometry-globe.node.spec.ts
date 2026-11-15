// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {prepareKeplerGeometrySource} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-gpu-vectors';
import {
  alignGlobePolygon,
  cutGlobePolygon,
  prepareGlobeGeometryParts,
  subdivideGlobePath,
  unwrapGlobePath
} from '../../../../examples/deck/kepler-compute/src/geometry-preparation/geometry-globe';

type Ring = [number, number, number][];

const getBounds = (ring: Ring) => [
  Math.min(...ring.map(point => point[0])),
  Math.min(...ring.map(point => point[1])),
  Math.max(...ring.map(point => point[0])),
  Math.max(...ring.map(point => point[1]))
];
const getArea = (ring: Ring) => {
  let area = 0;
  for (let index = 1; index < ring.length; index++)
    area += ring[index - 1][0] * ring[index][1] - ring[index][0] * ring[index - 1][1];
  return Math.abs(area) / 2;
};

it('unwraps paths across the antimeridian so no step exceeds 180 degrees', () => {
  const path = unwrapGlobePath([
    [170, 0, 0],
    [-170, 5, 0],
    [-160, 10, 0]
  ]);
  expect(path.map(point => point[0])).toEqual([170, 190, 200]);
});

it('keeps the z0 MVT water polygon wide and cuts its full surface into globe cells', () => {
  const ring: Ring = [
    [-180, -85, 0],
    [180, -85, 0],
    [180, 85, 0],
    [-180, 85, 0],
    [-180, -85, 0]
  ];
  const prepared = prepareGlobeGeometryParts(
    {paths: [], polygons: [[ring]]},
    {
      resolutionDegrees: 5,
      unwrapLongitude: false
    }
  );
  expect(prepared.polygons).toHaveLength(72 * 34);
  const totalArea = prepared.polygons.reduce((area, polygon) => area + getArea(polygon[0]), 0);
  expect(totalArea).toBeCloseTo(360 * 170, 6);
  for (const polygon of prepared.polygons) {
    const [west, south, east, north] = getBounds(polygon[0]);
    expect(east - west).toBeLessThanOrEqual(5);
    expect(north - south).toBeLessThanOrEqual(5);
  }
});

it('subdivides edges so none exceeds the resolution and endpoints are preserved', () => {
  const path = subdivideGlobePath(
    [
      [0, 0, 0],
      [60, 30, 120]
    ],
    5
  );
  expect(path[0]).toEqual([0, 0, 0]);
  expect(path[path.length - 1]).toEqual([60, 30, 120]);
  expect(path).toHaveLength(13);
  for (let index = 1; index < path.length; index++)
    expect(Math.abs(path[index][0] - path[index - 1][0])).toBeLessThanOrEqual(5 + 1e-9);
  expect(path[6][2]).toBeCloseTo(60);
});

it('cuts a wide polygon into grid cells that preserve its area', () => {
  const polygon: Ring[] = [
    [
      [0, 0, 0],
      [20, 0, 0],
      [20, 10, 0],
      [0, 10, 0],
      [0, 0, 0]
    ]
  ];
  const pieces = cutGlobePolygon(polygon, 5);
  expect(pieces).toHaveLength(8);
  for (const piece of pieces) {
    const [west, south, east, north] = getBounds(piece[0]);
    expect(east - west).toBeLessThanOrEqual(5 + 1e-9);
    expect(north - south).toBeLessThanOrEqual(5 + 1e-9);
  }
  expect(pieces.reduce((sum, piece) => sum + getArea(piece[0]), 0)).toBeCloseTo(200);
});

it('keeps a polygon that fits one cell unchanged and clips holes per cell', () => {
  const small: Ring[] = [
    [
      [1, 1, 0],
      [4, 1, 0],
      [4, 4, 0],
      [1, 1, 0]
    ]
  ];
  expect(cutGlobePolygon(small, 5)).toEqual([small]);
  const holed: Ring[] = [
    [
      [0, 0, 0],
      [10, 0, 0],
      [10, 10, 0],
      [0, 10, 0],
      [0, 0, 0]
    ],
    [
      [4, 4, 0],
      [6, 4, 0],
      [6, 6, 0],
      [4, 6, 0],
      [4, 4, 0]
    ]
  ];
  const pieces = cutGlobePolygon(holed, 5);
  expect(pieces).toHaveLength(4);
  const holeArea = pieces.reduce(
    (sum, piece) => sum + piece.slice(1).reduce((holes, hole) => holes + getArea(hole), 0),
    0
  );
  expect(holeArea).toBeCloseTo(4);
});

it('aligns an antimeridian polygon into one continuous ring', () => {
  const [ring] = alignGlobePolygon([
    [
      [170, 0, 0],
      [-170, 0, 0],
      [-170, 10, 0],
      [170, 10, 0],
      [170, 0, 0]
    ]
  ]);
  expect(getBounds(ring)).toEqual([170, 0, 190, 10]);
});

it('keeps polygon outlines on the source rings while cutting fills', () => {
  const parts = prepareGlobeGeometryParts({
    paths: [],
    polygons: [
      [
        [
          [0, 0, 0],
          [20, 0, 0],
          [20, 10, 0],
          [0, 10, 0],
          [0, 0, 0]
        ]
      ]
    ]
  });
  expect(parts.polygons).toHaveLength(8);
  expect(parts.outlines).toHaveLength(1);
  expect(parts.outlines[0][0].length).toBe(13);
});

it('prepares a globe variant that keeps source rows and the single source row per primitive', () => {
  const rows = [
    {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [30, 0],
          [30, 30],
          [0, 30],
          [0, 0]
        ]
      ]
    },
    {
      type: 'LineString',
      coordinates: [
        [0, 0],
        [90, 0]
      ]
    }
  ];
  const source = prepareKeplerGeometrySource([{id: 'rows', rows}], {globe: {}});
  const [batch] = source.batches;
  expect(batch.rowCount).toBe(2);
  expect(batch.polygons.length).toBe(2);
  expect(batch.polygons.get(0)!.length).toBe(36);
  expect(Array.from(batch.paths.sourceRows)).toEqual([1]);
  expect(batch.paths.valueOffsets[1]).toBe(19);
  expect(batch.polygonOutlines.sourceRows.length).toBe(1);
});

it('rejects projected coordinates instead of cutting an enormous grid', () => {
  const source = prepareKeplerGeometrySource(
    [
      {
        id: 'projected',
        rows: [
          {
            type: 'Polygon',
            coordinates: [
              [
                [0, 0],
                [2e7, 0],
                [2e7, 2e7],
                [0, 2e7],
                [0, 0]
              ]
            ]
          },
          {
            type: 'LineString',
            coordinates: [
              [0, 0],
              [10, 10]
            ]
          }
        ]
      }
    ],
    {globe: {}}
  );
  expect(source.metadata.invalidRows.map(row => row.sourceRowIndex)).toEqual([0]);
  expect(source.batches[0].paths.sourceRows.length).toBeGreaterThan(0);
});
