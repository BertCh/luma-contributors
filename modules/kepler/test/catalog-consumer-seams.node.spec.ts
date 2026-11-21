// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {runConvexClip} from '../src/analysis/catalog-geometry';
import {
  getCatalogRectangle,
  getCatalogPolygonArea,
  readCatalogPolygons
} from '../src/analysis/catalog-gpu-pack';
import type {CatalogFeatureCollection, CatalogPosition} from '../src/analysis/catalog-geojson';
const square: CatalogPosition[] = [
  [0, 0],
  [10, 0],
  [10, 10],
  [0, 10],
  [0, 0]
];
const collection = (
  geometry: CatalogFeatureCollection['features'][number]['geometry']
): CatalogFeatureCollection => ({
  type: 'FeatureCollection',
  features: [{geometry, properties: {}}]
});
it('UCO-02 rejects whole unsupported clip operands and crossed rectangles', () => {
  const subject = collection({
    type: 'LineString',
    coordinates: [
      [3, 3],
      [4, 4]
    ]
  });
  for (const geometry of [
    {
      type: 'Polygon',
      coordinates: [
        square,
        [
          [2, 2],
          [8, 2],
          [8, 8],
          [2, 8],
          [2, 2]
        ]
      ]
    },
    {type: 'MultiPolygon', coordinates: [[square], [square.map(([x, y]) => [x + 20, y])]]},
    {
      type: 'Polygon',
      coordinates: [
        [
          [0, 0],
          [1, 1],
          [1, 0],
          [0, 1],
          [0, 0]
        ]
      ]
    }
  ] as const)
    expect(() => runConvexClip(subject, collection(geometry as never))).toThrow();
  expect(
    getCatalogRectangle([
      [0, 0],
      [1, 1],
      [1, 0],
      [0, 1],
      [0, 0]
    ])
  ).toBeNull();
  expect(
    runConvexClip(subject, collection({type: 'Polygon', coordinates: [square]})).tables[0].rowCount
  ).toBe(1);
});
it('UCO-03 preserves translation invariant planar unit area', () => {
  // A translated unit square has area 1 by width times height.
  expect(
    getCatalogPolygonArea([
      [
        [1e8, 1e8],
        [1e8 + 1, 1e8],
        [1e8 + 1, 1e8 + 1],
        [1e8, 1e8 + 1],
        [1e8, 1e8]
      ]
    ])
  ).toBe(1);
});
it('UCO-08 never promotes a hole when the shell is unusable', () => {
  expect(readCatalogPolygons([0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [0, 2, 5], [0, 2], [0, 1])).toEqual(
    []
  );
});

import {createCatalogFrame, getCatalogBounds} from '../src/analysis/catalog-gpu-pack';
import {
  collectCatalogPositions,
  runCatalogLineGeometry
} from '../src/analysis/catalog-platform4-geometry-common';
import {collectPositions} from '../src/analysis/catalog-platform3-a-shared';
import type {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
it('UCO-12 visits a 150000 vertex line and a closed positive-area large ring in order', () => {
  const vertices: CatalogPosition[] = Array.from({length: 150000}, (_, index) => [
    index / 10000,
    index / 20000
  ]);
  const ring: CatalogPosition[] = Array.from({length: 150000}, (_, index) => {
    const edge = Math.floor(index / 37500),
      fraction = (index % 37500) / 37500;
    return edge === 0
      ? [3 * fraction, 0]
      : edge === 1
        ? [3, 2 * fraction]
        : edge === 2
          ? [3 * (1 - fraction), 2]
          : [0, 2 * (1 - fraction)];
  });
  ring.push(ring[0]);
  for (const [geometry, expected, bounds] of [
    [{type: 'LineString', coordinates: vertices}, vertices, [0, 0, 14.9999, 7.49995]],
    [{type: 'Polygon', coordinates: [ring]}, ring, [0, 0, 3, 2]]
  ] as const) {
    const input = collection(geometry);
    expect(collectCatalogPositions([input])).toEqual(expected);
    expect(collectPositions(input)).toEqual(expected);
    expect(getCatalogBounds([input])).toEqual(bounds);
    expect(
      createCatalogFrame('EPSG:4326', [input]).project(expected[0]).every(Number.isFinite)
    ).toBe(true);
  }
  // A densely sampled 3x2 rectangle has area 6 independently of vertex count.
  const input = collection({type: 'Polygon', coordinates: [ring]});
  for (const result of [
    runGeometryMeasures(input, 'planar'),
    runShapeDescriptors(input, 'planar')
  ]) {
    const area = result.tables[0].columns['area'][0] as number;
    expect(Number.isFinite(area)).toBe(true);
    expect(area).toBeCloseTo(6, 10);
  }
});
it('UCO-08 honors zero emitted count despite populated tail storage', async () => {
  const session = {
    output: () => ({view: {}}),
    input: () => ({}),
    graph: {add: () => {}},
    run: () => {},
    readScalar: async () => 0,
    readFloat32: async () => new Float32Array([1, 2, 3, 4]),
    readUint32: async () => new Uint32Array([0, 2])
  } as unknown as CatalogGpuSession;
  const result = await runCatalogLineGeometry(
    session,
    {positions: new Float32Array([1, 2, 3, 4]), lineOffsets: new Uint32Array([0, 2])},
    {vertices: 2, lines: 1},
    () => ({}) as never
  );
  expect(result.lineCount).toBe(0);
});

import {vi} from 'vitest';
import * as lineGeometry from '../src/analysis/catalog-platform4-geometry-common';
import * as sessions from '../src/analysis/catalog-gpu-session';
import {runGpuLineChunk, runGpuLineSmooth} from '../src/analysis/catalog-platform4-geometry-lines';
import {runGpuRectangleClip} from '../src/analysis/catalog-gpu-geometry';
it('UCO-03 centers all planar chunk parts and restores the emitted coordinates', async () => {
  const open = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockImplementation(async (_device, _name, callback) => callback({} as never));
  const run = vi
    .spyOn(lineGeometry, 'runCatalogLineGeometry')
    .mockImplementation(async (_session, packed) => {
      expect(Array.from(packed.positions)).toEqual([-0.5, 0, 0.5, 0]);
      return {...packed, sourceLineIndices: new Uint32Array([0]), lineCount: 1};
    });
  try {
    const result = await runGpuLineChunk(
      {device: {} as never},
      collection({
        type: 'LineString',
        coordinates: [
          [20000000, 0],
          [20000001, 0]
        ]
      }),
      {maximumChunkLength: 2, coordinateSystem: 'planar'},
      10
    );
    expect(JSON.parse(result.tables[0].columns['geometry'][0] as string).coordinates).toEqual([
      [20000000, 0],
      [20000001, 0]
    ]);
  } finally {
    run.mockRestore();
    open.mockRestore();
  }
});
it('UCO-08 smoothing ignores populated storage beyond emitted zero', async () => {
  const open = vi
    .spyOn(sessions, 'withCatalogGpuSession')
    .mockImplementation(async (_device, _name, callback) => callback({} as never));
  const run = vi.spyOn(lineGeometry, 'runCatalogLineGeometry').mockResolvedValue({
    positions: new Float32Array([1, 2, 3, 4]),
    lineOffsets: new Uint32Array([0, 2]),
    sourceLineIndices: new Uint32Array([0]),
    lineCount: 0
  });
  try {
    expect(
      (
        await runGpuLineSmooth(
          {device: {} as never},
          collection({
            type: 'LineString',
            coordinates: [
              [1, 2],
              [3, 4]
            ]
          }),
          {iterations: 1}
        )
      ).tables[0].rowCount
    ).toBe(0);
  } finally {
    run.mockRestore();
    open.mockRestore();
  }
});
it('UCO-02 GPU runner rejects a holed operand before allocating', async () => {
  await expect(
    runGpuRectangleClip(
      {device: {} as never},
      collection(null),
      collection({type: 'Polygon', coordinates: [square, square]})
    )
  ).rejects.toThrow(/one polygon, one ring/);
});

it('UCO-03 centers the shared planar frame used by buffer, overlay and repair', () => {
  const input = collection({
    type: 'LineString',
    coordinates: [
      [20000000, 0],
      [20000001, 0]
    ]
  });
  const frame = createCatalogFrame('planar', [input]);
  expect(frame.project([20000000, 0])).toEqual([-0.5, 0]);
  expect(frame.project([20000001, 0])).toEqual([0.5, 0]);
  expect(frame.unproject(-0.5, 0)).toEqual([20000000, 0]);
});

import {runGeometryMeasures} from '../src/analysis/catalog-geometry';
import {runShapeDescriptors} from '../src/analysis/catalog-platform3-a-measure';
