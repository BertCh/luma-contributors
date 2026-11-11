// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUSpatialContext} from '../../../src/analysis-core/contracts/spatial-context';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getGPURectangleClipParameterValues,
  GPURectangleClip
} from '../../../src/gpu-spatial-analysis/rectangle-clip/index';
import {
  createGeometryFixture,
  createRandom,
  expectClose
} from '../outline-geometry/geometry-fixture';
import {
  clipLines,
  clipRing,
  getRingArea,
  type Point,
  type Rectangle
} from './rectangle-clip-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

function flatten(lines: Point[][]) {
  const positions: number[] = [];
  const offsets = [0];
  for (const line of lines) {
    for (const point of line) {
      positions.push(point[0], point[1]);
    }
    offsets.push(positions.length / 2);
  }
  return {
    positions: new Float32Array(positions),
    offsets: new Uint32Array(offsets),
    // One polygon per feature with a single ring each.
    identity: Uint32Array.from({length: offsets.length}, (_, index) => index)
  };
}

function round(lines: Point[][]): Point[][] {
  return lines.map(line => line.map(([x, y]) => [Math.fround(x), Math.fround(y)] as Point));
}

function createFixture(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  lines: Point[][],
  geometryType: 'lines' | 'polygons',
  capacity: number,
  lineCapacity: number,
  spatialContext?: GPUSpatialContext
) {
  const flat = flatten(lines);
  return createGeometryFixture(device, {
    inputs: {
      positions: {values: flat.positions, format: 'float32x2'},
      lineOffsets: {values: flat.offsets, format: 'uint32'},
      ...(geometryType === 'polygons'
        ? {
            featureOffsets: {values: flat.identity, format: 'uint32' as const},
            polygonOffsets: {values: flat.identity, format: 'uint32' as const}
          }
        : {})
    },
    outputs: {
      positions: {format: 'float32x2', length: capacity},
      lineOffsets: {format: 'uint32', length: lineCapacity + 1},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      requiredCount: {format: 'uint32', length: 1},
      vertexCount: {format: 'uint32', length: 1},
      requiredVertexCount: {format: 'uint32', length: 1},
      ...(geometryType === 'lines'
        ? {sourceLineIndices: {format: 'uint32' as const, length: lineCapacity}}
        : {})
    },
    create: ({inputs, outputs, parameters}) =>
      new GPURectangleClip({
        spatialContext,
        positions: inputs['positions'] as never,
        ...(geometryType === 'polygons'
          ? {
              geometryType,
              featureOffsets: inputs['featureOffsets'] as never,
              polygonOffsets: inputs['polygonOffsets'] as never,
              ringOffsets: inputs['lineOffsets'] as never
            }
          : {geometryType, lineOffsets: inputs['lineOffsets'] as never}),
        parameters,
        output: {
          positions: outputs['positions'] as never,
          lineOffsets: outputs['lineOffsets'] as never,
          status: {
            count: outputs['count'] as never,
            overflow: outputs['overflow'] as never,
            requiredCount: outputs['requiredCount'] as never,
            vertexCount: outputs['vertexCount'] as never,
            requiredVertexCount: outputs['requiredVertexCount'] as never
          },
          sourceLineIndices: outputs['sourceLineIndices'] as never
        }
      })
  });
}

function readLines(result: Record<string, number[]>, lineCount: number): Point[][] {
  return Array.from({length: lineCount}, (_, line) =>
    Array.from(
      {length: result['lineOffsets'][line + 1] - result['lineOffsets'][line]},
      (__, vertex) => {
        const row = result['lineOffsets'][line] + vertex;
        return [result['positions'][2 * row], result['positions'][2 * row + 1]] as Point;
      }
    )
  );
}

function expectLinesClose(actual: Point[][], expected: Point[][], absolute: number) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((line, lineIndex) => {
    expect(line.length, `line ${lineIndex}`).toBe(expected[lineIndex].length);
    line.forEach((point, vertex) => {
      expectClose(point[0], expected[lineIndex][vertex][0], 1e-6, absolute);
      expectClose(point[1], expected[lineIndex][vertex][1], 1e-6, absolute);
    });
  });
}

it('GPURectangleClip clips lines like the f64 oracle and matches Shapely clip_by_rect', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Shapely 2.1.2 clip_by_rect(0, 0, 10, 10): lengths 16.2777, 12.1655 and 0 (total 28.4432).
  const fixed: Point[][] = [
    [
      [-5, 5],
      [5, 5],
      [15, 5],
      [15, 15],
      [5, 12],
      [-2, 2]
    ],
    [
      [2, 2],
      [8, 3],
      [9, 9]
    ],
    [
      [-3, -3],
      [-1, -1]
    ]
  ];
  const random = createRandom(9);
  const randomLines: Point[][] = Array.from({length: 150}, () => {
    const count = 2 + Math.floor(random() * 6);
    return Array.from({length: count}, () => [random() * 24 - 7, random() * 24 - 7] as Point);
  });
  for (const [lines, label] of [
    [fixed, 'fixed'],
    [randomLines, 'random']
  ] as const) {
    const capacity = 2048;
    const lineCapacity = 512;
    const fixture = createFixture(device, [...lines], 'lines', capacity, lineCapacity);
    for (const rect of [
      {minX: 0, minY: 0, maxX: 10, maxY: 10},
      {minX: 3.5, minY: -2, maxX: 8, maxY: 6}
    ] as Rectangle[]) {
      const result = await fixture.run(getGPURectangleClipParameterValues(rect));
      const expected = clipLines(round([...lines]), rect);
      expect(result['overflow'][0]).toBe(0);
      expect(result['count'][0]).toBe(expected.lines.length);
      expect(result['requiredCount'][0]).toBe(expected.lines.length);
      expect(result['vertexCount'][0]).toBe(expected.lines.flat().length);
      expect(result['requiredVertexCount'][0]).toBe(expected.lines.flat().length);
      expect(result['vertexCount'][0]).toBeGreaterThan(0);
      expectLinesClose(readLines(result, result['count'][0]), expected.lines, 1e-4);
      expect(result['sourceLineIndices'].slice(0, expected.lines.length)).toEqual(
        expected.sourceLineIndices
      );
      expect(result['lineOffsets'][result['count'][0]]).toBe(result['vertexCount'][0]);
      if (label === 'fixed' && rect.maxX === 10) {
        const length = readLines(result, result['count'][0]).reduce(
          (sum, line) =>
            sum +
            line
              .slice(1)
              .reduce(
                (lineSum, point, index) =>
                  lineSum + Math.hypot(point[0] - line[index][0], point[1] - line[index][1]),
                0
              ),
          0
        );
        expectClose(length, 28.443182234402343, 1e-5, 1e-4, 'Shapely length');
      }
    }
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  }
});

it('GPURectangleClip reports overflow when line or vertex capacity is exceeded', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const zigzag: Point[] = Array.from({length: 40}, (_, index) => [
    index,
    index % 2 === 0 ? -1 : 11
  ]);
  const fixture = createFixture(device, [zigzag], 'lines', 16, 4);
  const result = await fixture.run(
    getGPURectangleClipParameterValues({minX: 0, minY: 0, maxX: 40, maxY: 10})
  );
  expect(result['overflow'][0]).toBe(1);
  expect(result['vertexCount'][0]).toBeLessThanOrEqual(16);
  expect(result['requiredVertexCount'][0]).toBeGreaterThan(16);
  // Every vertex of the zigzag is outside the rectangle, so each of its 39 segments is its own piece.
  const pieceCount = clipLines(round([zigzag]), {minX: 0, minY: 0, maxX: 40, maxY: 10}).lines
    .length;
  expect(pieceCount).toBe(39);
  expect(result['count'][0]).toBe(4);
  expect(result['requiredCount'][0]).toBe(pieceCount);
  expect(result['lineOffsets'][4]).toBeLessThanOrEqual(16);
  fixture.destroy();
});

it('GPURectangleClip clips polygon rings like f64 Sutherland-Hodgman and matches Shapely areas', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Shapely 2.1.2 clip_by_rect(0, 0, 10, 10) areas: 100, 40 (C shape), 100, 7.5.
  const rings: Point[][] = [
    [
      [-5, -5],
      [15, -5],
      [15, 15],
      [-5, 15]
    ],
    [
      [2, 2],
      [20, 2],
      [20, 4],
      [4, 4],
      [4, 8],
      [20, 8],
      [20, 10],
      [2, 10]
    ],
    [
      [-10, 5],
      [5, -10],
      [20, 5],
      [5, 20]
    ],
    [
      [1, 1],
      [4, 1],
      [2.5, 6]
    ],
    // Fully outside: becomes an empty ring.
    [
      [30, 30],
      [31, 30],
      [30, 31]
    ]
  ];
  const shapelyAreas = [100, 40, 100, 7.5];
  const capacity = 256;
  const fixture = createFixture(device, rings, 'polygons', capacity, rings.length);
  const rect = {minX: 0, minY: 0, maxX: 10, maxY: 10};
  const result = await fixture.run(getGPURectangleClipParameterValues(rect));
  expect(result['overflow'][0]).toBe(0);
  // Polygons count rings: every input ring is kept, the one outside the rectangle as an empty ring.
  expect(result['count'][0]).toBe(rings.length);
  expect(result['requiredCount'][0]).toBe(rings.length);
  const actual = readLines(result, rings.length);
  const expected = round(rings).map(ring => clipRing(ring, rect));
  expectLinesClose(actual, expected, 1e-4);
  expect(actual[4].length).toBe(0);
  shapelyAreas.forEach((area, index) => {
    expectClose(getRingArea(actual[index]), area, 1e-5, 1e-3, `ring ${index} area`);
  });
  // Moving the rectangle per frame changes the result without recompiling.
  const moved = await fixture.run(
    getGPURectangleClipParameterValues({minX: 6, minY: 6, maxX: 12, maxY: 30})
  );
  const movedExpected = round(rings).map(ring =>
    clipRing(ring, {minX: 6, minY: 6, maxX: 12, maxY: 30})
  );
  expectLinesClose(readLines(moved, rings.length), movedExpected, 1e-4);
  expect(moved['vertexCount'][0]).toBe(movedExpected.flat().length);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
  const small = createFixture(device, rings, 'polygons', 6, rings.length);
  const clipped = await small.run(getGPURectangleClipParameterValues(rect));
  expect(clipped['overflow'][0]).toBe(1);
  expect(clipped['vertexCount'][0]).toBeLessThanOrEqual(6);
  small.destroy();
});

it('GPURectangleClip clips longitude-latitude degrees like planar coordinates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const spatialContext: GPUSpatialContext = {coordinateSpace: 'longitude-latitude', metric: 'none'};
  // Lines near Paris; the rectangle is in the same degrees as the input.
  const lines: Point[][] = [
    [
      [1.5, 48.5],
      [2.5, 48.9],
      [3.5, 49.5]
    ],
    [
      [2.1, 48.0],
      [2.2, 48.2]
    ]
  ];
  const rect: Rectangle = {minX: 2, minY: 48.25, maxX: 3, maxY: 49};
  const fixture = createFixture(device, lines, 'lines', 64, 8, spatialContext);
  const result = await fixture.run(getGPURectangleClipParameterValues(rect));
  const expected = clipLines(round(lines), rect);
  expect(expected.lines.length).toBe(1);
  expect(result['overflow'][0]).toBe(0);
  expect(result['count'][0]).toBe(expected.lines.length);
  expectLinesClose(readLines(result, result['count'][0]), expected.lines, 1e-5);
  fixture.destroy();
});

it('GPURectangleClip gives both polygons of a shared edge the same clip point (review 07)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Math correctness review 2026-11-10, polygon-topology-07. Each edge below is shared by two
  // triangles that traverse it in opposite directions. The first is the report's edge, which
  // gave y = 4.1434054 from one end and 4.1434059 from the other at x = 10; the others are
  // random edges across the clip lines x = 10 and y = 5.
  const random = createRandom(31);
  const sharedEdges: [Point, Point][] = [
    [
      [0.204, 0.165],
      [15.43, 6.349]
    ]
  ];
  for (let index = 0; index < 200; index++) {
    sharedEdges.push([
      [random() * 9, random() * 4],
      [11 + random() * 9, 6 + random() * 4]
    ]);
  }
  const rings: Point[][] = [];
  for (const [start, end] of sharedEdges) {
    // One triangle on each side of the edge, far apexes inside the rectangle.
    rings.push([start, end, [1, 4.5]], [end, start, [9, 0.5]]);
  }
  const rect = {minX: -1, minY: -1, maxX: 10, maxY: 5};
  const fixture = createFixture(device, rings, 'polygons', 4096, rings.length);
  const result = await fixture.run(getGPURectangleClipParameterValues(rect));
  expect(result['overflow'][0]).toBe(0);
  const clipped = readLines(result, rings.length);
  let comparedPoints = 0;
  sharedEdges.forEach(([start, end], edge) => {
    const [startX, startY, endX, endY] = [...start, ...end].map(Math.fround);
    // The edge leaves the rectangle through x = 10 or through y = 5, whichever comes first.
    const atRight = startY + ((rect.maxX - startX) / (endX - startX)) * (endY - startY);
    const atTop = startX + ((rect.maxY - startY) / (endY - startY)) * (endX - startX);
    const onBoundary = (point: Point) =>
      atRight <= rect.maxY
        ? point[0] === rect.maxX && Math.abs(point[1] - atRight) < 1e-5
        : point[1] === rect.maxY && Math.abs(point[0] - atTop) < 1e-5;
    const first = clipped[2 * edge].filter(onBoundary);
    const second = clipped[2 * edge + 1].filter(onBoundary);
    expect(first.length, `edge ${edge} first ring`).toBe(1);
    expect(second.length, `edge ${edge} second ring`).toBe(1);
    // Bit for bit: a coverage clipped this way still matches vertex for vertex.
    expect(second[0], `edge ${edge}`).toEqual(first[0]);
    comparedPoints++;
  });
  expect(comparedPoints).toBe(sharedEdges.length);
  fixture.destroy();
});
