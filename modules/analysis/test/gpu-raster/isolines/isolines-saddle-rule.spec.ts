// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterContours} from '@luma.gl/graph/gpu-raster';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {
  getGPUIsobandsParameterValues,
  getGPUIsolinesParameterValues,
  GPUIsobands,
  GPUIsolines
} from '../../../src/raster-analysis/isolines/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// raster-fields-06 and luma-official-01. A saddle cell (high corners on one diagonal) is resolved
// by the bilinear (asymptotic) decider, as the published GPURasterContours does: the high corners
// are joined when the saddle value of the bilinear surface, (v0 v2 - v1 v3) / (v0 + v2 - v1 - v3),
// is at or above the level. The old rule compared the mean of the four corners.

type Segment = [number, number, number, number];

type Result = {
  /** GPUIsolines segments `[x0, y0, x1, y1]`. */
  local: Segment[];
  /** Published GPURasterContours segments. */
  official: Segment[];
  /** Triangles of GPUIsobands for the single break, as `{band, area}`. */
  bandTriangles: {band: number; area: number}[];
};

/**
 * Runs `GPUIsolines`, the published `GPURasterContours` and `GPUIsobands` on one raster at one
 * level. The extent is `[0, 0, width, height]`, so the local world coordinates are the published
 * kernel's pixel-centre coordinates (sample `(column, row)` at `(column + 0.5, row + 0.5)`).
 */
async function runContours(
  device: Device,
  width: number,
  height: number,
  values: Float32Array,
  level: number
): Promise<Result> {
  const sampleCount = width * height;
  const capacity = 2 * (width - 1) * (height - 1);
  const triangleCapacity = 16 * (width - 1) * (height - 1);
  const graph = new GPUCommandGraph(device, {id: 'saddle-rule'});
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const grid = makeGPURasterMetadata({width, height});
  const extent = [0, 0, width, height] as const;
  const valuesView = importGraphBuffer(
    graph,
    'values',
    track(createInputBuffer(device, values)),
    'float32',
    sampleCount
  );
  const levelsView = importGraphBuffer(
    graph,
    'levels',
    track(createInputBuffer(device, Float32Array.of(level))),
    'float32',
    1
  );
  const segmentsBuffer = track(createOutputBuffer(device, 4 * capacity));
  const countBuffer = track(createOutputBuffer(device, 1));
  graph.add(
    new GPUIsolines({
      id: 'local',
      grid,
      values: valuesView,
      levels: levelsView,
      parameters: importGraphBuffer(
        graph,
        'isoline-parameters',
        track(
          createInputBuffer(device, getGPUIsolinesParameterValues({levelCount: 1, extent}, grid))
        ),
        'float32',
        8
      ),
      output: {
        segments: importGraphBuffer(graph, 'segments', segmentsBuffer, 'float32x4', capacity),
        segmentLevels: importGraphBuffer(
          graph,
          'segment-levels',
          track(createOutputBuffer(device, capacity)),
          'uint32',
          capacity
        ),
        status: {
          overflow: importGraphBuffer(
            graph,
            'overflow',
            track(createOutputBuffer(device, 1)),
            'uint32',
            1
          ),
          count: importGraphBuffer(graph, 'count', countBuffer, 'uint32', 1)
        }
      }
    })
  );
  const officialVerticesBuffer = track(createOutputBuffer(device, 4 * capacity));
  const officialCountBuffer = track(createOutputBuffer(device, 1));
  new GPURasterContours({
    id: 'official',
    width,
    height,
    input: {id: 'official-band', format: 'float32', storage: {kind: 'buffer', values: valuesView}},
    level,
    vertices: importGraphBuffer(
      graph,
      'official-vertices',
      officialVerticesBuffer,
      'float32x2',
      2 * capacity
    ),
    segmentCount: importGraphBuffer(graph, 'official-count', officialCountBuffer, 'uint32', 1),
    overflow: importGraphBuffer(
      graph,
      'official-overflow',
      track(createOutputBuffer(device, 1)),
      'uint32',
      1
    )
  }).addToGraph(graph);
  const trianglesBuffer = track(createOutputBuffer(device, 6 * triangleCapacity));
  const triangleBandsBuffer = track(createOutputBuffer(device, triangleCapacity));
  const triangleCountBuffer = track(createOutputBuffer(device, 1));
  const isobandParameterValues = getGPUIsobandsParameterValues({breakCount: 1, extent}, grid);
  graph.add(
    new GPUIsobands({
      id: 'bands',
      grid,
      values: valuesView,
      breaks: levelsView,
      parameters: importGraphBuffer(
        graph,
        'isoband-parameters',
        track(createInputBuffer(device, isobandParameterValues)),
        'float32',
        isobandParameterValues.length
      ),
      output: {
        triangles: importGraphBuffer(
          graph,
          'triangles',
          trianglesBuffer,
          'float32x2',
          3 * triangleCapacity
        ),
        triangleBands: importGraphBuffer(
          graph,
          'triangle-bands',
          triangleBandsBuffer,
          'uint32',
          triangleCapacity
        ),
        status: {
          overflow: importGraphBuffer(
            graph,
            'triangle-overflow',
            track(createOutputBuffer(device, 1)),
            'uint32',
            1
          ),
          count: importGraphBuffer(graph, 'triangle-count', triangleCountBuffer, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const toSegments = (floats: number[], count: number): Segment[] =>
      Array.from({length: count}, (_, index) => [
        floats[4 * index],
        floats[4 * index + 1],
        floats[4 * index + 2],
        floats[4 * index + 3]
      ]);
    const [count] = await readUint32(countBuffer, 1);
    const [officialCount] = await readUint32(officialCountBuffer, 1);
    const [triangleCount] = await readUint32(triangleCountBuffer, 1);
    const triangles = await readFloat32(trianglesBuffer, 6 * triangleCount);
    const triangleBands = await readUint32(triangleBandsBuffer, triangleCount);
    return {
      local: toSegments(await readFloat32(segmentsBuffer, 4 * count), count),
      official: toSegments(
        await readFloat32(officialVerticesBuffer, 4 * officialCount),
        officialCount
      ),
      bandTriangles: Array.from({length: triangleCount}, (_, index) => {
        const [ax, ay, bx, by, cx, cy] = triangles.slice(6 * index, 6 * index + 6);
        return {
          band: triangleBands[index],
          area: Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2
        };
      })
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

/** True when the two segments have the same endpoints, in either order, within `tolerance`. */
function isSameSegment(first: Segment, second: Segment, tolerance: number): boolean {
  const isClose = (a: number, b: number) => Math.abs(a - b) <= tolerance;
  return (
    (isClose(first[0], second[0]) &&
      isClose(first[1], second[1]) &&
      isClose(first[2], second[2]) &&
      isClose(first[3], second[3])) ||
    (isClose(first[0], second[2]) &&
      isClose(first[1], second[3]) &&
      isClose(first[2], second[0]) &&
      isClose(first[3], second[1]))
  );
}

/** Matches every segment of `first` with a distinct segment of `second`. */
function expectSameSegments(first: Segment[], second: Segment[], label: string): void {
  expect(first.length, `${label}: segment count`).toBe(second.length);
  const unused = second.slice();
  for (const segment of first) {
    const match = unused.findIndex(candidate => isSameSegment(segment, candidate, 1e-5));
    expect(
      match,
      `${label}: no partner for segment ${JSON.stringify(segment)}`
    ).toBeGreaterThanOrEqual(0);
    unused.splice(match, 1);
  }
}

/**
 * A 2 x 2 raster holding one cell with corners `v0..v3` in boundary order: row 0 holds `v0, v1`,
 * row 1 holds `v3, v2`. In the published kernel's names these are top-left, top-right,
 * bottom-right, bottom-left.
 */
function makeCell(v0: number, v1: number, v2: number, v3: number): Float32Array {
  return Float32Array.of(v0, v1, v3, v2);
}

it('the report saddles separate the high corners, as GPURasterContours does (raster-fields-06)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Corners 10, 0, 1.1, 0 at level 1. Mean 2.775 (the old rule joined the high corners v0 and v2).
  // Saddle value 10 * 1.1 / 11.1 = 0.991, below the level: the high corners are separated, and
  // each is cut off by its own segment. On the edges out of v0 the level is 0.9 of the way to the
  // low corner; on the edges out of v2 it is 0.1 / 1.1 = 0.0909 of the way.
  const first = await runContours(device, 2, 2, makeCell(10, 0, 1.1, 0), 1);
  const near = 0.9;
  const far = 0.1 / 1.1;
  const expectedSegments: Segment[] = [
    // Around v0 at (0.5, 0.5): the crossings on the edges to v1 (+x) and to v3 (+y).
    [0.5 + near, 0.5, 0.5, 0.5 + near],
    // Around v2 at (1.5, 1.5): the crossings on the edges to v3 (-x) and to v1 (-y).
    [1.5 - far, 1.5, 1.5, 1.5 - far]
  ];
  expectSameSegments(first.local, expectedSegments, 'local against hand-computed');
  expectSameSegments(first.official, expectedSegments, 'official against hand-computed');
  // GPUIsobands, one break at 1: band 1 (values >= 1) is the two corner triangles, of areas
  // 0.9^2 / 2 and 0.0909^2 / 2; band 0 is the hexagon between them. The old rule gave band 1 the
  // hexagon (area 0.909) and band 0 two triangles.
  const highTriangles = first.bandTriangles.filter(triangle => triangle.band === 1);
  const lowTriangles = first.bandTriangles.filter(triangle => triangle.band === 0);
  expect(highTriangles).toHaveLength(2);
  expect(lowTriangles).toHaveLength(4);
  const highArea = highTriangles.reduce((sum, triangle) => sum + triangle.area, 0);
  const lowArea = lowTriangles.reduce((sum, triangle) => sum + triangle.area, 0);
  expect(highArea).toBeCloseTo((near * near + far * far) / 2, 5);
  expect(lowArea).toBeCloseTo(1 - (near * near + far * far) / 2, 5);

  // The luma-official-01 cell: top-left 100, top-right -9, bottom-right 1.1, bottom-left -9 at
  // level 1. Mean 20.775; determinant (99)(0.1) - (-10)(-10) = -0.1 before scaling, so the high
  // corners are separated. Level crossings: 99 / 109 from v0 and 0.1 / 10.1 from v2.
  const second = await runContours(device, 2, 2, makeCell(100, -9, 1.1, -9), 1);
  const nearSecond = 99 / 109;
  const farSecond = 0.1 / 10.1;
  const expectedSecond: Segment[] = [
    [0.5 + nearSecond, 0.5, 0.5, 0.5 + nearSecond],
    [1.5 - farSecond, 1.5, 1.5, 1.5 - farSecond]
  ];
  expectSameSegments(second.local, expectedSecond, 'local against hand-computed');
  expectSameSegments(second.official, expectedSecond, 'official against hand-computed');
  expect(second.bandTriangles.filter(triangle => triangle.band === 1)).toHaveLength(2);

  // The other diagonal, mirrored: v1 and v3 high. Saddle value below the level again.
  const mirrored = await runContours(device, 2, 2, makeCell(0, 10, 0, 1.1), 1);
  expectSameSegments(
    mirrored.local,
    [
      [1.5 - near, 0.5, 1.5, 0.5 + near],
      [0.5 + far, 1.5, 0.5, 1.5 - far]
    ],
    'mirrored against hand-computed'
  );
  expectSameSegments(mirrored.local, mirrored.official, 'mirrored against official');

  // The opposite disagreement, a saddle whose bilinear value is above the level while the mean is
  // below it: corners 3, 0.99, 3, -9 at level 1 have mean -0.5025 (the old rule separated the high
  // corners) and saddle value (9 + 8.91) / 14.01 = 1.278, so the high corners v0 and v2 are joined
  // and the segments cut off the low corners v1 and v3.
  const joined = await runContours(device, 2, 2, makeCell(3, 0.99, 3, -9), 1);
  const towardV1 = 2 / 2.01;
  const towardV3 = 2 / 12;
  expectSameSegments(
    joined.local,
    [
      // Around v1 at (1.5, 0.5): crossings on the edge from v0 and on the edge to v2.
      [0.5 + towardV1, 0.5, 1.5, 1.5 - towardV1],
      // Around v3 at (0.5, 1.5): crossings on the edge from v0 and on the edge from v2.
      [0.5, 0.5 + towardV3, 1.5 - towardV3, 1.5]
    ],
    'joined against hand-computed'
  );
  expectSameSegments(joined.local, joined.official, 'joined against official');
});

it('GPUIsolines has the topology of the published GPURasterContours on a raster full of saddles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 12;
  const height = 9;
  const level = 1;
  // A checkerboard of high and low samples with unequal magnitudes: every interior cell is a
  // saddle, and the wide spread of values makes the mean and the bilinear value disagree often.
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const values = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const isHigh = (row + column) % 2 === 0;
      // High samples in (1, 1 + 10^k), low samples in (1 - 10^k, 1), k uniform in [-1.5, 1.5].
      const offset = 10 ** (3 * random() - 1.5) * (0.05 + random());
      values[row * width + column] = Math.fround(isHigh ? level + offset : level - offset);
    }
  }
  // The two report cells, embedded in the corner.
  values.set([10, 0], 0);
  values.set([0, 1.1], width);
  let saddleCount = 0;
  let disagreementCount = 0;
  for (let row = 0; row < height - 1; row++) {
    for (let column = 0; column < width - 1; column++) {
      const v0 = values[row * width + column];
      const v1 = values[row * width + column + 1];
      const v2 = values[(row + 1) * width + column + 1];
      const v3 = values[(row + 1) * width + column];
      const high = [v0, v1, v2, v3].map(value => value >= level);
      if (high[0] === high[2] && high[1] === high[3] && high[0] !== high[1]) {
        saddleCount++;
        const mean = (v0 + v1 + v2 + v3) / 4;
        const saddleValue = (v0 * v2 - v1 * v3) / (v0 + v2 - v1 - v3);
        disagreementCount += mean >= level !== saddleValue >= level ? 1 : 0;
      }
    }
  }
  expect(saddleCount).toBe((width - 1) * (height - 1));
  // The parity below is a test of the rule only if the two rules differ on this raster.
  expect(disagreementCount).toBeGreaterThan(10);
  const result = await runContours(device, width, height, values, level);
  expect(result.local).toHaveLength(2 * saddleCount);
  expectSameSegments(result.local, result.official, 'local against official');
});

it('an exact saddle tie on integer data joins the high corners', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Corners 7, 4, 7, 1 at level 5: d = (2, -1, 2, -4), determinant 2 * 2 - 1 * 4 = 0, so the
  // bilinear saddle value is exactly the level. A tie joins the high corners (v0 and v2), so the
  // segments cut off the low corners. The unscaled differences keep the tie exact in float32.
  const tie = await runContours(device, 2, 2, makeCell(7, 4, 7, 1), 5);
  expectSameSegments(
    tie.local,
    [
      // Around v1 at (1.5, 0.5): level 2 / 3 of the way from v0, and 1 / 3 of the way to v2.
      [0.5 + 2 / 3, 0.5, 1.5, 0.5 + 1 / 3],
      // Around v3 at (0.5, 1.5): level 2 / 6 of the way from v0, and 4 / 6 of the way to v2.
      [0.5, 0.5 + 2 / 6, 0.5 + 4 / 6, 1.5]
    ],
    'tie on the first diagonal'
  );
  // The same tie with v1 and v3 high: corners 4, 7, 1, 7. Joined, so the segments cut off the low
  // corners v0 and v2.
  const mirroredTie = await runContours(device, 2, 2, makeCell(4, 7, 1, 7), 5);
  expectSameSegments(
    mirroredTie.local,
    [
      // Around v0 at (0.5, 0.5): 1 / 3 of the way to v1 and 1 / 3 of the way to v3.
      [0.5 + 1 / 3, 0.5, 0.5, 0.5 + 1 / 3],
      // Around v2 at (1.5, 1.5): 4 / 6 of the way from v1 down the right edge, 2 / 6 from v3.
      [1.5, 0.5 + 2 / 6, 0.5 + 2 / 6, 1.5]
    ],
    'tie on the second diagonal'
  );
  // The published kernel is not asserted here. It divides the corners by the largest magnitude
  // before forming the determinant, so on an exact tie its choice depends on rounding (it agrees
  // on these two cells; with the same scaling the local kernel missed a tie of the random-field
  // spec in gpu-isolines.spec.ts).
});
