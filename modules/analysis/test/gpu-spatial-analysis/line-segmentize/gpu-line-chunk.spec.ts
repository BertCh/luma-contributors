// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getGPULineChunkParameterValues,
  GPULineChunk,
  type GPULineChunkSettings
} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {createFlatLines, createRandom, type FlatLines} from './line-segmentize-oracle';
import {chunkLines} from './line-chunk-oracle';
import {
  createLineLineFixture,
  expectLineLineParity,
  getSphericalErrorMeters
} from './line-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

function createChunkFixture(
  device: Device,
  lines: FlatLines,
  options: {
    mode: 'chunk' | 'substring';
    capacity: number;
    lineCapacity: number;
    spherical?: boolean;
  }
) {
  return createLineLineFixture(device, {
    inputs: {
      positions: {values: lines.positions, format: 'float32x2'},
      lineOffsets: {values: lines.lineOffsets, format: 'uint32'}
    },
    capacity: options.capacity,
    lineCapacity: options.lineCapacity,
    createContributor: (inputs, parameters, output) =>
      new GPULineChunk({
        positions: inputs['positions'] as GraphDataView<'float32x2'>,
        lineOffsets: inputs['lineOffsets'] as GraphDataView<'uint32'>,
        mode: options.mode,
        spatialContext: options.spherical
          ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
          : undefined,
        parameters,
        output
      })
  });
}

function createRandomLines(seed: number): FlatLines {
  const random = createRandom(seed);
  return createFlatLines(
    Array.from({length: 80}, (_, lineIndex) => {
      if (lineIndex % 19 === 3) {
        return [];
      }
      if (lineIndex % 17 === 5) {
        return [[random() * 50, random() * 50]];
      }
      if (lineIndex % 23 === 7) {
        // Zero-length line with repeated vertices.
        return [
          [5, 5],
          [5, 5]
        ];
      }
      let x = random() * 50;
      let y = random() * 50;
      return Array.from({length: 2 + Math.floor(random() * 9)}, (__, vertex) => {
        if (vertex > 0 && random() > 0.1) {
          x += (random() - 0.5) * 8;
          y += (random() - 0.5) * 8;
        }
        return [x, y];
      });
    })
  );
}

it('GPULineChunk splits lines into chunks like the f64 oracle, per-frame chunk length', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomLines(3);
  const capacity = 6000;
  const lineCapacity = 1500;
  const fixture = createChunkFixture(device, lines, {mode: 'chunk', capacity, lineCapacity});
  for (const chunkLength of [3.3, 7.77, 50, 0, 1.01]) {
    const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength}));
    const expected = chunkLines(lines, {mode: 'chunk', chunkLength, lineCapacity});
    // `count` counts pieces (output lines); requiredCount is independent of line capacity.
    expect(actual.count).toBe(Math.min(expected.pieceTotal, lineCapacity));
    expect(actual.requiredCount).toBe(expected.pieceTotal);
    expect(actual.sourceLineIndices).toEqual(expected.sourceLineIndices);
    expectLineLineParity(actual, expected, capacity, 2e-5 * 60, 2e-5);
    expect(actual.overflow).toBe(0);
  }
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULineChunk reports line and vertex capacity overflow with valid offsets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomLines(5);
  const lineCapacity = 40;
  const capacity = 150;
  const fixture = createChunkFixture(device, lines, {mode: 'chunk', capacity, lineCapacity});
  const chunkLength = 2.5;
  const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength}));
  const expected = chunkLines(lines, {mode: 'chunk', chunkLength, lineCapacity});
  expect(expected.pieceTotal).toBeGreaterThan(lineCapacity);
  expect(actual.overflow).toBe(1);
  expect(actual.count).toBe(lineCapacity);
  expect(actual.requiredCount).toBe(expected.pieceTotal);
  expectLineLineParity(actual, expected, capacity, 2e-3, 2e-5, undefined, 1);
  // A tiny chunk length is capped instead of wrapping counts.
  const tiny = await fixture.run(getGPULineChunkParameterValues({chunkLength: 1e-30}));
  expect(tiny.overflow).toBe(1);
  expect(tiny.count).toBe(lineCapacity);
  expect(tiny.requiredCount).toBe(0xffffffff);
  fixture.destroy();
});

it('GPULineChunk extracts substrings by measure range', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createRandomLines(7);
  const lineCount = lines.lineOffsets.length - 1;
  const capacity = 2000;
  const fixture = createChunkFixture(device, lines, {
    mode: 'substring',
    capacity,
    lineCapacity: lineCount
  });
  const ranges: GPULineChunkSettings[] = [
    {startMeasure: 1.25, endMeasure: 9.5},
    {startMeasure: -5, endMeasure: 1e9},
    {startMeasure: 4, endMeasure: 4},
    {startMeasure: 6, endMeasure: 2},
    {startMeasure: 0, endMeasure: 0.5}
  ];
  for (const range of ranges) {
    const actual = await fixture.run(getGPULineChunkParameterValues(range));
    const expected = chunkLines(lines, {mode: 'substring', ...range, lineCapacity: lineCount});
    expectLineLineParity(actual, expected, capacity, 2e-5 * 60, 2e-5);
  }
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULineChunk retains repeated vertices in row order between shared and terminal cuts', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [
      [0, 0],
      [0, 0],
      [2, 0],
      [2, 0],
      [4, 0],
      [4, 0]
    ]
  ]);
  const chunked = createChunkFixture(device, lines, {mode: 'chunk', capacity: 7, lineCapacity: 2});
  const substring = createChunkFixture(device, lines, {
    mode: 'substring',
    capacity: 4,
    lineCapacity: 1
  });
  try {
    // Measures are the x coordinates. The cuts at 0, 2 and 4 use source rows 0, 2 and 5.
    const chunks = await chunked.run(getGPULineChunkParameterValues({chunkLength: 2}));
    expect(chunks).toMatchObject({
      count: 2,
      requiredCount: 2,
      vertexCount: 7,
      requiredVertexCount: 7,
      overflow: 0,
      lineOffsets: [0, 3, 7],
      sourceRows: [0, 1, 1, 2, 3, 4, 4],
      measures: [0, 0, 2, 2, 2, 4, 4],
      positions: [
        [0, 0],
        [0, 0],
        [2, 0],
        [2, 0],
        [2, 0],
        [4, 0],
        [4, 0]
      ]
    });

    const slice = await substring.run(
      getGPULineChunkParameterValues({startMeasure: 2, endMeasure: 5})
    );
    expect(slice).toMatchObject({
      count: 1,
      requiredCount: 1,
      vertexCount: 4,
      requiredVertexCount: 4,
      overflow: 0,
      lineOffsets: [0, 4],
      sourceRows: [2, 3, 4, 4],
      measures: [2, 2, 4, 4],
      positions: [
        [2, 0],
        [2, 0],
        [4, 0],
        [4, 0]
      ]
    });
  } finally {
    chunked.destroy();
    substring.destroy();
  }
});

it('GPULineChunk retains the final moving row at a rounded terminal measure', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [
      [0, 0],
      [1, 0],
      [1, 2 ** -25]
    ]
  ]);
  const fixture = createChunkFixture(device, lines, {
    mode: 'substring',
    capacity: 3,
    lineCapacity: 1
  });
  try {
    // The last measure is 1 + 2^-25, which publishes as 1. The terminal cut uses row 2.
    const result = await fixture.run(
      getGPULineChunkParameterValues({startMeasure: 0, endMeasure: 1})
    );
    expect(result).toMatchObject({
      count: 1,
      requiredVertexCount: 3,
      overflow: 0,
      lineOffsets: [0, 3],
      measures: [0, 1, 1],
      positions: [
        [0, 0],
        [1, 0],
        [1, 2 ** -25]
      ]
    });
  } finally {
    fixture.destroy();
  }
});

it('GPULineChunk fills exactly three pieces on a length-21 line with chunk length seven', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [
      [0, 0],
      [21, 0]
    ]
  ]);
  const fixture = createChunkFixture(device, lines, {mode: 'chunk', capacity: 6, lineCapacity: 3});
  try {
    // The boundary at 3 × 7 reaches 21 exactly, so three slots suffice.
    const result = await fixture.run(getGPULineChunkParameterValues({chunkLength: 7}));
    expect(result).toMatchObject({
      count: 3,
      requiredCount: 3,
      vertexCount: 6,
      requiredVertexCount: 6,
      overflow: 0,
      lineOffsets: [0, 2, 4, 6],
      sourceLineIndices: [0, 0, 0],
      measures: [0, 7, 7, 14, 14, 21]
    });
    for (const [row, expectedX] of [0, 7, 7, 14, 14, 21].entries()) {
      expect(result.positions[row][0]).toBeCloseTo(expectedX, 5);
      expect(result.positions[row][1]).toBe(0);
    }
  } finally {
    fixture.destroy();
  }
});

it('GPULineChunk reports full piece requirements across empty lines and output padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [],
    [
      [0, 0],
      [100, 0]
    ],
    [],
    [[5, 5]],
    [
      [0, 0],
      [8, 0]
    ],
    []
  ]);
  const fixture = createChunkFixture(device, lines, {mode: 'chunk', capacity: 8, lineCapacity: 4});
  try {
    const limited = await fixture.run(getGPULineChunkParameterValues({chunkLength: 1}));
    // 100 + one copied singleton + 8, independent of the four output slots.
    expect(limited.requiredCount).toBe(109);
    expect(limited.count).toBe(4);
    expect(limited.overflow).toBe(1);
    for (const [row, expectedX] of [0, 1, 1, 2, 2, 3, 3, 4].entries()) {
      expect(limited.positions[row]![0]).toBeCloseTo(expectedX, 5);
      expect(limited.positions[row]![1]).toBe(0);
    }
    expect(limited.sourceLineIndices).toEqual([1, 1, 1, 1]);
    const tiny = await fixture.run(getGPULineChunkParameterValues({chunkLength: 1e-30}));
    expect(tiny.requiredCount).toBe(0xffffffff);
    expect(tiny.overflow).toBe(1);
    const complete = await fixture.run(getGPULineChunkParameterValues({chunkLength: 100}));
    expect(complete.requiredCount).toBe(3);
    expect(complete.count).toBe(3);
    expect(complete.overflow).toBe(0);
    expect(complete.lineOffsets).toEqual([0, 2, 3, 5, 5]);
    expect(complete.sourceLineIndices).toEqual([1, 3, 4, 0xffffffff]);
  } finally {
    fixture.destroy();
  }
});

it('GPULineChunk saturates the sum when individually representable piece counts wrap uint32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [
      [0, 0],
      [2 ** 31, 0]
    ],
    [
      [0, 0],
      [2 ** 31, 0]
    ]
  ]);
  const fixture = createChunkFixture(device, lines, {mode: 'chunk', capacity: 4, lineCapacity: 2});
  try {
    const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength: 1}));
    expect(actual.requiredCount).toBe(0xffffffff);
    expect(actual.count).toBe(2);
    expect(actual.overflow).toBe(1);
    expect(actual.lineOffsets).toEqual([0, 2, 4]);
    expect(actual.sourceLineIndices).toEqual([0, 0]);
  } finally {
    fixture.destroy();
  }
});

it('GPULineChunk preserves equal-measure endpoints on a nearly antipodal great-circle line', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const targetLongitude = 179.99998474121094;
  const lines = createFlatLines([
    [
      [0, 0],
      [targetLongitude, 0]
    ]
  ]);
  const fixture = createChunkFixture(device, lines, {
    mode: 'substring',
    capacity: 2,
    lineCapacity: 1,
    spherical: true
  });
  try {
    // Equal measures emit two identical positions. The end case clamps to the
    // line length and exercises fraction one, while [0, 0] exercises zero.
    for (const [measure, longitude] of [
      [0, 0],
      [1e9, targetLongitude]
    ]) {
      const result = await fixture.run(
        getGPULineChunkParameterValues({
          startMeasure: measure,
          endMeasure: measure
        })
      );
      expect(result.positions).toEqual([
        [longitude, 0],
        [longitude, 0]
      ]);
      expect(result.lineOffsets).toEqual([0, 2]);
      expect(result.count).toBe(1);
      expect(result.vertexCount).toBe(2);
      expect(result.overflow).toBe(0);
    }
  } finally {
    fixture.destroy();
  }
});

it('GPULineChunk chunks spherical lines across the antimeridian', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [
      [170, 10],
      [-175, 20],
      [-160, 15]
    ],
    [
      [2.35, 48.85],
      [2.36, 48.86],
      [2.38, 48.855]
    ]
  ]);
  const capacity = 3000;
  const lineCapacity = 400;
  const fixture = createChunkFixture(device, lines, {
    mode: 'chunk',
    capacity,
    lineCapacity,
    spherical: true
  });
  for (const chunkLength of [250000, 25000]) {
    const actual = await fixture.run(getGPULineChunkParameterValues({chunkLength}));
    const expected = chunkLines(lines, {
      mode: 'chunk',
      chunkLength,
      lineCapacity,
      spherical: true
    });
    expect(actual.overflow).toBe(0);
    expectLineLineParity(actual, expected, capacity, 5, 2e-5, getSphericalErrorMeters);
  }
  fixture.destroy();
});

/**
 * Axis-aligned staircase lines with integer step lengths: every row measure is an exact integer in
 * f32 whatever the summation order, so chunk boundaries cannot flip on rounding.
 */
function createWalkLines(seed: number, lengths: number[]): FlatLines {
  const random = createRandom(seed);
  return createFlatLines(
    lengths.map(length => {
      let x = Math.floor(random() * 50);
      let y = Math.floor(random() * 50);
      return Array.from({length}, (_, row) => {
        if (row > 0) {
          const step = (random() < 0.5 ? -1 : 1) * (1 + Math.floor(random() * 4));
          if (random() < 0.5) {
            x += step;
          } else {
            y += step;
          }
        }
        return [x, y];
      });
    })
  );
}

it('GPULineChunk handles long lines whose prefix spans many scan tiles and pieces spanning many rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Line lengths around the 64-row tile size (63, 64, 65, 129; every line is longer than the substring start)
  // and one 2600-row line, so the
  // cooperative prefix, the tile carry and the per-vertex emit all see partial and full tiles.
  const lines = createWalkLines(13, [30, 63, 2600, 64, 65, 129, 40]);
  const lineCount = lines.lineOffsets.length - 1;
  const capacity = 12000;
  const chunked = createChunkFixture(device, lines, {mode: 'chunk', capacity, lineCapacity: 800});
  for (const chunkLength of [900, 13, 13.5, 1e6]) {
    const actual = await chunked.run(getGPULineChunkParameterValues({chunkLength}));
    const expected = chunkLines(lines, {mode: 'chunk', chunkLength, lineCapacity: 800});
    expectLineLineParity(actual, expected, capacity, 2e-5 * 600, 2e-5);
  }
  chunked.destroy();
  const substring = createChunkFixture(device, lines, {
    mode: 'substring',
    capacity,
    lineCapacity: lineCount
  });
  // One piece per line that copies almost the whole 2600-row line.
  for (const range of [
    {startMeasure: 7.5, endMeasure: 1e9},
    {startMeasure: 12.5, endMeasure: 4000.25}
  ]) {
    const actual = await substring.run(getGPULineChunkParameterValues(range));
    const expected = chunkLines(lines, {mode: 'substring', ...range, lineCapacity: lineCount});
    expectLineLineParity(actual, expected, capacity, 2e-5 * 600, 2e-5);
  }
  substring.destroy();
});

it('GPULineChunk substring preserves the bend between measures one and three', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createFlatLines([
    [
      [0, 0],
      [2, 0],
      [2, 2]
    ]
  ]);
  const fixture = createChunkFixture(device, lines, {
    mode: 'substring',
    capacity: 3,
    lineCapacity: 1
  });
  try {
    const actual = await fixture.run(
      getGPULineChunkParameterValues({startMeasure: 1, endMeasure: 3})
    );
    // Hand lengths: the bend is at measure 2; interpolating endpoints gives (1,0) and (2,1).
    expect(actual.count).toBe(1);
    expect(actual.overflow).toBe(0);
    expect(actual.vertexCount).toBe(3);
    expect(Array.from(actual.lineOffsets)).toEqual([0, 3]);
    actual.positions.flat().forEach(value => expect(Number.isFinite(value)).toBe(true));
    expect(actual.positions).toEqual([
      [1, 0],
      [2, 0],
      [2, 1]
    ]);
    expect(Array.from(actual.measures)).toEqual([1, 2, 3]);
  } finally {
    fixture.destroy();
  }
});
