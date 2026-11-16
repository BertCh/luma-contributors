// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createVectorView} from '../../utils/gpu-contributor-test-utils';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  getGPUBoundsFilterParameterValues,
  GPUBoundsFilter,
  type GPUBoundsFilterMode
} from '../../../src/gpu-spatial-analysis/bounds-filter/index';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {requireTestDevice} from '../../utils/require-test-device';

// Pinned from geopandas 1.2.0 / shapely 2.1.2 (scratchpad build/D/gen_bounds.py): 42 features
// (points, lines, rectangles on a 0.5 grid, an empty polygon at row 40 and box(0, 0, 16, 16) at
// row 41). Expected rows are GeoSeries.envelope.intersects(box) and .contains(box); intersects also
// equals GeoSeries.cx[xmin:xmax, ymin:ymax] for every box here. 'within' is the closed comparison of
// GeoSeries.bounds (withinClosed): OGC envelope.within(box) rejects features whose bounds only touch
// the box border, so it is a strict subset of the closed result.
const BOUNDS = [
  15.0, 10.0, 17.5, 13.5, 9.0, 12.0, 12.0, 12.5, 0.5, 4.5, 1.5, 7.5, 14.5, 0.0, 16.0, 3.0, 2.0,
  12.5, 2.0, 14.0, 13.0, 4.5, 14.0, 5.5, 11.5, 4.0, 15.0, 5.5, 7.5, 8.0, 9.5, 10.0, 8.0, 15.5, 11.0,
  18.5, 11.0, 9.5, 12.0, 13.0, 7.0, 3.0, 10.0, 3.5, 13.5, 9.5, 13.5, 9.5, 7.0, 0.5, 7.5, 2.5, 15.5,
  7.0, 18.5, 10.5, 13.0, 10.0, 14.5, 12.0, 4.0, 7.5, 5.5, 8.0, 15.5, 0.0, 15.5, 0.5, 15.5, 11.0,
  19.0, 11.5, 11.5, 5.5, 13.0, 5.5, 9.5, 13.0, 12.0, 13.5, 8.5, 4.0, 12.0, 7.5, 2.5, 8.0, 6.0, 11.0,
  11.0, 10.0, 11.0, 12.5, 7.5, 1.0, 8.0, 3.0, 11.5, 8.0, 13.5, 11.0, 10.5, 5.5, 13.0, 7.5, 1.5, 0.5,
  4.0, 2.0, 9.5, 5.0, 10.0, 5.5, 6.0, 13.0, 7.5, 14.5, 9.0, 15.5, 10.5, 17.5, 7.0, 9.5, 8.5, 12.0,
  8.5, 10.5, 12.0, 11.0, 8.5, 7.0, 9.5, 7.5, 0.5, 6.0, 3.5, 6.0, 6.0, 15.0, 9.5, 15.5, 0.0, 10.5,
  0.0, 11.5, 7.5, 13.5, 7.5, 16.0, 8.5, 2.0, 11.5, 5.0, 7.5, 15.0, 9.5, 18.5, 15.0, 9.0, 16.0, 9.5,
  0, 0, 0, 0, 0.0, 0.0, 16.0, 16.0
];
const CASES: {
  box: number[];
  intersects: number[];
  within: number[];
  withinClosed: number[];
  contains: number[];
}[] = [
  {
    box: [4, 4, 10, 10],
    intersects: [7, 15, 20, 21, 27, 30, 32, 37, 41],
    within: [7, 15, 27, 32],
    withinClosed: [7, 15, 27, 32],
    contains: [41]
  },
  {
    box: [0, 0, 16, 16],
    intersects: [
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
      26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41
    ],
    within: [
      1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30,
      31, 32, 33, 34, 37, 39, 41
    ],
    withinClosed: [
      1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
      30, 31, 32, 33, 34, 35, 36, 37, 39, 41
    ],
    contains: [41]
  },
  {box: [6, 6, 8, 8], intersects: [7, 21, 41], within: [], withinClosed: [], contains: [41]},
  {box: [20, 20, 30, 30], intersects: [], within: [], withinClosed: [], contains: []}
];
const EMPTY_ROW = 40;

function createBoundsValues(): Float32Array {
  const values = new Float32Array(BOUNDS);
  // GeoPandas reports NaN bounds for empty geometries.
  values.fill(NaN, EMPTY_ROW * 4, EMPTY_ROW * 4 + 4);
  return values;
}

function createFixture(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  mode: GPUBoundsFilterMode,
  capacity: number,
  chunkLengths?: number[],
  includeMask = true
) {
  return createGeometryFixture(device, {
    inputs: {bounds: {values: createBoundsValues(), format: 'float32x4'}},
    outputs: {
      ids: {format: 'uint32', length: capacity},
      mask: {format: 'uint32', length: 42},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      requiredCount: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) => {
      function chunkView<Format extends 'float32x4' | 'uint32'>(view: GraphDataView<Format>) {
        if (!chunkLengths) return view;
        let firstRow = 0;
        return createVectorView(
          `chunks-${view.buffer.id}`,
          view.format,
          chunkLengths.map(length => {
            const chunk = (view.buffer.graph as GPUCommandGraph).createDataView(view.buffer, {
              format: view.format,
              length,
              byteOffset: firstRow * view.rowByteLength
            });
            firstRow += length;
            return chunk;
          })
        );
      }
      return new GPUBoundsFilter({
        bounds: chunkView(inputs['bounds'] as GraphDataView<'float32x4'>),
        parameters,
        mode,
        output: {
          outputMask: includeMask
            ? chunkView(outputs['mask'] as GraphDataView<'uint32'>)
            : undefined,
          ids: outputs['ids'] as never,
          status: {
            count: outputs['count'] as never,
            overflow: outputs['overflow'] as never,
            requiredCount: outputs['requiredCount'] as never
          }
        }
      });
    }
  });
}

for (const mode of ['intersects', 'within', 'contains'] as const) {
  it(`GPUBoundsFilter ${mode} matches geopandas envelope.${mode}(box) and moves per frame`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createFixture(device, mode, 64);
    for (const testCase of CASES) {
      const [minX, minY, maxX, maxY] = testCase.box;
      const result = await fixture.run(getGPUBoundsFilterParameterValues({minX, minY, maxX, maxY}));
      const expected = mode === 'within' ? testCase.withinClosed : testCase[mode];
      for (const row of testCase.within) {
        expect(testCase.withinClosed).toContain(row);
      }
      expect(result['overflow'][0]).toBe(0);
      expect(result['count'][0]).toBe(expected.length);
      expect(result['requiredCount'][0]).toBe(expected.length);
      expect(result['ids'].slice(0, expected.length)).toEqual(expected);
      expect(result['mask'].reduce((sum, value) => sum + value, 0)).toBe(expected.length);
      expect(result['mask'][EMPTY_ROW]).toBe(0);
    }
    expect(fixture.getCompileCount()).toBe(0);
    fixture.destroy();
  });
}

it('GPUBoundsFilter clamps to a smaller capacity and reports overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device, 'intersects', 5);
  const result = await fixture.run(
    getGPUBoundsFilterParameterValues({minX: 0, minY: 0, maxX: 16, maxY: 16})
  );
  expect(result['requiredCount'][0]).toBe(41);
  expect(result['count'][0]).toBe(5);
  expect(result['overflow'][0]).toBe(1);
  expect(result['ids']).toEqual(CASES[1].intersects.slice(0, 5));
  fixture.destroy();
});

it('getGPUBoundsFilterParameterValues rejects inverted and non-finite boxes', () => {
  expect(() => getGPUBoundsFilterParameterValues({minX: 2, minY: 0, maxX: 1, maxY: 1})).toThrow();
  expect(() =>
    getGPUBoundsFilterParameterValues({minX: 0, minY: 0, maxX: Infinity, maxY: 1})
  ).toThrow();
});

it('GPUBoundsFilter reads minX > maxX with wrapX as a box through the longitude seam', async () => {
  expect(() => getGPUBoundsFilterParameterValues({minX: 12, minY: 0, maxX: 5, maxY: 16})).toThrow();
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const box = {minX: 12, minY: 0, maxX: 5, maxY: 16, wrapX: true};
  const values = createBoundsValues();
  const rowCount = values.length / 4;
  const expected = {
    intersects: [] as number[],
    within: [] as number[],
    contains: [] as number[]
  };
  for (let row = 0; row < rowCount; row++) {
    const [fMinX, fMinY, fMaxX, fMaxY] = Array.from(values.slice(row * 4, row * 4 + 4));
    if (![fMinX, fMinY, fMaxX, fMaxY].every(Number.isFinite)) continue;
    const yIntersects = fMinY <= box.maxY && fMaxY >= box.minY;
    const yWithin = fMinY >= box.minY && fMaxY <= box.maxY;
    if ((fMaxX >= box.minX || fMinX <= box.maxX) && yIntersects) expected.intersects.push(row);
    if ((fMinX >= box.minX || fMaxX <= box.maxX) && yWithin) expected.within.push(row);
  }
  for (const mode of ['intersects', 'within', 'contains'] as const) {
    const fixture = createFixture(device, mode, 64);
    const result = await fixture.run(getGPUBoundsFilterParameterValues(box));
    expect(result['count'][0], mode).toBe(expected[mode].length);
    expect(result['ids'].slice(0, expected[mode].length), mode).toEqual(expected[mode]);
    fixture.destroy();
  }
});

for (const mode of ['intersects', 'within', 'contains'] as const) {
  for (const capacity of [5, 64]) {
    for (const includeMask of [false, true]) {
      it(`GPUBoundsFilter ${mode} packed and three uneven chunks agree (capacity ${capacity}, mask ${includeMask})`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const packed = createFixture(device, mode, capacity, undefined, includeMask);
        const chunked = createFixture(device, mode, capacity, [3, 11, 28], includeMask);
        try {
          for (const testCase of CASES) {
            const [minX, minY, maxX, maxY] = testCase.box;
            const parameters = getGPUBoundsFilterParameterValues({minX, minY, maxX, maxY});
            const packedResult = await packed.run(parameters);
            const chunkedResult = await chunked.run(parameters);
            const expected = mode === 'within' ? testCase.withinClosed : testCase[mode];
            expect(chunkedResult).toEqual(packedResult);
            expect(chunkedResult['ids'].slice(0, Math.min(capacity, expected.length))).toEqual(
              expected.slice(0, capacity)
            );
            expect(chunkedResult['requiredCount']).toEqual([expected.length]);
          }
        } finally {
          packed.destroy();
          chunked.destroy();
        }
      });
    }
  }
}

it('GPUBoundsFilter selects global rows across nine million bounds in three uneven chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // More bounds than fit in a single core 128MiB float32x4 storage binding (8,388,608 rows).
  // Only the last row of chunk 0, first row of chunk 1 and last row of chunk 2 are accepted.
  const lengths = [3_000_001, 2_000_000, 4_000_000];
  const values = lengths.map(length => new Float32Array(length * 4));
  values[0].fill(1, (lengths[0] - 1) * 4);
  values[1].fill(1, 0, 4);
  values[2].fill(1, (lengths[2] - 1) * 4);
  const fixture = createGeometryFixture(device, {
    inputs: Object.fromEntries(
      values.map((chunk, index) => [
        `bounds-${index}`,
        {values: chunk, format: 'float32x4' as const}
      ])
    ),
    outputs: {
      ids: {format: 'uint32', length: 2},
      count: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      requiredCount: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUBoundsFilter({
        bounds: createVectorView(
          'large-bounds',
          'float32x4',
          lengths.map((_, index) => inputs[`bounds-${index}`] as GraphDataView<'float32x4'>)
        ),
        parameters,
        output: {
          ids: outputs['ids'] as GraphDataView<'uint32'>,
          status: {
            count: outputs['count'] as GraphDataView<'uint32'>,
            overflow: outputs['overflow'] as GraphDataView<'uint32'>,
            requiredCount: outputs['requiredCount'] as GraphDataView<'uint32'>
          }
        }
      })
  });
  try {
    const result = await fixture.run(
      getGPUBoundsFilterParameterValues({minX: 1, minY: 1, maxX: 1, maxY: 1})
    );
    expect(result['ids']).toEqual([3_000_000, 3_000_001]);
    expect(result['count']).toEqual([2]);
    expect(result['requiredCount']).toEqual([3]);
    expect(result['overflow']).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});

it('bounds compaction crosses bit words, scan blocks and scratch binding ceilings and resets each frame', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Model a small binding device so the old full-row scratch exceeds one binding on 4097 rows.
  // Every bounds input chunk is at most 16KiB; the new bit masks and block counts are much smaller.
  const lengths = [1021, 7, 1024, 1024, 1021];
  const originalLimit = device.limits.maxStorageBufferBindingSize;
  Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
    configurable: true,
    value: 16384
  });
  let firstRow = 0;
  const selected: number[] = [];
  const values = lengths.map(length => {
    const chunk = new Float32Array(length * 4);
    for (let row = 0; row < length; row++) {
      const globalRow = firstRow + row;
      if (globalRow % 3 === 1 || globalRow % 256 === 255) {
        chunk.fill(1, row * 4, row * 4 + 4);
        selected.push(globalRow);
      }
    }
    firstRow += length;
    return chunk;
  });
  let fixture: ReturnType<typeof createGeometryFixture> | undefined;
  try {
    fixture = createGeometryFixture(device, {
      inputs: Object.fromEntries(
        values.map((chunk, index) => [
          `chunk-${index}`,
          {values: chunk, format: 'float32x4' as const}
        ])
      ),
      outputs: {
        ids: {format: 'uint32', length: 1000},
        count: {format: 'uint32', length: 1},
        requiredCount: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUBoundsFilter({
          bounds: createVectorView(
            'small-binding-bounds',
            'float32x4',
            lengths.map((_, index) => inputs[`chunk-${index}`] as GraphDataView<'float32x4'>)
          ),
          parameters,
          output: {
            ids: outputs['ids'] as GraphDataView<'uint32'>,
            status: {
              count: outputs['count'] as GraphDataView<'uint32'>,
              requiredCount: outputs['requiredCount'] as GraphDataView<'uint32'>,
              overflow: outputs['overflow'] as GraphDataView<'uint32'>
            }
          }
        })
    });
    const parameters = getGPUBoundsFilterParameterValues({minX: 1, minY: 1, maxX: 1, maxY: 1});
    const result = await fixture.run(parameters);
    expect(result['ids']).toEqual(selected.slice(0, 1000));
    expect(result['count']).toEqual([1000]);
    expect(result['requiredCount']).toEqual([selected.length]);
    expect(result['overflow']).toEqual([1]);
    for (const index of lengths.keys())
      fixture.writeInput(`chunk-${index}`, new Float32Array(lengths[index] * 4));
    const empty = await fixture.run(parameters);
    expect(empty['count']).toEqual([0]);
    expect(empty['requiredCount']).toEqual([0]);
    expect(empty['overflow']).toEqual([0]);
    for (const index of lengths.keys()) fixture.writeInput(`chunk-${index}`, values[index]);
    expect((await fixture.run(parameters))['ids']).toEqual(selected.slice(0, 1000));
  } finally {
    fixture?.destroy();
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
      configurable: true,
      value: originalLimit
    });
  }
});

for (const rowCount of [31, 32, 33, 255, 256, 257, 8191, 8192, 8193]) {
  for (const chunked of [false, true]) {
    it(`dense bounds compaction preserves offset views and truncated prefixes at ${rowCount} rows (chunked ${chunked})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const lengths = chunked
        ? [1, Math.floor(rowCount / 3), rowCount - 1 - Math.floor(rowCount / 3)]
        : [rowCount];
      const capacity = rowCount - 2;
      const fixture = createGeometryFixture(device, {
        inputs: Object.fromEntries(
          lengths.map((length, index) => {
            const values = new Float32Array((length + 2) * 4);
            values.fill(1, 4, (length + 1) * 4);
            return [`bounds-${index}`, {values, format: 'float32x4' as const}];
          })
        ),
        outputs: {
          ids: {format: 'uint32', length: capacity + 2},
          mask: {format: 'uint32', length: rowCount + 2},
          count: {format: 'uint32', length: 1},
          required: {format: 'uint32', length: 1},
          overflow: {format: 'uint32', length: 1}
        },
        create: ({inputs, outputs, parameters}) => {
          const graph = parameters.buffer.graph as GPUCommandGraph;
          let firstRow = 0;
          const masks = lengths.map(length => {
            const view = graph.createDataView(outputs['mask'].buffer, {
              format: 'uint32',
              length,
              byteOffset: (firstRow + 1) * 4
            });
            firstRow += length;
            return view;
          });
          const boundChunks = lengths.map((length, index) =>
            graph.createDataView(inputs[`bounds-${index}`].buffer, {
              format: 'float32x4',
              length,
              byteOffset: 16
            })
          );
          return new GPUBoundsFilter({
            bounds: chunked
              ? createVectorView('offset-bounds', 'float32x4', boundChunks)
              : boundChunks[0],
            parameters,
            output: {
              ids: graph.createDataView(outputs['ids'].buffer, {
                format: 'uint32',
                length: capacity,
                byteOffset: 4
              }),
              outputMask: chunked ? createVectorView('offset-masks', 'uint32', masks) : masks[0],
              status: {
                count: outputs['count'] as GraphDataView<'uint32'>,
                requiredCount: outputs['required'] as GraphDataView<'uint32'>,
                overflow: outputs['overflow'] as GraphDataView<'uint32'>
              }
            }
          });
        }
      });
      try {
        const result = await fixture.run(
          getGPUBoundsFilterParameterValues({minX: 1, minY: 1, maxX: 1, maxY: 1})
        );
        expect(result['ids']).toEqual([
          0,
          ...Array.from({length: capacity}, (_, index) => index),
          0
        ]);
        expect(result['mask']).toEqual([0, ...new Array(rowCount).fill(1), 0]);
        expect(result['count']).toEqual([capacity]);
        expect(result['required']).toEqual([rowCount]);
        expect(result['overflow']).toEqual([1]);
      } finally {
        fixture.destroy();
      }
    });
  }
}
