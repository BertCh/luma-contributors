// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import type {Buffer, Device} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUMarkedPointStatistics,
  type GPUMarkedPointStatisticsProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {GPURipley} from '../../../src/dataframe-analysis/pair-statistics/gpu-ripley';
import {getGPURipleyParameterValues} from '../../../src/dataframe-analysis/pair-statistics/ripley-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

function createContext(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    data: ArrayLike<number>
  ) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(data) : Float32Array.from(data)
    );
    buffers.push(buffer);
    const length = format === 'float32x2' ? data.length / 2 : data.length;
    return importGraphBuffer(graph, `input-${serial++}`, buffer, format, length);
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length)};
  };
  let compiled: ReturnType<typeof graph.compile> | undefined;
  return {
    graph,
    input,
    output,
    run: () => {
      compiled = graph.compile();
      submitGraph(device, compiled, undefined);
    },
    destroy: () => {
      compiled?.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectClose(actual: ArrayLike<number>, expected: ArrayLike<number>, label: string): void {
  expect(actual.length, label).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    expect(
      Math.abs(actual[index] - expected[index]),
      `${label}[${index}]: ${actual[index]} vs ${expected[index]}`
    ).toBeLessThanOrEqual(1e-5 * Math.abs(expected[index]) + 1e-6);
  }
}

// Four points in the window [0, 4] x [0, 4] (area 16). Within radius 1.5 only the neighbours along
// y = 1 are paired: (1, 1)-(2, 1) and (2, 1)-(3, 1), four ordered pairs.
const POSITIONS = [1, 1, 2, 1, 3, 1, 1, 3];
const WINDOW = [0, 0, 4, 4] as const;
const RADIUS = 1.5;
const AREA = 16;

it('GPUMarkedPointStatistics same-type K uses A / (n (n - 1)), as GPURipley and spatstat Kest', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createContext(device);
  try {
    const positions = context.input('float32x2', POSITIONS);
    const statistics = context.output('float32', 4);
    const pairCounts = context.output('uint32', 1);
    const status = context.output('uint32', 1);
    context.graph.add(
      new GPUMarkedPointStatistics({
        id: 'untyped',
        positions,
        parameters: context.input('float32', [...WINDOW, RADIUS, 0]),
        radiusCount: 1,
        edgeCorrection: 'none',
        output: {statistics: statistics.view, pairCounts: pairCounts.view, status: status.view}
      })
    );
    // The same pattern with two extra points of another type: K of type 1 against itself.
    const typedStatistics = context.output('float32', 4);
    context.graph.add(
      new GPUMarkedPointStatistics({
        id: 'same-type',
        positions: context.input('float32x2', [...POSITIONS, 2.5, 1, 3.5, 3.5]),
        types: context.input('uint32', [1, 1, 1, 1, 0, 0]),
        typePair: [1, 1],
        parameters: context.input('float32', [...WINDOW, RADIUS, 0]),
        radiusCount: 1,
        edgeCorrection: 'none',
        output: {
          statistics: typedStatistics.view,
          pairCounts: context.output('uint32', 1).view,
          status: context.output('uint32', 1).view
        }
      })
    );
    const ripleyK = context.output('float32', 1);
    context.graph.add(
      new GPURipley({
        id: 'ripley',
        positions,
        parameters: context.input(
          'float32',
          getGPURipleyParameterValues({
            bounds: WINDOW,
            maximumDistance: RADIUS,
            edgeCorrection: 'none'
          })
        ),
        gridSize: [2, 2],
        radiusCount: 1,
        output: {
          k: ripleyK.view,
          l: context.output('float32', 1).view,
          lMinusR: context.output('float32', 1).view
        }
      })
    );
    context.run();
    // Four ordered pairs, n = 4: K = 16 * 4 / (4 * 3) = 16 / 3. The n^2 form gave 16 * 4 / 16 = 4.
    const k = (AREA * 4) / (4 * 3);
    const expected = [k, Math.sqrt(k / Math.PI), k / (Math.PI * RADIUS * RADIUS), 1];
    expectClose(await readFloat32(statistics.buffer, 4), expected, 'untyped');
    expectClose(await readFloat32(typedStatistics.buffer, 4), expected, 'same type');
    expect(await readUint32(pairCounts.buffer, 1)).toEqual([4]);
    expect(await readUint32(status.buffer, 1)).toEqual([0]);
    expectClose(await readFloat32(ripleyK.buffer, 1), [k], 'GPURipley');
  } finally {
    context.destroy();
  }
});

it('GPUMarkedPointStatistics cross-type K keeps A / (n_i n_j) and one same-type point has no K', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createContext(device);
  try {
    // Types 0, 1, 0, 1: focus points (1, 1) and (3, 1), neighbours (2, 1) and (1, 3). Ordered
    // focus-to-neighbour pairs within 1.5: (1, 1)-(2, 1) and (3, 1)-(2, 1), so
    // K = 16 * 2 / (2 * 2) = 8.
    const cross = context.output('float32', 4);
    context.graph.add(
      new GPUMarkedPointStatistics({
        id: 'cross',
        positions: context.input('float32x2', POSITIONS),
        types: context.input('uint32', [0, 1, 0, 1]),
        typePair: [0, 1],
        parameters: context.input('float32', [...WINDOW, RADIUS, 0]),
        radiusCount: 1,
        edgeCorrection: 'none',
        output: {
          statistics: cross.view,
          pairCounts: context.output('uint32', 1).view,
          status: context.output('uint32', 1).view
        }
      })
    );
    // One point of type 2: the same-type estimator divides by n (n - 1) = 0 and is undefined.
    const single = context.output('float32', 4);
    const singleStatus = context.output('uint32', 1);
    context.graph.add(
      new GPUMarkedPointStatistics({
        id: 'single',
        positions: context.input('float32x2', POSITIONS),
        types: context.input('uint32', [0, 1, 2, 1]),
        typePair: [2, 2],
        parameters: context.input('float32', [...WINDOW, RADIUS, 0]),
        radiusCount: 1,
        edgeCorrection: 'none',
        output: {
          statistics: single.view,
          pairCounts: context.output('uint32', 1).view,
          status: singleStatus.view
        }
      })
    );
    context.run();
    expectClose(
      await readFloat32(cross.buffer, 4),
      [8, Math.sqrt(8 / Math.PI), 8 / (Math.PI * RADIUS * RADIUS), 1],
      'cross'
    );
    expect((await readFloat32(single.buffer, 4)).every(Number.isNaN)).toBe(true);
    expect(await readUint32(singleStatus.buffer, 1)).toEqual([1]);
  } finally {
    context.destroy();
  }
});

for (const scene of [
  {firstCategoryMark: 2e38, secondCategoryMark: 1e-20},
  {firstCategoryMark: -2e38, secondCategoryMark: 1e-20},
  {firstCategoryMark: 2e38, secondCategoryMark: 1e-20, typePair: [1, 0] as const},
  {firstCategoryMark: 3.4028234663852886e38, secondCategoryMark: 1e-20, firstCategoryCount: 3},
  {firstCategoryMark: -3.4028234663852886e38, secondCategoryMark: 1e-20, firstCategoryCount: 3},
  {firstCategoryMark: 1e-35, secondCategoryMark: 1e20}
]) {
  it(`marked category means remain finite for constant marks ${JSON.stringify(scene)}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const firstCategoryCount = scene.firstCategoryCount ?? 2;
    const options: Partial<GPUMarkedPointStatisticsProps>[] = [
      {},
      {gridSize: [4, 4]},
      {searchStructure: 'brute-force'}
    ];
    for (const searchOptions of options) {
      const context = createContext(device);
      try {
        const statistics = context.output('float32', 4);
        const pairCounts = context.output('uint32', 1);
        const status = context.output('uint32', 1);
        context.graph.add(
          new GPUMarkedPointStatistics({
            ...searchOptions,
            positions: context.input('float32x2', [
              ...new Array(firstCategoryCount * 2).fill(0),
              1,
              0
            ]),
            types: context.input('uint32', [...new Array(firstCategoryCount).fill(0), 1]),
            typePair: scene.typePair ?? [0, 1],
            marks: context.input('float32', [
              ...new Array(firstCategoryCount).fill(scene.firstCategoryMark),
              scene.secondCategoryMark
            ]),
            parameters: context.input('float32', [-1, -1, 2, 1, 2, 0]),
            radiusCount: 1,
            edgeCorrection: 'none',
            output: {statistics: statistics.view, pairCounts: pairCounts.view, status: status.view}
          })
        );
        context.run();
        // Each category has constant marks, so mean pair products / product of category means = 1.
        // Every category-0 point meets the category-1 point: K = area * n / (n * 1) = 6.
        expectClose(
          await readFloat32(statistics.buffer, 4),
          [6, Math.sqrt(6 / Math.PI), 6 / (4 * Math.PI), 1],
          JSON.stringify(searchOptions)
        );
        expect(await readUint32(pairCounts.buffer, 1)).toEqual([firstCategoryCount]);
        expect(await readUint32(status.buffer, 1)).toEqual([0]);
      } finally {
        context.destroy();
      }
    }
  });
}

it('marked means retain small marks after large opposite marks cancel outside the pair neighborhood', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const context = createContext(device);
  try {
    const statistics = context.output('float32', 4);
    const pairCounts = context.output('uint32', 1);
    const status = context.output('uint32', 1);
    context.graph.add(
      new GPUMarkedPointStatistics({
        positions: context.input('float32x2', [2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        types: context.input('uint32', [0, 0, 0, 0, 0, 0, 1]),
        typePair: [0, 1],
        marks: context.input('float32', [2e38, -2e38, 1e-35, 1e-35, 1e-35, 1e-35, 1e20]),
        parameters: context.input('float32', [-1, -1, 4, 1, 0.5, 0]),
        radiusCount: 1,
        edgeCorrection: 'none',
        output: {statistics: statistics.view, pairCounts: pairCounts.view, status: status.view}
      })
    );
    context.run();
    // The isolated opposite marks cancel in the category mean. Four pairs have the same product;
    // the focus mean is 4/6 of their marks, so mark correlation = 6/4, independently of magnitude.
    const estimate = (10 * 4) / 6;
    expectClose(
      await readFloat32(statistics.buffer, 4),
      [estimate, Math.sqrt(estimate / Math.PI), estimate / (Math.PI * 0.25), 1.5],
      'cancelling isolated marks'
    );
    expect(await readUint32(pairCounts.buffer, 1)).toEqual([4]);
    expect(await readUint32(status.buffer, 1)).toEqual([0]);
  } finally {
    context.destroy();
  }
});
