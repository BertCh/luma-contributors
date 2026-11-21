// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGammaStatistic,
  GPUSpatialPearson
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

for (const Contributor of [GPUGammaStatistic, GPUSpatialPearson]) {
  for (const chunked of [false, true]) {
    it(`${Contributor.name} ${chunked ? 'chunked' : 'whole'} statistic depends on both producer-loss flags`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `edge-audit-${serial++}`, format, length);
      const createWeights = (rows: number) => ({
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows * 2),
        weights: view('float32', rows * 2)
      });
      const base = chunked
        ? {
            rowCount: 8,
            chunks: [
              {rowStart: 0, rowCount: 3, weights: createWeights(3)},
              {rowStart: 3, rowCount: 5, weights: createWeights(5)}
            ]
          }
        : createWeights(8);
      try {
        const nodes = new Contributor({
          weights: {
            ...base,
            status: {overflow: view('uint32', 1), candidateOverflow: view('uint32', 1)}
          },
          values: view('float32', 8),
          secondValues: view('float32', 8),
          output: {statistic: view('float32', 1)}
        }).getCommandNodes(graph);
        const source = getGPUCommandNodeKernelSources(nodes).find(record =>
          record.id.endsWith('-statistic')
        )!.source;
        // Both flags must be shader inputs and guard publication, not mere host metadata.
        expect(source).toContain('inputOverflow[inputOverflowOffset]');
        expect(source).toContain('inputCandidateOverflow[inputCandidateOverflowOffset]');
        expect(source).toContain('answer = getQuietNaN(0u)');
      } finally {
        device.destroy();
      }
    });
  }

  it(`${Contributor.name} refuses a loss flag hidden in a CSR chunk`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `hidden-edge-audit-${serial++}`, format, length);
    try {
      expect(
        () =>
          new Contributor({
            weights: {
              rowCount: 8,
              chunks: [
                {
                  rowStart: 0,
                  rowCount: 8,
                  weights: {
                    offsets: view('uint32', 9),
                    neighbors: view('uint32', 8),
                    weights: view('float32', 8),
                    status: {candidateOverflow: view('uint32', 1)}
                  }
                }
              ]
            },
            values: view('float32', 8),
            secondValues: view('float32', 8),
            output: {statistic: view('float32', 1)}
          })
      ).toThrow(/chunks\[0\].*status.*outer/);
    } finally {
      device.destroy();
    }
  });
}

it('Pearson protects finite sums of squares whose product overflows float32', () => {
  // Eight directed edges with x_i = y_neighbor = (i+1)*2**32. Closed form covariance and
  // both square sums are 42*2**64; r=1 exactly. All these sums fit f32, their product does not.
  const squareSum = Math.fround(42 * 2 ** 64);
  const oldDenominator = Math.fround(Math.sqrt(Math.fround(squareSum * squareSum)));
  expect(Math.fround(squareSum / oldDenominator)).toBe(0);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `pearson-units-${serial++}`, format, length);
  try {
    const nodes = new GPUSpatialPearson({
      weights: {
        offsets: view('uint32', 9),
        neighbors: view('uint32', 8),
        weights: view('float32', 8)
      },
      values: view('float32', 8),
      secondValues: view('float32', 8),
      output: {statistic: view('float32', 1)}
    }).getCommandNodes(graph);
    const source = getGPUCommandNodeKernelSources(nodes).find(record =>
      record.id.endsWith('-statistic')
    )!.source;
    expect(
      source.includes('isFiniteFloat(squareProduct)'),
      'overflow must select separate square roots'
    ).toBe(true);
    // Metal can regroup successive floating divisions into the overflowing product again.
    // Bounded frexp significands and integer exponents must carry the fallback's scale.
    expect(source).toContain('getScaledQuotient(numerator, sqrt(squareX), sqrt(squareY), 0)');
    expect(source).toContain('numerator.exp - firstDivisor.exp - secondDivisor.exp');
  } finally {
    device.destroy();
  }
});

it('FLT-05 keeps edge-end means in offset coordinates at a large baseline', () => {
  // Two reversed edges (0,1),(1,0): deviations +/-0.5 give covariance -0.25 and r=-1.
  const roundedMean = Math.fround(Math.fround(10_000_000 + 10_000_001) / 2);
  expect(Math.fround((10_000_000 - roundedMean) * (10_000_001 - roundedMean))).toBe(0);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'translated-values', 'float32', 2);
  const weights = {
    offsets: createTransientView(graph, 'translated-offsets', 'uint32', 3),
    neighbors: createTransientView(graph, 'translated-neighbors', 'uint32', 2),
    weights: createTransientView(graph, 'translated-weights', 'float32', 2)
  };
  try {
    for (const Contributor of [GPUSpatialPearson, GPUGammaStatistic]) {
      const nodes = new Contributor({
        values,
        secondValues: values,
        weights,
        output: {statistic: createTransientView(graph, Contributor.name, 'float32', 1)}
      }).getCommandNodes(graph);
      expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
      for (const kernel of getGPUCommandNodeKernelSources(nodes))
        expect(getStorageBindingCount(kernel.source)).toBeLessThanOrEqual(8);
      const source = getGPUCommandNodeKernelSources(nodes).find(record =>
        record.id.endsWith('-centered')
      )!.source;
      expect(source).toContain('getRoundedDifference(x, anchorX, runtimeZero) - meanX');
      expect(source).toContain('getRoundedDifference(y, anchorY, runtimeZero) - meanY');
      expect(source).toContain('bitcast<u32>(minuend - subtrahend) ^ runtimeZero');
      expect(source).toContain('bitcast<u32>(anchors[anchorsOffset + TOTAL_GROUPS * 4u])');
    }
  } finally {
    device.destroy();
  }
});

it('B1 a distant anchor does not collapse distinct near-zero observations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'float32' | 'uint32'>(id: string, format: Format, length: number) =>
    createTransientView(graph, id, format, length);
  try {
    const nodes = new GPUGammaStatistic({
      weights: {
        offsets: view('offsets', 'uint32', 4),
        neighbors: view('neighbors', 'uint32', 3),
        weights: view('weights', 'float32', 3)
      },
      values: view('values', 'float32', 3),
      secondValues: view('second', 'float32', 3),
      output: {statistic: view('statistic', 'float32', 1)}
    }).getCommandNodes(graph);
    const source = getGPUCommandNodeKernelSources(nodes).find(kernel =>
      kernel.id.endsWith('-centered')
    )!.source;
    // Evaluate the emitted reference selector, then the literal f32 moment arithmetic.
    const selection = source.match(/fn getEdgeCenteringAnchor[^]*?return ([^;]+);/)?.[1];
    const selectAnchor = selection
      ? new Function(
          'minimum',
          'maximum',
          `const select = (no, yes, condition) => condition ? yes : no; return ${selection};`
        )
      : (minimum: number) => minimum;
    const sum = (values: number[]) => {
      const lanes = Array.from({length: 256}, (_, index) => values[index] ?? 0);
      for (let stride = 128; stride >= 1; stride /= 2)
        for (let lane = 0; lane < stride; lane++)
          lanes[lane] = Math.fround(lanes[lane] + lanes[lane + stride]);
      return lanes[0];
    };
    for (const [x, y, expected] of [
      [[-16777216, 0, 1], [0, -3, 3], 1],
      [[10000000, 10000001], [10000001, 10000000], -0.25],
      [[-10000000, -10000001], [-10000001, -10000000], -0.25]
    ] as const) {
      const offsetX = x.map(value =>
        Math.fround(value - selectAnchor(Math.min(...x), Math.max(...x)))
      );
      const offsetY = y.map(value =>
        Math.fround(value - selectAnchor(Math.min(...y), Math.max(...y)))
      );
      const meanX = Math.fround(sum(offsetX) / x.length);
      const meanY = Math.fround(sum(offsetY) / y.length);
      const actual = Math.fround(
        sum(
          offsetX.map((value, row) =>
            Math.fround(Math.fround(value - meanX) * Math.fround(offsetY[row] - meanY))
          )
        ) / x.length
      );
      // Ring y ends have zero mean, so sum(x*y)/3 = 3/3 = 1 exactly.
      expect(Number.isFinite(actual)).toBe(true);
      expect(actual).toBe(expected);
    }
  } finally {
    device.destroy();
  }
});
